"""
金融知识库 RAGAS 评估脚本：五大核心指标全链路评估

五大指标（RAGAS，LLM-as-a-Judge + Embedding）：
  1. faithfulness        忠实度      —— 答案是否基于检索上下文（低分=编造）
  2. answer_relevancy    答案相关性  —— 答案是否回答了问题（低分=跑题）
  3. context_precision   上下文精确率 —— 检索结果中有用内容占比（低分=噪声大）
  4. context_recall      上下文召回率 —— 证据是否拿全（低分=检索没找全）
  5. answer_correctness  答案正确性  —— 与参考答案对齐程度（低分=表达/事实偏差）

流程六步：初始化组件 → 加载 qa.csv → 跑 query_app 全链路 → 构建数据集
         → RAGAS 评估 → 导出结果 + 筛 Bad Case

用法：
  python eval/evaluate_ragas.py                       # 默认 eval/eval_doc/qa.csv
  python eval/evaluate_ragas.py --limit 10            # 只评估前 10 条（evaluate 吃内存，先小跑）
  python eval/evaluate_ragas.py --threshold 0.5       # Bad Case 阈值
"""
import argparse
import re
import sys
import types
import uuid
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# --- 兼容垫片 -------------------------------------------------------------
# ragas 0.4.3 无条件导入 langchain_community.chat_models.vertexai，
# 而该模块已在 langchain-community 0.4 中被移除（拆到 langchain-google-vertexai）。
# 注入占位模块让 import ragas 通过：评估走 OpenAI 兼容路径，VertexAI 永不被使用。
# 未来升级 ragas 修复后，此 try/except 会自动变成空操作。
try:
    import langchain_community.chat_models.vertexai  # noqa: F401,E402
except ModuleNotFoundError:
    _shim = types.ModuleType("langchain_community.chat_models.vertexai")

    class _ChatVertexAI:  # 占位类
        pass

    _shim.ChatVertexAI = _ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _shim
# --------------------------------------------------------------------------

from datasets import Dataset  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402
from ragas import evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402  # ragas 0.4 起模块名为 ragas.llms（复数）
from ragas.metrics import (  # noqa: E402  # 0.4 提示未来迁往 metrics.collections，当前路径可用
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

# 注意：评估脚本不引入项目 logger（loguru 文件 sink 在后台任务环境下句柄异常，
# 且评估是独立工具，print 到 stdout 即可）
from processor.query_processor.main_graph import query_app  # noqa: E402
from processor.query_processor.state import create_query_default_state  # noqa: E402
from utils.lm.lm_utils import get_llm_client  # noqa: E402
from utils.lm.embedding_utils import generate_embeddings  # noqa: E402

EVAL_DOC_DIR = Path(__file__).resolve().parent / "eval_doc"
DEFAULT_CSV = EVAL_DOC_DIR / "qa.csv"


# ============================================================
# Step 1: 初始化评估组件
# ============================================================
class FinanceBGEEmbeddings(Embeddings):
    """把项目的 BGE-M3 向量服务适配成 LangChain Embeddings 接口。

    只取 dense：sparse 是检索阶段 hybrid search（Milvus 双路）用的，
    RAGAS 只评最终 contexts 文本的质量，交 dense 即可不失真。
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return generate_embeddings(texts)["dense"]

    def embed_query(self, text: str) -> List[float]:
        dense = self.embed_documents([text])
        return dense[0] if dense else []


def step_1_init_eval_components():
    llm = LangchainLLMWrapper(get_llm_client())
    embeddings = LangchainEmbeddingsWrapper(FinanceBGEEmbeddings())
    metrics = [
        faithfulness,        # 忠实度：答案是否基于检索到的上下文
        answer_relevancy,    # 答案相关性：答案是否回答了问题
        context_precision,   # 上下文精确率：相关文档的排名情况
        context_recall,      # 上下文召回率：是否检索到所有相关信息
        answer_correctness,  # 答案正确性：与参考答案的匹配程度
    ]
    return llm, embeddings, metrics


# ============================================================
# Step 2: 从 CSV 加载测试样本（answer 列映射为 ground_truth）
# ============================================================
def step_2_load_test_samples_from_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    samples = []
    for _, row in df.iterrows():
        q, a = str(row["question"]).strip(), str(row["answer"]).strip()
        if q and q != "nan" and a and a != "nan":
            samples.append({"question": q, "ground_truth": a})
    return samples


# ============================================================
# Step 3: 跑 RAG 全链路，采集 answer 与 contexts
# ============================================================
def strip_image_suffix_for_eval(text: str) -> str:
    """去掉答案中的 markdown 图片标记与 <ctx-meta> 元信息块，避免干扰 LLM 裁判与 embedding。
    （后端 node_answer_output 已做正文/元信息分离，此处为双保险——应对提示词未被执行的输出）"""
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"<ctx-meta>.*?</ctx-meta>", "", text, flags=re.DOTALL)
    return text.strip()


def step_3_run_pipeline(questions):
    answers, contexts = [], []
    for i, question in enumerate(questions, 1):
        # eval 前缀 session 隔离生产会话；单轮、关闭流式
        state = create_query_default_state(
            session_id=f"eval-{uuid.uuid4().hex[:8]}",
            original_query=question,
            rewritten_query=question,
            item_names=[],
            is_stream=False,
        )
        try:
            result = query_app.invoke(state)
            ans = result.get("answer", "")
            # contexts 取 rerank 之后的最终文档（RAGAS 要求 list[str]）
            # 注意：文档正文字段是 text（与 node_answer_output 的 step_4 一致）
            docs = [d.get("text", "") for d in result.get("reranked_docs", [])]
        except Exception as e:
            print(f"[{i}/{len(questions)}] 流水线失败：{question[:30]} -> {e}")
            ans, docs = "", []

        answers.append(strip_image_suffix_for_eval(ans))
        contexts.append([t for t in docs if t])
        print(f"  [{i}/{len(questions)}] answer={len(ans)}字 contexts={len(docs)}条 | {question[:30]}")

    return {"answer": answers, "contexts": contexts}


# ============================================================
# Step 4: 构建评估数据集（四要素：question/answer/contexts/ground_truth）
# ============================================================
def step_4_build_dataset(test_samples, pipeline_results):
    return Dataset.from_dict({
        "question": [item["question"] for item in test_samples],
        "answer": pipeline_results["answer"],
        "contexts": pipeline_results["contexts"],
        "ground_truth": [item["ground_truth"] for item in test_samples],  # str，非嵌套列表
    })


# ============================================================
# Step 5: 执行 RAGAS 评估
# ============================================================
def step_5_run_ragas_evaluation(dataset, metrics, llm, embeddings):
    return evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,  # 单条失败记 NaN，不中断整体（NaN 会进 Bad Case）
    )


# ============================================================
# Step 6: 导出结果 + 筛 Bad Cases（answer_correctness < 阈值 或 NaN）
# ============================================================
def safe_to_csv(df, path):
    """CSV 被占用（Excel/WPS 打开时的 Windows 文件锁）时改写 _new 后缀文件，避免整轮评估白跑"""
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
    except PermissionError:
        alt = path.with_name(path.stem + "_new" + path.suffix)
        print(f"[警告] {path.name} 被占用（请关闭 Excel/WPS 后重跑以覆盖），本次结果改写入 {alt.name}")
        df.to_csv(alt, index=False, encoding="utf-8-sig")


def step_6_export_results_and_bad_cases(results, metrics, threshold: float):
    df = results.to_pandas()
    metric_names = [m.name for m in metrics]

    # 末尾追加平均值行，CSV 里直接看整体表现
    avg_row = {col: "" for col in df.columns}
    for m in metric_names:
        if m in df.columns:
            avg_row[m] = pd.to_numeric(df[m], errors="coerce").mean()
    # 兜底：先把完整分数打到 stdout —— 即使 CSV 写盘失败，结果也能从日志恢复
    metric_cols = [m for m in metric_names if m in df.columns]
    print("\n===== 逐条分数 =====")
    print(df[metric_cols].to_string())
    print("\n===== 平均分 =====")
    for m in metric_cols:
        print(f"  {m}: {avg_row[m]:.4f}")

    # 文件名带时间戳：每轮评估独立存档，天然保留优化前后历史版本，也避免 Excel 锁定固定名文件
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    safe_to_csv(pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True),
                EVAL_DOC_DIR / f"eval_results_{ts}.csv")

    bad_case_metric = "answer_correctness"
    if bad_case_metric in df.columns:
        scores = pd.to_numeric(df[bad_case_metric], errors="coerce")
        bad_df = df[scores.lt(threshold) | scores.isna()]
        safe_to_csv(bad_df, EVAL_DOC_DIR / f"bad_cases_{ts}.csv")

    return df, avg_row


# ============================================================
# 入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="金融知识库 RAGAS 五大指标评估")
    parser.add_argument("--csv", type=str, default=str(DEFAULT_CSV), help="评估集 qa.csv 路径")
    parser.add_argument("--limit", type=int, default=0, help="只评估前 N 条（0=全部；evaluate 吃内存，建议先小跑）")
    parser.add_argument("--threshold", type=float, default=0.5, help="Bad Case 判定阈值（answer_correctness 低于此值）")
    args = parser.parse_args()

    # Step 1: 初始化评估组件
    llm, embeddings, metrics = step_1_init_eval_components()

    # Step 2: 加载测试数据
    samples = step_2_load_test_samples_from_csv(args.csv)
    if args.limit > 0:
        samples = samples[: args.limit]
    if not samples:
        print(f"评估集为空：{args.csv}")
        sys.exit(1)
    print(f"加载评估集：{len(samples)} 条\n")

    # Step 3: 运行 RAG pipeline
    print("运行 RAG 全链路（检索→重排→生成）...")
    pipeline_results = step_3_run_pipeline([s["question"] for s in samples])

    # Step 4: 构建评估数据集
    dataset = step_4_build_dataset(samples, pipeline_results)

    # Step 5: 执行 RAGAS 评估
    print("\n执行 RAGAS 五大指标评估（LLM-as-a-Judge，需要一些时间）...")
    results = step_5_run_ragas_evaluation(dataset, metrics, llm, embeddings)

    # Step 6: 导出结果与 Bad Case
    df, avg_row = step_6_export_results_and_bad_cases(results, metrics, args.threshold)

    print(f"\n{'='*60}")
    print(f" 五大指标平均分（N={len(df)}）")
    print(f"{'='*60}")
    for m in [x.name for x in metrics]:
        if m in avg_row and avg_row[m] != "":
            print(f"  {m:20s}: {avg_row[m]:.4f}")
    print(f"\n结果文件：")
    print(f"  {EVAL_DOC_DIR / 'eval_results.csv'}")
    print(f"  {EVAL_DOC_DIR / 'bad_cases.csv'}（answer_correctness < {args.threshold} 或 NaN）")
    print(f"\n低分定位口诀：Precision 低查噪声 | Recall 低查召回 | Faithfulness 低查编造")
    print(f"四高但 correctness 低 → 检索没问题，是回答罗嗦/表达问题 → 改生成提示词")


if __name__ == "__main__":
    main()
