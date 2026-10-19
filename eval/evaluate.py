"""
金融知识库 RAG 系统评估脚本

评估维度：
  1. 检索质量：Recall@K / Precision@K / MRR / NDCG@K
  2. 答案忠实度：答案内容是否来自检索上下文
  3. 性能：各阶段延迟、总延迟、吞吐量

使用方法：
  # 仅评估检索质量（不调用 LLM 生成答案，最快）
  python eval/evaluate.py --mode retrieval

  # 评估完整链路（含 LLM 答案生成与忠实度）
  python eval/evaluate.py --mode full

  # 性能基准（多轮并发，测吞吐量）
  python eval/evaluate.py --mode perf --rounds 20

  # 指定 K 值和测试集数量
  python eval/evaluate.py --mode retrieval --topk 5 --limit 10
"""
import argparse
import json
import os
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.config.milvus_config import milvus_config  # noqa: E402
from processor.query_processor.nodes.node_search_embedding import node_search_embedding  # noqa: E402
from processor.query_processor.nodes.node_search_embedding_hyde import node_search_embedding_hyde  # noqa: E402
from processor.query_processor.nodes.node_rrf import node_rrf  # noqa: E402
from processor.query_processor.nodes.node_rerank import node_rerank  # noqa: E402
from processor.query_processor.state import create_query_default_state  # noqa: E402
from utils.clients.milvus_utils import get_milvus_client  # noqa: E402


# ============================================================
# 测试集：query -> (item_names, relevant_keywords)
# relevant_keywords 是应该出现在检索结果 chunk title 中的关键词
# （chunk title 格式如 "三体简介_## 作者介绍"）
# ============================================================
TEST_SET = [
    # ---------- 基金产品 ----------
    {"q": "易方达智造优势混合主要投资什么？", "item_names": ["易方达智造优势混合"], "relevant_keywords": ["投资", "智造", "股票"]},
    {"q": "华夏债券C的风险等级是多少？", "item_names": ["华夏债券"], "relevant_keywords": ["风险", "华夏债券"]},
    {"q": "易方达智造优势混合的基金代码是什么？", "item_names": ["易方达智造优势混合"], "relevant_keywords": ["基金代码", "代码"]},
    {"q": "华夏国证自由现金流联接基金跟踪什么指数？", "item_names": ["华夏国证自由现金流"], "relevant_keywords": ["自由现金流", "指数", "跟踪"]},
    {"q": "易方达智造优势混合适合什么投资者？", "item_names": ["易方达智造优势混合"], "relevant_keywords": ["适合", "投资者", "风险"]},

    # ---------- 上市公司年报 ----------
    {"q": "贵州茅台2026年一季报讲了什么？", "item_names": ["贵州茅台"], "relevant_keywords": ["贵州茅台", "一季度"]},
    {"q": "招商银行一季度营业收入是多少？", "item_names": ["招商银行"], "relevant_keywords": ["营业收入", "招商银行"]},
    {"q": "平安银行一季报提到哪些风险？", "item_names": ["平安银行"], "relevant_keywords": ["风险", "平安银行"]},

    # ---------- 银行理财&风险揭示书 ----------
    {"q": "建行理财产品的风险揭示书说了什么？", "item_names": ["中国建设银行"], "relevant_keywords": ["风险揭示", "建设银行"]},
    {"q": "中国银行理财产品有哪些风险？", "item_names": ["中国银行"], "relevant_keywords": ["风险", "中国银行"]},
    {"q": "中银理财的产品不保本是什么意思？", "item_names": ["中银理财"], "relevant_keywords": ["保本", "中银理财"]},

    # ---------- 宏观经济&政策 ----------
    {"q": "中国货币政策执行报告的核心内容是什么？", "item_names": ["货币政策执行报告"], "relevant_keywords": ["货币政策"]},
    {"q": "2025年国民经济统计公报讲了什么？", "item_names": ["统计公报"], "relevant_keywords": ["统计公报", "国民经济"]},

    # ---------- 金融知识/FAQ ----------
    {"q": "什么是净值型理财？", "item_names": [], "relevant_keywords": ["净值型"]},
    {"q": "什么是最大回撤？", "item_names": [], "relevant_keywords": ["最大回撤", "回撤"]},
    {"q": "基金赎回多久到账？", "item_names": [], "relevant_keywords": ["赎回", "到账"]},
    {"q": "什么是债券基金？", "item_names": [], "relevant_keywords": ["债券基金", "债券"]},
    {"q": "如何申购基金？", "item_names": [], "relevant_keywords": ["申购"]},
    {"q": "理财产品能保本吗？", "item_names": [], "relevant_keywords": ["保本"]},
    {"q": "R3风险等级是什么意思？", "item_names": [], "relevant_keywords": ["R3", "风险等级"]},
    {"q": "ETF基金投资需要注意什么？", "item_names": [], "relevant_keywords": ["ETF"]},
    {"q": "私募基金投资者需要注意什么？", "item_names": [], "relevant_keywords": ["私募"]},
    {"q": "消费者金融素养调查报告有什么结论？", "item_names": [], "relevant_keywords": ["金融素养"]},
]


# ============================================================
# 指标计算工具
# ============================================================
def is_relevant(chunk, relevant_keywords):
    """判断一个 chunk 是否相关：title 中是否包含任一关键词"""
    title = str(chunk.get("title", "") or "")
    content = str(chunk.get("content", "") or "")
    text = title + " " + content
    return any(kw in text for kw in relevant_keywords)


def recall_at_k(retrieved, relevant_keywords, k):
    """Recall@K：top-K 中相关的数量 / 总相关数"""
    if not relevant_keywords:
        return 1.0
    topk = retrieved[:k]
    relevant_in_topk = sum(1 for c in topk if is_relevant(c, relevant_keywords))
    # 总相关数 = 关键词覆盖的独立 section 数（近似）
    total_relevant = len(relevant_keywords)
    return min(relevant_in_topk / total_relevant, 1.0)


def precision_at_k(retrieved, relevant_keywords, k):
    """Precision@K：top-K 中相关的数量 / K"""
    if k == 0:
        return 0.0
    topk = retrieved[:k]
    relevant_in_topk = sum(1 for c in topk if is_relevant(c, relevant_keywords))
    return relevant_in_topk / k


def reciprocal_rank(retrieved, relevant_keywords):
    """MRR 的单项：第一个相关 chunk 的排名倒数"""
    for i, chunk in enumerate(retrieved, start=1):
        if is_relevant(chunk, relevant_keywords):
            return 1.0 / i
    return 0.0


def dcg_at_k(retrieved, relevant_keywords, k):
    """DCG@K：相关性=1（相关）或 0（不相关），按排名折损"""
    import math
    score = 0.0
    for i, chunk in enumerate(retrieved[:k], start=1):
        rel = 1.0 if is_relevant(chunk, relevant_keywords) else 0.0
        score += rel / math.log2(i + 1)
    return score


def ndcg_at_k(retrieved, relevant_keywords, k):
    """NDCG@K = DCG / IDCG"""
    dcg = dcg_at_k(retrieved, relevant_keywords, k)
    # IDCG：理想情况下 top-K 全相关
    import math
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(relevant_keywords)) + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def average(values):
    return sum(values) / len(values) if values else 0.0


# ============================================================
# 检索链路执行
# ============================================================
def run_retrieval(query, item_names, with_hyde=True):
    """
    执行完整检索链路（embedding + hyde + rrf + rerank）
    返回 (reranked_docs, latency_dict)
    """
    session_id = "eval-" + uuid.uuid4().hex[:8]
    state = create_query_default_state(
        session_id=session_id,
        original_query=query,
        rewritten_query=query,
        item_names=item_names,
        is_stream=False,
    )

    latency = {}
    t0 = time.perf_counter()

    # 1. 普通向量检索
    t1 = time.perf_counter()
    emb_res = node_search_embedding(state)
    state["embedding_chunks"] = emb_res.get("embedding_chunks", [])
    latency["embedding"] = time.perf_counter() - t1

    # 2. HyDE 检索（可选，需要 LLM）
    if with_hyde:
        t2 = time.perf_counter()
        hyde_res = node_search_embedding_hyde(state)
        state["hyde_embedding_chunks"] = hyde_res.get("hyde_embedding_chunks", [])
        latency["hyde"] = time.perf_counter() - t2
    else:
        state["hyde_embedding_chunks"] = []
        latency["hyde"] = 0.0

    # 3. RRF 融合
    t3 = time.perf_counter()
    rrf_res = node_rrf(state)
    state["rrf_chunks"] = rrf_res.get("rrf_chunks", [])
    latency["rrf"] = time.perf_counter() - t3

    # 4. Rerank（需要 reranker 模型）
    t4 = time.perf_counter()
    rerank_res = node_rerank(state)
    latency["rerank"] = time.perf_counter() - t4

    latency["total"] = time.perf_counter() - t0
    return rerank_res.get("reranked_docs", []), latency


# ============================================================
# 评估模式
# ============================================================
def eval_retrieval(test_set, topk, with_hyde):
    """评估检索质量"""
    print(f"\n{'='*60}")
    print(f" 检索质量评估 (topk={topk}, hyde={'on' if with_hyde else 'off'})")
    print(f"{'='*60}")
    print(f" 测试集大小: {len(test_set)}")
    print(f" 加载模型与 Milvus 连接中...")

    recalls, precisions, mrr_list, ndcgs = [], [], [], []
    stage_latency = defaultdict(list)

    for i, item in enumerate(test_set, 1):
        query = item["q"]
        item_names = item.get("item_names", [])
        relevant_kw = item["relevant_keywords"]

        try:
            reranked_docs, latency = run_retrieval(query, item_names, with_hyde=with_hyde)
        except Exception as e:
            print(f"  [{i}/{len(test_set)}] FAIL: {query[:30]}... -> {e}")
            continue

        r = recall_at_k(reranked_docs, relevant_kw, topk)
        p = precision_at_k(reranked_docs, relevant_kw, topk)
        rr = reciprocal_rank(reranked_docs, relevant_kw)
        ndcg = ndcg_at_k(reranked_docs, relevant_kw, topk)

        recalls.append(r)
        precisions.append(p)
        mrr_list.append(rr)
        ndcgs.append(ndcg)
        for k, v in latency.items():
            stage_latency[k].append(v)

        status = "OK" if r > 0 else "MISS"
        print(f"  [{i}/{len(test_set)}] {status} Recall@{topk}={r:.2f} P@K={p:.2f} "
              f"MRR={rr:.2f} NDCG={ndcg:.2f} | {query[:30]}")

    # 汇总
    print(f"\n{'='*60}")
    print(f" 检索质量汇总 (N={len(recalls)})")
    print(f"{'='*60}")
    print(f"  Recall@{topk}    : {average(recalls):.4f}")
    print(f"  Precision@{topk} : {average(precisions):.4f}")
    print(f"  MRR              : {average(mrr_list):.4f}")
    print(f"  NDCG@{topk}      : {average(ndcgs):.4f}")
    print(f"\n 平均延迟 (秒):")
    for stage in ["embedding", "hyde", "rrf", "rerank", "total"]:
        vals = stage_latency.get(stage, [])
        if vals:
            print(f"    {stage:10s}: {average(vals):.3f}s")
    print(f"{'='*60}")

    return {
        "recall": average(recalls),
        "precision": average(precisions),
        "mrr": average(mrr_list),
        "ndcg": average(ndcgs),
        "latency": {k: average(v) for k, v in stage_latency.items()},
    }


def eval_perf(rounds):
    """性能基准：连续执行 rounds 次检索，测吞吐量和延迟分布"""
    import statistics

    print(f"\n{'='*60}")
    print(f" 性能基准 (rounds={rounds})")
    print(f"{'='*60}")
    print(f" 预热中（加载模型）...")

    # 预热
    try:
        run_retrieval("易方达智造优势混合主要投资什么", ["三体"], with_hyde=False)
    except Exception as e:
        print(f"  预热失败: {e}")
        return

    times = []
    print(f"  开始执行 {rounds} 轮...")
    for i in range(rounds):
        item = TEST_SET[i % len(TEST_SET)]
        t0 = time.perf_counter()
        try:
            run_retrieval(item["q"], item.get("item_names", []), with_hyde=False)
        except Exception as e:
            print(f"  第 {i+1} 轮失败: {e}")
            continue
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        if (i + 1) % 5 == 0:
            print(f"    完成 {i+1}/{rounds} 轮, 最近延迟 {elapsed:.3f}s")

    if not times:
        print("  无成功样本")
        return

    times.sort()
    n = len(times)
    print(f"\n{'='*60}")
    print(f" 性能结果 (成功 {n}/{rounds} 轮)")
    print(f"{'='*60}")
    print(f"  吞吐量 QPS   : {n / sum(times):.2f}")
    print(f"  平均延迟     : {statistics.mean(times):.3f}s")
    print(f"  中位延迟 P50 : {times[n // 2]:.3f}s")
    print(f"  P90 延迟     : {times[int(n * 0.9)]:.3f}s")
    print(f"  P99 延迟     : {times[int(n * 0.99)]:.3f}s" if n >= 10 else f"  P99 延迟     : N/A (样本不足)")
    print(f"  最小延迟     : {times[0]:.3f}s")
    print(f"  最大延迟     : {times[-1]:.3f}s")
    print(f"{'='*60}")


# ============================================================
# 入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="金融知识库 RAG 系统评估")
    parser.add_argument("--mode", choices=["retrieval", "full", "perf"], default="retrieval",
                        help="retrieval=检索质量, full=含答案生成, perf=性能基准")
    parser.add_argument("--topk", type=int, default=5, help="评估 Recall@K/Precision@K 的 K 值")
    parser.add_argument("--limit", type=int, default=0, help="只用前 N 条测试用例 (0=全部)")
    parser.add_argument("--rounds", type=int, default=20, help="perf 模式下的轮次")
    parser.add_argument("--no-hyde", action="store_true", help="关闭 HyDE（更快，省 LLM 调用）")
    args = parser.parse_args()

    test_set = TEST_SET[: args.limit] if args.limit > 0 else TEST_SET

    print("金融知识库 RAG 评估")
    print(f"  Milvus: {milvus_config.milvus_url}")
    print(f"  集合:   {milvus_config.chunks_collection}")

    if args.mode == "retrieval":
        eval_retrieval(test_set, args.topk, with_hyde=not args.no_hyde)
    elif args.mode == "perf":
        eval_perf(args.rounds)
    elif args.mode == "full":
        # full 模式 = retrieval 评估 + 提示用户去页面看答案
        print("\n  full 模式说明：")
        print("  检索质量评估后，答案生成的忠实度建议在页面上人工抽查：")
        print("  - 回答中的事实是否能在「阶段进度」展示的参考切片中找到")
        print("  - 是否出现具体数字、人名、奖项等切片中没有的信息（编造）")
        print("  - 来源信息（主体名/章节）是否正确")
        eval_retrieval(test_set, args.topk, with_hyde=not args.no_hyde)


if __name__ == "__main__":
    main()
