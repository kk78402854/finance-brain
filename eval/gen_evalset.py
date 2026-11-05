"""
评估集生成脚本：从 eval/eval_doc/ 下的知识库 Markdown 文档，
用 LLM 批量生成「question + answer」QA 对，写入 qa.csv 供 RAGAS 评估使用。

用法：
  # 1. 把知识库的原始文档(.md)放入 eval/eval_doc/ 目录
  #    （PDF 可用导入链路的 MinerU 产物，或任何已转 MD 的文档）
  # 2. 生成评估集（默认每篇文档 6 条 QA）
  python eval/gen_evalset.py

  # 自定义每篇问题数 / 文档截断长度 / 输出路径
  python eval/gen_evalset.py --num 6 --max-chars 6000 --out eval/eval_doc/qa.csv

输出 CSV 结构（两列，utf-8-sig，Excel 直接打开不乱码）：
  question,answer
  "易方达智造优势混合：该基金主要投资什么行业？","该基金主要投资智能制造相关行业的股票..."
"""
import argparse
import json
import re
import sys
import uuid
from pathlib import Path

import pandas as pd

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.lm.lm_utils import get_llm_client  # noqa: E402
from common.logging.logger import logger  # noqa: E402

# 文档目录与默认输出（均在 eval/eval_doc/ 下）
DOC_DIR = Path(__file__).resolve().parent / "eval_doc"
DEFAULT_OUT = DOC_DIR / "qa.csv"

# 生成 QA 的提示词：约束问题格式（主体名前缀）与答案来源（严格取自文档）
GEN_PROMPT = """你是一个金融知识库的评测集构造专家。请根据下面的文档内容，生成 {num} 条高质量的「问题 + 参考答案」对，用于评估 RAG 问答系统。

要求：
1. question 必须以「主体名：」开头，主体名取文档中的产品名/机构名/报告名（如「易方达智造优势混合：该基金主要投资什么？」「HAK180：为什么不要将HAK180烫金机放在化学品附近？」）。
2. answer 必须严格来自文档中明确出现的事实，禁止编造、禁止扩写；简洁准确，2~4 句以内。
3. 问题覆盖文档的不同侧面：基本信息、风险、规则、操作/流程、数字类事实各至少 1 条（文档没有的侧面跳过）。
4. 数字类事实问题（代码、费率、日期、阈值）必须保留原文的准确数值。

输出 JSON 数组，不要输出任何其他内容：
[{{"question": "...", "answer": "..."}}, ...]

文档内容：
{doc}"""


def extract_json(text: str):
    """从 LLM 输出中提取 JSON 数组（容错：剥掉 ```json 代码块标记）"""
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group())
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def generate_qa_for_doc(md_path: Path, num: int, max_chars: int):
    """对单篇文档调用 LLM 生成 QA 对"""
    content = md_path.read_text(encoding="utf-8")[:max_chars]
    if not content.strip():
        logger.warning(f"文档为空，跳过：{md_path.name}")
        return []

    client = get_llm_client(json_mode=True)
    prompt = GEN_PROMPT.format(num=num, doc=content)
    resp = client.invoke(prompt)
    qa_list = extract_json(resp.content if hasattr(resp, "content") else str(resp))

    valid = []
    for item in qa_list:
        q = str(item.get("question", "")).strip()
        a = str(item.get("answer", "")).strip()
        if q and a:
            valid.append({"question": q, "answer": a})
    logger.info(f"{md_path.name}: 生成 {len(valid)}/{num} 条有效 QA")
    return valid


def main():
    parser = argparse.ArgumentParser(description="从知识库 MD 文档生成 RAGAS 评估集")
    parser.add_argument("--num", type=int, default=6, help="每篇文档生成的问题数（默认 6）")
    parser.add_argument("--max-chars", type=int, default=6000,
                        help="每篇文档参与生成的最大字符数（默认 6000，超长截断）")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="输出 CSV 路径")
    args = parser.parse_args()

    md_files = sorted(DOC_DIR.glob("*.md"))
    if not md_files:
        print(f"未找到文档：请先把知识库 .md 文档放入 {DOC_DIR}")
        sys.exit(1)

    print(f"发现 {len(md_files)} 篇文档，开始生成评估集（每篇 {args.num} 条）...")
    all_qa = []
    for i, md_path in enumerate(md_files, 1):
        print(f"  [{i}/{len(md_files)}] {md_path.name}")
        try:
            all_qa.extend(generate_qa_for_doc(md_path, args.num, args.max_chars))
        except Exception as e:
            logger.error(f"{md_path.name} 生成失败：{e}")

    # 按 question 去重（不同文档可能生成重复问题）
    seen, deduped = set(), []
    for item in all_qa:
        if item["question"] not in seen:
            seen.add(item["question"])
            deduped.append(item)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(deduped, columns=["question", "answer"]).to_csv(
        out_path, index=False, encoding="utf-8-sig")

    print(f"\n完成：共 {len(deduped)} 条（去重前 {len(all_qa)}）")
    print(f"输出：{out_path}")
    print("下一步：python eval/evaluate_ragas.py --csv " + str(out_path))


if __name__ == "__main__":
    main()
