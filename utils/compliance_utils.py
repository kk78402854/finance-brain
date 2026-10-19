"""
    @Author:MaYuhang
    @Time:2026/10/19
    @DESC:金融智库-合规护栏工具
    输出侧安全引擎：
    1. 违规表述检测（保证收益/稳赚不赔等禁用词命中时追加合规更正）
    2. 风险话题检测（涉及收益/风险/投资时强制追加风险提示）
    设计原则：只追加、不改写已生成的答案正文，保证流式输出与落库内容一致。
"""
from common.logging.logger import logger

# 禁用词表：金融监管场景下禁止出现的收益承诺类表述
FORBIDDEN_CLAIMS = [
    "保证收益", "保本无风险", "稳赚不赔", "稳赚不亏", "一定赚钱",
    "肯定会赚", "一定会上涨", "必然上涨", "保本保息", "零风险",
    "无风险高收益", "必须买入", "放心梭哈", "闭眼买",
]

# 风险提示触发关键词：问题或答案命中任一关键词即追加风险提示
RISK_TRIGGER_KEYWORDS = [
    "风险", "收益", "亏损", "亏钱", "赚钱", "投资", "理财",
    "基金", "赎回", "申购", "买入", "卖出", "保本", "净值", "回报",
]

# 强制风险提示文本（合规要求：不承诺收益 + 提示风险 + 免责声明）
RISK_DISCLAIMER = (
    "\n\n---\n**风险提示**：金融产品存在风险，历史业绩不代表未来表现，"
    "投资需结合自身风险承受能力。本回答仅为知识库资料查询结果，不构成投资建议；"
    "正式信息以产品合同、招募说明书、公告原文和监管文件为准。"
)

# 违规表述命中后的合规更正文本
def _build_correction_note(hits: list[str]) -> str:
    words = "、".join(f"「{w}」" for w in hits)
    return (
        f"\n\n> **合规说明**：上述回答中出现了{words}等表述，特此说明："
        f"金融产品不存在保证收益，任何收益相关表述请以正式产品文件为准，投资需谨慎。"
    )


def find_forbidden_claims(text: str) -> list[str]:
    """
    检测文本中命中的禁用词列表。
    :param text: 待检测文本（LLM 生成的答案）
    :return: 命中的禁用词列表（去重，保持出现顺序）
    """
    hits = []
    for word in FORBIDDEN_CLAIMS:
        if word in text and word not in hits:
            hits.append(word)
    return hits


def is_risk_related(question: str, answer: str) -> bool:
    """
    判断问题或答案是否涉及风险/收益/投资话题，需要追加风险提示。
    """
    combined = f"{question}{answer}"
    return any(keyword in combined for keyword in RISK_TRIGGER_KEYWORDS)


def apply_compliance_guardrail(answer: str, question: str) -> tuple[str, str]:
    """
    对 LLM 生成的答案施加合规护栏（只追加，不改写正文）。
    :param answer: LLM 生成的原始答案
    :param question: 用户问题（重写后）
    :return: (施加护栏后的完整答案, 追加的合规文本)；无追加时第二项为空字符串
    """
    appended_parts: list[str] = []

    # 1. 禁用词检测：命中则追加合规更正说明
    hits = find_forbidden_claims(answer)
    if hits:
        logger.warning(f"合规护栏：答案中检测到违规表述 {hits}")
        appended_parts.append(_build_correction_note(hits))

    # 2. 风险话题检测：命中则强制追加风险提示
    if is_risk_related(question, answer):
        appended_parts.append(RISK_DISCLAIMER)

    appended_text = "".join(appended_parts)
    if appended_text:
        return answer + appended_text, appended_text
    return answer, ""


if __name__ == "__main__":
    # 单元测试
    test_cases = [
        ("这个基金收益怎么样？", "该基金预期年化收益不错，稳赚不赔，建议加大投入。"),
        ("什么是基金净值？", "基金净值是指每份基金份额的资产净值，计算公式为总资产减去负债后除以份额总数。"),
        ("今天天气如何？", "今天天气晴朗。"),
    ]
    for q, a in test_cases:
        guarded, appended = apply_compliance_guardrail(a, q)
        print(f"\n问题: {q}")
        print(f"追加内容: {appended!r}")
        print(f"最终答案: {guarded}")
