"""
    @Author:MaYuhang
    @Time:2026/10/19
    @DESC:金融智库-导入子图状态定义
"""
from typing import TypedDict
import copy
from common.logging.logger import logger

class ImportGraphState(TypedDict):
    """
    金融导入子图的状态定义，包含所有节点产生和消费的数据字段。
    字段结构统一使用 item_name 表示主实体名（产品名/机构名/公司名等），贯穿导入全流程。
    使用字典式访问（如 state["task_id"]、state.get("chunks")）。
    """
    task_id: str

    # --- 流程控制标记 ---
    is_md_read_enabled: bool   # .md / .docx / .doc 输入：内容已就绪，直接切分
    is_pdf_read_enabled: bool  # .pdf 输入：需要先走 MinerU 解析链路

    # --- 路径相关 ---
    local_dir: str          # 输出文件夹地址
    local_file_path: str    # 传入文件地址（.md/.pdf/.docx/.doc）
    file_title: str         # 文件名（如"招商银行2026年第一季度报告"）
    pdf_path: str           # PDF 输入时的源文件地址
    md_path: str            # Markdown 文件地址

    # --- 内容数据 ---
    md_content: str         # MD 文件全文
    chunks: list            # 切分后的切片列表
    item_name: str          # 【语义】主实体名（如"易方达智造优势混合"），字段名沿用 item_name
    doc_metadata: dict      # LLM 抽取的文档级元数据（content_type/product_code/risk_level等）

    # --- 数据库相关 ---
    embeddings_content: list  # 向量化处理后的数据

# 初始状态
graph_default_state: ImportGraphState = {
    "task_id": "",
    "is_pdf_read_enabled": False,
    "is_md_read_enabled": False,
    "local_dir": "",
    "local_file_path": "",
    "pdf_path": "",
    "md_path": "",
    "file_title": "",
    "md_content": "",
    "chunks": [],
    "item_name": "",
    "doc_metadata": {},
    "embeddings_content": [],
}


def create_default_state(**overrides) -> ImportGraphState:
    """
    创建默认状态，支持覆盖。
    """
    state = copy.deepcopy(graph_default_state)
    state.update(overrides)
    return state


def get_default_state() -> ImportGraphState:
    """
    返回一个新的状态实例，避免全局变量污染。
    """
    return copy.deepcopy(graph_default_state)

if __name__ == "__main__":
    """
    测试
    """
    # 创建默认状态
    state = create_default_state(local_file_path="../../doc/金融数据/用户FAQ/投资者问答.pdf")
    logger.info(state)
