"""
    @Author:MaYuhang
    @Time:2026/10/19
    @DESC:金融智库-入口节点（按文件类型分发：.md直读 / .pdf走MinerU / .docx|.doc先转MD）
"""
from pathlib import Path

from common.logging.logger import logger, node_log
from processor.import_processor.state import ImportGraphState, create_default_state
from utils.path_util import PROJECT_ROOT
from utils.docx_to_md_utils import convert_office_to_md
from utils.task_utils import add_running_task, add_done_task


@node_log("node_entry")
def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 入口节点 (node_entry)
    金融项目支持三种输入，按扩展名分发：
    1. .md      -> 直接读取内容，is_md_read_enabled=True，后续直接切分
    2. .pdf     -> 只登记路径，is_pdf_read_enabled=True，由 node_pdf_to_md（MinerU）解析
    3. .docx/.doc -> 先转换为 Markdown 落盘，再按 .md 流程走
    同时设置 local_dir=output/{task_id}/{file_title}，规范后续输出路径，
    避免 JSON 等中间文件污染源文件目录。
    """
    # 1. 添加节点到运行时列表
    add_running_task(state.get("task_id"), "node_entry")

    # 2. 从状态获取 local_file_path
    local_file_path = state.get("local_file_path")

    # 3. local_file_path 判空
    if not local_file_path:
        logger.warning("从状态中没有获取到 local_file_path，导入流程终止")
        return state

    file_path_obj = Path(local_file_path)
    if not file_path_obj.is_file():
        logger.error(f"文件不存在: {local_file_path}，导入流程终止")
        return state

    # 4. 公共字段：file_title + 输出目录 local_dir
    state["file_title"] = file_path_obj.stem
    task_id = state.get("task_id") or "default_task"
    local_dir = str(PROJECT_ROOT / "output" / task_id / state["file_title"])
    local_dir_obj = Path(local_dir)
    if not local_dir_obj.is_dir():
        logger.warning(f"local_dir:{local_dir_obj}不存在，主动创建!")
        local_dir_obj.mkdir(parents=True, exist_ok=True)
    state["local_dir"] = local_dir
    logger.info(f"输出目录 local_dir: {local_dir}")

    # 5. 按扩展名分发
    suffix = file_path_obj.suffix.lower()

    if suffix == ".md":
        # ---- Markdown：直接读取 ----
        state["is_md_read_enabled"] = True
        state["is_pdf_read_enabled"] = False
        state["md_path"] = local_file_path
        state["pdf_path"] = None
        state["md_content"] = file_path_obj.read_text(encoding="utf-8")
        logger.info(f"已读取 MD 文件内容，长度: {len(state['md_content'])} 字符")

    elif suffix == ".pdf":
        # ---- PDF：交给 MinerU 解析链路 ----
        state["is_md_read_enabled"] = False
        state["is_pdf_read_enabled"] = True
        state["pdf_path"] = local_file_path
        state["md_path"] = ""
        state["md_content"] = ""  # 由 node_pdf_to_md / node_md_img 填充
        logger.info(f"检测到 PDF 输入: {file_path_obj.name}，将走 MinerU 解析链路")

    elif suffix in (".docx", ".doc"):
        # ---- Office 文档：先转 Markdown 再走 MD 流程 ----
        state["is_pdf_read_enabled"] = False
        try:
            md_text = convert_office_to_md(file_path_obj)
        except Exception as e:
            logger.error(f"Office 文档转换失败: {e}，导入流程终止")
            return state
        converted_md_path = local_dir_obj / f"{file_path_obj.stem}.md"
        converted_md_path.write_text(md_text, encoding="utf-8")
        state["is_md_read_enabled"] = True
        state["md_path"] = str(converted_md_path)
        state["pdf_path"] = None
        state["md_content"] = md_text
        logger.info(f"Office 文档已转换为 MD: {converted_md_path}，长度: {len(md_text)} 字符")

    else:
        logger.warning(f"不支持的文件类型: {suffix}，金融项目仅支持 .md/.pdf/.docx/.doc")
        return state

    # 6. 将节点添加到已完成列表
    add_done_task(state.get("task_id"), "node_entry")
    return state

if __name__ == '__main__':
    logger.info("===== 开始 node_entry 节点单元测试 =====")

    # 测试1: PDF 文件（走 MinerU 分支）
    test_state1 = create_default_state(
        task_id="test_task_001",
        local_file_path="doc/金融数据/用户FAQ/基金基础知识.pdf"
    )
    node_entry(test_state1)

    # 测试2: docx 文件（走 Office 转换分支）
    test_state2 = create_default_state(
        task_id="test_task_002",
        local_file_path="doc/金融数据/用户FAQ/私募基金投资者教育知识问答.docx"
    )
    node_entry(test_state2)
    logger.info(f"docx 分发结果: is_md={test_state2.get('is_md_read_enabled')}, md_path={test_state2.get('md_path')}")

    # 测试3: 空路径
    test_state3 = create_default_state(
        task_id="test_task_003",
    )
    node_entry(test_state3)

    logger.info("===== 结束 node_entry 节点单元测试 =====")
