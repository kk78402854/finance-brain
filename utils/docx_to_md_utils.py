"""
    @Author:MaYuhang
    @Time:2026/10/19
    @DESC:金融智库-Office文档转Markdown工具
    支持 .docx（python-docx 解析）和 .doc（Word COM 自动化）两种格式，
    供导入链路入口节点统一转换为 Markdown 后走标准切分流程。
"""
from pathlib import Path

from common.logging.logger import logger

# docx 段落样式名 -> Markdown 前缀 的映射
# 兼容英文（Heading 1）与中文（标题 1）两种 Word 样式命名
_HEADING_STYLE_MAP = {
    "heading 1": "# ", "heading 2": "## ", "heading 3": "### ",
    "heading 4": "#### ", "heading 5": "##### ", "heading 6": "###### ",
    "标题 1": "# ", "标题 2": "## ", "标题 3": "### ",
    "标题 4": "#### ", "标题 5": "##### ", "标题 6": "###### ",
}


def _table_to_md(table) -> str:
    """
    将 python-docx 的 Table 对象转换为 Markdown 管道表格。
    第一行作为表头，空表格返回空字符串。
    """
    rows_text: list[list[str]] = []
    for row in table.rows:
        cells = [cell.text.strip().replace("\n", " ").replace("|", "\\|") for cell in row.cells]
        rows_text.append(cells)
    if not rows_text:
        return ""

    lines = []
    header = rows_text[0]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows_text[1:]:
        # 列数对齐表头（合并单元格可能导致行长度不一致）
        row = row + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _docx_to_md(docx_path: Path) -> str:
    """
    解析 .docx 文件为 Markdown 文本。
    按文档原始顺序遍历段落和表格（不破坏阅读顺序），
    标题样式转 MD 标题，列表样式转 MD 列表，表格转管道表格。
    """
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = Document(str(docx_path))

    md_lines: list[str] = []
    # 遍历 body 的子元素，保证段落与表格按文档真实顺序输出
    for element in document.element.body:
        if element.tag.endswith("}p"):
            paragraph = Paragraph(element, document)
            text = paragraph.text.strip()
            if not text:
                continue
            style_name = (paragraph.style.name or "").strip().lower()
            if style_name in _HEADING_STYLE_MAP:
                md_lines.append(_HEADING_STYLE_MAP[style_name] + text)
            elif style_name.startswith("list bullet"):
                md_lines.append("- " + text)
            elif style_name.startswith("list number"):
                md_lines.append("1. " + text)
            else:
                md_lines.append(text)
        elif element.tag.endswith("}tbl"):
            table_md = _table_to_md(Table(element, document))
            if table_md:
                md_lines.append(table_md)

    return "\n\n".join(md_lines)


def _doc_to_md_via_word(doc_path: Path) -> str:
    """
    通过 Word COM 自动化将 .doc 转为 .docx 后再解析。
    仅 Windows + 本机安装 Word 时可用；失败时抛出带明确指引的异常。
    """
    import pythoncom

    pythoncom.CoInitialize()
    word = None
    temp_docx = doc_path.with_suffix(".tmp_conv.docx")
    try:
        import win32com.client

        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        word.DisplayAlerts = 0
        doc = word.Documents.Open(str(doc_path), ReadOnly=True)
        # FileFormat=16 表示 docx 格式
        doc.SaveAs2(str(temp_docx), FileFormat=16)
        doc.Close(False)
        md_text = _docx_to_md(temp_docx)
        logger.info(f"通过 Word COM 完成 .doc 转换: {doc_path.name}")
        return md_text
    except Exception as e:
        raise RuntimeError(
            f".doc 文件需要本机安装 Word 才能自动转换（{doc_path.name}）：{e}。"
            f"请用 Word 手动另存为 .docx 后重新导入。"
        ) from e
    finally:
        if word is not None:
            try:
                word.Quit()
            except Exception:
                pass
        pythoncom.CoUninitialize()
        if temp_docx.exists():
            try:
                temp_docx.unlink()
            except Exception:
                pass


def convert_office_to_md(file_path) -> str:
    """
    统一入口：将 .docx/.doc 文件转换为 Markdown 文本。
    :param file_path: Office 文件路径（str 或 Path）
    :return: Markdown 文本
    """
    path_obj = Path(file_path)
    suffix = path_obj.suffix.lower()

    if suffix == ".docx":
        md_text = _docx_to_md(path_obj)
    elif suffix == ".doc":
        md_text = _doc_to_md_via_word(path_obj)
    else:
        raise ValueError(f"不支持的 Office 文件类型: {path_obj}")

    if not md_text.strip():
        raise ValueError(f"文档内容为空，无法转换: {path_obj}")
    logger.info(f"Office 文档转换完成: {path_obj.name}，MD 长度 {len(md_text)} 字符")
    return md_text


if __name__ == "__main__":
    # 单元测试：转换指定 docx 文件
    import sys

    if len(sys.argv) > 1:
        result = convert_office_to_md(sys.argv[1])
        print(result[:500])
        print(f"\n... 共 {len(result)} 字符")
    else:
        print("用法: python docx_to_md_utils.py <docx或doc文件路径>")
