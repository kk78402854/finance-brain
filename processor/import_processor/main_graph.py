"""
    @Author:MaYuhang
    @Time:2026/10/19
    @DESC:金融智库-导入子图（7节点：入口分发→[PDF链路]→切分→主体识别→向量化→入库）
"""
import json
from langgraph.graph import StateGraph, END, START

from common.logging.logger import logger
from dotenv import load_dotenv
from processor.import_processor.state import ImportGraphState, create_default_state
from processor.import_processor.nodes.node_entry import node_entry
from processor.import_processor.nodes.node_pdf_to_md import node_pdf_to_md
from processor.import_processor.nodes.node_md_img import node_md_img
from processor.import_processor.nodes.node_document_split import node_document_split
from processor.import_processor.nodes.node_item_name_recognition import node_item_name_recognition
from processor.import_processor.nodes.node_bge_embedding import node_bge_embedding
from processor.import_processor.nodes.node_import_milvus import node_import_milvus

load_dotenv()

# ===================== 1. 初始化LangGraph状态图 =====================
workflow = StateGraph(ImportGraphState)

# ===================== 2. 注册所有业务节点 =====================

"""
金融数据包含 Markdown、PDF（季报/基金概要/风险揭示书）、docx/doc（FAQ）三类，
导入流水线共 7 个节点：

start → node_entry → (条件分发)
                          ├─ .md/.docx/.doc → node_document_split
                          └─ .pdf → node_pdf_to_md → node_md_img → node_document_split
       → node_item_name_recognition → node_bge_embedding → node_import_milvus → end

1.node_entry：入口分发（.md直读 / .pdf登记待解析 / .docx|.doc先转MD）
2.node_pdf_to_md：MinerU 云端解析 PDF 为 Markdown（保留表格结构）
3.node_md_img：图片处理（VLM生成摘要 + MinIO上传换URL）
4.node_document_split：智能文档切片（按 Markdown 标题切分）
5.node_item_name_recognition：主体识别（LLM抽取主实体名+金融元数据）
6.node_bge_embedding：混合向量化（BGE-M3 稠密+稀疏）
7.node_import_milvus：数据持久化（写入 Milvus）
"""

workflow.add_node("node_entry", node_entry)
workflow.add_node("node_pdf_to_md", node_pdf_to_md)
workflow.add_node("node_md_img", node_md_img)
workflow.add_node("node_document_split", node_document_split)
workflow.add_node("node_item_name_recognition", node_item_name_recognition)
workflow.add_node("node_bge_embedding", node_bge_embedding)
workflow.add_node("node_import_milvus", node_import_milvus)

# ===================== 3. 设置工作流入口节点 =====================
workflow.set_entry_point("node_entry")

# ===================== 4. 定义条件路由函数（入口节点后的分支逻辑） =====================
# PDF 走 MinerU 解析链路，MD/Office 直接切分，其余（不支持的类型）直接结束
def route_after_entry(state: ImportGraphState):
    if state.get("is_pdf_read_enabled"):
        return "node_pdf_to_md"
    elif state.get("is_md_read_enabled"):
        return "node_document_split"
    else:
        return END

# 唯一的条件边
workflow.add_conditional_edges(
    "node_entry",
    route_after_entry,
    {
        "node_pdf_to_md": "node_pdf_to_md",
        "node_document_split": "node_document_split",
        END: END
    }
)

# ===================== 5. 注册静态顺序边 =====================
# PDF 解析链路：MinerU 转MD -> 图片处理（无图自动跳过）
workflow.add_edge("node_pdf_to_md", "node_md_img")
workflow.add_edge("node_md_img", "node_document_split")
# 公共链路：切分 -> 主体识别 -> 向量化 -> 入库
workflow.add_edge("node_document_split", "node_item_name_recognition")
workflow.add_edge("node_item_name_recognition", "node_bge_embedding")
workflow.add_edge("node_bge_embedding", "node_import_milvus")
workflow.add_edge("node_import_milvus", END)

# ===================== 6. 编译工作流为可执行对象 =====================
kb_import_app = workflow.compile()

# ***************************************************************************************************

if __name__ == "__main__":
    logger.info("===== 开始测试金融导入子图 =====")

    initial_state = create_default_state(local_file_path="../../doc/金融数据/用户FAQ/投资者问答.pdf")
    final_state = kb_import_app.invoke(initial_state)

    logger.info(f"最终状态: {json.dumps(final_state, indent=4, ensure_ascii=False, default=str)}")

    logger.info("图结构:")
    kb_import_app.get_graph().print_ascii()

    logger.info("===== 测试结束 =====")
