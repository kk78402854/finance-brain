"""
    @Author:MaYuhang
    @Time:2026/10/19
    @DESC:查询子图编排（7节点）
"""
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from processor.query_processor.nodes.node_answer_output import node_answer_output
from processor.query_processor.nodes.node_item_name_confirm import node_item_name_confirm
from processor.query_processor.nodes.node_rerank import node_rerank
from processor.query_processor.nodes.node_rrf import node_rrf
from processor.query_processor.nodes.node_search_embedding import node_search_embedding
from processor.query_processor.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from processor.query_processor.nodes.node_web_search_mcp import node_web_search_mcp
from processor.query_processor.state import QueryGraphState

# 创建StateGraph对象
builder = StateGraph(QueryGraphState)

# 注册节点
builder.add_node("node_item_name_confirm", node_item_name_confirm)
builder.add_node("node_search_embedding", node_search_embedding)
builder.add_node("node_search_embedding_hyde", node_search_embedding_hyde)
builder.add_node("node_web_search_mcp", node_web_search_mcp)
builder.add_node("node_rrf", node_rrf)
builder.add_node("node_rerank", node_rerank)
builder.add_node("node_answer_output", node_answer_output)

# 定义路由函数
def after_node_item_name_confirm(state: QueryGraphState):
    if state.get("answer"):
        return "node_answer_output"

    return "node_search_embedding", "node_search_embedding_hyde", "node_web_search_mcp"

# 指定入口
builder.set_entry_point("node_item_name_confirm")

# 条件边：主体名确认后，若已有答案（无确认/无可选）直接输出，否则三路并行检索
builder.add_conditional_edges(
    "node_item_name_confirm",
    after_node_item_name_confirm,
    {
        "node_answer_output": "node_answer_output",
        "node_search_embedding": "node_search_embedding",
        "node_search_embedding_hyde": "node_search_embedding_hyde",
        "node_web_search_mcp": "node_web_search_mcp"
    }
)
# 三路检索汇流到 RRF
builder.add_edge("node_search_embedding", "node_rrf")
builder.add_edge("node_search_embedding_hyde", "node_rrf")
builder.add_edge("node_web_search_mcp", "node_rrf")
# RRF → Rerank → 答案输出 → END
builder.add_edge("node_rrf", "node_rerank")
builder.add_edge("node_rerank", "node_answer_output")
builder.add_edge("node_answer_output", END)

query_app = builder.compile()