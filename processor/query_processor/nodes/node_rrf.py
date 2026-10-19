"""
    @Author:MaYuhang
    @Time:2026/10/19
    @DESC:RRF融合排序节点
"""
import time
import sys

from processor.query_processor.nodes.node_search_embedding import node_search_embedding
from processor.query_processor.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from processor.query_processor.state import QueryGraphState
from utils.task_utils import add_running_task, add_done_task
from common.logging.logger import logger, node_log, step_log


@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state):
    """
      获取核心参数并且校验
    """
    embedding_chunks = state.get("embedding_chunks") or []
    hyde_embedding_chunks = state.get("hyde_embedding_chunks") or []
    # 不抛错，允许空列表流入后续节点（由 rerank 节点统一处理空结果场景）
    return embedding_chunks, hyde_embedding_chunks


@step_log("step_2_use_rrf_rank")
def step_2_use_rrf_rank(data_list: list[tuple[float, list]], k: int = 60, limit: int = 5):
    # 用于存放chunk对应的得分字典
    score_dict: dict[str, float] = {}
    # 用于存放chunk
    chunk_dict: dict[str, dict] = {}

    for weight, chunks in data_list:
        for rank, chunk in enumerate(chunks, start=1):
            rrf_score = weight * (1 / (k + rank))
            chunk_id = chunk.get("chunk_id")
            score_dict[chunk_id] = score_dict.get(chunk_id, 0.0) + rrf_score
            chunk_dict[chunk_id] = chunk

    rrf_chunks: list[dict] = []
    for chunk_id, chunk in chunk_dict.items():
        chunk["score"] = score_dict.get(chunk_id, 0.0)
        rrf_chunks.append(chunk)

    logger.debug(f"排序之前,长度:{len(rrf_chunks)},内容: {rrf_chunks}")
    rrf_chunks.sort(key=lambda x: x.get("score"), reverse=True)
    logger.debug(f"排序之后,长度:{len(rrf_chunks)},内容: {rrf_chunks}")

    rrf_chunks = rrf_chunks[:limit]
    return rrf_chunks


@node_log("node_rrf")
def node_rrf(state: QueryGraphState):
    """
    节点功能：Reciprocal Rank Fusion
    将多路召回的结果（向量、HyDE）进行加权融合排序。
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    embedding_chunks, hyde_embedding_chunks = step_1_validate_and_get_data(state)
    data_list: list[tuple[float, list]] = [
        (0.5, embedding_chunks),
        (0.5, hyde_embedding_chunks)
    ]
    rrf_chunks = step_2_use_rrf_rank(data_list)
    state["rrf_chunks"] = rrf_chunks
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    return state


if __name__ == "__main__":
    mock_state = {
        "session_id": "test_rrf_session",
        "is_stream": False,
        "original_query": "三体讲了什么？",
        "rewritten_query": "三体的主要内容是什么？",
        "item_names": ["三体"],
    }

    emb_res = node_search_embedding(mock_state)
    hyde_res = node_search_embedding_hyde(mock_state)
    mock_state["embedding_chunks"] = emb_res.get("embedding_chunks") or []
    mock_state["hyde_embedding_chunks"] = hyde_res.get("hyde_embedding_chunks") or []

    result = node_rrf(mock_state)
    print(result)