"""
    @Author:MaYuhang
    @Time:2026/10/19
    @DESC:向量检索节点
"""
import time
import sys

from common.config.milvus_config import milvus_config
from common.logging.logger import logger, node_log, step_log
from processor.query_processor.state import QueryGraphState
from utils.clients.milvus_utils import create_hybrid_search_requests, hybrid_search, get_milvus_client
from utils.lm.embedding_utils import generate_embeddings
from utils.task_utils import add_done_task, add_running_task


@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state):
    item_names: list[str] = state.get("item_names") or []
    rewritten_query: str = state.get("rewritten_query")

    if not rewritten_query:
        logger.warning(f"rewritten_query为空,业务无法继续,提前终止!")
        raise ValueError(f"rewritten_query为空,业务无法继续,提前终止!")
    return item_names, rewritten_query


@step_log("step_2_select_chunks_in_milvus")
def step_2_select_chunks_in_milvus(item_names, rewritten_query) -> list[dict]:
    """
    向量化查询
    :param item_names:          确认的主体名列表
    :param rewritten_query:     重写的问题
    :return:                    [{"id":xx,"distance":xxx,"entity":""},...]
    """
    result = generate_embeddings([rewritten_query])
    dense_vector = result.get("dense")[0]
    sparse_vector = result.get("sparse")[0]

    # item_names 为空时不加过滤条件，进行全局检索；否则限定在指定金融产品内
    expr = f"item_name in {item_names}" if item_names else None
    reqs_list = create_hybrid_search_requests(
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        expr=expr
    )

    response = hybrid_search(
        client=get_milvus_client(),
        collection_name=milvus_config.chunks_collection,
        reqs=reqs_list,
        ranker_weights=(0.5, 0.5),
        norm_score=True,
        output_fields=[
            "chunk_id",
            "file_title",
            "parent_title",
            "title",
            "part",
            "item_name",
            "content"
        ]
    )
    if not response:
        return []
    return response[0]


@step_log("step_3_after_deal_milvus_result")
def step_3_after_deal_milvus_result(real_response) -> list[dict]:
    """
    对查询结果进行格式化处理
    """
    embedding_chunks: list[dict] = []

    if real_response:
        for item in real_response:
            chunk = item.get("entity")
            chunk["score"] = item.get("distance", 0.0)
            chunk["type"] = "milvus"
            embedding_chunks.append(chunk)
    logger.info(f"完成了问题向量检索!检索的数量:{len(embedding_chunks)}")
    return embedding_chunks


@node_log("node_search_embedding")
def node_search_embedding(state: QueryGraphState):
    """
    节点功能：进行向量内容检索
    """
    add_running_task(state.get("session_id"), sys._getframe().f_code.co_name, state.get("is_stream"))
    item_names, rewritten_query = step_1_validate_and_get_data(state)
    real_response: list[dict] = step_2_select_chunks_in_milvus(item_names, rewritten_query)
    embedding_chunks = step_3_after_deal_milvus_result(real_response)
    add_done_task(state.get("session_id"), sys._getframe().f_code.co_name, state.get("is_stream"))
    return {"embedding_chunks": embedding_chunks}


if __name__ == "__main__":
    test_state = {
        "session_id": "test_search_embedding_001",
        "rewritten_query": "三体讲了什么？",
        "item_names": ["三体"],
        "is_stream": False
    }

    print("\n>>> 开始测试 node_search_embedding 节点...")
    try:
        result = node_search_embedding(test_state)
        logger.info(f"检索结果汇总：{result}")
        chunks = result.get("embedding_chunks", [])
        print(f"\n>>> 测试完成！检索到 {len(chunks)} 条结果")

        if chunks:
            print("\n>>> Top 1 结果详情:")
            top1 = chunks[0]
            print(f"chunk_id: {top1.get('chunk_id')}")
            print(f"score(相似度): {top1.get('score')}")
            print(f"type: {top1.get('type')}")
            print(f"主体名: {top1.get('item_name')}")
            print(f"file_title: {top1.get('file_title')}")
            print(f"Content Preview: {top1.get('content', '')[:120]}...")
        else:
            print("\n>>> 警告：未检索到任何结果")

    except Exception as e:
        logger.error(f"测试运行失败: {e}", exc_info=True)