"""
    @Author:MaYuhang
    @Time:2026/10/19
    @DESC:HyDE检索节点
"""
import time
import sys

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from common.config.milvus_config import milvus_config
from common.logging.logger import node_log, logger, step_log
from processor.query_processor.state import QueryGraphState
from utils.clients.milvus_utils import create_hybrid_search_requests, hybrid_search, get_milvus_client
from utils.lm.embedding_utils import generate_embeddings
from utils.lm.lm_utils import get_llm_client
from utils.load_prompt import load_prompt
from utils.task_utils import add_done_task, add_running_task


@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state: QueryGraphState):
    item_names: list[str] = state.get("item_names") or []
    rewritten_query: str = state.get("rewritten_query")
    if not rewritten_query:
        logger.error(f"rewritten_query为空,业务无法继续,提前终止!")
        raise ValueError(f"rewritten_query为空,业务无法继续,提前终止!")
    return item_names, rewritten_query


@step_log("step_2_call_llm_by_rewritten_query")
def step_2_call_llm_by_rewritten_query(rewritten_query):
    llm_model = get_llm_client()

    prompt_text = load_prompt("hyde_prompt", rewritten_query=rewritten_query)
    message = HumanMessage(content=prompt_text)
    chains = llm_model | StrOutputParser()
    hyde_answer = chains.invoke([message])
    logger.info(f"基于:{rewritten_query},模型给与的假设性回答:{hyde_answer}")
    return hyde_answer


@step_log("step_3_select_chunks_in_milvus")
def step_3_select_chunks_in_milvus(item_names, rewritten_query, hyde_answer):
    result = generate_embeddings(["问题:" + rewritten_query + "\n 假设性回答:" + hyde_answer])
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

    return response[0]


@step_log("step_4_after_deal_milvus_result")
def step_4_after_deal_milvus_result(real_response) -> list[dict]:
    hyde_embedding_chunks: list[dict] = []

    if real_response:
        for item in real_response:
            chunk = item.get("entity")
            chunk["score"] = item.get("distance", 0.0)
            chunk["type"] = "milvus"
            hyde_embedding_chunks.append(chunk)

    logger.info(f"完成了问题向量检索!检索的数量:{len(hyde_embedding_chunks)}")
    return hyde_embedding_chunks


@node_log("node_search_embedding_hyde")
def node_search_embedding_hyde(state: QueryGraphState):
    """
    节点功能：HyDE (Hypothetical Document Embedding)
    先让 LLM 生成假设性答案，再对答案进行向量检索，提高召回率。
    """
    add_running_task(state.get("session_id"), sys._getframe().f_code.co_name, state.get("is_stream"))
    item_names, rewritten_query = step_1_validate_and_get_data(state)
    hyde_answer = step_2_call_llm_by_rewritten_query(rewritten_query)
    real_response: list[dict] = step_3_select_chunks_in_milvus(item_names, rewritten_query, hyde_answer)
    hyde_embedding_chunks = step_4_after_deal_milvus_result(real_response)
    add_done_task(state.get("session_id"), sys._getframe().f_code.co_name, state.get("is_stream"))
    return {"hyde_embedding_chunks": hyde_embedding_chunks}


if __name__ == "__main__":
    mock_state = {
        "session_id": "test_hyde_session_001",
        "original_query": "三体讲了什么？",
        "rewritten_query": "三体的主要内容是什么？",
        "item_names": ["三体"],
        "is_stream": False,
    }
    result = node_search_embedding_hyde(mock_state)
    print(result)