"""
    @Author:MaYuhang
    @Time:2026/10/19
    @DESC:重排序节点
"""
import time
import sys

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from common.logging.logger import node_log, logger, step_log
from processor.query_processor.state import QueryGraphState
from utils.lm.lm_utils import get_llm_client
from utils.lm.reranker_utils import compute_token_number, get_reranker_model
from utils.load_prompt import load_prompt
from utils.task_utils import add_running_task, add_done_task

# -----------------------------
# Rerank / TopK 全局常量
# -----------------------------
RERANK_MAX_TOPK: int = 10
RERANK_MIN_TOPK: int = 1
RERANK_GAP_RATIO: float = 0.25
RERANK_GAP_ABS: float = 0.5

RERANK_MAX_INPUT_TOKENS: int = 512
RERANK_SUMMARY_CHAR_RATIO: float = 1.3
RERANK_MIN_SUMMARY_CHARS: int = 50

# Milvus 知识库结果的分数加成系数：reranker 打分后乘以该系数，
# 让知识库内容优先于联网搜索结果，减少 LLM 依赖外部信息编造
MILVUS_SCORE_BOOST: float = 1.2


@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state):
    web_search_docs = state.get("web_search_docs") or []
    rrf_chunks = state.get("rrf_chunks") or []
    rewritten_query = state.get("rewritten_query")

    if not rewritten_query:
        logger.error(f"rewritten_query参数为空,业务无法继续,提前终止!")
        raise ValueError(f"rewritten_query参数为空,业务无法继续,提前终止!")

    return rrf_chunks, web_search_docs, rewritten_query


@step_log("step_2_merge_rrf_and_web")
def step_2_merge_rrf_and_web(rrf_chunks, web_search_docs) -> list[dict]:
    """
    合并 RRF 结果和网络搜索结果，统一结构
    """
    merged_list = []

    for chunk in rrf_chunks:
        merged_list.append({
            "title": chunk.get("title"),
            "text": chunk.get("content"),
            "score": 0.0,
            "type": "milvus",
            "url": ""
        })
    for doc in web_search_docs:
        merged_list.append({
            "title": doc.get("title"),
            "text": doc.get("snippet"),
            "score": 0.0,
            "type": "web",
            "url": doc.get("url")
        })
    logger.info(f"完成两路{len(rrf_chunks)}:{len(web_search_docs)}数据融合,融合后的数量:{len(merged_list)}")
    return merged_list


@step_log("step_3_create_question_answer_pair")
def step_3_create_question_answer_pair(merged_list, rewritten_query) -> list[list[str]]:
    """
    封装问题-答案对，超长答案调用 LLM 压缩
    """
    question_answer_pair: list[list[str]] = []

    question_token_len = compute_token_number(rewritten_query)

    for chunk in merged_list:
        answer = chunk.get("text")
        answer_token_len = compute_token_number(answer)

        if answer_token_len > (RERANK_MAX_INPUT_TOKENS - 4 - question_token_len):
            char_length = max((RERANK_MAX_INPUT_TOKENS - 4 - question_token_len) / RERANK_SUMMARY_CHAR_RATIO,
                              RERANK_MIN_SUMMARY_CHARS)

            llm_model = get_llm_client()
            prompt_text = load_prompt("rerank_text_refine",
                                      question=rewritten_query,
                                      answer=answer,
                                      limit=char_length)
            message = HumanMessage(content=prompt_text)
            chains = llm_model | StrOutputParser()

            logger.debug(f"已经触发压缩流程,压缩前{len(answer)}:{answer}")
            answer = chains.invoke([message])
            logger.debug(f"已经触发压缩流程,压缩后{len(answer)}:{answer}")

        question_answer_pair.append([rewritten_query, answer])

    return question_answer_pair


@step_log("step_4_list_score_and_rank")
def step_4_list_score_and_rank(merged_list, question_answer_pair):
    reranker_model = get_reranker_model()
    score_list = reranker_model.compute_score(question_answer_pair, normalize=True)
    logger.debug(f"完成了数据打分:{score_list}")

    for chunk, score in zip(merged_list, score_list):
        # Milvus 知识库结果应用分数加成，使其优先于联网搜索结果
        if chunk.get("type") == "milvus":
            chunk["score"] = score * MILVUS_SCORE_BOOST
        else:
            chunk["score"] = score

    logger.debug(f"未排序之前的合并列表:{merged_list}")
    merged_list.sort(key=lambda c: c.get('score', 0.0), reverse=True)
    logger.debug(f"排序之后的合并列表:{merged_list}")


@step_log("step_5_dynamic_topk")
def step_5_dynamic_topk(merged_list):
    max_top_k = min(RERANK_MAX_TOPK, len(merged_list))
    min_top_k = RERANK_MIN_TOPK
    top_k = max_top_k

    if max_top_k > min_top_k:
        for index in range(min_top_k - 1, max_top_k - 1):
            current = merged_list[index]
            next = merged_list[index + 1]

            abs_v = current.get("score") - next.get("score")
            radio_v = abs_v / current.get("score")

            if (abs_v > RERANK_GAP_ABS) or (radio_v > RERANK_GAP_RATIO):
                top_k = index + 1
                logger.debug(f"下标:{index}位置发生了断崖!当前值:{current.get('score')},下一个值:{next.get('score', 0.0)}")
                break
    reranked_docs = merged_list[:top_k]
    return reranked_docs


@node_log("node_rerank")
def node_rerank(state: QueryGraphState):
    """
    节点功能：使用 Cross-Encoder 模型对 RRF 后的结果进行精确打分重排。
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    rrf_chunks, web_search_docs, rewritten_query = step_1_validate_and_get_data(state)
    merged_list: list[dict] = step_2_merge_rrf_and_web(rrf_chunks, web_search_docs)

    # 所有检索结果均为空（知识库和联网都没找到相关内容），直接返回引导语
    if not merged_list:
        state["reranked_docs"] = []
        state["answer"] = (
            "抱歉，知识库中暂无与您问题相关的内容。我是金融智库助手，主要回答金融产品相关问题，你可以问我：\n"
            "- 推荐类：「推荐一本适合睡前听的治愈系金融产品」\n"
            "- 详情类：「三体这本书的详细信息」\n"
            "- 内容类：「活着讲了什么故事」\n"
            "- 笔记类：「小王子有哪些精彩金融笔记」"
        )
        add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
        return state

    question_answer_pair: list[list[str]] = step_3_create_question_answer_pair(merged_list, rewritten_query)
    step_4_list_score_and_rank(merged_list, question_answer_pair)
    reranked_docs = step_5_dynamic_topk(merged_list)

    state["reranked_docs"] = reranked_docs
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    return state


if __name__ == "__main__":
    mock_rrf_chunks = [
        {"chunk_id": "local_1", "content": "RRF是一种倒数排名融合算法", "title": "算法介绍"},
        {"chunk_id": "local_2", "content": "BGE是一个强大的重排序模型", "title": "模型介绍"},
        {"chunk_id": "local_3", "content": "路漫漫其修远兮，吾将上下而求索", "title": "励志诗句"},
        {"chunk_id": "local_4", "content": "让天下没有难学的技术", "title": "广告"}
    ]
    mock_web_docs = [
        {"title": "Rerank技术详解", "url": "http://web.com/1", "snippet": "Rerank即重排序，常用于RAG系统的第二阶段"},
    ]
    mock_state = {
        "session_id": "test_rerank_session",
        "rewritten_query": "什么是RRF和Rerank？",
        "rrf_chunks": mock_rrf_chunks,
        "web_search_docs": mock_web_docs,
        "is_stream": False,
    }
    result = node_rerank(mock_state)
    print(result)