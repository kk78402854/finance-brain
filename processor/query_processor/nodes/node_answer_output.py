"""
    @Author:MaYuhang
    @Time:2026/10/19
    @DESC:金融智库-答案输出节点（含合规护栏：禁用词更正 + 强制风险提示）
"""
import re
import time
import sys
from common.logging.logger import logger, node_log, step_log
from processor.query_processor.state import QueryGraphState
from utils.clients.mongo_history_utils import save_chat_message, get_recent_messages
from utils.compliance_utils import apply_compliance_guardrail
from utils.lm.lm_utils import get_llm_client
from utils.load_prompt import load_prompt
from utils.sse_utils import push_to_session, SSEEvent
from utils.task_utils import add_running_task, add_done_task


@step_log("step_1_has_answer")
def step_1_has_answer(state):
    answer = state.get("answer")
    return True if answer else False


@step_log("step_2_validate_and_get_data")
def step_2_validate_and_get_data(state):
    reranked_docs = state.get("reranked_docs")
    session_id = state.get("session_id")
    item_names = state.get("item_names")
    rewritten_query = state.get("rewritten_query")
    is_stream = state.get("is_stream", False)

    # item_names 允许为空（开放式问题走全局检索）
    item_names = item_names or []
    if (not reranked_docs) or (not session_id) or (not rewritten_query):
        logger.error("核心参数为空,业务无法继续,提前终止!")
        raise ValueError("核心参数为空,业务无法继续,提前终止!")
    return reranked_docs, session_id, item_names, rewritten_query, is_stream


@step_log("step_3_get_history_by_session_id")
def step_3_get_history_by_session_id(session_id):
    history_list: list[dict] = get_recent_messages(session_id=session_id, limit=10)
    logger.debug(f"根据:{session_id}查询到的聊天记录:{len(history_list)}")
    history_list = [item for item in history_list if item.get("item_names")]
    logger.debug(f"根据:{session_id}过滤后的有效聊天记录:{len(history_list)}")
    return history_list


@step_log("step_4_create_answer_prompt")
def step_4_create_answer_prompt(reranked_docs, history_list, item_names, rewritten_query):
    context: str = ""
    for doc in reranked_docs:
        source_type = '联网搜索' if doc.get('type') == 'web' else '向量数据库'
        # 收集来源信息：主实体名、内容类型、文件名、章节标题
        item_name = doc.get('item_name', '')
        content_type = doc.get('content_type', '')
        file_title = doc.get('file_title', '')
        parent_title = doc.get('parent_title', '')
        source_parts = []
        if item_name:
            source_parts.append(f"主体:{item_name}")
        if content_type:
            source_parts.append(f"类型:{content_type}")
        if file_title:
            source_parts.append(f"文件名:{file_title}")
        if parent_title:
            source_parts.append(f"章节:{parent_title}")
        source_info = "，".join(source_parts) if source_parts else "无"
        context += (f"标题:{doc.get('title')},数据来源:{source_type},来源信息:[{source_info}], "
                    f"置信度:{doc.get('score')},内容:{doc.get('text')} \n")

    history_text: str = ""
    if history_list:
        history_text_list: list[str] = []

        for index, history in enumerate(history_list, start=1):
            if history.get("role") == 'user':
                history_text_list.append((
                    f"编号:{index},用户提问,原始问题:{history.get('text')},重写后的问题:{history.get('rewritten_query')},"
                    f"关联主体:{','.join(history.get('item_names'))}"))
            else:
                history_text_list.append((
                    f"编号:{index},助手回答,重写问题:{history.get('rewritten_query')},回答结果:{history.get('text')[:50]},"
                    f"关联主体:{','.join(history.get('item_names'))}"))
        history_text = "\n".join(history_text_list)
    else:
        history_text = "没有有效的历史聊天记录!"

    item_names_str: str = ",".join(item_names) if item_names else "全局检索（未指定具体产品）"

    answer_prompt_text = load_prompt("answer_out", context=context, history=history_text, item_names=item_names_str,
                                     question=rewritten_query)
    return answer_prompt_text


@step_log("step_5_call_llm_create_answer")
def step_5_call_llm_create_answer(answer_prompt, is_stream, session_id):
    llm_model = get_llm_client()

    answer: str = ""
    if is_stream:
        for chunk in llm_model.stream(answer_prompt):
            chunk_str = chunk.content

            push_to_session(
                session_id=session_id,
                event=SSEEvent.DELTA,
                data={"delta": chunk_str}
            )

            answer += chunk_str
    else:
        response = llm_model.invoke(answer_prompt)
        answer = response.content

    return answer


@step_log("step_5_5_apply_compliance_guardrail")
def step_5_5_apply_compliance_guardrail(answer, question, is_stream, session_id):
    """
    步骤 5.5: 合规护栏
    1. 禁用词检测：答案出现"保证收益/稳赚不赔"等违规表述时追加合规更正
    2. 风险话题检测：涉及收益/风险/投资时强制追加风险提示
    追加的合规文本同步推送到 SSE 流，保证前端展示与落库内容一致。
    """
    guarded_answer, appended_text = apply_compliance_guardrail(answer, question)
    if appended_text:
        logger.info(f"合规护栏已生效，追加内容长度: {len(appended_text)}")
        if is_stream:
            push_to_session(
                session_id=session_id,
                event=SSEEvent.DELTA,
                data={"delta": appended_text}
            )
    return guarded_answer


# 占位/示例域名黑名单：这些域名下的图片 URL 一律视为编造，不展示
_IMAGE_PLACEHOLDER_DOMAINS = {"example.com", "example.org", "example.net", "example.edu"}


def _is_placeholder_image_url(url: str) -> bool:
    """判断图片 URL 是否来自占位/示例域名（LLM 编造的假链接）"""
    try:
        from urllib.parse import urlparse
        host = urlparse(url).hostname or ""
        host = host.lower()
        return any(host == d or host.endswith("." + d) for d in _IMAGE_PLACEHOLDER_DOMAINS)
    except Exception:
        return False


@step_log("step_6_extract_chunk_and_url_image")
def step_6_extract_chunk_and_url_image(reranked_docs):
    image_urls: list[str] = []
    rep = re.compile(r"\!\[.*?\]\((.*?)\)")

    for chunk in reranked_docs:
        url: str = chunk.get("url")
        text: str = chunk.get("text")

        if url:
            if url.endswith((".png", ".jpg", ".gif", ".jpeg", ".svg")) and not _is_placeholder_image_url(url):
                image_urls.append(url)

        if text:
            image_url_list: list[str] = rep.findall(text)
            if image_url_list:
                image_urls.extend([u for u in image_url_list if not _is_placeholder_image_url(u)])
    logger.info(f"完成切片中的图片提取,提取数量:{len(image_urls)},内容:{image_urls}")
    return image_urls


@step_log("step_7_save_assistant_message")
def step_7_save_assistant_message(state):
    save_chat_message(
        session_id=state.get("session_id"),
        role="assistant",
        text=state.get("answer"),
        rewritten_query=state.get("rewritten_query"),
        item_names=state.get("item_names"),
        image_urls=state.get("image_urls")
    )


@node_log("node_answer_output")
def node_answer_output(state: QueryGraphState):
    """
    节点功能：生成最终答案（含合规护栏），支持流式/非流式输出
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    has_answer = step_1_has_answer(state)
    if not has_answer:
        reranked_docs, session_id, item_names, rewritten_query, is_stream = step_2_validate_and_get_data(state)
        history_list = step_3_get_history_by_session_id(session_id)
        answer_prompt: str = step_4_create_answer_prompt(reranked_docs, history_list, item_names, rewritten_query)
        answer: str = step_5_call_llm_create_answer(answer_prompt, is_stream, session_id)
        # 合规护栏：禁用词更正 + 强制风险提示（流式场景同步推送追加内容）
        answer: str = step_5_5_apply_compliance_guardrail(answer, rewritten_query, is_stream, session_id)
        image_urls: list[str] = step_6_extract_chunk_and_url_image(reranked_docs)
        state["answer"] = answer
        state["image_urls"] = image_urls

    step_7_save_assistant_message(state)
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    logger.success("~~~全剧终~~~")
    return state


if __name__ == "__main__":
    mock_reranked_docs = [
        {
            "chunk_id": "local_101",
            "type": "milvus",
            "title": "产品概况",
            "score": 0.95,
            "text": "易方达智造优势混合型证券投资基金（基金代码：020988）由易方达基金管理有限公司管理，"
                    "主要投资于智造优势主题相关上市公司股票，风险等级为中风险(R3)。",
            "item_name": "易方达智造优势混合",
            "content_type": "产品说明书",
            "file_title": "易方达智造优势混合基金产品资料概要更新.md",
            "parent_title": "产品概况"
        }
    ]
    mock_history = [
        {"role": "user", "text": "易方达智造优势混合主要投资什么？", "rewritten_query": "易方达智造优势混合基金的投资方向是什么？",
         "item_names": ["易方达智造优势混合"]},
    ]
    mock_state = {
        "session_id": "test_answer_session_001",
        "original_query": "易方达智造优势混合主要投资什么？",
        "rewritten_query": "易方达智造优势混合基金的投资方向是什么？",
        "item_names": ["易方达智造优势混合"],
        "history": mock_history,
        "reranked_docs": mock_reranked_docs,
        "is_stream": False,
        "answer": None,
    }
    result = node_answer_output(mock_state)
    print(result)
