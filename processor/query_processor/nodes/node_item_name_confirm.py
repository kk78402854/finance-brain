"""
    @Author:MaYuhang
    @Time:2026/10/19
    @DESC:主体名确认节点
"""
import re
import time
import sys
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from common.config.milvus_config import milvus_config
from common.logging.logger import node_log, logger, step_log
from processor.query_processor.state import QueryGraphState
from utils.clients.milvus_utils import create_hybrid_search_requests, get_milvus_client, hybrid_search
from utils.clients.mongo_history_utils import save_chat_message, get_recent_messages
from utils.lm.embedding_utils import generate_embeddings
from utils.lm.lm_utils import get_llm_client
from utils.load_prompt import load_prompt
from utils.task_utils import add_running_task, add_done_task

# 主体名确认阈值：高于该分数 → 直接确认 [0.70]
ITEM_NAME_CONFIRM_THRESHOLD = 0.70
# 主体名候选阈值：介于两者之间 → 让用户选择
ITEM_NAME_CANDIDATE_THRESHOLD = 0.60
# 给用户选择时，最多展示几个候选
ITEM_NAME_OPTIONS_TOPK = 2
ITEM_NAME_CONFIRMED_TOPK = 1


@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state):
    """
    步骤1：校验并获取原始查询（original_query）和会话 ID（session_id）。
    """
    original_query = state.get("original_query")
    session_id = state.get("session_id")
    if (not original_query) or (not session_id):
        logger.error(f"session_id或者original_query为空,业务无法继续,提前终止!")
        raise ValueError(f"session_id或者original_query为空,业务无法继续,提前终止!")

    return original_query, session_id


@step_log("step_2_get_history_by_session_id")
def step_2_get_history_by_session_id(session_id):
    """
    步骤2：根据会话 ID 获取历史聊天记录。
        返回的记录结构说明：
            - _id:            记录的唯一标识
            - session_id:     所属会话 ID
            - role:           消息角色（如 user / assistant）
            - text:           消息文本内容
            - rewritten_query: 重写后的查询语句
            - item_names:     消息涉及的主体名列表
            - image_urls:     消息附带的图片 URL 列表
            - ts:             消息时间戳

        Returns:
            list[dict]: 过滤后的历史聊天记录列表，仅包含 item_names 非空的记录。
    """
    history_list = get_recent_messages(session_id)
    logger.debug(f"根据:{session_id}查询到的聊天记录:{len(history_list)}")

    # 过滤出包含 item_names 的有效记录（即与金融产品相关的对话）
    history_list = [item for item in history_list if item.get("item_names")]
    logger.debug(f"根据:{session_id}过滤后的有效聊天记录:{len(history_list)}")

    return history_list


@step_log("step_3_call_llm_rewritten_and_extract_itemnames")
def step_3_call_llm_rewritten_and_extract_itemnames(history_list, original_query):
    """
    步骤3：调用大语言模型对原始查询进行重写，并提取关联的主体名（item_names）。

    面试题：如何保证大语言模型返回的是 JSON？
        1. 提示词中约定返回 JSON 格式。
        2. 提示词中提供 JSON 示例。
        3. 在创建模型时指定 json_mode，例如：
           model_kwargs["response_format"] = {"type": "json_object"}
        4. 在输出时使用 JsonOutputParser 解析器。

    Args:
        history_list (list[dict]): 过滤后的历史聊天记录。
        original_query (str): 用户当前输入的原始问题。

    Returns:
        dict: 包含以下键：
            - rewritten_query (str): 重写后的查询；缺失时回退为 original_query。
            - item_names (list[str]): 关联主体名列表；缺失时为空列表。
    """
    llm_model = get_llm_client(json_mode=True)
    history_text: str = None
    if history_list:
        history_text_list: list[str] = []
        for index, item in enumerate(history_list, start=1):
            if item.get("role") == "user":
                history_text_list.append(
                    f"编号:{index},用户提问,原始问题:{item.get('text')},"
                    f"重写后的问题:{item.get('rewritten_query')}，"
                    f"关联主体名：{','.join(item.get('item_names'))}"
                )
            else:
                history_text_list.append(
                    f"编号:{index},助手回答,重写问题:{item.get('rewritten_query')},"
                    f"回答结果:{item.get('text')[:50]},关联主体名:{','.join(item.get('item_names'))}"
                )
        history_text = "\n".join(history_text_list)
    else:
        history_text = "当前会话id没有对应的历史记录"

    prompt_text = load_prompt(
        "rewritten_query_and_itemnames",
        query=original_query,
        history_text=history_text
    )
    message = HumanMessage(content=prompt_text)
    chains = llm_model | JsonOutputParser()
    json_dict = chains.invoke([message])

    if "item_names" not in json_dict:
        json_dict["item_names"] = []
    if "rewritten_query" not in json_dict:
        json_dict["rewritten_query"] = original_query
    # 对主体名去除内部空格，避免"华夏债券 C"与库中"华夏债券C"匹配失败
    json_dict["item_names"] = [name.replace(" ", "") for name in json_dict["item_names"]]

    # 【历史污染防护】：
    # 判断原始问题是否包含代词（这/那/它/该产品/这只基金等），只有代词句才允许从历史继承主体名；
    # 否则主体名必须出现在原始问题文本中，否则清空，防止 LLM 把历史主体名张冠李戴到当前问题。
    _pronoun_pattern = re.compile(r"(这|那|它|该)[本个部册只]?(书|产品|基金|理财|银行|公司|股票)?|该产品|这只基金|这个产品|那只基金|该公司")
    has_pronoun = bool(_pronoun_pattern.search(original_query))
    if not has_pronoun:
        _original_clean = original_query.replace(" ", "")
        json_dict["item_names"] = [
            name for name in json_dict["item_names"]
            if name and name in _original_clean
        ]
    logger.error(json_dict)
    return json_dict


@step_log("step_4_select_item_names_milvus")
def step_4_select_item_names_milvus(item_names) -> dict[str, list[dict]]:
    """
    对 LLM 识别出的主体名进行向量化，并在 Milvus 的主体名集合中进行混合检索。
    功能说明：
        1. 将 LLM 抽取出的主体名列表批量生成 dense/sparse 向量；
        2. 对每个主体名分别构造稠密 + 稀疏混合检索请求；
        3. 到 Milvus 的 item_name_collection 中检索相似主体名；
        4. 返回"原始主体名 -> 候选主体名及相似度分数列表"的映射。
    参数:
        item_names: list[str]
            LLM 从用户问题中识别出的主体名列表
    返回值:
        dict[str, list[dict]]
            每个原始主体名对应一个候选结果列表
    """
    milvus_search_result_dict: dict[str, list[dict]] = {}

    result = generate_embeddings(item_names)

    for index, llm_item_name in enumerate(item_names):
        dense_vector = result.get("dense")[index]
        sparse_vector = result.get("sparse")[index]

        reqs_list = create_hybrid_search_requests(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector
        )

        response = hybrid_search(
            client=get_milvus_client(),
            collection_name=milvus_config.item_name_collection,
            reqs=reqs_list,
            ranker_weights=(0.4, 0.6),
            norm_score=True,
            output_fields=["item_name"]
        )

        real_response = response[0]

        if not real_response:
            logger.warning(
                f"根据主体名:{llm_item_name}进行向量数据库的相似度搜索,值为空,"
                f"向量数据库为空,跳出所有查询!"
            )
            break

        item_name_score_list: list[dict] = []

        for item in real_response:
            item_name_score_list.append({
                "item_name": item.get("entity", []).get("item_name"),
                "score": item.get("distance")
            })

        milvus_search_result_dict[llm_item_name] = item_name_score_list

    return milvus_search_result_dict


@step_log("step_5_select_confirmed_and_option_item_names")
def step_5_select_confirmed_and_option_item_names(milvus_dict):
    """
    根据 Milvus 混合检索结果，将候选主体名划分为"确认"和"可选"。

    功能：
        1. 遍历每个 LLM 原始主体名对应的候选主体名列表；
        2. 分数 >= ITEM_NAME_CONFIRM_THRESHOLD 的候选进入 confirmed_list；
        3. ITEM_NAME_CANDIDATE_THRESHOLD <= 分数 < ITEM_NAME_CONFIRM_THRESHOLD
           的候选进入 option_list；
        4. 每个主体名优先取确认结果；无确认结果时再取可选结果。

    返回:
        dict[str, list[str]]
            {"confirmed": ["金融产品A", "金融产品B"], "option": ["金融产品a", "金融产品ab"]}
    """
    confirmed_list: list[str] = []
    option_list: list[str] = []

    for llm_item_name, item_name_score_list in milvus_dict.items():
        high_list = [
            item.get("item_name")
            for item in item_name_score_list
            if item.get("score") >= ITEM_NAME_CONFIRM_THRESHOLD
        ]
        middle_list = [
            item.get("item_name")
            for item in item_name_score_list
            if ITEM_NAME_CANDIDATE_THRESHOLD <= item.get("score", 0.0) < ITEM_NAME_CONFIRM_THRESHOLD
        ]

        if high_list:
            confirmed_list.extend(high_list[:ITEM_NAME_CONFIRMED_TOPK])
            logger.info(f"{llm_item_name}已经获取到可信的主体名:{high_list[:ITEM_NAME_CONFIRMED_TOPK]}")
            continue

        if middle_list:
            option_list.extend(middle_list[:ITEM_NAME_OPTIONS_TOPK])
            logger.info(f"{llm_item_name}没有可信的主体名,但是有可选的主体名:{middle_list[:ITEM_NAME_OPTIONS_TOPK]}")

    return {
        "confirmed": confirmed_list,
        "option": option_list
    }


@step_log("step_6_change_state_property")
def step_6_change_state_property(state, rewritten_query, confirmed_option_dict):
    """
    根据 step_5 的确认/可选结果更新图状态 state。

    支持 4 种业务场景：
        1. 金融产品推荐场景：用户未指定具体金融产品（如"推荐悬疑小说"），item_names 为空，
           进入全局检索，在所有金融产品切片中匹配相关内容。
        2. 金融产品详情场景：用户指定了具体金融产品，item_names 非空，在该书切片内检索。
        3. 内容检索场景：同详情场景，在指定金融产品内检索内容。
        4. 金融笔记场景：同详情场景，在指定金融产品内检索笔记/评论。

    不再拒绝用户提问，始终进入检索流程（全局或指定金融产品内）。
    """
    confirmed_list = confirmed_option_dict.get("confirmed", [])
    option_list = confirmed_option_dict.get("option", [])

    # 场景0：问候语/闲聊，直接回复，不进入检索流程
    _greeting_pattern = re.compile(r"^(你好|您好|嗨|hi|hello|在吗|在不在|哈喽|哈喽啊|嘿)[\s!！。\.~～]*$", re.IGNORECASE)
    _original_query = state.get("original_query", "").strip()
    if _greeting_pattern.match(_original_query):
        state["answer"] = "你好！我是金融知识库助手，可以问我关于金融产品的推荐、详情、内容或金融笔记，比如「推荐一本悬疑小说」「三体讲了什么」。"
        logger.info(f"检测到问候语:{_original_query},直接回复!")
        return

    # 场景1：存在确认主体名，在指定金融产品内检索。
    if confirmed_list:
        state["item_names"] = confirmed_list
        state["rewritten_query"] = rewritten_query
        logger.info(f"已确认主体名列表:{confirmed_list},进入指定金融产品检索!")
        return

    # 场景2：无确认但有可选主体名，使用候选主体名检索（放宽匹配）。
    if option_list:
        state["item_names"] = option_list
        state["rewritten_query"] = rewritten_query
        logger.info(f"无确认主体名,使用候选主体名:{option_list}进行检索!")
        return

    # 场景3：无确认也无可选主体名（推荐/开放式问题），item_names 置空，进入全局检索。
    state["item_names"] = []
    state["rewritten_query"] = rewritten_query
    logger.info(f"未识别到具体主体名,进入全局检索模式!")


@step_log("step_7_save_user_chat_message")
def step_7_save_user_chat_message(state):
    """
    将当前用户问题及相关状态保存到 MongoDB 聊天记录。
    """
    save_chat_message(
        session_id=state.get("session_id"),
        role="user",
        text=state.get("original_query"),
        rewritten_query=state.get("rewritten_query"),
        item_names=state.get("item_names", []),
        image_urls=[]
    )


@node_log("node_item_name_confirm")
def node_item_name_confirm(state: QueryGraphState):
    """
    节点功能：确认用户问题中的核心主体名。
    """
    add_running_task(state.get("session_id"), sys._getframe().f_code.co_name, state.get("is_stream"))
    original_query, session_id = step_1_validate_and_get_data(state)
    history_list: list[dict] = step_2_get_history_by_session_id(session_id)
    llm_dict: dict[str, Any] = step_3_call_llm_rewritten_and_extract_itemnames(history_list, original_query)
    confirmed_option_dict = {}
    if llm_dict.get("item_names"):
        milvus_dict: dict[str, list[dict]] = step_4_select_item_names_milvus(llm_dict.get("item_names"))
        confirmed_option_dict: dict[str, list[str]] = step_5_select_confirmed_and_option_item_names(milvus_dict)
    step_6_change_state_property(state, llm_dict.get("rewritten_query"), confirmed_option_dict)
    step_7_save_user_chat_message(state)
    add_done_task(state.get("session_id"), sys._getframe().f_code.co_name, state.get("is_stream"))
    return state


if __name__ == "__main__":
    mock_state = {
        "session_id": "test_session_001",
        "original_query": "三体讲了什么？",
        "is_stream": False,
    }
    result_state = node_item_name_confirm(mock_state)
    print(result_state)