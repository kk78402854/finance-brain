"""
    @Author:MaYuhang
    @Time:2026/10/19
    @DESC:联网搜索节点（MCP）
"""
import asyncio
import json
import time
import sys

from agents.mcp import MCPServerStreamableHttp

from common.config.bailian_mcp_config import mcp_config
from processor.query_processor.state import QueryGraphState
from utils.task_utils import add_done_task, add_running_task
from common.logging.logger import logger, node_log, step_log


@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state):
    # 从状态中获取重写的问题
    rewritten_query = state.get("rewritten_query")
    if not rewritten_query:
        logger.error(f"重写的问题为空,业务无法继续,提前终止!")
        raise ValueError(f"重写的问题为空,业务无法继续,提前终止!")
    return rewritten_query


@step_log("step_2_call_mcp_tool")
async def step_2_call_mcp_tool(rewritten_query):
    # 通过open-ai提供的mcp调用sdk进行调用
    # 创建连接mcp的客户端实例  注意：别被名字迷惑，虽然叫MCPServerStreamableHttp，但是属于客户端
    mcp_client = MCPServerStreamableHttp(
        # 指定连接名
        name="web_search_mcp",
        params={
            "url": mcp_config.mcp_base_url,
            "headers": {"Authorization": f"Bearer {mcp_config.api_key}"},
            "timeout": 50
        },
        cache_tools_list=True,
        max_retry_attempts=3,
    )

    # 连接mcp服务器
    await mcp_client.connect()
    try:
        # 查看mcp服务提供的工具列表
        list_tool = await mcp_client.list_tools()
        logger.debug(f"查询指定服务器:{mcp_config.mcp_base_url},获取工具列表:{list_tool}")

        # 调用工具
        result = await mcp_client.call_tool(
            tool_name="bailian_web_search",
            arguments={
                "query": rewritten_query,
                "count": 3
            }
        )
        return result
    except Exception as e:
        logger.exception(f"连接mcp报错,错误信息:{str(e)}")
    finally:
        # 关闭连接  释放资源
        await mcp_client.cleanup()


@node_log("node_web_search_mcp")
def node_web_search_mcp(state: QueryGraphState):
    """
    节点功能，调用外部搜索引擎补充信息
    :param state:
    :return:
    """
    # TODO 1. 添加节点到运行时列表
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    # TODO 2. 从状态中获取数据并进行校验
    rewritten_query: str = step_1_validate_and_get_data(state)
    # TODO 3. 调用mcp服务获取网络搜索结果
    result = asyncio.run(step_2_call_mcp_tool(rewritten_query))
    # TODO 4. 对结果进行解析
    text_result = result.content[0].text
    web_search_docs = json.loads(text_result).get("pages", [])
    # TODO 5. 添加节点到已完成列表
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])

    return {"web_search_docs": web_search_docs}


if __name__ == "__main__":
    test_state = {
        "session_id": "xxxx",
        "is_stream": False,
        "rewritten_query": "三体这本书讲了什么",
    }
    result_state = node_web_search_mcp(test_state)
    print(result_state)
