"""
    @Author:MaYuhang
    @Time:2026/10/19
    @DESC:金融知识库查询服务
"""
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# 将项目根目录注入 sys.path，保证 `python api/query_service.py` 直接启动时能找到本地包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse, FileResponse

from processor.query_processor.main_graph import query_app
from processor.query_processor.state import create_query_default_state
from utils.clients.mongo_history_utils import get_recent_messages, clear_history
from utils.path_util import PROJECT_ROOT
from common.logging.logger import logger
from utils.sse_utils import sse_generator, push_to_session, SSEEvent, create_sse_queue
from fastapi import Request

from utils.task_utils import get_done_task_list, update_task_status, TASK_STATUS_PROCESSING, TASK_STATUS_COMPLETED, \
    TASK_STATUS_FAILED, clear_task

# 创建FastAPI应用实例
app = FastAPI()

# 跨域的配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# 跳转到查询页面接口  /chat.html
@app.get("/chat.html")
def chat():
    chat_html_path_obj = PROJECT_ROOT / "page" / "chat.html"
    if not chat_html_path_obj.exists():
        logger.error(f"{chat_html_path_obj}不存在")
        raise FileNotFoundError(f"{chat_html_path_obj}不存在")

    logger.info(f"跳转到{chat_html_path_obj}")
    return FileResponse(chat_html_path_obj)

# 健康检查接口    /health
@app.get("/health")
def check_health():
    """服务健康检查"""
    logger.info(f"{datetime.now()}:完成本次健康状态检查!")
    return {"ok": True}


# 定义一个类，用于封装客户端查询的时候传递给服务器端的参数
class QueryRequest(BaseModel):
    query: str = Field(..., description="查询内容")
    session_id: str = Field(None, description="会话ID")
    is_stream: bool = Field(False, description="是否流式返回")

# 运行langgraph查询图整体流程
def run_query_graph(session_id: str, is_stream: bool,user_query:str):
    try:
        # 在执行图节点前，应该先将task_utils中维护的数据清除掉
        clear_task(session_id)

        # 更新全局状态为运行中
        update_task_status(session_id,TASK_STATUS_PROCESSING,is_stream)
        # 初始化状态
        state = create_query_default_state(
            session_id=session_id,
            is_stream=is_stream,
            original_query = user_query
        )
        logger.info(f"查询图开始执行,传入参数:\n{json.dumps(state, indent=4, ensure_ascii=False)}")
        # 调用编译后的图对象
        final_state = query_app.invoke(state)
        logger.info(f"查询图执行结束,返回结果:\n{json.dumps(final_state, indent=4, ensure_ascii=False)}")
        # 更新全局状态为已完成
        update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)
        if is_stream:
            push_to_session(
                session_id,
                SSEEvent.FINAL,
                {
                    "answer": final_state.get("answer"),
                    "status": "completed",
                    "image_urls": final_state.get("image_urls")
                }
            )
        return final_state
    except Exception as e:
        # 更新全局状态为失败
        update_task_status(session_id, TASK_STATUS_FAILED, is_stream)
        logger.exception(f"{user_query}:执行图流程报错!")
        # 向客户端通过SSE推送事件类型为"error"的消息
        push_to_session(session_id=session_id, event=SSEEvent.ERROR, data={
            "error": str(e)
        })

# 查询接口   /query  --- post    {"session_id":"","query":"","is_stream":xxx}
@app.post("/query")
def query(query_request:QueryRequest,background_tasks:BackgroundTasks):
    # 获取会话id
    session_id = query_request.session_id
    # 获取用户查询的问题
    user_query = query_request.query
    # 获取是否流式输出的标记
    is_stream = query_request.is_stream
    logger.info(f"用户发起提问: {user_query} 关联的session_id:{session_id},流式状态:{is_stream}")

    # 判断是不是流式输出
    if is_stream:
        # 提前将当前会话对应的队列创建出来
        create_sse_queue(session_id)

        # 以异步的方式 开启后台任务， 调用langgraph图 执行各个节点
        background_tasks.add_task(run_query_graph,session_id,is_stream,user_query)

        logger.info(f"[{session_id}] 流式任务已提交至后台执行")
        return {
            "message": "结果正在处理中...",
            "session_id": session_id
        }
    else:
        # 以同步的方式 不需要开启后台任务， 直接调用langgraph图 执行各个节点
        final_state = run_query_graph(session_id, is_stream, user_query)
        # 获取当前会话已完成的节点
        done_list = get_done_task_list(session_id)

        # 向客户端返回数据
        return {
            "session_id": session_id,
            "message": f"{session_id}已经完成问题解析!",
            "answer": final_state.get("answer"),
            "done_list": done_list,
            "image_urls": final_state.get("image_urls", [])
        }


# 通过SSE向客户端推流接口    /stream/${session_id}
@app.get("/stream/{session_id}")
def stream(session_id:str,request: Request):
    logger.info(f"{datetime.now()}完成{session_id}对应的流式响应接口!")
    return StreamingResponse(
        sse_generator(session_id,request),
        media_type="text/event-stream",
    )

# 获取当前会话历史记录   /history/${sessionId} --- get
@app.get("/history/{session_id}")
def history(session_id:str):
    records : list[dict[str,Any]] = get_recent_messages(session_id)
    logger.info(f"查询{session_id}的聊天记录,查询到:{len(records)}条!")
    items = []
    for r in records:
        items.append({
            "_id": str(r.get("_id")) if r.get("_id") is not None else "",
            "session_id": session_id,
            "role": r.get("role", ""),
            "text": r.get("text", ""),
            "rewritten_query": r.get("rewritten_query", ""),
            "item_names": r.get("item_names", []),
            "image_urls": r.get("image_urls", []),
            "ts": r.get("ts")
        })
    return {
        "session_id": session_id,
        "items": items
    }
# 清空当前会话历史记录   /history/${sessionId} --- delete
@app.delete("/history/{session_id}")
def clear_chat_history(session_id:str):
    deleted_count = clear_history(session_id)
    logger.info(f"已经清空:{session_id}对应的聊天记录,删除的数量为:{deleted_count}")
    return {"message": "History cleared", "deleted_count": deleted_count}
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
