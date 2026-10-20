"""
    @Author:MaYuhang
    @Time:2026/10/19
    @DESC:金融智库-文件导入服务（FastAPI + LangGraph 后台任务）
"""
import sys
import uuid
from datetime import datetime
from pathlib import Path

# 将项目根目录注入 sys.path，保证 `python api/file_import_service.py` 直接启动时能找到本地包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn
from fastapi import FastAPI, UploadFile, BackgroundTasks
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse

from processor.import_processor.main_graph import kb_import_app
from processor.import_processor.state import get_default_state
from utils.path_util import PROJECT_ROOT
from utils.task_utils import get_running_task_list, get_done_task_list, get_task_status, add_running_task, \
    add_done_task, update_task_status, TASK_STATUS_PROCESSING, TASK_STATUS_COMPLETED, TASK_STATUS_FAILED
from common.logging.logger import logger

# 创建 FastAPI 应用实例
app = FastAPI()
# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# 健康检查接口    /health
@app.get("/health")
def check_health():
    """服务健康检查，供前端展示导入服务连接状态"""
    return {"ok": True}

# 路由：GET /import.html
# 功能：返回导入页面
@app.get("/import.html")
def import_page():
    import_page_path = PROJECT_ROOT / "page" / "import.html"
    return FileResponse(import_page_path, media_type="text/html")


# 后台任务：调用 LangGraph 执行导入全流程
def run_graph_task(task_id: str, local_dir: str, local_file_path: str):
    """
    在后台异步执行 LangGraph 导入流程。

    更新任务状态、初始化状态对象，通过 LangGraph 应用流式执行，
    记录每个节点的完成日志。若执行过程中发生异常，则捕获并标记任务失败。

    Args:
        task_id (str): 当前任务的唯一标识符。
        local_dir (str): 本地目录路径，供导入流程使用。
        local_file_path (str): 本地文件完整路径，供导入流程使用。
    """
    try:
        # 更新全局任务状态为"处理中"
        update_task_status(task_id, TASK_STATUS_PROCESSING)

        # 创建默认状态对象，作为 LangGraph 的初始输入
        init_state = get_default_state()

        # 将任务相关参数注入初始状态
        init_state["task_id"] = task_id
        init_state["local_dir"] = local_dir
        init_state["local_file_path"] = local_file_path

        # 通过 LangGraph 应用流式执行，stream_mode="updates" 表示每次返回节点更新事件
        for event in kb_import_app.stream(init_state, stream_mode="updates"):
            for node_name, node_result in event.items():
                logger.info(f"[{task_id}] LangGraph 节点执行完成：{node_name}")

        # 全流程执行成功，更新全局任务状态为"已完成"
        update_task_status(task_id, TASK_STATUS_COMPLETED)
        logger.info(f"[{task_id}] LangGraph 全流程执行完毕，任务完成")

    except Exception as e:
        # 发生异常时，更新全局任务状态为"失败"
        update_task_status(task_id, TASK_STATUS_FAILED)
        logger.error("[{task_id}] 执行异常，异常信息：{e}", task_id=task_id, e=e, exc_info=True)


# 路由：POST /upload
# 功能：接收上传的 Markdown 文件，保存到本地并启动后台导入流程
@app.post("/upload")
async def upload_file(files: list[UploadFile], background_tasks: BackgroundTasks):
    """
    处理文件上传的 POST 接口。

    接收前端上传的一个或多个 Markdown 文件，为每个文件生成唯一 task_id，
    保存到本地目录（output/YYYYMMDD/task_id/文件名），
    并将 LangGraph 处理流程作为后台任务异步执行。

    Args:
        files (list[UploadFile]): 前端上传的文件列表。
        background_tasks (BackgroundTasks): FastAPI 后台任务对象。

    Returns:
        dict: 包含状态码、消息和所有任务 ID 列表的响应。
    """
    # 获取当前日期字符串，格式：年月日
    today_str = datetime.now().strftime("%Y%m%d")
    # 构建基于日期的根目录：PROJECT_ROOT/output/YYYYMMDD
    date_based_root_dir: Path = PROJECT_ROOT / "output" / today_str

    # 用于收集本次上传所有任务的 ID
    task_ids = []

    # 遍历每一个上传的文件
    for file in files:
        # 生成全局唯一的任务 ID
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)
        logger.info(f"[{task_id}] 开始处理上传文件，文件名：{file.filename}，文件类型：{file.content_type}")

        # 记录一个正在运行的任务（用于任务状态跟踪）
        add_running_task(task_id, "upload_file")

        # 构建该任务的本地独立目录：output/YYYYMMDD/task_id，避免多文件重名冲突
        task_local_dir: Path = date_based_root_dir / task_id
        task_local_dir.mkdir(parents=True, exist_ok=True)

        # 构建上传文件的本地保存绝对路径
        local_file_abs_path: Path = task_local_dir / file.filename

        # 读取上传文件的字节数据并写入本地文件
        data = await file.read()
        local_file_abs_path.write_bytes(data)
        logger.info(f"[{task_id}] 文件已保存至本地，路径：{local_file_abs_path}")

        # 标记上传任务已完成
        add_done_task(task_id, "upload_file")

        # 将 LangGraph 调用流程加入后台任务，以异步方式执行
        background_tasks.add_task(
            run_graph_task,
            task_id,
            str(task_local_dir),
            str(local_file_abs_path)
        )
        logger.info(f"[{task_id}] 已将 LangGraph 全流程加入后台任务，任务已启动")

    # 返回响应，包含本次上传的所有任务 ID
    return {
        "code": 200,
        "message": f"Files uploaded successfully, total: {len(files)}",
        "task_ids": task_ids
    }


# 路由：GET /status/{task_id}
# 功能：根据任务 ID 查询该任务的节点执行进度和全局状态
@app.get("/status/{task_id}")
def get_status(task_id: str):
    """
    查询指定任务的执行状态。

    返回当前任务正在运行的节点列表、已完成的节点列表，
    以及任务的全局状态（processing/completed/failed）。

    Args:
        task_id (str): 任务的唯一标识符。

    Returns:
        dict: 包含任务 ID、已完成节点列表、正在运行节点列表和任务全局状态的响应。
    """
    running_task_list: list[str] = get_running_task_list(task_id)
    done_task_list: list[str] = get_done_task_list(task_id)
    task_status = get_task_status(task_id)

    return {
        "code": 200,
        "task_id": task_id,
        "done_list": done_task_list,
        "running_list": running_task_list,
        "status": task_status
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)