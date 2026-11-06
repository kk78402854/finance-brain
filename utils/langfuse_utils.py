"""Langfuse 追踪工具：统一初始化客户端、提供 LangChain 回调与 trace 属性上下文"""
import atexit
from contextlib import contextmanager
from typing import Any, Iterator

from common.config.langfuse_config import langfuse_config
from common.logging.logger import logger

_langfuse_client = None
_init_attempted = False


def get_langfuse_client():
    """
    首次调用时初始化全局 Langfuse 客户端。

    - 未配置密钥：只告警一次并返回 None，业务照常运行
    - 初始化失败：记录错误并返回 None，不向上抛异常
    """
    global _langfuse_client, _init_attempted
    if _init_attempted:
        return _langfuse_client
    _init_attempted = True

    if not langfuse_config.enabled:
        logger.warning(
            "[Langfuse] 未配置 LANGFUSE_BASE_URL / LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY，本次不开启链路追踪"
        )
        return None

    if "0.0.0.0" in langfuse_config.base_url:
        logger.warning(
            f"[Langfuse] LANGFUSE_BASE_URL={langfuse_config.base_url} 是服务端监听地址，客户端无法连接，"
            f"请改为 Langfuse 所在虚拟机的实际 IP，例如 http://192.168.10.8:3000"
        )

    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=langfuse_config.public_key,
            secret_key=langfuse_config.secret_key,
            base_url=langfuse_config.base_url,
        )
        # 进程退出前统一 flush，短生命周期脚本也不会丢数据
        atexit.register(flush_langfuse)
        logger.info(f"[Langfuse] 客户端初始化完成，上报地址={langfuse_config.base_url}")
    except Exception as e:
        logger.error(f"[Langfuse] 客户端初始化失败，本次不开启链路追踪：{e}")
        _langfuse_client = None

    return _langfuse_client


def get_langchain_callback():
    """返回可挂到 LangChain / LangGraph 的回调处理器；未启用时返回 None"""
    if get_langfuse_client() is None:
        return None
    try:
        from langfuse.langchain import CallbackHandler

        return CallbackHandler(public_key=langfuse_config.public_key)
    except Exception as e:
        logger.error(f"[Langfuse] LangChain 回调创建失败，本次不开启链路追踪：{e}")
        return None


def langchain_trace_config() -> dict:
    """构造 invoke/stream 的 config 参数：未启用时返回空字典，调用方式保持不变"""
    callback = get_langchain_callback()
    return {"callbacks": [callback]} if callback else {}


@contextmanager
def trace_attributes(
    trace_name: str,
    session_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[None]:
    """统一设置 trace 名称 / 会话 id / 标签 / metadata；未启用时为空上下文"""
    if get_langfuse_client() is None:
        yield
        return

    try:
        from langfuse import propagate_attributes

        ctx = propagate_attributes(
            trace_name=trace_name,
            session_id=session_id,
            tags=tags,
            metadata=metadata,
        )
    except Exception as e:
        # 追踪初始化失败不影响业务主流程
        logger.error(f"[Langfuse] trace 属性设置失败：{e}")
        yield
        return

    # 业务异常必须原样抛出，不能被追踪逻辑吞掉
    with ctx:
        yield


def flush_langfuse() -> None:
    """把缓冲区中的追踪数据发送出去（服务关闭、脚本结束前调用）"""
    if _langfuse_client is None:
        return
    try:
        _langfuse_client.flush()
        logger.info("[Langfuse] 已 flush 待上报的追踪数据")
    except Exception as e:
        logger.warning(f"[Langfuse] flush 失败：{e}")
