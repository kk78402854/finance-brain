"""从项目根目录运行一次真实 LangGraph 查询，并向 Langfuse 发送 Trace。

用法（项目根目录）：
    python perf_trace/trade_demo.py

密钥从 .env 读取（LANGFUSE_BASE_URL / LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY），
不要把 Key 直接写进本文件，避免随代码提交进 git。
"""

import sys
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

# 当前文件位于 perf_trace/trade_demo.py，父级目录为项目根目录。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from langfuse import Langfuse, get_client, propagate_attributes  # noqa: E402
from langfuse.langchain import CallbackHandler  # noqa: E402

from common.config.langfuse_config import langfuse_config  # noqa: E402

TRACE_NAME = "finance-perf-demo"
QUESTION = "易方达智造优势混合基金主要投资什么方向？"


def main() -> None:
    if not langfuse_config.enabled:
        raise RuntimeError(
            "Langfuse 配置缺失：请在项目根目录的 .env 中填写 LANGFUSE_BASE_URL / LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY"
        )

    # 仅在运行测试时加载查询图，供 Web 服务导入配置时不执行此步骤。
    from processor.query_processor.main_graph import query_app

    Langfuse(
        public_key=langfuse_config.public_key,
        secret_key=langfuse_config.secret_key,
        base_url=langfuse_config.base_url,
    )
    client = get_client(public_key=langfuse_config.public_key)
    if not client.auth_check():
        raise RuntimeError(f"Langfuse 认证失败：检查地址 {langfuse_config.base_url}、项目 Key 与网络")

    run_id = uuid4().hex
    session_id = f"class-{run_id}"
    with propagate_attributes(
        trace_name=TRACE_NAME,
        session_id=session_id,
        tags=["classroom", "single-query"],
        metadata={"user_query": QUESTION},
    ):
        state = query_app.invoke(
            {
                "original_query": QUESTION,
                "session_id": session_id,
                "is_stream": False,
            },
            config={"callbacks": [CallbackHandler(public_key=langfuse_config.public_key)]},
        )
    client.flush()

    print("run_id:", run_id)
    print("answer_chars:", len(str(state.get("answer") or "")))
    print(f"在 Langfuse Traces 中搜索 {TRACE_NAME}")


if __name__ == "__main__":
    main()
