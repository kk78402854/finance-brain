"""Langfuse 链路追踪配置：从 .env 读取上报地址与项目密钥"""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

# 提前加载.env配置文件（必须在读取环境变量前执行，确保os.getenv能获取到值）
load_dotenv()


@dataclass
class LangfuseConfig:
    base_url: str
    public_key: str
    secret_key: str

    @property
    def enabled(self) -> bool:
        """三项配置齐全才上报追踪数据，缺任意一项则整体跳过（不影响业务）"""
        return bool(self.base_url and self.public_key and self.secret_key)


langfuse_config = LangfuseConfig(
    base_url=(os.getenv("LANGFUSE_BASE_URL") or "").strip(),
    public_key=(os.getenv("LANGFUSE_PUBLIC_KEY") or "").strip(),
    secret_key=(os.getenv("LANGFUSE_SECRET_KEY") or "").strip(),
)
