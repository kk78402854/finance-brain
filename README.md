# Finance-Brain · 金融知识库智能问答系统

> 基于 LangGraph 编排的金融领域 RAG 系统：支持多格式研报导入、混合检索、流式问答与合规护栏

Python 3.12 · LangGraph · FastAPI · Milvus · BGE-M3 · MongoDB · MinIO

## 项目简介

Finance-Brain 是一套面向金融场景（货币政策报告、基金产品概要、风险揭示书等）的检索增强生成（RAG）系统，包含**数据导入**与**智能问答**两条 LangGraph 编排的处理链路，提供 DeepSeek 风格的 Web 交互界面，支持多轮会话、流式输出与回答配图。

## 系统架构

### 数据导入链路（端口 8003）

```
文件上传(.md/.pdf/.docx/.doc)
   │
   ├─► PDF 解析 ──────── MinerU 云端 API，PDF → Markdown
   ├─► 图片处理 ──────── 图片上传 MinIO + Qwen-VL 视觉摘要（保留图表语义）
   ├─► 智能切分 ──────── 结构感知的文档分块
   ├─► 实体识别 ──────── LLM 抽取金融实体（产品名/机构名），构建检索过滤条件
   ├─► 混合向量化 ────── BGE-M3 同时生成 dense + sparse 向量
   └─► Milvus 入库 ──── 双集合存储（切片集合 + 实体名集合）
```

### 检索问答链路（端口 8002）

```
用户提问
   │
   ├─► 实体确认 ──────── 产品名/机构名识别与确认，支持多轮澄清
   ├─► 向量检索 ──────── BGE-M3 混合检索（dense + sparse），支持 HyDE 变体
   ├─► RRF 融合 ──────── 多路召回结果融合排序
   ├─► 精排 ─────────── bge-reranker-large 重排序
   ├─► 联网搜索 ──────── MCP WebSearch（可选，补充时效性信息）
   └─► 回答生成 ──────── Qwen3 流式生成 + 合规护栏（风险提示/拒答兜底）
```

## 核心技术亮点

| 能力 | 实现方式 |
|------|----------|
| **混合检索** | BGE-M3 单模型同时产出稠密与稀疏向量，Milvus 多路召回后 RRF 融合，兼顾语义理解与关键词精确匹配 |
| **图编排** | LangGraph 状态图管理两条链路，节点职责单一、可独立测试与替换 |
| **流式体验** | SSE 推送 `progress / delta / final` 三类事件，前端实时渲染节点进度与打字机效果 |
| **多模态处理** | PDF 图表经 MinerU 提取后存入 MinIO，Qwen-VL 生成图表摘要参与检索与回答配图 |
| **会话管理** | MongoDB 聚合管道（`$group + $first/$min/$max`）实现会话列表、日期分组与多轮上下文 |
| **合规护栏** | 金融场景定制：禁用词检测、强制风险提示、超出知识范围拒答 |
| **工程细节** | 上传前端防 XSS（textContent 写入）、拖拽事件单一入口 + 文件级去重兜底、服务健康探测、异步任务状态机 |

## 技术栈

- **编排框架**：LangGraph + LangChain
- **Web 服务**：FastAPI + Uvicorn（SSE 流式响应）
- **向量数据库**：Milvus（混合检索）
- **Embedding**：BGE-M3（稠密 + 稀疏双通道，本地 ModelScope 部署）
- **重排序**：bge-reranker-large
- **LLM / VLM**：Qwen3-Flash / Qwen-VL-Plus（DashScope OpenAI 兼容接口）
- **文档解析**：MinerU 云端 API（PDF）、python-docx（Word）
- **存储**：MongoDB（会话与消息）、MinIO（图片对象存储）
- **前端**：原生 HTML/JS（零构建依赖，DeepSeek 风格交互）

## 快速开始

```bash
# 1. 安装依赖（uv 管理）
uv sync

# 2. 配置环境变量
cp .env.example .env   # 填入 MinerU / DashScope API Key，修改 Milvus/MongoDB/MinIO 地址

# 3. 下载本地模型（BGE-M3 与 reranker）
python utils/download_model/download_bgem3.py
python utils/download_model/download_reranker.py

# 4. 启动服务
python api/file_import_service.py   # 导入服务 http://127.0.0.1:8003/import.html
python api/query_service.py         # 查询服务 http://127.0.0.1:8002/chat.html
```

## 目录结构

```
finance-brain/
├── api/                        # FastAPI 服务层
│   ├── file_import_service.py  #   导入服务（上传 / 任务状态 / SSE 进度）
│   └── query_service.py        #   查询服务（问答 / 会话列表 / 历史记录）
├── processor/
│   ├── import_processor/       # 导入链路（LangGraph）
│   │   └── nodes/              #   解析→切分→实体识别→向量化→入库
│   └── query_processor/        # 查询链路（LangGraph）
│       └── nodes/              #   实体确认→检索→RRF→重排→生成
├── utils/
│   ├── clients/                # Milvus / MongoDB / MinIO 客户端封装
│   ├── lm/                     # LLM / Embedding / Reranker 封装
│   └── compliance_utils.py     # 合规护栏
├── page/                       # 前端页面（chat.html / import.html）
├── eval/                       # 检索效果评估
└── doc/                        # 项目文档
```

## 关键设计决策

**为什么选 BGE-M3 而不是 OpenAI Embedding？**
金融文本含大量专业术语与代码（如基金代码、政策文件号），BGE-M3 的稀疏通道可精确匹配这类 token，同时稠密通道负责语义泛化；本地部署还避免了金融数据出域的合规问题。

**为什么用 RRF 而非加权分数融合？**
稠密与稀疏检索的分数量纲不同，直接加权需反复调参；RRF 基于排名倒数融合，无超参、对分数分布不敏感，在多路召回场景下更稳健。

**为什么 SSE 而非 WebSocket？**
问答场景是单向推送（服务端→客户端），SSE 无需协议升级、自带断线重连、经过 HTTP 代理更友好；双向需求（如取消生成）可用 HTTP 接口补充。

**为什么导入与查询拆成两个服务？**
导入是 CPU/GPU 密集型批处理（解析、向量化），查询是低延迟在线服务；拆分后可独立扩缩容，导入高峰不影响问答响应。

## License

MIT
