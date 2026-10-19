"""
    @Author:MaYuhang
    @Time:2026/10/19
    @DESC:金融智库-主体识别节点（LLM抽取主实体名+金融元数据：类型/代码/风险等级/机构/日期）
"""
import json
import re
import threading
from pathlib import Path
from typing import Any

from pymilvus import DataType

from common.config.milvus_config import milvus_config
from common.logging.logger import logger, node_log, step_log
from processor.import_processor.state import ImportGraphState
from utils.clients.milvus_utils import get_milvus_client
from utils.escape_milvus_string_utils import escape_milvus_string
from utils.lm.embedding_utils import generate_embeddings
from utils.lm.lm_utils import get_llm_client
from utils.load_prompt import load_prompt
from utils.rate_limit_utils import apply_api_rate_limit
from utils.task_utils import add_running_task, add_done_task

# Milvus 集合创建/加载锁，防止并发导入时重复创建或未load就insert
_collection_lock = threading.Lock()

# 主体识别上下文切片数：取前 K 个切片作为 LLM 抽取上下文
ITEM_NAME_CONTEXT_CHUNK_K = 3
# 主体识别上下文总字符数上限
ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS = 2000

# 元数据中允许回填到 chunk 的字段（空值不回填）
_METADATA_FIELDS = ("content_type", "product_code", "risk_level", "institution", "publish_date")


@step_log("step_1_validata_and_get_data")
def step_1_validata_and_get_data(state) -> tuple[list[dict[str, Any]], str, str]:
    """
    步骤 1: 获取输入并校验
    从 State 中提取 chunks、file_title、md_content，并进行基础校验：
    - 若 chunks 为空，则尝试从 md_path 对应的 JSON 备份文件中恢复；
    - 若 file_title 为空，则用 md_path 的文件名（去后缀）作为默认标题。
    """
    chunks: list[dict[str, Any]] = state.get("chunks")
    file_title: str = state.get('file_title')
    md_content: str = state.get("md_content", "")
    md_path = state.get("md_path")
    md_path_obj = Path(md_path)

    # 如果chunks为空，尝试从 local_dir 下的 JSON 备份文件中恢复
    if not chunks:
        if md_path_obj.is_file():
            local_dir = state.get("local_dir")
            json_path_obj: Path = Path(local_dir) / f"{md_path_obj.stem}.json"
            if json_path_obj.is_file():
                chunks = json.loads(json_path_obj.read_text(encoding='utf-8'))
                state['chunks'] = chunks
                logger.warning("从备份文件中读取chunks")
            else:
                logger.error(f"chunks没有值,同时也没有读取到对应json备份数据,抛出异常!")
                raise ValueError(f"chunks没有值,同时也没有读取到对应json备份数据,抛出异常!")
        else:
            logger.error(f"chunks没有值,同时也没有读取到对应md_path,抛出异常!")
            raise ValueError(f"chunks没有值,同时也没有读取到对应md_path,抛出异常!")

    # 如果file_title为空，用md_path的文件名(去后缀）作为默认标题
    if not file_title:
        file_title = md_path_obj.stem or "default_title"
        state['file_title'] = file_title
        logger.warning(f"file_title不存在,给与默认值:{file_title}")

    # md_content 为空时从 md_path 兜底读取
    if not md_content and md_path_obj.is_file():
        md_content = md_path_obj.read_text(encoding="utf-8")

    return chunks, file_title, md_content


@step_log("step_2_build_llm_context")
def step_2_build_llm_context(chunks, md_content) -> str:
    """
    步骤 2: 构建 LLM 抽取上下文
    优先取前 K 个切片的正文（经过切分，正文密度高）；
    切片不可用时回退使用 md_content 头部。
    总长度不超过 ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS。
    """
    parts: list[str] = []
    for chunk in chunks[:ITEM_NAME_CONTEXT_CHUNK_K]:
        text = str(chunk.get("content", "")).strip()
        if text:
            parts.append(text)

    context = "\n\n".join(parts)
    if not context:
        # 切片内容缺失时，回退取 MD 全文头部
        context = md_content[:ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS]

    if len(context) > ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS:
        context = context[:ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS]
    logger.info(f"主体识别上下文构建完成，长度: {len(context)} 字符")
    return context


def _parse_llm_json(raw_text: str) -> dict:
    """
    健壮解析 LLM 输出的 JSON：
    1. 剥离 ```json 代码围栏；
    2. 截取首个 { 到最后一个 } 之间的内容；
    3. 解析失败返回空 dict，由调用方兜底。
    """
    text = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start:end + 1]
    try:
        return json.loads(text)
    except Exception:
        return {}


@step_log("step_3_extract_metadata_by_llm")
def step_3_extract_metadata_by_llm(context, file_title) -> dict:
    """
    步骤 3: 调用 LLM 抽取主实体名与金融元数据
    输出结构:
        {
            "item_name": "易方达智造优势混合",
            "content_type": "产品说明书",
            "product_code": "020988",
            "risk_level": "R3",
            "institution": "易方达基金",
            "publish_date": "2026-03-31"
        }
    LLM 调用失败或解析失败时，回退 item_name=file_title、其余字段为空。
    """
    prompt_text = load_prompt(
        name="finance_entity_recognition",
        file_title=file_title,
        context=context,
    )
    llm = get_llm_client()
    try:
        apply_api_rate_limit()
        raw_output = llm.invoke(prompt_text).content
        metadata = _parse_llm_json(raw_output)
    except Exception as e:
        logger.error(f"LLM 主体识别调用失败: {e}，回退使用 file_title")
        metadata = {}

    # 兜底与规范化
    item_name = str(metadata.get("item_name", "")).strip() or file_title
    metadata["item_name"] = item_name
    for field in _METADATA_FIELDS:
        metadata[field] = str(metadata.get(field, "")).strip()

    logger.info(f"主体识别结果: item_name={metadata['item_name']}, "
                f"content_type={metadata['content_type']}, risk_level={metadata['risk_level']}, "
                f"institution={metadata['institution']}")
    return metadata


@step_log("step_4_padding_item_name_to_chunks")
def step_4_padding_item_name_to_chunks(chunks, metadata):
    """
    步骤 4: 将识别出的主实体名与元数据回填到每个 chunk 中
    item_name 必填；其余元数据字段非空才回填（Milvus 动态字段，按需存储）。
    """
    for chunk in chunks:
        chunk["item_name"] = metadata["item_name"]
        for field in _METADATA_FIELDS:
            if metadata.get(field):
                chunk[field] = metadata[field]


@step_log("step_5_prepared_item_name_collection")
def step_5_prepared_item_name_collection():
    """
    步骤 5: 准备主实体名集合
    确保 Milvus 中用于存储主实体名的 collection 已存在：
    - 若已存在，直接复用；
    - 若不存在，则创建 schema、索引并建集合。
    该集合用于存放主实体名及其向量，便于后续按产品/机构名检索。
    """
    global _collection_lock
    with _collection_lock:
        # 获取操作 Milvus 数据库的客户端
        milvus_client = get_milvus_client()
        # 判断集合是否存在：存在则直接复用，不存在才走创建流程
        has_collection = milvus_client.has_collection(collection_name=milvus_config.item_name_collection)
        if not has_collection:
            logger.info(f"{milvus_config.item_name_collection}不存在,进行集合的创建!")

            # 创建 schema：开启主键自增 + 允许动态字段
            schema = milvus_client.create_schema(
                auto_id=True,
                enable_dynamic_field=True
            )
            # 通过 schema 维护表中的字段
            schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
            schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=512)
            schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=512)
            schema.add_field(field_name="content_type", datatype=DataType.VARCHAR, max_length=128)
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            # 创建索引参数
            index_params = milvus_client.prepare_index_params()
            # 稠密向量字段：HNSW 图索引，COSINE 相似度
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_vector_index",
                index_type="HNSW",
                metric_type="COSINE",
                params={
                    "M": 64,
                    "efConstruction": 100
                }
            )
            # 稀疏向量字段：倒排索引，IP 内积，DAAT_MAXSCORE 加速
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_vector_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"inverted_index_algo": "DAAT_MAXSCORE"}
            )

            # 创建collection
            milvus_client.create_collection(
                collection_name=milvus_config.item_name_collection,
                schema=schema,
                index_params=index_params
            )
            logger.success(f"集合{milvus_config.item_name_collection}创建成功")
        else:
            logger.info(f"{milvus_config.item_name_collection}已经存在,可以直接使用!")
        # 无论新建还是已存在，都确保集合已加载到内存，避免后续 delete/insert 报 collection not loaded
        milvus_client.load_collection(collection_name=milvus_config.item_name_collection)


@step_log("step_6_insert_item_name_data")
def step_6_insert_item_name_data(metadata, file_title):
    """
    步骤 6: 将主实体名写入 Milvus
    流程：
    1. 先按 file_title 删除旧数据（保证同一文件只保留最新主体）；
    2. 对主实体名做向量化（稠密 + 稀疏）；
    3. 将 file_title、主实体名、文档类型及向量写入集合。
    """
    # 获取Milvus 客户端
    milvus_client = get_milvus_client()
    # 先删除该 file_title 下的旧数据，避免重复插入（覆盖式更新）
    milvus_client.delete(
        collection_name=milvus_config.item_name_collection,
        filter=f"file_title=='{file_title}'"
    )
    item_name = metadata["item_name"]
    # 对主实体名做向量化处理：同时得到稠密向量和稀疏向量
    result = generate_embeddings([item_name])
    dense_vector = result.get("dense")[0]
    sparse_vector = result.get("sparse")[0]

    # 将元数据与向量一并插入集合中
    milvus_client.insert(
        collection_name=milvus_config.item_name_collection,
        data=[
            {
                "file_title": file_title,
                "item_name": item_name,
                "content_type": metadata.get("content_type") or "未知",
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector
            }
        ]
    )
    logger.success(f"向{milvus_config.item_name_collection}插入主实体「{item_name}」成功")


@node_log("node_item_name_recognition")
def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 主体识别 (node_item_name_recognition)
    金融项目通过 LLM 从文档切片中抽取主实体名与金融元数据，
    并将元数据回填到每个 chunk（Milvus 动态字段）。
    """
    # TODO 1.添加节点到运行时列表
    add_running_task(state.get("task_id"), "node_item_name_recognition")
    # TODO 2.从状态中获取数据并进行校验
    chunks, file_title, md_content = step_1_validata_and_get_data(state)
    # TODO 3.构建 LLM 抽取上下文
    context = step_2_build_llm_context(chunks, md_content)
    # TODO 4.LLM 抽取主实体名与元数据
    metadata = step_3_extract_metadata_by_llm(context, file_title)
    # TODO 5.向 chunk 中回填 item_name 与元数据属性
    step_4_padding_item_name_to_chunks(chunks, metadata)
    # TODO 6.在 milvus 中创建存储主实体名的 collection
    step_5_prepared_item_name_collection()
    # TODO 7.将数据写到 milvus 的 collection 中
    step_6_insert_item_name_data(metadata, file_title)
    # TODO 8.更新状态
    state["item_name"] = metadata["item_name"]
    state["doc_metadata"] = metadata
    state["chunks"] = chunks
    # TODO 9.添加节点到已完成列表
    add_done_task(state.get("task_id"), "node_item_name_recognition")
    return state

# ===================== 本地测试方法 =====================
import os
from utils.path_util import PROJECT_ROOT

def test_node_item_name_recognition():
    """
    主体识别节点本地测试
    """
    logger.info("=== 开始执行主体识别节点本地测试 ===")
    try:
        # 构造模拟状态（以基金产品资料概要为例）
        mock_state = {
            "task_id": "test_task_123456",
            "file_title": "易方达智造优势混合基金产品资料概要更新",
            "md_path": os.path.join(PROJECT_ROOT, "doc", "基金概要.md"),
            "local_dir": os.path.join(PROJECT_ROOT, "output", "test_task_123456", "基金概要"),
            "md_content": "",
            "chunks": [
                {
                    "title": "产品概况",
                    "parent_title": "产品概况",
                    "content": "易方达智造优势混合型证券投资基金（基金代码：020988，简称：易方达智造优势混合C）"
                               "由易方达基金管理有限公司管理，风险等级为中风险(R3)。"
                }
            ]
        }

        result_state = node_item_name_recognition(mock_state)

        logger.info(f"最终识别主实体名：{result_state.get('item_name')}")
        logger.info(f"文档元数据：{result_state.get('doc_metadata')}")
        logger.info(f"切片数量：{len(result_state.get('chunks', []))}")

        # 验证 Milvus 存储
        milvus_client = get_milvus_client()
        collection_name = milvus_config.item_name_collection
        if milvus_client and collection_name:
            milvus_client.flush(collection_name=collection_name)
            item_name = result_state.get("item_name")
            safe_name = escape_milvus_string(item_name)
            res = milvus_client.query(
                collection_name=collection_name,
                filter=f'item_name=="{safe_name}"',
                output_fields=["file_title", "item_name", "content_type"],
                consistency_level="Strong"
            )
            logger.info(f"Milvus中检索到的数据：{res}")

    except Exception as e:
        logger.error(f"主体识别节点本地测试失败，原因：{str(e)}", exc_info=True)


if __name__ == "__main__":
    test_node_item_name_recognition()
