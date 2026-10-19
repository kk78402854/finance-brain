"""
    @Author:MaYuhang
    @Time:2026/10/19
    @DESC:金融智库-Milvus入库节点
"""
import threading

from pymilvus import DataType

from common.config.milvus_config import milvus_config
from common.logging.logger import logger, node_log, step_log
from processor.import_processor.state import ImportGraphState
from utils.clients.milvus_utils import get_milvus_client
from utils.task_utils import add_running_task, add_done_task

# Milvus 集合创建/加载锁，防止并发导入时重复创建或未load就insert
_collection_lock = threading.Lock()

@step_log("step_1_validate_get_data")
def step_1_validate_get_data(state):
    """
    步骤 1: 校验并获取 embeddings_content
    从 state 中取 embeddings_content，若为空则记录日志并抛异常终止。
    """
    #从状态中获取 embeddings_content
    embeddings_content = state.get("embeddings_content")
    #判断是否存在
    if not embeddings_content:
        logger.error(f"embeddings_content没有值,业务无法继续,提前终止!")
        raise ValueError(f"embeddings_content没有值,业务无法继续,提前终止!")
    return embeddings_content

@step_log("step_2_prepared_chunks_collection")
def step_2_prepared_chunks_collection():
    """
    步骤 2: 准备 chunks 集合
    确保 Milvus 中用于存储 chunk 的集合存在：
    已存在则直接复用，不存在则创建 schema、索引并建集合。
    """
    global _collection_lock
    with _collection_lock:
        #获取操作 milvus 的客户端
        milvus_client = get_milvus_client()
        #判断集合是否存在：存在直接用，不存在继续执行创建操作
        has_collection = milvus_client.has_collection(collection_name=milvus_config.chunks_collection)
        if not has_collection:
            logger.info(f"{milvus_config.chunks_collection}不存在,进行集合的创建!")

            #创建 schema：开启主键自增和动态字段
            schema = milvus_client.create_schema(
                auto_id=True,
                enable_dynamic_field=True,
            )
            # 添加字段
            schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=512)
            schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=512)
            schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=512)
            schema.add_field(field_name="part", datatype=DataType.INT8)
            schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=512)
            schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            # 创建索引参数
            index_params = milvus_client.prepare_index_params()
            # 稠密向量字段：HNSW 图索引，COSINE 相似度
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_vector_index",
                index_type="HNSW",
                params={
                    "M": 64,
                    "efConstruction": 100
                },
                metric_type="COSINE"
            )
            # 稀疏向量字段：倒排索引，IP 内积，DAAT_MAXSCORE 加速
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_vector_index",
                index_type="SPARSE_INVERTED_INDEX",
                params={"inverted_index_algo": "DAAT_MAXSCORE"},
                metric_type="IP"
            )
            # 创建集合
            milvus_client.create_collection(
                collection_name=milvus_config.chunks_collection,
                schema=schema,
                index_params=index_params
            )
            logger.success(f"集合{milvus_config.chunks_collection}创建成功")
        else:
            logger.info(f"{milvus_config.chunks_collection}已经存在,可以直接使用!")
        # 无论新建还是已存在，都确保集合已加载到内存，避免后续 delete/insert 报 collection not loaded
        milvus_client.load_collection(collection_name=milvus_config.chunks_collection)

@step_log("step_3_insert_chunks_data")
def step_3_insert_chunks_data(embeddings_content):
    """
    步骤 3: 向 Milvus 插入 chunk 数据
    业务逻辑：同一份文档重新入库时，先删掉它的旧数据，再插入新数据，
    保证一个文件在库里只保留最新的一份，避免重复。
    """
    #获取 milvus 客户端
    milvus_client = get_milvus_client()
    # 同一批数据里 file_title 都一样，取第一条代表整批
    file_title = embeddings_content[0].get("file_title")

    # 按 file_title 删除 Milvus 中该文件的旧数据
    milvus_client.delete(
        collection_name=milvus_config.chunks_collection,
        filter=f"file_title=='{file_title}'"
    )

    #把本次切好并已完成向量的 chunk 批量插入
    milvus_client.insert(
        collection_name=milvus_config.chunks_collection,
        data=embeddings_content
    )
    logger.success(f"向{milvus_config.chunks_collection}插入数据成功，共{len(embeddings_content)}条")

@node_log("node_import_milvus")
def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 导入向量库 (node_import_milvus)
    """
    # TODO 1. 添加节点到运行时列表
    add_running_task(state.get("task_id"),"node_import_milvus")
    # TODO 2. 从状态中获取数据并进行校验
    embeddings_content = step_1_validate_get_data(state)
    # TODO 3. 在milvus中创建集合
    step_2_prepared_chunks_collection()
    # TODO 4. 向milvus集合中插入数据
    step_3_insert_chunks_data(embeddings_content)
    # TODO 5. 添加节点到已完成列表
    add_done_task(state.get("task_id"),"node_import_milvus")
    return state

if __name__ == '__main__':
    # --- 单元测试 ---
    import os
    from dotenv import load_dotenv

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    load_dotenv(os.path.join(project_root, ".env"))

    # 构造测试数据
    dim = 1024
    test_state = {
        "task_id": "test_milvus_task",
        "item_name":"三体",
        "file_title": "三体简介.md",
        "embeddings_content": [
            {
                "content": "《三体》是刘慈欣创作的长篇科幻小说",
                "title": "内容正文",
                "item_name": "三体",
                "parent_title":"内容正文",
                "part":1,
                "file_title": "三体简介.md",
                "dense_vector": [0.1] * dim,
                "sparse_vector": {1: 0.5, 10: 0.8}
            }
        ]
    }

    print("正在执行 Milvus 导入节点测试...")
    try:
        result_state = node_import_milvus(test_state)
        print("测试成功")
    except Exception as e:
        print(f"测试失败: {e}")
