"""
    @Author:MaYuhang
    @Time:2026/10/19
    @DESC:金融智库-混合向量化节点（BGE-M3稠密+稀疏）
"""
import json
from pathlib import Path

from common.logging.logger import logger, node_log, step_log
from processor.import_processor.state import ImportGraphState
from utils.lm.embedding_utils import generate_embeddings
from utils.task_utils import add_running_task, add_done_task
# 向量化批次大小：每批处理 5 条切片，避免显存溢出
EMBEDDING_BATCH_SIZE = 5

@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state):
    """
    步骤 1: 校验并获取 chunks 数据
    获取策略（优先级从高到低）：
      1. 优先从 state 中直接取 chunks；
      2. 若 state 中没有 chunks，则尝试从 md_path 同目录下的 JSON 备份文件中恢复，
         并为恢复出来的每个 chunk 补充 item_name 属性；
      3. 若两条路径都取不到数据，则记录错误日志并抛异常终止流程。
    """
    #从状态中获取chunks数据
    chunks = state.get("chunks")

    if not chunks:
        #state 里没有 chunks，尝试从 local_dir 下的 JSON 备份恢复
        md_path = state.get("md_path")
        md_path_obj = Path(md_path)
        #判断 md 文件是否存在
        if md_path_obj.is_file():
            #推算 local_dir 下、同主名的 JSON 备份路径
            local_dir = state.get("local_dir")
            json_path_obj: Path = Path(local_dir) / f"{md_path_obj.stem}.json"
            #判断 JSON 备份是否存在
            if json_path_obj.is_file():
                #从 JSON 读取 chunks
                chunks = json.loads(json_path_obj.read_text(encoding="utf-8"))
                #从恢复出来的 chunks 缺少 item_name,统一补上（语义为主体名）
                for chunk in chunks:
                    chunk["item_name"] = state.get("item_name")
                state["chunks"] = chunks
            else:
                logger.error(f"chunks没有值,json备份文件不存在,抛出异常!")
                raise ValueError(f"chunks没有值,json备份文件不存在,抛出异常!")
        else:
            logger.error(f"chunks没有值,同时也没有读取到md_path,抛出异常!")
            raise ValueError(f"chunks没有值,同时也没有读取到对应md_path,抛出异常!")
    return chunks

@step_log("step_2_generate_vector")
def step_2_batch_generate_vector(chunks):
    """
    步骤 2: 批量生成向量
    把每个 chunk 的 item_name（主体名）和 content 拼接后做向量化，
    将生成的稠密向量和稀疏向量回填到对应 chunk 中。
    采用攒批方式处理，避免一次性拼接全部内容导致内存压力。
    """
    # 按 EMBEDDING_BATCH_SIZE 分批遍历 chunks
    for i in range(0, len(chunks),EMBEDDING_BATCH_SIZE):
        #取出当前批次
        current_chunks = chunks[i:i+EMBEDDING_BATCH_SIZE]
        #拼接 主体名 和 content，作为向量化输入
        content_list = [chunk.get("item_name") + "_" + chunk.get("content") for chunk in current_chunks]
        #向量化处理，返回 {"dense": [...], "sparse": [...]}
        result = generate_embeddings(content_list)
        dense_list = result.get("dense")
        sparse_list = result.get("sparse")
        # 把本批的向量按顺序回填到对应 chunk 中
        for index, chunk in enumerate(current_chunks):
            chunk["dense_vector"] = dense_list[index]
            chunk["sparse_vector"] = sparse_list[index]

    logger.info(f"chunks已经批量生成向量! 共{len(chunks)}条")


@node_log("node_bge_embedding")
def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 向量化 (node_bge_embedding)
    """
    # TODO 1. 添加节点到运行时列表
    add_running_task(state.get("task_id"), "node_bge_embedding")
    # TODO 2. 从状态中获取数据并进行校验
    chunks = step_1_validate_and_get_data(state)
    # TODO 3. 对chunk中的 主体名 + 内容 进行嵌入操作
    step_2_batch_generate_vector(chunks)
    # TODO 4. 更新状态
    state["embeddings_content"] = chunks
    # TODO 5. 添加节点到已完成列表
    add_done_task(state.get("task_id"), "node_bge_embedding")
    return state
