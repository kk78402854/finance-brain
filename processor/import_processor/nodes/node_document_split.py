"""
    @Author:MaYuhang
    @Time:2026/10/19
    @DESC:金融智库-文档切分节点（按Markdown标题切分）
"""
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from common.logging.logger import logger, node_log, step_log
from processor.import_processor.state import ImportGraphState
from utils.task_utils import add_running_task, add_done_task

# ====================== 全局配置（可根据模型调整）======================
# 文本切块最大长度：单个文本块最多包含 1000 字符（防止过长导致向量失真）
CHUNK_MAX_SIZE = 1000
# 文本切块基准长度：单个文本块理想大小为 600 字符（兼顾语义完整性 + 检索精度）
CHUNK_SIZE = 600
# 文本块重叠长度：相邻块之间重叠 50 字符，保证语义不被切断、上下文连贯
CHUNK_OVERLAP = 50
# 最小碎片阈值：低于这个长度判定为短碎片，需要尝试合并
CHUNK_MIN = 400

@step_log("step_1_validate_get_data")
def step_1_validate_get_data(state):
    #从状态中获取md_content
    md_content = state.get("md_content")
    #从状态中获取md_path
    md_path = state.get("md_path")
    #从状态中获取file_title
    file_title = state.get("file_title")

    #判断md_content
    if not md_content:
        from pathlib import Path
        if (not md_path) or (not Path(md_path).is_file()):
            logger.error(f"md_content内容为空,md_path也为空或者没有对应的文件,业务无法继续,提前终止!")
            raise ValueError(f"md_content内容为空,md_path也为空或者没有对应的文件,业务无法继续,提前终止!")
        #从md_path中读取文件内容
        md_content = Path(md_path).read_text(encoding="utf-8")
        logger.info(f"从 md_path 读取文件内容，长度: {len(md_content)} 字符")
        #file_title为空时，用md_path的文件名兜底
        if not file_title:
            file_title = Path(md_path).stem or "default"
            state['file_title'] = file_title
            logger.warning(f"file_title为空,给与默认值:{file_title}")

    #对md进行处理,统一不同操作系统的换行符
    md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")

    return md_content,file_title,md_path

@step_log("step_2_split_document_by_title")
def step_2_split_document_by_title(md_content, file_title):
    """
  按 Markdown 标题对文档进行切割，尽量保持标题下的语义完整。

    处理规则：
    1. 识别 # ~ ###### 形式的 Markdown 标题，并按层级维护标题栈。
    2. 代码块（``` 或 ~~~）内的 # 不会被识别为标题。
    3. 每个返回块包含当前标题路径、标题行及其下正文。
    4. 首个标题出现前的内容作为"孤儿内容"，在结算首个标题块时合并进去；
       若全文没有标题，则以 file_title 作为标题输出一个块。
    5. 空行会被跳过，不保留在正文中。

    :param md_content: Markdown 文件内容。
    :param file_title: 文件标题/文件名，用于结果字段和全文无标题时的回退标题。
    :return: list[dict[str, Any]]，每个元素形如：
             {
                 "file_title": "...",
                 "title": "# 一级标题_## 二级标题",
                 "content": "# 一级标题\n正文内容..."
             }
    """
    # 用于存储返回数据的列表
    chunks: list[dict[str, Any]] = []
    # 当前激活的完整标题路径，例如 "# 标题1_## 标题2"；None 表示尚未遇到任何标题
    current_title: str = None
    # 正文缓冲区
    current_content_lines: list[str] = []
    # 孤儿数据缓冲区：第一个标题出现前的内容，或代码块在无标题状态下出现的内容
    orphan_lines: list[str] = []
    # 代码块标记位：True 表示当前位于三个反引号或 ~~~ 代码块内部
    is_code: bool = False
    # 标题栈 ：按层级保存标题行，例如 ["# A", "## B", None, "#### D"]
    # 索引 0 对应 h1，索引 1 对应 h2，None 表示该层级暂时缺失
    heading_stack: list[str] = []

    # 将Markdown 内容按行切分，后续逐行处理
    doucument_lines: list[str] = md_content.splitlines()

    # 匹配Markdown 标题行：1——6 个 #，后跟空白和至少一个字符
    title_reg = re.compile(r"^#{1,6}\s.+")

    # 逐行遍历文档
    for line in doucument_lines:
        # 去除前后空格，用于判断标题,代码块标记等；正文仍使用有原始line
        line_strip = line.strip()
        # 跳过空行，空行不会保留到最终content 中
        if not line_strip:
            continue

        # 判断是否为代码块起始或结束标记；遇到标记就切换 is_code 状态
        if line_strip.startswith("```") or line_strip.startswith("~~~"):
            # 标记位状态取反: 进入代码块 / 跳出代码块
            is_code = not is_code

            # 如果还没有激活标题，标记放入孤儿区；否则放入当前标题正文区
            if not current_title:
                orphan_lines.append(line_strip)
            else:
                current_content_lines.append(line_strip)
            continue

        # 若当前在代码块内，则代码块内容不参与标题识别
        if is_code:
            # 如果还没有激活标题，代码块内容放入孤儿区；否则放入当前标题正文区
            if not current_title:
                orphan_lines.append(line)
            else:
                current_content_lines.append(line)
            continue

        if title_reg.match(line_strip):
            # 标题处理逻辑：先结算上一个标题块，再更新标题栈并激活新标题
            # 对上级标题进行结算：仅当已有激活标题且正文缓存非空时生成块
            if current_title and len(current_content_lines) > 0:
                # 若存在孤儿数据，则合并到当前正文前
                if len(orphan_lines) > 0:
                    # 将孤儿数据和正文数据进行拼接，保证首个标题块包含前置内容
                    current_content_lines = orphan_lines + current_content_lines
                    orphan_lines = []
                full_content = f"{current_title}\n" + "\n".join(current_content_lines)
                chunks.append({
                    "title": current_title,
                    "content": full_content,
                    "file_title": file_title
                })
                current_content_lines = []
            # 维护标题栈内容：根据当前标题层级生成新的标题路径
            # 获取当前标题层级：行首 # 的数量，h1 为 1.h6 为6
            heading_level = len(line_strip) - len(line_strip.lstrip("#"))
            # 处理标题跳级：例如从 h1 直接到 h3，中间缺失层级补 None
            while len(heading_stack) < heading_level:
                heading_stack.append(None)
            # 截断标题栈，只保留到当前层级，丢弃更深层级的旧标题
            heading_stack = heading_stack[:heading_level]
            # 将当前标题行放入标题栈对应层级
            heading_stack[heading_level - 1] = line_strip
            # 激活标题：用标题栈中非空标题拼接当前完整标题路径
            current_title = "_".join([h for h in heading_stack if h])
            continue
        else:
            # 正文处理逻辑：无激活标题时进孤儿区，有激活标题时进正文区
            # 判断是否有激活的标题
            if not current_title:
                orphan_lines.append(line)
            else:
                current_content_lines.append(line)

    # 循环结束后，结算最后一个标题块
    if current_title and (len(current_content_lines) > 0 or (len(orphan_lines) > 0)):
        if len(orphan_lines) > 0:
            current_content_lines = orphan_lines + current_content_lines
            orphan_lines = []
        full_content = f"{current_title}\n" + "\n".join(current_content_lines)
        chunks.append({
            "title": current_title,
            "content": full_content,
            "file_title": file_title
        })
    # 全文没有标题，但存在孤儿内容：以文件名作为标题输出一个块\
    elif (not current_title) and len(orphan_lines) > 0:
        chunks.append({
            "title": file_title,
            "content": "\n".join(orphan_lines),
            "file_title": file_title
        })
    logger.info(f"已经根据标题进行多级切块（极简稳健版），现有的块: {len(chunks)}")
    return chunks

@step_log("_split_chunk_content")
def _split_chunk_content(chunk):
    """
    使用递归切分器对单个超长 chunk 进行二次切分。

    切分时会保留原 chunk 的标题前缀，并为每个子块补充元数据：
    - file_title：原文件标题
    - parent_title：原 chunk 的标题
    - part：子块序号，从 1 开始
    - title：原标题 + "_" + 序号
    - content：标题前缀 + 子块正文

    :param chunk:     需要切分的 chunk，通常结构示例：
     {'title': '## 警告',
     'content': '## 警告\n不遵守说明和警告可能导致人员死亡或严重受伤。遵守这些指引以避免冒烟、发热、爆炸、火灾或人员受伤的风险。',
     'file_title': 'hak180产品安全手册'
     }
    :return:          chunk 列表
    """
    sub_chunks:list[dict[str,Any]] = []

    #获取 chunk 的完整内容
    chunk_content = chunk.get("content")
    #获取 chunk 的标题前缀（标题+换行符）
    prefix = chunk.get("title")+"\n"
    #去掉内容开头的前缀，只保留正文部分用于二次切分
    deal_content=chunk_content[len(prefix):]

    #创建递归切分器对象
    splitter= RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE - len(prefix),
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " "]
    )

    #对切分后的文本片段进行遍历，并组装成子 chunk
    for index,text in enumerate(splitter.split_text(deal_content),start=1):
        sub_chunks.append({
            "file_title": chunk.get("file_title"),
            "parent_title": chunk.get("title"),
            "part": index,
            "title": f"{chunk.get('title')}_{index}",
            "content": prefix + text
        })

    return sub_chunks

@step_log("_merge_chunk_content")
def _merge_chunk_content(refine_chunks)->list[dict[str,Any]]:
    """
    对递归切分后的 chunk 列表进行合并处理，减少碎片块。

    输入 chunk 结构说明：
    - 如果没有做递归切分：{title, file_title, content}
    - 如果做了递归切分：{title, file_title, content, parent_title, part}

    合并规则：
    - 以 base_chunk 为基准，依次与 next_chunk 比较；
    - 仅当 base_chunk 内容未超过 CHUNK_MIN，且两者 parent_title 相同，
      且合并后内容未超过 CHUNK_MAX_SIZE 时，才进行合并；
    - 合并时，next_chunk 的 content 会去掉其 parent_title + "\n" 前缀，避免重复标题；
    - 否则将 base_chunk 放入结果，并将 next_chunk 作为新的基准 chunk。

    :param refine_chunks: 经过递归切分或未切分的 chunk 列表
    :return: 合并后的 chunk 列表
    """
    #用于存储合并后的 chunk 列表
    merge_refine_chunks: list[dict[str, Any]] = []

    # 当前用于尝试合并的基准 chunk
    base_chunk:dict[str,Any] = None

    # 对 refine_chunks 列表进行遍历
    for next_chunk in refine_chunks:
        # 首次进入循环时，base_chunk 为空，先把第一个 chunk 设为基准
        if base_chunk is None:
            base_chunk = next_chunk
            continue
            # 先判断当前基准 chunk 的内容长度是否超过 CHUNK_MIN
        is_too_long = len(base_chunk.get("content")) > CHUNK_MIN
        if not is_too_long:
            # 未超过 CHUNK_MIN，尝试与下一个 chunk 合并
            # 判断基准 chunk 和当前遍历到的 chunk 是否属于同一个父标题
            is_same_parent_title = (base_chunk.get("parent_title") and base_chunk.get("parent_title") == next_chunk.get(
                "parent_title"))
            if is_same_parent_title:
                # 属于同一个父标题，尝试进行合并
                # 计算合并后的长度，判断是否会超过 CHUNK_MAX_SIZE(1000)
                base_content: str = base_chunk.get("content")
                next_cleared_content: str = next_chunk.get("content")[len(next_chunk.get("parent_title")) + 1:]
                is_merged_too_long = (len(base_content) + len(next_cleared_content)) > CHUNK_MAX_SIZE
                if not is_merged_too_long:
                    # 合并后未超限，执行合并
                    base_chunk['content'] = base_content + "\n" + next_cleared_content
                else:
                    # 合并后会超限，先结算基准 chunk，再以下一个 chunk 为新基准
                    merge_refine_chunks.append(base_chunk)
                    base_chunk = next_chunk
            else:
                # 不属于同一个父标题，不能合并，先结算基准 chunk
                merge_refine_chunks.append(base_chunk)
                # 将当前遍历到的 chunk 设为新的基准 chunk
                base_chunk = next_chunk
        else:
            # 基准 chunk 已超过 CHUNK_MIN，不再合并，先结算基准 chunk
            merge_refine_chunks.append(base_chunk)
            # 将当前遍历到的 chunk 设为新的基准 chunk
            base_chunk = next_chunk

    if base_chunk:
        merge_refine_chunks.append(base_chunk)

    logger.info(f"完成小于400的chunk的合并,合并后的数量:{len(merge_refine_chunks)}")

    return merge_refine_chunks

@step_log("step_3_refine_split_and_merge_chunks")
def step_3_refine_split_and_merge_chunks(title_chunks):
    """
    对按标题切分后的 chunks 进行精细化处理：
    1. 对内容长度超过 CHUNK_SIZE 的 chunk，使用递归切分器再次切分；
    2. 对同一父标题下、内容较小的 chunk 进行合并，避免块过碎。

    :param title_chunks: 经过按标题切分后得到的 chunk 列表
    :return: 经过精细化切分与合并后的 chunk 列表
    """
    refine_chunks: list[dict[str,Any]] = []

    #遍历标题切分阶段产出的 chunk 列表
    for chunk in title_chunks:
        if len(chunk.get("content")) > CHUNK_SIZE:
            #内容超过 CHUNK_SIZE ：调用递归切分器进一步切分
            refine_chunks.extend(_split_chunk_content(chunk))
        else:
            # 内容未超过 CHUNK_SIZE ：无需切分，直接保留
            refine_chunks.append(chunk)

    logger.info(f"chunks经过超长以后向短切割处理！切割后的数量：{len(refine_chunks)}")
    # 对递归切分后的 chunks 进行合并。
    # 合并前提：
    # 1. 属于同一个父标题（parent_title）下；
    # 2. 待合并的 chunk 内容小于 CHUNK_MIN（400）；
    # 3. 合并后的内容小于 CHUNK_MAX_SIZE（1000）。
    refine_chunks = _merge_chunk_content(refine_chunks)
    return refine_chunks

@step_log("step_4_padding_chunks_metadata")
def step_4_padding_chunks_metadata(refine_chunks):
    """
       为切分后的 chunk 补充元数据字段，保证每个 chunk 都包含 parent_title 和 part。

       refine_chunks: list[dict[str, Any]] 中元素结构：
       - 未做递归切分：{title, file_title, content}
       - 已做递归切分：{title, file_title, content, parent_title, part}

       补充规则：
       - 若缺少 parent_title，则用当前 chunk 的 title 填充；
       - 若缺少 part，则默认填充为 1。

       :param refine_chunks: 经过精细化切分与合并后的 chunk 列表
       :return: 无返回值，函数直接原地修改 refine_chunks
       """
    # 遍历 refine_chunks，为每个 chunk 补齐必要元数据
    for chunk in refine_chunks:
        if "parent_title" not in chunk:
            # 若没有 parent_title，则将自身 title 作为父标题
            chunk["parent_title"] = chunk.get("title")

        if "part" not in chunk:
            # 若没有 part，则默认视为第 1 部分
            chunk["part"] = 1

    logger.info(f"完成chunks的元素补充,所有属性都完整!")

@step_log("step_5_backup_chunks_json")
def step_5_backup_chunks_json(refine_chunks, md_path, local_dir):
    """
    将 chunks 数据备份为 JSON 文件。
    金融项目规范：输出到 local_dir 目录下，避免污染源文件目录。
    """
    # 将 md_path 转为 Path 对象，取文件名 stem
    md_path_obj = Path(md_path)
    # 输出 JSON 路径：local_dir/{stem}.json
    local_dir_obj = Path(local_dir)
    local_dir_obj.mkdir(parents=True, exist_ok=True)
    json_path_obj = local_dir_obj / f"{md_path_obj.stem}.json"
    # 将 refine_chunks 序列化为 JSON 并写入目标文件；保留中文，缩进 4 空格
    json_path_obj.write_text(data=json.dumps(refine_chunks,ensure_ascii=False,indent=4),encoding="utf-8")
    # 记录备份完成日志，输出备份位置
    logger.info(f"完成chunks数据的备份,备份位置:{str(json_path_obj)}")

@node_log("node_document_split")
def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 文档切分 (node_document_split)
    """
    #TODO 1. 将节点添加到运行时列表
    add_running_task(state.get("task_id"),"node_document_split")
    #TODO 2. 从状态中获取数据并进行校验（md_content、file_title、md_path）
    md_content,file_title,md_path = step_1_validate_get_data(state)
    #TODO 3. 按照标题对文档进行切割————保语义
    title_chunks:list[dict[str,Any]] = step_2_split_document_by_title(md_content,file_title)
    #TODO 4. 精细化切分————保大小，边界
    refine_chunks: list[dict[str, Any]] = step_3_refine_split_and_merge_chunks(title_chunks)
    #TODO 5. 补充一些元数据信息(part,parent_title)————可追溯
    step_4_padding_chunks_metadata(refine_chunks)
    #TODO 6. 将chunk持久化到磁盘文件
    step_5_backup_chunks_json(refine_chunks, md_path, state.get("local_dir"))
    #TODO 7. 更新状态
    state["chunks"] = refine_chunks
    #TODO 8. 将节点添加到已完成列表
    add_done_task(state.get("task_id"),"node_document_split")
    return state

if __name__ == '__main__':
    """本地测试入口：单独运行该文件时，执行文档切分节点测试"""
    from utils.path_util import PROJECT_ROOT

    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径
    test_md_name = os.path.join(r"doc", "三体简介.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
    else:
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": "",
            "file_title": "三体简介",
            "local_dir": os.path.join(PROJECT_ROOT, "output", "test_task_123456", "三体简介"),
        }
        logger.info("开始本地测试 - 文档切分节点")
        final_state = node_document_split(test_state)
        final_chunks = final_state.get("chunks", [])
        logger.info(f"测试成功：最终生成{len(final_chunks)}个有效Chunk")
