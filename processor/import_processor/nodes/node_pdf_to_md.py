"""
    @Author:MaYuhang
    @Time:2026/9/23
    @DESC:
"""
import os
import shutil
import sys
import time
from pathlib import Path

from utils.path_util import PROJECT_ROOT
import requests

from common.config.mineru_config import mineru_config
from common.logging.logger import logger, node_log, step_log
from processor.import_processor.state import ImportGraphState, create_default_state
from utils.task_utils import add_running_task, add_done_task

"""
初始状态：
        task_id="test_pdf2md_task_001",
        pdf_path=os.path.join(PROJECT_ROOT,os.path.join("doc", "金融数据", "用户FAQ", "投资者问答.pdf")) 
        local_dir=os.path.join(PROJECT_ROOT, "output")
"""
@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state):

    #第一次判断：状态值是否为空 / 是否传了

    #从状态中获取pdf_path和local_dir
    pdf_path=state.get("pdf_path")
    local_dir=state.get("local_dir")
    #判断pdf_path是否为空
    if not pdf_path:
        logger.error(f"pdf_path的值为空,无法读取文件,直接抛出异常!")
        raise ValueError(f"pdf_path的呵值为空,无法读取呵呵文件,直接抛出异常!")
    # 判断local_dir是否为空
    if not local_dir:
        # 如果local_dir在状态中没有找到，那么指定默认的路径作为输出路径
        logger.warning(f"没有传入local_dir地址,给与默认值!")
        local_dir = PROJECT_ROOT / "output"
        # 更新到状态中
        state["local_dir"] = local_dir


    #第二次判断：路径在文件系统里是否真实存在、类型对不对

    # 将pdf_path以及local_dir转换为path对象
    pdf_path_obj = Path(pdf_path)
    local_dir_obj = Path(local_dir)
    # 判断pdf_path文件是否存在
    if not pdf_path_obj.is_file():
        # pdf地址不存在,抛出异常
        logger.error(f"pdf_path:{pdf_path_obj},不存在或者不是一个文件!")
        raise ValueError(f"pdf_path:{pdf_path_obj},不存在或者不是一个文件!")
    # 判断local_dir文件是否存在
    if not local_dir_obj.is_dir():
        logger.warning(f"local_dir:{local_dir_obj}不存在，或者不是文件夹,我们主动创建!")
        # parents=True  可以创建多层结构文件夹
        # exist_ok=True 存在也不报错! 没有才会创建
        local_dir_obj.mkdir(parents=True,exist_ok=True)

    return pdf_path_obj,local_dir_obj

@step_log("step_2_upload_and_poll")
def step_2_upload_and_poll(pdf_path_obj):
    # =====1.对env中MinerU相关的配置进行校验=====
    if (not mineru_config.api_key) or (not mineru_config.base_url):
        logger.error(f"minerU配置错误,请检查minerU配置!")
        raise ValueError("minerU配置错误,请检查minerU配置!")

    # =====2.申请文件上传地址=====
    token = mineru_config.api_key
    url = f"{mineru_config.base_url}/file-urls/batch"
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "files": [
            {"name": pdf_path_obj.name, "data_id": pdf_path_obj.stem}
        ],
        "model_version": "vlm"
    }
    http_response = requests.post(url, headers=header, json=data)
    http_response_status_code = http_response.status_code
    if http_response_status_code != 200:
        logger.error(f"申请上传地址失败,返回状态码为:{http_response_status_code},请检查minerU配置!")
        raise RuntimeError(f"申请上传地址失败,返回状态码为:{http_response_status_code},请检查minerU配置!")

    # 获取响应的数据
    # http_response.text  以字符串的形式获取响应数据   http_response.content  以字节的形式获取响应数据  http_response.json()以json形式获取服务器数据-->dict
    response_dict = http_response.json()
    # 获取接口响应状态码
    response_dict_code = response_dict.get("code")
    # 获取接口响应信息
    response_dict_msg = response_dict.get("msg")
    if response_dict_code != 0:
        logger.error(f"申请地址网络状态成功!但是业务失败!错误码:{response_dict_code},失败信息:{response_dict_msg}")
        raise RuntimeError(
            f"申请地址网络状态成功!但是业务失败!错误码:{response_dict_code},失败信息:{response_dict_msg}")

    # 获取批量操作的id
    batch_id = response_dict.get("data").get("batch_id")
    # 获取批量上传的文件地址
    file_upload_urls = response_dict.get("data").get("file_urls")

    if not batch_id:
        logger.error(f"没有批量操作标识")
        raise RuntimeError(
            f"没有批量操作标识")

    if not file_upload_urls:
        logger.error(f"没有申请到上传地址")
        raise RuntimeError(
            f"没有申请到上传地址")

    file_upload_url = file_upload_urls[0]
    logger.info(f">>>申请地址成功{file_upload_url}>>>")

    # =====3.上传文件=====
    # 注意：如果通过这种方式进行上传，有很大的概率是不成功
    # requests.put(file_upload_url, data=pdf_path_obj.read_bytes())
    pdf_file_data = pdf_path_obj.read_bytes()
    with requests.session() as session:
        # 纯净版的请求头,不随意携带代理的参数
        session.trust_env = False
        upload_response = session.put(url=file_upload_url, data=pdf_file_data)
        if upload_response.status_code != 200:
            logger.error(f"上传文件失败,返回状态码为:{upload_response.status_code},请检查minerU配置!")
            raise RuntimeError(f"上传文件失败,返回状态码为:{upload_response.status_code},请检查minerU配置!")
    logger.info(f">>>向{file_upload_url}上传{pdf_path_obj}成功>>>")

    # =====4.轮询获取结果=====
    # 轮询可以使用wile循环实现      循环终止条件： 1.获取到zip_url     2. 解析失败     3. 超时了
    # 定义一个超时时间  单位秒       一般情况下解析1页pdf用时大概在0.5~1s，如果pdf文件有600页  那就需要600s
    timeout = 600
    # 每隔多长时间 轮询一次
    interval_time = 3
    # 起始时间
    start_time = time.time()
    # 拉取数据的url地址
    poll_url = f"{mineru_config.base_url}/extract-results/batch/{batch_id}"
    while True:
        # 判断是否超时
        if time.time() - start_time > timeout:
            logger.error(f"轮询超时,请检查minerU配置!")
            raise TimeoutError(f"轮询超时,请检查minerU配置!")

        # 发送请求获取结果
        try:
            poll_response = requests.get(poll_url, headers=header)
        except Exception as e:
            logger.warning(f"请求出现异常!可以稍后重试!!")
            time.sleep(interval_time)
            continue

        # 获取响应的状态码
        poll_response_status_code = poll_response.status_code
        if poll_response_status_code != 200:
            if 500 <= poll_response_status_code < 600:
                # 如果是服务器报错  给机会继续重试
                logger.warning(f"可有修复的网络异常,状态码为:{poll_response_status_code}")
                time.sleep(interval_time)
                continue
            else:
                logger.error(f"不可修复的网络状态异常,状态码为:{poll_response_status_code}")
                raise RuntimeError(f"不可修复的网络状态异常,状态码为:{poll_response_status_code}")

        # 以json字典形式获取响应结果
        poll_response_dict = poll_response.json()

        # 获取业务响应的状态码
        poll_response_dict_code = poll_response_dict.get("code")
        # 获取业务响应的状态信息
        poll_response_dict_msg = poll_response_dict.get("msg")

        if poll_response_dict_code != 0:
            logger.error(f"轮询业务异常,错误码:{poll_response_dict_code},失败信息:{poll_response_dict_msg}")
            raise RuntimeError(f"轮询业务异常,错误码:{poll_response_dict_code},失败信息:{poll_response_dict_msg}")

        extract_result = poll_response_dict.get("data").get("extract_result")[0]
        # 获取解析的状态
        extract_result_state = extract_result.get("state")
        if extract_result_state == "done":
            extract_result_url = extract_result.get("full_zip_url")
            if not extract_result_url:
                logger.error(f"已经完成了解析,但是zip地址为空!!")
                raise RuntimeError(f"已经完成了解析,但是zip地址为空!!")
                # 5. 返回压缩地址即可
            logger.info(f">>>获取mineru服务器的解析结果：{extract_result_url}>>>")
            return extract_result_url
        elif extract_result_state == "failed":
            logger.error(f"已经完成了解析,但是失败了!!失败信息:{extract_result['err_msg']}")
            raise RuntimeError(f"已经完成了解析,但是失败了!!失败信息:{extract_result['err_msg']}")
        else:
            logger.warning(f"解析正在进行中,状态:{extract_result_state}!")
            time.sleep(interval_time)
            continue

@step_log("step_3_download_and_extract")
def step_3_download_and_extract(zip_url, local_dir_obj, stem):
    # 发送请求，从指定的路径中下载zip压缩包
    response = requests.get(zip_url, timeout=60)
    if response.status_code != 200:
        logger.error(f"从{zip_url}下载文件失败!")
        raise RuntimeError(f"从{zip_url}下载文件失败!")

    # 将下载的zip压缩包  保存到指定的文件中
    md_zip_path_obj: Path = local_dir_obj / f"{stem}_result.zip"
    md_zip_path_obj.write_bytes(response.content)

    # 解压
    # 指定解压的目录
    extract_dir_obj: Path = local_dir_obj / stem
    # 先删除目录内容，避免旧的数据对新的结果造成污染
    if extract_dir_obj.is_dir():
        shutil.rmtree(extract_dir_obj)
    # 创建新的目录
    extract_dir_obj.mkdir(parents=True, exist_ok=True)
    # 真正的解压操作
    # 解压 ZIP 包到目标目录
    # 参数1: 压缩文件
    # 参数2: 解压到哪文件夹
    shutil.unpack_archive(md_zip_path_obj, extract_dir_obj)

    # 到解压的目录中找到md文件
    md_file_list = list(extract_dir_obj.rglob("*.md"))
    # 判断是否存在md文件
    if not md_file_list:
        logger.error(f"文件解压失败,在:{extract_dir_obj}没有任何md文件!")
        raise RuntimeError(f"文件解压失败,在:{extract_dir_obj}没有任何md文件!")

    # 找到对应的解析的结果md   优先级1: 和pdf同名       优先级2: full.md    优先级3: 拿列表第一个
    for md_file_path_obj in md_file_list:
        if md_file_path_obj.stem == stem:
            logger.info(f"文件{md_file_path_obj}解压成功")
            return md_file_path_obj

    target_file_obj = None
    for md_file_path_obj in md_file_list:
        # 如果没有找到和pdf同名md文件，再去找full.md，但是需要改名
        if md_file_path_obj.name.lower() == "full.md":
            target_file_obj = md_file_path_obj
            logger.info(f"文件解压成功,但是名字是full.md,后续需要重命名")
            break

    if not target_file_obj:
        # 没有同名的，也没有叫full.md   从列表中取第一个
        target_file_obj = md_file_list[0]

    # 如果md的名字要是和pdf不一致，需要改名
    final_md_path_obj = target_file_obj.rename(target_file_obj.with_name(f"{stem}.md"))

    logger.info(f"文件{md_zip_path_obj}解压成功")

    return final_md_path_obj



@node_log("node_pdf_to_md")
def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    节点: PDF转Markdown (node_pdf_to_md)
    为什么叫这个名字: 核心任务是将 PDF 非结构化数据转换为 Markdown 结构化数据。
    未来要实现:
    1. 调用 MinerU (magic-pdf) 工具。
    2. 将 PDF 转换成 Markdown 格式。
    3. 将结果保存到 state["md_content"]。
    """
    #TODO.节点添加运行时列表
    add_running_task(state.get("task_id"),"node_pdf_to_md")
    #TODO.状态中获取数据并校验
    pdf_path_obj,local_dir_obj=step_1_validate_and_get_data(state)
    #TODO.上传文件轮询获取结果
    zip_url = step_2_upload_and_poll(pdf_path_obj)
    #TODO.下载并进行解压
    md_path = step_3_download_and_extract(zip_url, local_dir_obj, pdf_path_obj.stem)
    #TODO.更新状态的mdpath
    state["md_path"] = md_path
    #TODO. 节点添加完成列表
    add_done_task(state.get("task_id"),"node_pdf_to_md")
    return state

if __name__ == "__main__":

    # 单元测试：验证PDF转MD全流程
    logger.info("===== 开始node_pdf_to_md节点单元测试 =====")

    logger.info(f"测试获取根地址：{PROJECT_ROOT}")

    test_pdf_name = os.path.join("doc", "金融数据", "用户FAQ", "投资者问答.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)

    # 构造测试状态
    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    node_pdf_to_md(test_state)

    logger.info("===== 结束node_pdf_to_md节点单元测试 =====")