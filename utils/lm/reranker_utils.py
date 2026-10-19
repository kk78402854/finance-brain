import huggingface_hub.utils._validators as _hf_validators

# 修复 huggingface_hub 对本地模型路径的误校验：
# @validate_hf_hub_args 装饰的函数（如 try_to_load_from_cache）会把传入的
# 本地路径 "D:/ai_models/..." 当作 repo_id 校验，而 Windows 本地路径含多个
# "/" 触发 HFValidationError。对包含路径分隔符的字符串跳过 repo_id 校验。
_orig_validate_repo_id = _hf_validators.validate_repo_id


def _patched_validate_repo_id(repo_id):
    if isinstance(repo_id, str) and ("/" in repo_id or "\\" in repo_id):
        return
    return _orig_validate_repo_id(repo_id)


_hf_validators.validate_repo_id = _patched_validate_repo_id

import threading

from FlagEmbedding import FlagReranker
from common.config.reranker_config import reranker_config

_reranker_model = None
_reranker_lock = threading.Lock()

def get_reranker_model():
    global _reranker_model, _reranker_lock
    if _reranker_model is not None:
        return _reranker_model
    with _reranker_lock:
        if _reranker_model is not None:
            return _reranker_model
        _reranker_model = FlagReranker(
            model_name_or_path=reranker_config.bge_reranker_large,
            device=reranker_config.bge_reranker_device,
            use_fp16=reranker_config.bge_reranker_fp16
        )
        return _reranker_model

def compute_token_number( content: str):
    """
      计算字符串对应的token的数量
    :param content:
    :return:
    """
    reranker_model = get_reranker_model()
    tokenizer = reranker_model.tokenizer
    token_list = tokenizer.encode(content, add_special_tokens=False)
    # for token in token_list:
    #     print(token,tokenizer.decode(token))
    return len(token_list)

if __name__ == "__main__":
    print(compute_token_number("RRF是一种倒数排名融合算法"))