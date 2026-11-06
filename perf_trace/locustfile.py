"""金融知识库查询服务 Locust 压测脚本（Locust 2.x）

覆盖两类真实任务：
  1) 非流式问答：POST /query (is_stream=False)，同步等待整条链路返回 answer
  2) 流式问答：POST /query (is_stream=True) + GET /stream/{session_id}，完整消费 SSE 到 final，
     并单独记录「首 token 延迟」自定义指标

说明（与参考文档的差异，均按本项目真实接口实现）：
  - 服务端口 8002：locust -f perf_trace/locustfile.py --host http://localhost:8002
  - /query 返回体只有 session_id，没有 task_id
  - SSE 事件名以 utils/sse_utils.py 为准：ready / progress / delta / final / error
    · delta 帧 data 为 {"delta": "增量文本"}
    · final 帧 data 为 {"answer": ..., "answer_meta": ..., "status": "completed", "image_urls": [...]}
    · 服务端不会发关闭帧，收到 final 后客户端必须主动断开

用法：
    # 冒烟（headless，2 用户 3 分钟；单次回答 25~40s，-t 太短拿不到完整样本）
    locust -f perf_trace/locustfile.py --host http://localhost:8002 \
           --headless -u 2 -r 1 -t 180s --stop-timeout 190 --only-summary

    # 正式压测（开 Web UI：浏览器访问 http://localhost:8089）
    locust -f perf_trace/locustfile.py --host http://localhost:8002 \
           -u 10 -r 1 -t 5m --stop-timeout 190 --web-port 8089
"""

import json
import os
import random
import time
import uuid

import requests
from locust import HttpUser, between, events, task

# ---------------- 可调参数 ----------------
# 单次请求允许的最长等待（秒）。链路基线约 30s，留足余量避免误判失败
QUERY_TIMEOUT = int(os.getenv("LOCUST_QUERY_TIMEOUT", "180"))
# 流式消费的兜底超时（秒），超过仍未收到 final 记为失败
STREAM_DEADLINE = int(os.getenv("LOCUST_STREAM_DEADLINE", "170"))

# 贴近真实使用的问题池，避免所有请求命中同一条缓存/同一条切片
QUESTIONS = [
    "易方达智造优势混合基金主要投资什么方向？",
    "华夏沪深300ETF联接C跟踪的指数是什么？",
    "南方中证500ETF的基金费率是多少？",
    "招商中证白酒指数A的基金经理是谁？",
    "工银前沿医疗股票A的投资范围包括哪些？",
    "易方达蓝筹精选混合的风险收益特征是什么？",
]


def _new_session_id(kind: str) -> str:
    """每次请求用独立会话，模拟「新对话」基线，避免历史消息累积导致时延漂移"""
    return f"locust-{kind}-{uuid.uuid4().hex[:12]}"


def _pick_question() -> str:
    return random.choice(QUESTIONS)


@events.test_start.add_listener
def _precheck_service(environment, **_kwargs):
    """压测前自检：服务没起来时直接给明确提示，避免整轮只看到「连接被拒、0 字节」的统计"""
    host = (environment.host or "").rstrip("/")
    if not host:
        return
    try:
        resp = requests.get(f"{host}/health", timeout=5)
        print(f"[precheck] {host}/health -> {resp.status_code} {resp.text[:120]}")
    except Exception as e:
        print(f"[precheck] 无法连接 {host}：{type(e).__name__}: {e}")
        print("[precheck] 请先启动查询服务：.venv\\Scripts\\python.exe api\\query_service.py（端口 8002）")


class FinanceQueryUser(HttpUser):
    """模拟一个正在使用金融问答系统的用户"""

    wait_time = between(
        float(os.getenv("LOCUST_WAIT_MIN", "1")),
        float(os.getenv("LOCUST_WAIT_MAX", "3")),
    )

    @task(2)
    def query_non_stream(self):
        """非流式问答：POST /query 同步阻塞到整条链路结束"""
        session_id = _new_session_id("sync")
        payload = {"query": _pick_question(), "session_id": session_id, "is_stream": False}

        with self.client.post(
            "/query",
            json=payload,
            name="04 complete answer [非流式]",
            timeout=QUERY_TIMEOUT,
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:200]}")
                return
            try:
                data = resp.json()
            except Exception as e:
                resp.failure(f"返回体不是合法 JSON: {e}")
                return
            if not data.get("answer"):
                resp.failure("answer 为空")
                return
            resp.success()

    @task(3)
    def query_stream(self):
        """流式问答：先提交后台任务，再消费 SSE 直到 final"""
        session_id = _new_session_id("stream")
        payload = {"query": _pick_question(), "session_id": session_id, "is_stream": True}
        start = time.perf_counter()

        with self.client.post(
            "/query",
            json=payload,
            name="01 submit /query [流式]",
            timeout=30,
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"提交失败 HTTP {resp.status_code}: {resp.text[:200]}")
                return
            try:
                data = resp.json()
            except Exception as e:
                resp.failure(f"提交返回体不是合法 JSON: {e}")
                return
            if data.get("session_id") != session_id:
                resp.failure(f"返回 session_id 不匹配: {data.get('session_id')}")
                return
            resp.success()

        first_token_ms = None
        answer_chars = 0
        final_seen = False
        error_msg = None
        current_event = None
        deadline = time.time() + STREAM_DEADLINE

        # 2. 消费 SSE 流直到 final；整链路耗时从「提交问题前」开始计算
        with self.client.get(
            f"/stream/{session_id}",
            name="03 complete answer [流式]",
            stream=True,
            timeout=QUERY_TIMEOUT,
            catch_response=True,
        ) as response:
            # 服务端未声明 charset，显式按 UTF-8 解码，保证中文按字符统计
            response.encoding = "utf-8"
            if response.status_code != 200:
                response.failure(f"SSE 连接失败 HTTP {response.status_code}")
                return

            try:
                for line in response.iter_lines(chunk_size=1, decode_unicode=True):
                    if time.time() > deadline:
                        error_msg = "超时未收到 final 事件"
                        break
                    if not line:
                        continue
                    line = line.strip()
                    if line.startswith("event:"):
                        current_event = line[len("event:"):].strip()
                        continue
                    if not line.startswith("data:"):
                        continue

                    raw_data = line[len("data:"):].strip()
                    try:
                        payload_data = json.loads(raw_data) if raw_data else {}
                    except json.JSONDecodeError:
                        payload_data = {}

                    if current_event == "delta":
                        if first_token_ms is None and (payload_data.get("delta") or "").strip():
                            # 首段答案延迟：首个非空增量到达时记录一次；
                            # 单独 fire 一条自定义指标，与整链路耗时区分开
                            first_token_ms = (time.perf_counter() - start) * 1000
                            events.request.fire(
                                request_type="ANSWER",
                                name="02 first answer delta",
                                response_time=first_token_ms,
                                response_length=0,
                                exception=None,
                                context={},
                                url=f"/stream/{session_id}",
                            )
                        answer_chars += len(payload_data.get("delta") or "")
                    elif current_event == "final":
                        final_seen = True
                        answer_chars = max(
                            answer_chars, len(payload_data.get("answer") or "")
                        )
                        break
                    elif current_event == "error":
                        error_msg = payload_data.get("error") or "服务端返回 error 事件"
                        break
            except Exception as e:
                error_msg = f"读取 SSE 失败: {e}"

            if error_msg:
                response.failure(error_msg)
            elif not final_seen:
                response.failure("未收到 final 事件")
            elif answer_chars == 0:
                response.failure("final 到达但 answer 为空")
            else:
                response.success()

            # 3. 覆盖响应头计时：stream=True 时 Locust 默认只记到响应头（几十 ms），
            #    改写为「提交问题 → 收到 final」的整链路耗时，并主动断开 SSE
            response.request_meta["response_time"] = (time.perf_counter() - start) * 1000
            response.close()
