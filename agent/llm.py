"""언어모델 연결.

1순위  OpenRouter의 무료 오픈소스 모델 (인터넷 API, 도구 호출 기능 사용)
2순위  같은 OpenRouter 모델에게 글로 도구를 고르게 함 (도구 호출 기능이 막힌 무료 모델 대비)
3순위  이 서버에 직접 설치한 Qwen2.5-1.5B (llama.cpp, 과제 3에서 설치한 모델)

모두 '오픈소스' 모델이다. OpenRouter는 모델을 빌려 쓰는 통로일 뿐, 고르는 모델은 공개된 가중치의 모델이다.
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OPENROUTER_URL = os.environ.get("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
# 무료 모델 목록 (앞에서부터 시도). 가중치가 공개된 오픈소스 모델을 이름으로 지정한다.
#   Nemotron 3.5 Super VL (NVIDIA, 120B·활성 12B, 공개 가중치) — 배포 후 실제로 도구 호출에 성공한 모델
#   Qwen3-Coder (Alibaba, Apache 2.0) · gpt-oss-20b (OpenAI 공개 가중치, Apache 2.0) · Llama 3.3 70B (Meta)
# 모두 응답하지 않을 때만 마지막으로 openrouter/free(OpenRouter가 무료 모델 중 하나를 자동 선택)를 쓴다.
DEFAULT_MODELS = ["nvidia/nemotron-3.5-super-vl:free", "qwen/qwen3-coder:free", "openai/gpt-oss-20b:free",
                  "meta-llama/llama-3.3-70b-instruct:free", "openrouter/free"]
LOCAL_MODEL = {"repo": "lmstudio-community/Qwen2.5-1.5B-Instruct-GGUF", "file": "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"}


class LLMError(Exception):
    pass


class RateLimited(LLMError):
    pass


class ToolsUnsupported(LLMError):
    pass


# =============================================================
# 1. OpenRouter (인터넷 API)
# =============================================================
class OpenRouter:
    def __init__(self, key, models=None, timeout=90):
        self.key = key
        self.models = models or DEFAULT_MODELS
        self.timeout = timeout
        self.cooldown_until = 0.0          # 무료 사용 횟수를 다 쓰면 잠시 쉬고 예비 모델을 씀
        self.last_model = None
        self.bad = set()                   # 응답하지 않은 모델은 다음부터 뒤로 미뤄 무료 횟수를 아낌

    @property
    def available(self):
        return bool(self.key) and time.time() >= self.cooldown_until

    def _post(self, payload):
        req = urllib.request.Request(
            OPENROUTER_URL, data=json.dumps(payload).encode("utf-8"), method="POST",
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                     "HTTP-Referer": "https://github.com/csw33400-art/-2", "X-Title": "LLM Data Analysis System"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:400]
            if e.code == 429:
                raise RateLimited(f"무료 사용 한도 초과 (429): {body}")
            if "tool" in body.lower() and e.code in (400, 404):
                raise ToolsUnsupported(body)
            raise LLMError(f"HTTP {e.code}: {body}")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise LLMError(f"연결 실패: {e}")
        if data.get("error"):
            msg = str(data["error"])[:400]
            if "429" in msg or "rate" in msg.lower():
                raise RateLimited(msg)
            raise LLMError(msg)
        if not data.get("choices"):
            raise LLMError(f"빈 응답: {str(data)[:200]}")
        return data

    def chat(self, messages, tools=None, max_tokens=1200, temperature=0.3):
        """메시지를 보내고 (assistant 메시지, 사용한 모델)을 돌려준다. 모델이 안 되면 다음 무료 모델로 넘어간다."""
        if not self.available:
            raise RateLimited("잠시 쉬는 중")
        errors = []
        order = [m for m in self.models if m not in self.bad] + [m for m in self.models if m in self.bad]
        for model in order:
            payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
            if tools:
                payload.update(tools=tools, tool_choice="auto")
            try:
                data = self._post(payload)
                self.bad.discard(model)
                self.last_model = data.get("model", model)
                return data["choices"][0]["message"], self.last_model
            except RateLimited as e:
                errors.append(str(e))
                if "per-day" in str(e) or "free-models-per-day" in str(e):
                    self.cooldown_until = time.time() + 3600   # 하루 한도: 한 시간 뒤 다시 시도
                    raise
                continue                                       # 분당 한도나 모델별 혼잡: 다음 모델
            except ToolsUnsupported:
                errors.append(f"{model}: 도구 호출 미지원")
                if tools:
                    self.bad.add(model)
                continue
            except LLMError as e:
                errors.append(f"{model}: {e}")
                self.bad.add(model)
                continue
        if errors and all("도구 호출 미지원" in x for x in errors):
            raise ToolsUnsupported("; ".join(errors))
        if errors and all("429" in x or "한도" in x for x in errors):
            self.cooldown_until = time.time() + 120
            raise RateLimited("; ".join(errors)[:300])
        raise LLMError("; ".join(errors)[:600])


# =============================================================
# 2. 서버에 설치한 Qwen2.5-1.5B (과제 3의 llm_worker.py를 별도 프로세스로 실행)
# =============================================================
class LocalQwen:
    def __init__(self):
        self.proc = None
        self.lock = threading.Lock()
        self.error = None

    def _threads(self):
        n = os.cpu_count() or 1
        try:
            quota, period = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
            if quota != "max":
                n = min(n, max(1, round(int(quota) / int(period))))
        except (OSError, ValueError):
            pass
        return max(1, min(n, 4))

    def _read(self):
        while True:
            line = self.proc.stdout.readline()
            if not line:
                code = self.proc.poll()
                self.proc = None
                raise LLMError(f"서버 모델 프로세스가 멈췄어요 (종료 코드 {code})")
            try:
                return json.loads(line)
            except ValueError:
                continue

    def start(self):
        if self.proc is not None and self.proc.poll() is None:
            return
        worker = ROOT / "llm_chat" / "llm_worker.py"
        self.proc = subprocess.Popen([sys.executable, str(worker), LOCAL_MODEL["repo"], LOCAL_MODEL["file"], str(self._threads())],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     text=True, encoding="utf-8", bufsize=1)
        msg = self._read()
        if not msg.get("ready"):
            self.proc = None
            raise LLMError(msg.get("error", "서버 모델을 불러오지 못했어요"))

    def chat(self, messages, max_tokens=400, temperature=0.4):
        if not self.lock.acquire(timeout=240):
            raise LLMError("다른 답변을 만드는 중이에요")
        try:
            self.start()
            self.proc.stdin.write(json.dumps({"messages": messages, "temperature": temperature,
                                              "max_tokens": max_tokens}, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
            out = []
            while True:
                msg = self._read()
                if "t" in msg:
                    out.append(msg["t"])
                    continue
                if "error" in msg:
                    raise LLMError(msg["error"])
                return "".join(out)
        except (BrokenPipeError, OSError) as e:
            self.proc = None
            raise LLMError(f"서버 모델과 연결이 끊겼어요 ({e})")
        finally:
            self.lock.release()


# =============================================================
# 3. 글로 도구를 고르게 하는 방식 (도구 호출 기능이 없는 모델용)
# =============================================================
TOOL_TAG = re.compile(r"<tool>\s*(\{.*?\})\s*</tool>", re.S)
JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def tools_as_text(tools):
    lines = []
    for t in tools:
        f = t["function"]
        params = ", ".join(f["parameters"]["properties"]) or "없음"
        lines.append(f"- {f['name']}({params}): {f['description']}")
    return "\n".join(lines)


def text_protocol(tools):
    return ("\n\n[도구 사용법]\n프로그램을 실행해야 하면 다른 말 없이 아래 형식 한 줄만 출력한다.\n"
            '<tool>{"name": "도구이름", "arguments": {"인자": 값}}</tool>\n'
            "도구 결과는 다음 메시지로 받는다. 도구가 필요 없거나 결과를 다 받았으면 한국어로 답한다.\n\n"
            f"[쓸 수 있는 도구]\n{tools_as_text(tools)}")


def parse_text_tool(text):
    """'<tool>{...}</tool>' 또는 ```json {...}``` 에서 도구 이름과 인자를 꺼낸다. 없으면 None."""
    for pat in (TOOL_TAG, JSON_BLOCK):
        m = pat.search(text or "")
        if m:
            try:
                obj = json.loads(m.group(1))
            except ValueError:
                continue
            name = obj.get("name") or obj.get("tool")
            if name:
                return name, obj.get("arguments") or obj.get("args") or {}
    return None
