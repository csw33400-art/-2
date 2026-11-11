"""채팅으로만 쓰는 데이터 분석 챗봇 (데이터처리개론 과제)

- 화면에는 채팅창 하나만 있습니다. 버튼·메뉴 없이 말로 시키면 선형회귀, 비선형회귀,
  인공신경망, 딥러닝을 실행하고 그래프·표·해석을 채팅 안에 보여 줍니다.
- 오픈소스 언어모델(GGUF 4비트 양자화)을 이 서버에 내려받아 llama.cpp로 직접 실행합니다.
  외부 AI API를 쓰지 않습니다. 언어모델은 결과 해설과 자유 질문에 답합니다.
- 분석 명령(무엇을 실행할지)은 키워드로 인식해 tools.py의 함수가 정확히 계산합니다.

로컬 실행:
    pip install -r llm_chat/requirements.txt
    streamlit run llm_chat/app.py
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tools as T  # noqa: E402

st.set_page_config(page_title="데이터 분석 챗봇", page_icon="💬", layout="centered")

# =============================================================
# 1. 언어모델 목록 (서버 사양: CPU 2코어 · 메모리 약 2.7GB에 맞춰 선정)
# =============================================================
DEFAULT_MODEL = "Qwen2.5-1.5B-Instruct"
MODELS = {
    "Qwen2.5-1.5B-Instruct": {
        "repo": "lmstudio-community/Qwen2.5-1.5B-Instruct-GGUF",
        "file": "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
        "aliases": ("qwen", "큐원", "큐웬", "알리바바"),
        "developer": "Alibaba Qwen 팀 (중국)", "params": "약 15억 개 (1.5B)", "n_params": 1.5e9,
        "size": "약 0.99GB", "license": "Apache 2.0",
        "feature": "29개 이상 언어 지원(한국어 포함). 작고 빨라서 CPU 서버에 적합",
    },
    "EXAONE-3.5-2.4B-Instruct": {
        "repo": "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct-GGUF",
        "file": "EXAONE-3.5-2.4B-Instruct-Q4_K_M.gguf",
        "aliases": ("exaone", "엑사원", "엑사온", "엘지"),
        "developer": "LG AI연구원 (한국)", "params": "약 24억 개 (2.4B)", "n_params": 2.4e9,
        "size": "약 1.64GB", "license": "EXAONE AI Model License (연구·교육용)",
        "feature": "한국어 특화. 답변이 자연스럽지만 크기가 커서 느림",
    },
}

SYSTEM_PROMPT = """너는 '데이터처리개론' 수업의 데이터 분석 조교 챗봇이다.
규칙: 반드시 한국어로 짧게 답한다. 모르면 모른다고 말한다. 아래에 없는 숫자는 만들지 않는다.

[수업 핵심 개념]
- 선형회귀: y=ax+b. 잔차 제곱합을 최소로 하는 a, b를 최소제곱법으로 구한다.
- 비선형회귀: 다항식·거듭제곱·지수·로그 곡선으로 적합한다.
- 뉴런: y=φ(Σwx+b). 가중치 w, 편향 b, 활성화 함수 φ(시그모이드, ReLU).
- 경사하강법: 오차 E=(t-o)²가 줄도록 w ← w - η·∂E/∂w.
- 오류역전파: 출력의 오차를 거꾸로 전달해 모든 가중치를 고친다(합성함수의 미분법).
- 정규화: z=(x-μ)/σ 또는 MinMax. 과적합 방지: 학습·검증·테스트 분할.
- R²는 1에 가까울수록 좋다. 학습 R²가 테스트 R²보다 많이 높으면 과적합이다.
- 이 챗봇에서 "선형회귀 해줘", "인공신경망 학습해줘"처럼 말하면 분석이 실행된다."""

COMMENT_PROMPT = ("너는 데이터 분석 조교다. 분석 결과를 한국어 두세 문장으로 쉽게 해설한다. "
                  "주어진 숫자만 쓰고 새 숫자를 만들지 않는다. R²는 1에 가까울수록 좋고, "
                  "학습 R²가 테스트 R²보다 많이 높으면 과적합이다.")

WELCOME = """안녕하세요! 저는 **채팅으로만 쓰는 데이터 분석 챗봇**이에요. 버튼 없이 말로 시키면 돼요.

**이렇게 말해 보세요**
- "콘크리트 데이터 불러와" (다른 데이터: "데이터 목록 보여줘")
- 내 데이터 쓰기: 📎로 파일 첨부, 데이터 주소 붙여넣기, 표를 그대로 붙여넣기
- "선형회귀 해줘" → "비선형회귀 해줘"
- "인공신경망 학습해줘" → "딥러닝으로 해줘"
- "전부 비교해줘" → "보고서 만들어줘"
- "시멘트 300, 물 180, 재령 28일 때 예측해줘"

수업 개념도 물어보세요. 예: "오류역전파가 뭐야?"
더 많은 사용법은 **"도움말"** 이라고 입력하세요."""

HELP = """**채팅으로 할 수 있는 것**

| 하고 싶은 것 | 이렇게 말하세요 |
| --- | --- |
| 공개 데이터 쓰기 | "데이터 목록 보여줘", "자동차 연비 데이터 불러와", "펭귄 데이터 불러와", "집값 데이터 불러와" |
| 내 데이터 쓰기 | 입력창 📎로 파일 첨부(CSV·Excel·TSV·TXT·JSON), 데이터 주소 붙여넣기(CSV·구글 스프레드시트·GitHub), 표를 채팅창에 그대로 붙여넣기 |
| 숫자 직접 입력 | "x: 1, 2, 3, 4, 5 / y: 2.1, 3.9, 6.2, 8.1, 9.8" |
| 데이터 살펴보기 | "데이터 요약 보여줘", "열 목록 보여줘", "상관관계 보여줘", "마력 평균 알려줘", "무게랑 연비 산점도 그려줘", "연비 분포 보여줘" |
| 변수 정하기 | "Y는 압축강도, X는 시멘트랑 재령", "마력으로 연비 예측해줘", "무게에 따른 연비 분석해줘", "물 빼고 해줘", "모든 변수로 해줘" |
| 변수 이름 | 일부만("슬래그"), 한국어로("몸무게" → weight), 순서로("첫 번째 열") 말해도 찾아요 |
| 선형회귀 | "선형회귀 해줘", "시멘트로 선형회귀 해줘" |
| 비선형회귀 | "비선형회귀 해줘", "3차 다항 회귀 해줘" |
| 인공신경망 | "인공신경망 학습해줘", "은닉층 32-16, tanh, 에포크 500으로 신경망 학습해줘" |
| 딥러닝 | "딥러닝으로 해줘", "128-64-32-16 딥러닝 돌려줘" |
| 모델 비교 | "전부 비교해줘" |
| 예측 | "시멘트 300, 물 180, 재령 28일 때 예측해줘" |
| 보고서 | "보고서 만들어줘" |
| 기억 | "지난번에 뭐 했지?", "기억 지워줘" |
| 현재 상태 | "지금 상태 알려줘" |
| 언어모델 | "엑사원으로 바꿔줘", "모델 정보 알려줘", "서버 사양 알려줘", "LLM 원리 설명해줘", "해설 꺼줘" |
| 기타 | "대화 지워줘", 수업 개념 질문은 그냥 물어보세요 |"""

LOAD = ("불러", "가져", "로드", "열어", "데이터로", "데이터써", "데이터사용", "데이터바꿔", "자료로", "가지고", "쓸래", "쓰자")
STRONG = ("돌려", "실행", "학습해", "학습시", "훈련", "적합", "그려", "분석해", "해봐", "시켜", "만들어", "비교해")
GENERIC = ("해줘", "해주", "하자", "해라", "ㄱㄱ", "부탁", "보여")
EXPLAIN = ("설명", "뭐야", "뭔지", "무엇", "이유", "왜", "차이", "의미", "뜻", "개념", "원리", "어떻게")
MAX_HISTORY_MESSAGES = 6


def has(text, *words):
    return any(w in text for w in words)


# =============================================================
# 2. 언어모델 설치(내려받기)와 실행
# =============================================================
def server_spec():
    cpus = os.cpu_count() or 0
    mem_gb = None
    try:
        for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
            if os.path.exists(path):
                raw = open(path).read().strip()
                if raw.isdigit() and int(raw) < 1 << 50:
                    mem_gb = int(raw) / 1024 ** 3
                    break
        if mem_gb is None:
            with open("/proc/meminfo") as f:
                mem_gb = int(f.readline().split()[1]) / 1024 ** 2
    except (OSError, ValueError):
        pass
    return cpus, mem_gb


def cpu_threads():
    """이 서버가 실제로 쓸 수 있는 CPU 코어 수 (컨테이너 제한을 반영, 최대 4)."""
    n = os.cpu_count() or 1
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        if quota != "max":
            n = min(n, max(1, round(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return max(1, min(n, 4))


SIGNALS = {-9: "메모리 부족으로 강제 종료됨", -4: "이 서버 CPU가 지원하지 않는 명령어", -11: "모델 내부 오류(세그멘테이션 폴트)"}


class LLMWorker:
    """언어모델을 별도 프로세스(llm_worker.py)에서 돌린다.

    모델이 죽어도 웹 앱은 죽지 않고, 한 번에 한 사람의 질문만 처리한다(llama.cpp는 동시 실행 불가).
    """

    def __init__(self):
        self.proc, self.name, self.busy = None, None, False
        self.lock = threading.Lock()

    def _read(self):
        while True:
            line = self.proc.stdout.readline()
            if not line:                                   # 프로세스가 죽었음
                code = self.proc.poll()
                self.stop()
                raise RuntimeError(f"언어모델 프로세스가 멈췄어요 ({SIGNALS.get(code, f'종료 코드 {code}')})")
            try:
                return json.loads(line)
            except ValueError:
                continue                                   # JSON이 아닌 줄은 무시

    def stop(self):
        if self.proc is not None:
            try:
                self.proc.kill()
                self.proc.wait(timeout=10)
            except Exception:
                pass
        self.proc, self.name, self.busy = None, None, False

    def ensure(self, name):
        """name 모델이 준비된 상태로 만든다. 다른 모델이 떠 있으면 먼저 끝내 메모리를 비운다."""
        if not self.lock.acquire(timeout=180):
            raise RuntimeError("다른 답변을 만드는 중이에요. 잠시 뒤에 다시 물어봐 주세요.")
        try:
            if self.proc is not None and (self.name != name or self.proc.poll() is not None):
                self.stop()
            if self.proc is None:
                info = MODELS[name]
                worker = str(Path(__file__).resolve().parent / "llm_worker.py")
                self.proc = subprocess.Popen(
                    [sys.executable, worker, info["repo"], info["file"], str(cpu_threads())],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1)
                msg = self._read()
                if not msg.get("ready"):
                    self.stop()
                    raise RuntimeError(msg.get("error", "언어모델을 불러오지 못했어요"))
                self.name = name
        finally:
            self.lock.release()

    def chat(self, messages, temperature, max_tokens):
        """답을 글자 조각 단위로 내보낸다."""
        if not self.lock.acquire(timeout=180):
            raise RuntimeError("다른 답변을 만드는 중이에요. 잠시 뒤에 다시 물어봐 주세요.")
        try:
            if self.proc is None or self.proc.poll() is not None:
                raise RuntimeError("언어모델이 꺼져 있어요. 다시 물어보면 새로 켤게요.")
            while self.busy:                               # 중간에 끊긴 이전 답변의 남은 글자를 비움
                if "t" not in self._read():
                    self.busy = False
            self.proc.stdin.write(json.dumps({"messages": messages, "temperature": temperature,
                                              "max_tokens": max_tokens}, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
            self.busy = True
            while True:
                msg = self._read()
                if "t" in msg:
                    yield msg["t"]
                    continue
                self.busy = False
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return
        except (BrokenPipeError, OSError) as e:
            self.stop()
            raise RuntimeError(f"언어모델 프로세스와 연결이 끊겼어요 ({e})")
        finally:
            self.lock.release()


@st.cache_resource(show_spinner=False)
def worker():
    return LLMWorker()


def get_llm():
    """준비된 언어모델 작업자를 돌려준다. 실패하면 None (분석 기능은 계속 동작)."""
    ss = st.session_state
    w = worker()
    try:
        if w.name == ss.model_name and w.proc is not None and w.proc.poll() is None:
            return w
        with st.spinner(f"언어모델 '{ss.model_name}' 준비 중… (처음 한 번은 내려받느라 1~2분 걸려요)"):
            w.ensure(ss.model_name)
        ss.llm_error = None
        return w
    except Exception as e:  # 설치 실패, 내려받기 실패, 메모리 부족 등
        if ss.model_name != DEFAULT_MODEL:
            st.warning(f"'{ss.model_name}' 모델을 불러오지 못해서 기본 모델로 되돌려요. ({e})")
            ss.model_name = DEFAULT_MODEL
            return get_llm()
        ss.llm_error = str(e)
        return None


def stream_answer(llm, messages, temperature, max_tokens, stats):
    t0 = time.perf_counter()
    first, n_tokens = None, 0
    for text in llm.chat(messages, temperature, max_tokens):
        if first is None:
            first = time.perf_counter() - t0
        n_tokens += 1
        yield text
    total = time.perf_counter() - t0
    gen_time = max(total - (first or 0), 1e-6)
    stats.update(tokens=n_tokens, total=total, first=first or total, model=st.session_state.model_name,
                 tps=(n_tokens - 1) / gen_time if n_tokens > 1 else 0.0)


def ask_llm(llm, messages, temperature, max_tokens, prefix=""):
    """답을 한 토큰씩 화면에 흘려보내고 (답, 측정값)을 돌려준다."""
    stats = {}
    with st.spinner("생각 중… (CPU로 계산해서 몇 초 걸려요)"):
        gen = stream_answer(llm, messages, temperature, max_tokens, stats)
        first_piece = next(gen, "")

    def full_stream():
        yield prefix + first_piece
        yield from gen

    answer = st.write_stream(full_stream())
    return (answer or "")[len(prefix):], stats


def stats_caption(s):
    return (f"🤖 {s['model']} · {s['tokens']}토큰 · 첫 글자까지 {s['first']:.1f}초 · "
            f"전체 {s['total']:.1f}초 · 초당 {s['tps']:.1f}토큰")


# =============================================================
# 3. 채팅 명령 해석: 사용자의 말 → 실행할 도구
# =============================================================
def settings_command(t):
    ss = st.session_state
    if has(t, "바꿔", "바꾸", "변경", "전환", "써줘", "사용해", "쓰자"):
        for name, info in MODELS.items():
            if has(t, *info["aliases"]):
                if name == ss.model_name:
                    return f"이미 **{name}** 모델을 쓰고 있어요."
                ss.model_name, ss.llm_error = name, None
                return (f"🔄 언어모델을 **{name}**({info['developer']})으로 바꿨어요. "
                        f"다음 답변 때 파일({info['size']})을 준비하느라 1~2분 걸릴 수 있어요.")
    if has(t, "해설") and has(t, "꺼", "끄", "하지마", "빼"):
        ss.commentary = False
        return "🔇 분석 뒤에 붙는 AI 해설을 껐어요. 다시 켜려면 \"해설 켜줘\"라고 하세요."
    if has(t, "해설") and has(t, "켜", "다시"):
        ss.commentary = True
        return "🔊 분석 뒤에 AI 해설을 붙일게요."
    if has(t, "사양", "스펙"):
        cpus, mem_gb = server_spec()
        mem = f"{mem_gb:.1f}GB" if mem_gb else "확인 불가"
        return (f"**이 서버의 사양**\n\n- CPU: {cpus}코어 (GPU 없음)\n- 메모리: {mem}\n\n"
                "작은 서버라서 4비트로 압축한 15억~24억 개 매개변수 모델을 골랐어요. "
                "DeepSeek 원본(671B)처럼 큰 모델은 여기서 돌릴 수 없어요.")
    if has(t, "모델") and has(t, "정보", "뭐써", "무슨", "어떤", "목록", "종류") and not has(t, "회귀", "신경망", "비교"):
        i = MODELS[ss.model_name]
        others = ", ".join(n for n in MODELS if n != ss.model_name)
        return (f"**지금 언어모델: {ss.model_name}**\n\n- 개발: {i['developer']}\n- 매개변수: {i['params']}\n"
                f"- 양자화: Q4_K_M (4비트) · 파일 {i['size']}\n- 라이선스: {i['license']}\n- 특징: {i['feature']}\n"
                f"- 모델 페이지: https://huggingface.co/{i['repo']}\n\n바꿀 수 있는 모델: **{others}**")
    if has(t, "llm", "언어모델", "챗봇") and has(t, "원리", "어떻게대답", "어떻게작동", "어떻게동작", "어떻게돌아"):
        i = MODELS[ss.model_name]
        return f"""**LLM은 어떻게 대답할까?**

1. **LLM도 인공신경망이에요.** 수업의 뉴런 y = φ(Σwx + b)가 수십 층 쌓여 있어요. 지금 모델은 가중치와 편향이 {i['params']}예요. 이 챗봇의 기본 인공신경망(은닉층 64-32)보다 수십만 배 커요.
2. **글자를 숫자로 바꿔서 넣어요.** 문장을 '토큰'(글자 조각)으로 자르고 숫자 벡터로 바꿔요.
3. **다음 토큰을 '분류'해요.** "다음에 올 토큰이 무엇일까?"를 수만 개 후보 중에서 고르는 일을 반복해요. 그래서 글자가 하나씩 나와요.
4. **4비트 양자화로 작게 만들었어요.** 16비트로 저장하면 약 {i['n_params'] * 2 / 1e9:.1f}GB라 이 서버(메모리 약 2.7GB)에 안 들어가서, 4비트 파일({i['size']})을 써요.
5. **대화 중에는 가중치가 바뀌지 않아요.** 새로 학습하지 않고, 이전 대화와 작업 기록(메모)을 함께 다시 읽고 답해요."""
    return None


def detect_intents(t, raw):
    if has(t, "비교") or (has(t, "전부", "모두", "싹다", "네가지", "4가지", "다해") and has(t, "모델", "회귀", "돌려", "해줘", "해봐")):
        return ["compare"]
    intents = []
    t2 = t.replace("비선형", "§").replace("nonlinear", "§")
    if has(t2, "선형", "linear", "직선"):
        intents.append("linear")
    if "§" in t2 or has(t, "다항", "곡선", "polynomial", "거듭제곱", "지수회귀", "로그회귀"):
        intents.append("nonlinear")
    deep = has(t, "딥러닝", "deep", "심층", "딥뉴럴")
    if deep:
        intents.append("deep")
    elif has(t, "신경망", "mlp", "뉴럴", "퍼셉트론") or re.search(r"\bann\b", raw.lower()):
        intents.append("ann")
    if not intents and "회귀" in t:
        intents = ["linear", "nonlinear"]
    return intents


def ensure_data(out):
    an = st.session_state.an
    if an["df"] is None:
        rep = T.load_dataset(an, "concrete")
        rep["text"] = "ℹ️ 데이터를 정하지 않아서 기본 데이터를 불러왔어요.\n\n" + rep["text"]
        out.append(rep)


def route(text, files):
    """(답 목록, 언어모델에 넘길지)를 돌려준다. 대화 지우기는 ("CLEAR", False)."""
    ss = st.session_state
    an = ss.an
    t = T.norm(text)
    out = []
    url = re.search(r"https?://\S+", text)

    if not files and not url:
        if has(t, "도움말", "help", "명령어", "사용법", "뭘할수있", "뭐할수있", "무엇을할수있"):
            return [T.reply(HELP)], False
        if has(t, "대화", "채팅") and has(t, "지워", "지우", "삭제", "초기화", "리셋"):
            return "CLEAR", False
        if has(t, "기억", "기록") and has(t, "지워", "지우", "삭제", "잊어"):
            T.forget_all()
            return [T.reply("🧹 기억해 둔 작업 기록을 모두 지웠어요.")], False
        if has(t, "기억", "지난번", "저번에", "아까뭐", "이전에뭐", "뭐했"):
            return [T.recall()], False
        s = settings_command(t)
        if s:
            return [T.reply(s)], False
        if has(t, "데이터", "자료") and has(t, "목록", "종류", "뭐있", "뭐가있", "어떤데이터", "어떤자료", "리스트", "쓸수있"):
            return [T.dataset_list()], False
        pasted = T.parse_pasted(text)                   # 채팅창에 직접 적거나 붙여넣은 데이터
        if pasted is not None:
            out.append(T.load_pasted(an, pasted))
            text = " ".join(ln for ln in text.splitlines() if not re.search(r"\d", ln))
            t = T.norm(text)
            if not t:
                return out, False

    # --- 데이터 불러오기 (파일 첨부, 주소, 내장 데이터) ---
    for f in files:
        out.append(T.load_file(an, f))
    if url:
        link = url.group().rstrip(".,)")
        out.append(T.load_url(an, link))
        text = text.replace(url.group(), " ")
        t = T.norm(text)
    key = T.match_dataset(t)
    if key and not files and not url and not out and (has(t, *LOAD) or an["df"] is None):
        out.append(T.load_dataset(an, key) if key != an["key"]
                   else T.reply(f"이미 **{an['name']}** 데이터를 쓰고 있어요."))
        if out[-1]["text"].startswith("⚠️"):           # 내려받기 실패: 다른 데이터로 넘어가지 않고 알림
            return out, False

    explain_q, strong, generic = has(t, *EXPLAIN), has(t, *STRONG), has(t, *GENERIC)

    # --- 예측 ---
    if "예측" in t and re.search(r"\d", t):
        ensure_data(out)
        vals = T.parse_inputs(an, text)
        if vals:
            rep = T.run_predict(an, vals)
            T.remember(rep["summary"])
            return out + [rep], False

    # --- 분석 실행 ---
    intents = detect_intents(t, text)
    run = strong or (generic and not explain_q) or (bool(intents) and len(t) <= 12 and not explain_q)
    plot_q = has(t, "산점도", "그래프", "그림", "플롯", "plot", "히스토그램", "분포", "시각화")
    if (not intents and run and not explain_q and not plot_q and not re.search(r"\d", t)
            and has(t, "예측", "맞춰", "추정", "에따른", "에따라", "관계", "영향")):
        ensure_data(out)                                # "마력으로 연비 예측해줘" → 변수를 정하고 세 모델로 분석
        if T.find_cols(text, T.numeric_cols(an)):
            intents = ["linear", "nonlinear", "ann"]
    if intents and run:
        ensure_data(out)
        note = T.parse_vars(an, text)
        if note:
            out.append(T.reply(note))
        opts = T.parse_ann_options(text)
        deg = re.search(r"(\d)\s*차", text)
        for it in intents:
            try:
                if it == "compare":
                    rep = T.run_compare(an)
                elif it == "linear":
                    rep = T.run_linear(an)
                elif it == "nonlinear":
                    rep = T.run_nonlinear(an, int(deg.group(1)) if deg else None)
                else:
                    rep = T.run_ann(an, deep=(it == "deep"), layers=opts.get("layers"), act=opts.get("act", "relu"),
                                    epochs=opts.get("epochs"), lr=opts.get("lr", 0.001))
            except Exception as e:
                rep = T.reply(f"⚠️ 분석 중 오류가 났어요: {e}")
            if rep.get("summary"):
                T.remember(rep["summary"])
            out.append(rep)
        if out[-1].get("summary"):
            out[-1]["explain"] = True          # 마지막 결과에만 언어모델 해설을 붙임
        return out, False

    # --- 그 밖의 도구 ---
    if has(t, "보고서", "리포트"):
        return out + [T.report(an)], False
    if has(t, "열", "컬럼", "칼럼", "변수", "column") and has(t, "목록", "뭐있", "뭐가있", "어떤", "리스트", "이름", "종류") and not explain_q:
        ensure_data(out)
        return out + [T.columns(an)], False
    if plot_q and not has(t, "상관") and has(t, "그려", "보여", "해줘", "만들어", "plot", "봐"):
        ensure_data(out)
        return out + [T.plot_cols(an, text)], False
    if T.which_stats(t) and an["df"] is not None and T.find_cols(text, T.numeric_cols(an)):
        return out + [T.column_stats(an, text)], False
    if has(t, "상관") and has(t, "보여", "그려", "분석", "히트맵", "해줘", "알려"):
        ensure_data(out)
        return out + [T.correlation(an)], False
    if has(t, "데이터") and has(t, "보여", "요약", "미리", "통계", "확인", "어떻게생"):
        ensure_data(out)
        return out + [T.describe(an)], False
    if T.has_var_marker(t) or has(t, "빼고", "제외", "빼줘") or (has(t, "변수") and has(t, "설정", "바꿔", "로해", "사용", "만써", "전체", "모든")):
        ensure_data(out)
        note = T.parse_vars(an, text)
        return out + [T.reply(note) if note else T.status(an)], False
    if has(t, "현재상태", "지금상태", "상태알려", "상태보여", "설정알려", "설정보여", "변수뭐", "변수알려", "변수보여", "변수확인"):
        return out + [T.status(an)], False
    if out:
        return out, False
    if "예측" in t and (strong or generic) and not explain_q and not intents:
        ex = ", ".join(f"{T.short(c)} {an['df'][c].mean():.0f}" for c in an["x"][:3]) if an["df"] is not None else "시멘트 300, 물 180, 재령 28"
        return [T.reply(f"예측할 입력값을 같이 말해 주세요. 예: \"{ex}일 때 예측해줘\"")], False
    return [], True


def llm_context():
    an = st.session_state.an
    lines = []
    if an["df"] is not None:
        lines.append(f"- 현재 설정: Y={T.short(an['y'])}, X={', '.join(T.short(c) for c in an['x'][:12])}")
        lines += T.data_brief(an)
    for r in an["results"].values():
        lines.append(f"- {r['label']}: 테스트 R² {r['test_r2']:.3f}, RMSE {r['rmse']:.2f}, 학습 R² {r['train_r2']:.3f}")
    for m in T.recall_all()[-3:]:
        lines.append(f"- 기록({m['time']}): {m['summary']}")
    return ("\n\n[지금까지의 분석 상태와 기록]\n" + "\n".join(lines)) if lines else ""


# =============================================================
# 4. 화면: 채팅창 하나
# =============================================================
ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("an", T.new_state())
ss.setdefault("model_name", DEFAULT_MODEL)
ss.setdefault("commentary", True)

st.title("💬 데이터 분석 챗봇")
st.caption("선형회귀 · 비선형회귀 · 인공신경망 · 딥러닝을 채팅으로 실행해요. "
           "이 서버에 설치한 오픈소스 언어모델이 해설하고 질문에 답해요.")


def render(m, idx):
    st.markdown(m["content"])
    for j, (kind, obj) in enumerate(m.get("items") or []):
        if kind == "fig":
            st.plotly_chart(obj, key=f"fig_{idx}_{j}", width="stretch")
        else:
            st.dataframe(obj, hide_index=True, width="stretch")
    if m.get("commentary"):
        st.markdown(f"🤖 **AI 해설**: {m['commentary']}")
    if m.get("stats"):
        st.caption(stats_caption(m["stats"]))


if not ss.messages:
    with st.chat_message("assistant"):
        st.markdown(WELCOME)
        last = T.recall_all()[-1:]
        if last:
            st.caption(f"📝 기억: 마지막 작업은 {last[0]['time']} · {last[0]['summary']} (\"지난번에 뭐 했지?\"로 더 보기)")

for i, m in enumerate(ss.messages):
    with st.chat_message(m["role"]):
        render(m, i)

inp = st.chat_input("메시지를 입력하세요 (예: 선형회귀 해줘 / 도움말)", accept_file=True, file_type=["csv", "xlsx", "xls", "tsv", "txt", "json"])

if inp:
    text = inp if isinstance(inp, str) else (getattr(inp, "text", "") or "")
    files = [] if isinstance(inp, str) else list(getattr(inp, "files", None) or [])
    shown = text or f"📎 {files[0].name}"
    if text and files:
        shown += f"\n\n📎 {files[0].name}"

    try:
        replies, to_llm = route(text, files)
    except Exception as e:                      # 예상하지 못한 입력에도 앱이 멈추지 않게
        replies, to_llm = [T.reply(f"⚠️ 요청을 처리하다 오류가 났어요: {e}\n\n다르게 말해 보거나 \"도움말\"이라고 입력해 보세요.")], False
    if replies == "CLEAR":
        ss.messages = []
        st.rerun()

    user_msg = {"role": "user", "content": shown, "kind": None if to_llm else "tool"}
    with st.chat_message("user"):
        st.markdown(shown)
    ss.messages.append(user_msg)

    for rep in replies:                         # 도구가 실행한 결과
        msg = {"role": "assistant", "content": rep["text"], "items": rep["items"], "kind": "tool"}
        with st.chat_message("assistant"):
            render(msg, len(ss.messages))
            if rep.get("explain") and ss.commentary:
                llm = get_llm()
                if llm is not None:
                    try:
                        prompt = [{"role": "system", "content": COMMENT_PROMPT},
                                  {"role": "user", "content": f"분석 결과: {rep['summary']}\n이 결과가 무슨 뜻인지 해설해줘."}]
                        msg["commentary"], msg["stats"] = ask_llm(llm, prompt, 0.3, 160, prefix="🤖 **AI 해설**: ")
                        st.caption(stats_caption(msg["stats"]))
                    except Exception as e:
                        st.caption(f"AI 해설을 만들지 못했어요: {e}")
        ss.messages.append(msg)

    if to_llm:                                  # 일반 질문 → 언어모델이 답함
        with st.chat_message("assistant"):
            llm = get_llm()
            if llm is None:
                answer, stats = ("⚠️ 언어모델을 불러오지 못해서 자유 질문에는 지금 답할 수 없어요. "
                                 f"({ss.get('llm_error')})\n\n분석 기능은 그대로 쓸 수 있어요. \"도움말\"이라고 입력해 보세요."), None
                st.markdown(answer)
            else:
                history = [{"role": m["role"], "content": m["content"]}
                           for m in ss.messages if m.get("kind") != "tool"][-MAX_HISTORY_MESSAGES:]
                messages = [{"role": "system", "content": SYSTEM_PROMPT + llm_context()}] + history
                try:
                    answer, stats = ask_llm(llm, messages, 0.5, 384)
                    st.caption(stats_caption(stats))
                except Exception as e:
                    answer, stats = f"⚠️ 답변 생성 중 오류가 났어요: {e}", None
                    st.markdown(answer)
        ss.messages.append({"role": "assistant", "content": answer, "stats": stats})
