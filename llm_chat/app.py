"""오픈소스 LLM 설치·실행 및 채팅 앱 (데이터처리개론 과제)

- 오픈소스 언어모델(GGUF 4비트 양자화 파일)을 Hugging Face에서 내려받아 이 서버에 설치하고,
  llama.cpp(llama-cpp-python)로 CPU에서 직접 실행합니다. 외부 AI API를 쓰지 않습니다.
- 사용자는 아무것도 설치할 필요 없이 웹 주소로 접속해 바로 대화할 수 있습니다.
- 화면에는 채팅창 하나만 있습니다. 모델 바꾸기, 설정 변경, 대화 지우기도 모두 채팅으로 합니다.

로컬 실행:
    pip install -r llm_chat/requirements.txt
    streamlit run llm_chat/app.py
"""
import gc
import os
import re
import time

import streamlit as st

st.set_page_config(page_title="오픈소스 LLM 채팅", page_icon="💬", layout="centered")

# =============================================================
# 1. 모델 목록 (서버 사양: CPU 2코어 · 메모리 약 2.7GB에 맞춰 선정)
# =============================================================
DEFAULT_MODEL = "Qwen2.5-1.5B-Instruct"
MODELS = {
    "Qwen2.5-1.5B-Instruct": {
        "repo": "lmstudio-community/Qwen2.5-1.5B-Instruct-GGUF",
        "file": "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
        "aliases": ("qwen", "큐원", "큐웬", "알리바바"),
        "developer": "Alibaba Qwen 팀 (중국)",
        "params": "약 15억 개 (1.5B)",
        "n_params": 1.5e9,
        "quant": "Q4_K_M (4비트)",
        "size": "약 0.99GB",
        "license": "Apache 2.0 (상업적 이용 가능)",
        "feature": "29개 이상 언어 지원(한국어 포함). 작고 빨라서 CPU 서버에 적합",
    },
    "EXAONE-3.5-2.4B-Instruct": {
        "repo": "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct-GGUF",
        "file": "EXAONE-3.5-2.4B-Instruct-Q4_K_M.gguf",
        "aliases": ("exaone", "엑사원", "엑사온", "엘지"),
        "developer": "LG AI연구원 (한국)",
        "params": "약 24억 개 (2.4B)",
        "n_params": 2.4e9,
        "quant": "Q4_K_M (4비트)",
        "size": "약 1.64GB",
        "license": "EXAONE AI Model License (연구·교육용, 비상업)",
        "feature": "한국어·영어 특화. 한국어 답변이 자연스럽지만 크기가 커서 답이 느림",
    },
}

SYSTEM_PROMPT = """너는 '데이터처리개론' 수업의 AI 조교다. 이 수업은 동아대학교 조선해양공학과 과목이다.
너는 학생이 만든 '콘크리트 압축강도 예측 프로그램'을 안내하고 수업 개념을 쉽게 설명한다.

[규칙]
- 반드시 한국어로, 짧고 핵심 위주로 답한다.
- 모르는 것은 모른다고 말하고 절대 지어내지 않는다.
- 숫자는 아래 정보에 있는 것만 사용한다.

[프로그램 정보]
- Streamlit 웹 앱. 선형·비선형 회귀 13종과 인공신경망(ANN)으로 콘크리트 압축강도를 예측한다.
- 탭: 데이터 탐색, 성능 비교, 잔차 진단, 모델 해석, 예측, 내보내기.

[데이터] UCI 공개 데이터 '콘크리트 압축강도' 1,030개.
- 입력 8개: 시멘트, 고로슬래그, 플라이애시, 물, 고성능감수제, 굵은골재, 잔골재, 재령(일). 출력: 압축강도(MPa).
- 강도와의 상관계수: 시멘트 0.50, 고성능감수제 0.37, 재령 0.33, 물 -0.29.
- 변수 중요도(순열 중요도): 재령이 가장 크고, 다음이 시멘트, 물 순서.

[실험 결과] 테스트 R² (10번 분할 평균)
- 랜덤 포레스트 0.907, 그래디언트 부스팅 0.903, 인공신경망(은닉층 64-32, ReLU) 0.895, 선형 회귀 0.581.
- 인공신경망은 선형 회귀보다 R²가 0.31 높다. 데이터가 20행 이하로 적으면 과적합된다.

[수업 핵심 개념]
- 뉴런: y = φ(Σ wx + b). 가중치 w, 편향 b, 활성화 함수 φ(시그모이드, ReLU).
- 경사하강법: 오차 E=(t-o)²를 줄이도록 w ← w - η·∂E/∂w.
- 오류역전파: 출력의 오차를 거꾸로 전달해 모든 가중치를 한 번에 고친다(합성함수의 미분법).
- 정규화: 표준화 z=(x-μ)/σ 또는 MinMax로 단위를 맞춘다.
- 과적합 방지: 학습·검증·테스트 데이터 분할."""

WELCOME = """안녕하세요! 저는 이 서버에 직접 설치된 **오픈소스 언어모델**이에요. 버튼 없이 **채팅으로만** 쓰는 앱이에요.

**이런 걸 물어보세요**
- "오류역전파를 쉽게 설명해줘"
- "콘크리트 강도에 가장 큰 영향을 주는 변수는?"
- "선형회귀보다 인공신경망이 나은 이유는?"

**설정도 말로 바꿔요**
- "엑사원으로 바꿔줘" / "큐원으로 바꿔줘" (모델 바꾸기)
- "지금 모델 정보 알려줘" · "서버 사양 알려줘"
- "창의성 0.2로 해줘" · "답변 길게 해줘"
- "LLM 원리 설명해줘" · "대화 지워줘" · "도움말"

처음 질문은 모델을 준비하느라 조금 느릴 수 있어요."""

HELP = """**채팅으로 할 수 있는 것**

| 하고 싶은 것 | 이렇게 말하세요 |
| --- | --- |
| 수업·프로그램 질문 | 그냥 물어보세요 |
| 모델 바꾸기 | "엑사원으로 바꿔줘", "큐원으로 바꿔줘" |
| 모델 정보·목록 | "지금 모델 정보 알려줘" |
| 서버 사양 | "서버 사양 알려줘" |
| 창의성(temperature) | "창의성 0.2로 해줘", "창의성 높여줘" |
| 답변 길이 | "답변 길게 해줘", "답변 짧게 해줘", "답변 길이 500토큰" |
| LLM 동작 원리 | "LLM 원리 설명해줘" |
| 대화 지우기 | "대화 지워줘" |"""

MAX_HISTORY_MESSAGES = 8   # 문맥 창(2,048토큰)에 맞추기 위해 최근 대화만 모델에 전달
CLEARED = "🗑️ 대화를 모두 지웠어요. 새로 시작해요!"


# =============================================================
# 2. 서버 사양 읽기
# =============================================================
def server_spec():
    cpus = os.cpu_count() or 0
    mem_gb = None
    try:  # 컨테이너 메모리 제한(cgroup)이 있으면 그 값을, 없으면 전체 메모리를 사용
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


# =============================================================
# 3. 모델 설치(내려받기)와 불러오기
# =============================================================
@st.cache_resource(show_spinner=False, max_entries=1)
def load_model(name):
    from llama_cpp import Llama  # 설치 실패 시 아래에서 안내하도록 함수 안에서 불러옴

    info = MODELS[name]
    return Llama.from_pretrained(
        repo_id=info["repo"],
        filename=info["file"],
        n_ctx=2048,                          # 한 번에 볼 수 있는 토큰 수(문맥 창)
        n_threads=max(1, os.cpu_count() or 1),
        n_batch=256,
        verbose=False,
    )


def stream_answer(llm, messages, temperature, max_tokens, stats):
    """모델이 만드는 토큰을 하나씩 화면에 흘려보내고, 속도를 측정한다."""
    t0 = time.perf_counter()
    first = None
    n_tokens = 0
    for chunk in llm.create_chat_completion(
        messages=messages, stream=True, temperature=temperature,
        top_p=0.9, repeat_penalty=1.1, max_tokens=max_tokens,
    ):
        text = chunk["choices"][0].get("delta", {}).get("content")
        if text:
            if first is None:
                first = time.perf_counter() - t0
            n_tokens += 1
            yield text
    total = time.perf_counter() - t0
    gen_time = max(total - (first or 0), 1e-6)
    stats.update(tokens=n_tokens, total=total, first=first or total,
                 tps=(n_tokens - 1) / gen_time if n_tokens > 1 else 0.0)


# =============================================================
# 4. 채팅 명령 처리 (버튼·슬라이더 대신 말로 앱을 조작)
#    앱 설정을 바꾸는 말은 프로그램이 직접 처리하고, 나머지는 모두 언어모델이 답한다.
# =============================================================
def has(text, *words):
    return any(w in text for w in words)


def model_info_text(name):
    i = MODELS[name]
    others = [n for n in MODELS if n != name]
    return (f"**지금 모델: {name}**\n\n- 개발: {i['developer']}\n- 매개변수: {i['params']}\n- 양자화: {i['quant']}\n"
            f"- 파일 크기: {i['size']}\n- 라이선스: {i['license']}\n- 특징: {i['feature']}\n"
            f"- 모델 페이지: https://huggingface.co/{i['repo']}\n\n"
            f"바꿀 수 있는 다른 모델: **{', '.join(others)}** (\"○○으로 바꿔줘\"라고 말하세요)")


def about_text(name):
    i = MODELS[name]
    return f"""**LLM은 어떻게 대답할까?**

1. **LLM도 인공신경망이에요.** 수업에서 배운 뉴런 y = φ(Σwx + b)가 수십 층 쌓여 있어요. 지금 모델은 가중치와 편향이 {i['params']}예요. 과제 2의 콘크리트 신경망(2,689개)보다 약 {i['n_params'] / 2689 / 1e4:.0f}만 배 커요.
2. **글자를 숫자로 바꿔서 넣어요.** 문장을 '토큰'(글자 조각)으로 자르고, 토큰마다 숫자 벡터로 바꿔 신경망에 넣어요. 4강의 '흑백 이미지를 행렬로 바꾸기'와 같은 생각이에요.
3. **다음 토큰을 '분류'해요.** 답 전체를 한 번에 만들지 않고, "다음에 올 토큰이 무엇일까?"를 수만 개 후보 중에서 고르는 일을 반복해요. 마지막 층의 소프트맥스가 후보별 확률을 구해요. 그래서 글자가 하나씩 나와요. 창의성(temperature)을 높이면 확률이 낮은 후보도 뽑혀서 답이 다양해져요.
4. **4비트 양자화로 작게 만들었어요.** 가중치를 16비트로 저장하면 이 모델은 약 {i['n_params'] * 2 / 1e9:.1f}GB라서 이 서버(메모리 약 2.7GB)에 안 들어가요. 그래서 4비트로 줄인 파일({i['size']})을 써요.
5. **대화 중에는 가중치가 바뀌지 않아요.** 모델은 새로 학습하지 않고, 이전 대화를 함께 다시 읽고 답해요. 교수님이 말씀하신 '메모 보고 대답하기' 방식이에요. 그래서 문맥 창(2,048토큰)을 넘는 오래된 대화는 잊어요."""


def handle_command(raw):
    """앱을 조작하는 말이면 처리하고 답 문장을 돌려준다. 일반 질문이면 None."""
    ss = st.session_state
    t = raw.lower().replace(" ", "")
    num = re.search(r"\d+(?:\.\d+)?", raw)

    if has(t, "도움말", "help", "명령어", "사용법", "뭘할수있", "뭐할수있", "무엇을할수있"):
        return HELP

    if has(t, "대화", "기록", "채팅") and has(t, "지워", "지우", "삭제", "초기화", "리셋"):
        return CLEARED

    if has(t, "바꿔", "바꾸", "변경", "전환", "써줘", "사용해", "로해줘", "쓰자"):
        for name, info in MODELS.items():
            if has(t, *info["aliases"]):
                if name == ss.model_name:
                    return f"이미 **{name}** 모델을 쓰고 있어요."
                ss.model_name = name
                return (f"🔄 모델을 **{name}**({info['developer']})으로 바꿨어요. "
                        f"파일({info['size']})을 준비하느라 처음 한 번은 1~2분 걸릴 수 있어요.")

    if has(t, "창의성", "temperature", "온도"):
        if num:
            ss.temperature = min(max(float(num.group()), 0.0), 1.5)
        elif has(t, "높", "올려", "올리"):
            ss.temperature = min(ss.temperature + 0.3, 1.5)
        elif has(t, "낮", "내려", "내리", "줄여"):
            ss.temperature = max(ss.temperature - 0.3, 0.0)
        else:
            return (f"창의성(temperature)은 지금 **{ss.temperature:.1f}**이에요 (0~1.5). 낮으면 같은 질문에 같은 답을, "
                    "높으면 다양한 답을 해요. \"창의성 0.2로 해줘\"처럼 말하면 바꿔요.")
        return f"🎛️ 창의성(temperature)을 **{ss.temperature:.1f}**로 맞췄어요."

    if has(t, "답변", "대답", "답을", "답길이") and has(t, "길게", "짧게", "길이", "토큰"):
        if num:
            ss.max_tokens = int(min(max(float(num.group()), 64), 1024))
        elif has(t, "길게"):
            ss.max_tokens = 768
        elif has(t, "짧게"):
            ss.max_tokens = 160
        else:
            return f"최대 답변 길이는 지금 **{ss.max_tokens}토큰**이에요 (64~1024). \"답변 길이 500토큰\"처럼 말하면 바꿔요."
        return f"📏 최대 답변 길이를 **{ss.max_tokens}토큰**으로 맞췄어요."

    if has(t, "사양", "스펙"):
        cpus, mem_gb = server_spec()
        mem = f"{mem_gb:.1f}GB" if mem_gb else "확인 불가"
        return (f"**이 서버의 사양**\n\n- CPU: {cpus}코어 (GPU 없음)\n- 메모리: {mem}\n\n"
                "작은 서버라서 4비트로 압축한 15억~24억 개 매개변수 모델을 골랐어요. "
                "DeepSeek 원본(671B)처럼 큰 모델은 여기서 돌릴 수 없어요.")

    if has(t, "모델") and has(t, "정보", "뭐써", "무슨", "어떤", "지금", "현재", "목록", "종류"):
        return model_info_text(ss.model_name)

    if has(t, "원리", "어떻게대답", "어떻게작동", "어떻게동작", "어떻게돌아") and \
            has(t, "llm", "언어모델", "너는", "넌", "네가", "니가", "챗봇"):
        return about_text(ss.model_name)

    return None


# =============================================================
# 5. 화면: 채팅창 하나
# =============================================================
ss = st.session_state
ss.setdefault("messages", [])            # 대화 기록 (+ 답변별 측정값)
ss.setdefault("model_name", DEFAULT_MODEL)
ss.setdefault("temperature", 0.7)
ss.setdefault("max_tokens", 384)

st.title("💬 오픈소스 LLM 채팅")
st.caption(f"모델: {ss.model_name} · 창의성 {ss.temperature:.1f} · 최대 {ss.max_tokens}토큰 · "
           "이 서버에 설치된 오픈소스 모델이 CPU로 직접 답해요 (외부 AI 서비스 사용 안 함)")

# 모델이 바뀌면 이전 모델을 먼저 메모리에서 지운다 (메모리 2.7GB에 두 모델이 동시에 올라가지 않도록)
if ss.get("loaded_model") not in (None, ss.model_name):
    load_model.clear()
    gc.collect()

try:
    with st.status(f"'{ss.model_name}' 모델 준비 중… (처음 한 번은 내려받기 때문에 1~2분 걸려요)",
                   expanded=False) as status:
        llm = load_model(ss.model_name)
        status.update(label=f"✅ 모델 준비 완료: {ss.model_name}", state="complete")
    ss.loaded_model = ss.model_name
except ImportError:
    st.error("llama-cpp-python(모델 실행 도구)이 이 서버에 설치되지 않았어요. requirements.txt를 확인해 주세요.")
    st.stop()
except Exception as e:  # 내려받기 실패, 메모리 부족 등
    if ss.model_name != DEFAULT_MODEL:   # 큰 모델이 실패하면 기본 모델로 되돌아가 대화를 이어간다
        failed, ss.model_name = ss.model_name, DEFAULT_MODEL
        ss.loaded_model = None
        ss.messages.append({"role": "assistant", "kind": "command",
                            "content": f"⚠️ **{failed}** 모델을 불러오지 못해서 **{DEFAULT_MODEL}**로 되돌렸어요. ({e})"})
        st.rerun()
    st.error(f"모델을 불러오지 못했어요: {e}")
    st.info("잠시 뒤 새로고침해 주세요.")
    st.stop()

if not ss.messages:
    with st.chat_message("assistant"):
        st.markdown(WELCOME)

for m in ss.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("stats"):
            s = m["stats"]
            st.caption(f"🤖 {s['model']} · {s['tokens']}토큰 · 첫 글자까지 {s['first']:.1f}초 · "
                       f"전체 {s['total']:.1f}초 · 초당 {s['tps']:.1f}토큰")
        elif m["role"] == "assistant" and m.get("kind") == "command":
            st.caption("⚙️ 앱 설정 명령이라 프로그램이 처리했어요")

prompt = st.chat_input("메시지를 입력하세요 (예: 오류역전파가 뭐야? / 엑사원으로 바꿔줘 / 도움말)")

if prompt:
    reply = handle_command(prompt)
    if reply is not None:                      # 앱 조작 명령
        if reply == CLEARED:
            ss.messages = []
        else:
            ss.messages.append({"role": "user", "content": prompt, "kind": "command"})
        ss.messages.append({"role": "assistant", "content": reply, "kind": "command"})
        st.rerun()

    ss.messages.append({"role": "user", "content": prompt})   # 일반 질문 → 언어모델이 답함
    with st.chat_message("user"):
        st.markdown(prompt)

    # 설정 명령은 빼고, 최근 대화만 시스템 프롬프트와 함께 모델에 전달
    history = [{"role": m["role"], "content": m["content"]}
               for m in ss.messages if m.get("kind") != "command"][-MAX_HISTORY_MESSAGES:]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    with st.chat_message("assistant"):
        stats = {}
        try:
            with st.spinner("생각 중… (CPU로 계산해서 몇 초 걸려요)"):
                gen = stream_answer(llm, messages, ss.temperature, ss.max_tokens, stats)
                first_piece = next(gen, "")      # 첫 토큰이 나올 때까지 스피너 표시

            def full_stream():
                yield first_piece
                yield from gen

            answer = st.write_stream(full_stream())
        except Exception as e:
            answer, stats = f"⚠️ 답변 생성 중 오류가 났어요: {e}", {}
            st.markdown(answer)
        if stats:
            stats["model"] = ss.model_name
            st.caption(f"🤖 {stats['model']} · {stats['tokens']}토큰 · 첫 글자까지 {stats['first']:.1f}초 · "
                       f"전체 {stats['total']:.1f}초 · 초당 {stats['tps']:.1f}토큰")
    ss.messages.append({"role": "assistant", "content": answer or "", "stats": stats or None})
