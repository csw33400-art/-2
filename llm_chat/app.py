"""오픈소스 LLM 설치·실행 및 채팅 앱 (데이터처리개론 과제)

- 오픈소스 언어모델(GGUF 4비트 양자화 파일)을 Hugging Face에서 내려받아 이 서버에 설치하고,
  llama.cpp(llama-cpp-python)로 CPU에서 직접 실행합니다. 외부 AI API를 쓰지 않습니다.
- 사용자는 아무것도 설치할 필요 없이 웹 주소로 접속해 바로 대화할 수 있습니다.

로컬 실행:
    pip install -r llm_chat/requirements.txt
    streamlit run llm_chat/app.py
"""
import gc
import json
import os
import time
from datetime import datetime

import streamlit as st

st.set_page_config(page_title="오픈소스 LLM 채팅", page_icon="💬", layout="wide")

# =============================================================
# 1. 모델 목록 (서버 사양: CPU 2코어 · 메모리 약 2.7GB에 맞춰 선정)
# =============================================================
MODELS = {
    "Qwen2.5-1.5B-Instruct (빠름, 기본)": {
        "repo": "lmstudio-community/Qwen2.5-1.5B-Instruct-GGUF",
        "file": "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
        "developer": "Alibaba Qwen 팀 (중국)",
        "params": "약 15억 개 (1.5B)",
        "n_params": 1.5e9,
        "quant": "Q4_K_M (4비트)",
        "size": "약 0.99GB",
        "license": "Apache 2.0 (상업적 이용 가능)",
        "feature": "29개 이상 언어 지원(한국어 포함). 작고 빨라서 CPU 서버에 적합",
    },
    "EXAONE-3.5-2.4B-Instruct (한국어 특화, 느림)": {
        "repo": "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct-GGUF",
        "file": "EXAONE-3.5-2.4B-Instruct-Q4_K_M.gguf",
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

EXAMPLES = [
    "안녕! 너는 누구야?",
    "오류역전파를 쉽게 설명해줘",
    "콘크리트 강도에 가장 큰 영향을 주는 변수는?",
    "선형회귀보다 인공신경망이 나은 이유는?",
]
MAX_HISTORY_MESSAGES = 8   # 문맥 창(2,048토큰)에 맞추기 위해 최근 대화만 모델에 전달


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
    t0 = time.perf_counter()
    llm = Llama.from_pretrained(
        repo_id=info["repo"],
        filename=info["file"],
        n_ctx=2048,                          # 한 번에 볼 수 있는 토큰 수(문맥 창)
        n_threads=max(1, os.cpu_count() or 1),
        n_batch=256,
        verbose=False,
    )
    return llm, time.perf_counter() - t0


def stream_answer(llm, messages, temperature, max_tokens, stats):
    """모델이 만드는 토큰을 하나씩 화면에 흘려보내고, 속도를 측정한다."""
    t0 = time.perf_counter()
    first = None
    n_tokens = 0
    for chunk in llm.create_chat_completion(
        messages=messages, stream=True, temperature=temperature,
        top_p=0.9, repeat_penalty=1.1, max_tokens=max_tokens,
    ):
        delta = chunk["choices"][0].get("delta", {})
        text = delta.get("content")
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
# 4. 화면 구성
# =============================================================
st.title("💬 오픈소스 LLM 채팅")
st.caption("오픈소스 언어모델을 이 웹 서버에 직접 설치해 실행합니다. ChatGPT 같은 외부 AI 서비스를 쓰지 않아요. "
           "처음 접속하면 모델을 내려받느라 1~2분 걸릴 수 있어요.")

if "messages" not in st.session_state:
    st.session_state.messages = []      # 화면에 보이는 대화 기록 (+ 답변별 측정값)
if "pending" not in st.session_state:
    st.session_state.pending = None

with st.sidebar:
    st.header("⚙️ 설정")
    model_name = st.selectbox("언어모델 선택", list(MODELS))
    temperature = st.slider("창의성 (temperature)", 0.0, 1.5, 0.7, 0.1,
                            help="낮을수록 같은 질문에 같은 답, 높을수록 다양한 답")
    max_tokens = st.slider("최대 답변 길이 (토큰)", 64, 1024, 384, 64)

    info = MODELS[model_name]
    st.subheader("📘 모델 정보")
    st.markdown(
        f"- **개발**: {info['developer']}\n- **매개변수**: {info['params']}\n- **양자화**: {info['quant']}\n"
        f"- **파일 크기**: {info['size']}\n- **라이선스**: {info['license']}\n- **특징**: {info['feature']}"
    )
    st.markdown(f"[모델 페이지 (Hugging Face)](https://huggingface.co/{info['repo']})")

    cpus, mem_gb = server_spec()
    st.subheader("🖥️ 서버 사양")
    st.markdown(f"- CPU: **{cpus}코어**\n- 메모리: **{mem_gb:.1f}GB**" if mem_gb else f"- CPU: **{cpus}코어**")
    st.caption("GPU가 없는 작은 서버라서, 4비트로 압축한 15억~24억 개 매개변수 모델을 골랐어요.")

    st.divider()
    if st.button("🗑️ 대화 지우기", width="stretch"):
        st.session_state.messages = []
        st.rerun()
    if st.session_state.messages:
        md = "\n\n".join(f"**{'나' if m['role'] == 'user' else 'AI'}**: {m['content']}"
                         for m in st.session_state.messages)
        st.download_button("💾 대화 내려받기 (.md)", md.encode("utf-8"),
                           f"chat_{datetime.now():%Y%m%d_%H%M}.md", "text/markdown",
                           width="stretch")

# 모델이 바뀌면 이전 모델을 먼저 메모리에서 지운다 (메모리 2.7GB에 두 모델이 동시에 올라가지 않도록)
if st.session_state.get("loaded_model") not in (None, model_name):
    load_model.clear()
    gc.collect()

try:
    with st.status(f"'{model_name.split(' (')[0]}' 모델 준비 중… (처음 한 번은 내려받기 때문에 1~2분 걸려요)",
                   expanded=False) as status:
        llm, load_sec = load_model(model_name)
        status.update(label=f"✅ 모델 준비 완료: {model_name.split(' (')[0]}", state="complete")
    st.session_state.loaded_model = model_name
except ImportError:
    st.error("llama-cpp-python(모델 실행 도구)이 이 서버에 설치되지 않았어요. requirements.txt를 확인해 주세요.")
    st.stop()
except Exception as e:  # 내려받기 실패, 메모리 부족 등
    st.error(f"모델을 불러오지 못했어요: {e}")
    st.info("잠시 뒤 새로고침하거나, 사이드바에서 더 작은 모델(Qwen2.5-1.5B)을 골라 보세요.")
    st.stop()

tab_chat, tab_about = st.tabs(["💬 채팅", "📖 LLM은 어떻게 대답할까?"])

with tab_chat:
    if not st.session_state.messages:
        st.markdown("**이런 걸 물어보세요** (누르면 바로 질문해요)")
        cols = st.columns(len(EXAMPLES))
        for col, q in zip(cols, EXAMPLES):
            if col.button(q, width="stretch"):
                st.session_state.pending = q
                st.rerun()

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("stats"):
                s = m["stats"]
                st.caption(f"🤖 {s['model']} · {s['tokens']}토큰 · 첫 글자까지 {s['first']:.1f}초 · "
                           f"전체 {s['total']:.1f}초 · 초당 {s['tps']:.1f}토큰")

    prompt = st.chat_input("메시지를 입력하세요") or st.session_state.pending
    st.session_state.pending = None

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        history = [{"role": m["role"], "content": m["content"]}
                   for m in st.session_state.messages[-MAX_HISTORY_MESSAGES:]]
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

        with st.chat_message("assistant"):
            stats = {}
            try:
                with st.spinner("생각 중… (CPU로 계산해서 몇 초 걸려요)"):
                    gen = stream_answer(llm, messages, temperature, max_tokens, stats)
                    first_piece = next(gen, "")      # 첫 토큰이 나올 때까지 스피너 표시

                def full_stream():
                    yield first_piece
                    yield from gen

                answer = st.write_stream(full_stream())
            except Exception as e:
                answer, stats = f"⚠️ 답변 생성 중 오류가 났어요: {e}", {}
                st.markdown(answer)
            if stats:
                stats["model"] = model_name.split(" (")[0]
                st.caption(f"🤖 {stats['model']} · {stats['tokens']}토큰 · 첫 글자까지 {stats['first']:.1f}초 · "
                           f"전체 {stats['total']:.1f}초 · 초당 {stats['tps']:.1f}토큰")
        st.session_state.messages.append({"role": "assistant", "content": answer or "", "stats": stats or None})

with tab_about:
    st.markdown(f"""
### 1. LLM도 결국 인공신경망이에요
수업에서 배운 뉴런 **y = φ(Σwx + b)** 가 수십 층 쌓인 거대한 인공신경망이에요.
지금 쓰는 모델은 가중치(w)와 편향(b)이 **{info['params']}** 있어요.
과제 2의 콘크리트 신경망(가중치 2,689개)보다 약 **{info['n_params'] / 2689 / 1e4:.0f}만 배** 커요.

### 2. 글자를 숫자로 바꿔서 넣어요
컴퓨터는 숫자만 계산할 수 있어요. 그래서 문장을 '토큰'(글자 조각)으로 자르고, 각 토큰을 숫자 벡터로 바꿔 신경망에 넣어요.
4강의 '흑백 이미지를 행렬로 바꾸기'와 같은 생각이에요.

### 3. 다음에 올 토큰을 '분류'해요
LLM은 답 전체를 한 번에 만들지 않아요. **"다음에 올 토큰이 무엇일까?"** 를 수만 개 후보 중에서 고르는 분류를 반복해요.
마지막 층에서 소프트맥스로 각 후보의 확률을 구하고, 하나를 뽑아 이어 붙여요.
아래 채팅에서 글자가 하나씩 나오는 이유예요. 'temperature'를 높이면 확률이 낮은 후보도 뽑혀서 답이 다양해져요.

### 4. 4비트 양자화로 작게 만들었어요
가중치를 보통 16비트(2바이트)로 저장하면 이 모델은 약 **{info['n_params'] * 2 / 1e9:.1f}GB**예요. 이 서버(메모리 약 2.7GB)에는 안 들어가요.
그래서 가중치를 **4비트**로 줄인 파일(Q4_K_M, {info['size']})을 써요. 정확도는 조금 떨어지지만 CPU에서도 돌아가요.

### 5. 대화 중에는 가중치가 바뀌지 않아요
대화하는 동안 모델은 새로 '학습'하지 않아요. 가중치는 그대로이고, **이전 대화를 함께 다시 읽고** 답할 뿐이에요.
교수님이 말씀하신 '메모 보고 대답하기(컨닝)' 방식이에요. 그래서 문맥 창(2,048토큰)을 넘는 오래된 대화는 잊어요.
최종 과제에서는 대화 기록을 따로 저장해 두고 다시 읽게 해서, 컴퓨터를 꺼도 기억하게 만들 계획이에요.
""")
