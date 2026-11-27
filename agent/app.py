"""LLM 기반 자동 데이터분석 시스템 (데이터처리개론 마지막 과제).

오픈소스 LLM이 대화를 듣고, 지금까지 만든 프로그램(과제 1·2 대시보드, 과제 3 분석 도구)을
도구 호출로 직접 실행한 뒤 결과를 받아 설명한다. 화면에는 채팅창만 있다.

실행:  streamlit run agent/app.py
설정:  Streamlit의 Secrets에 OPENROUTER_API_KEY = "sk-or-..." (없으면 서버 설치 모델로 동작)
"""
import importlib
import os
import sys
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "llm_chat"))

import knowledge as K  # noqa: E402
import tools as T  # noqa: E402
import knowledge_base as KB  # noqa: E402
import programs as P  # noqa: E402
import llm as L  # noqa: E402
import core as C  # noqa: E402
import memory_store as M  # noqa: E402

# 코드를 새로 올리면 서버는 app.py만 다시 읽으므로, 다른 파일이 바뀌었으면 다시 불러온다.
_MODS = (K, T, KB, P, L, C, M)
_STAMP = tuple(os.path.getmtime(m.__file__) for m in _MODS)
if getattr(sys, "_agent_stamp", _STAMP) != _STAMP:
    for m in _MODS:
        importlib.reload(m)
sys._agent_stamp = _STAMP

TITLE = "LLM 기반 자동 데이터분석 시스템"
st.set_page_config(page_title=TITLE, page_icon="🤖", layout="centered")

TOOL_LABEL = {
    "list_datasets": ("📋", "데이터 목록", "과제 3 도구"), "load_dataset": ("📂", "데이터 불러오기", "과제 3 도구"),
    "load_data_url": ("🌐", "주소에서 데이터 불러오기", "과제 3 도구"), "make_data": ("🧪", "식으로 데이터 만들기", "과제 3 도구"),
    "data_summary": ("🔎", "데이터 요약", "과제 3 도구"), "set_variables": ("🎯", "변수 정하기", "과제 3 도구"),
    "run_linear_regression": ("📈", "선형 회귀", "과제 1 회귀"), "run_nonlinear_regression": ("📉", "비선형 회귀", "과제 1 회귀"),
    "run_neural_network": ("🧠", "인공신경망", "과제 2 인공신경망"), "run_deep_learning": ("🧬", "딥러닝", "과제 2 인공신경망"),
    "compare_models": ("🏆", "4개 모델 비교", "과제 1·2"), "run_dashboard_models": ("🗂️", "대시보드 모델 전체 학습", "과제 1·2 대시보드"),
    "predict": ("🔮", "예측", "학습한 모델"), "sweep_variable": ("📐", "변화 곡선", "학습한 모델"),
    "correlation": ("🌡️", "상관관계", "과제 3 도구"), "plot": ("📊", "그래프", "과제 3 도구"),
    "column_stats": ("🧮", "통계 계산", "과제 3 도구"), "add_variable": ("➕", "새 변수 만들기", "과제 3 도구"),
    "make_report": ("📄", "보고서", "과제 3 도구"), "current_status": ("ℹ️", "현재 상태", "과제 3 도구"),
    "search_lecture": ("📚", "강의 자료 검색 (RAG)", "지식 검색"), "search_my_code": ("💻", "내 코드 검색", "지식 검색"),
    "remember": ("📝", "기억하기", "기억"), "recall_memory": ("🗒️", "기억 읽기", "기억"),
}

WELCOME = f"""안녕하세요! **{TITLE}**이에요. 클릭 없이 대화로만 운용해요.

오픈소스 LLM이 말을 알아듣고, 지금까지 만든 프로그램을 **직접 실행**한 뒤 결과를 받아 설명해요.
실행 과정은 `🔧 프로그램 실행` 줄로 보여 드려요.

**이렇게 말해 보세요**
- "콘크리트 데이터로 선형회귀랑 인공신경망 돌려서 비교해줘"
- "대시보드 모델 전부 학습시키고 제일 좋은 모델 알려줘"
- "시멘트 350, 물 170, 재령 28일 때 강도 예측해줘" → "물을 200으로 바꾸면?"
- "물이 늘어나면 강도가 어떻게 변해?"
- "역전파가 뭐야? 내 코드 어디에 들어 있어?"
- "y = 3*sin(x) + x 데이터 만들어서 딥러닝 해줘"
- 📎로 CSV·Excel 파일을 첨부해도 돼요.

"진단"이라고 하면 연결 상태를, "대화 지워줘"라고 하면 처음부터 다시 시작해요."""


def secret(name, default=""):
    try:
        return st.secrets.get(name, os.environ.get(name, default))
    except Exception:
        return os.environ.get(name, default)


@st.cache_resource(show_spinner=False)
def engines(key, models):
    router = L.OpenRouter(key, [m.strip() for m in models.split(",") if m.strip()] or None) if key else None
    return router, L.LocalQwen()


ROUTER, LOCAL = engines(secret("OPENROUTER_API_KEY"), secret("OPENROUTER_MODELS"))


@st.cache_resource(show_spinner=False)
def memory_store(token):
    store = M.MemoryStore(token)
    store.pull()                       # 서버가 켜질 때 GitHub에 저장해 둔 기억을 읽어 옴
    return store


MEMORY = memory_store(secret("GITHUB_TOKEN"))

ss = st.session_state
if "state" not in ss:
    ss.state = T.new_state()
    ss.messages = [{"role": "assistant", "content": WELCOME, "events": []}]


# =============================================================
# 화면 그리기
# =============================================================
def draw_rep(rep, key):
    if rep.get("text"):
        st.markdown(rep["text"])
    for j, (kind, obj) in enumerate(rep.get("items", [])):
        if kind == "fig":
            st.plotly_chart(obj, width="stretch", key=f"{key}-f{j}")
        elif kind == "table":
            st.dataframe(obj, width="stretch", hide_index=True, key=f"{key}-t{j}")


def draw_call(ev):
    icon, label, prog = TOOL_LABEL.get(ev["name"], ("🔧", ev["name"], ""))
    args = ", ".join(f"{k}={v}" for k, v in (ev.get("args") or {}).items())
    st.caption(f"🔧 프로그램 실행 · {icon} **{label}** `{ev['name']}({args[:120]})` · {prog}")


def draw_message(m, idx):
    with st.chat_message(m["role"]):
        for n, ev in enumerate(m.get("events", [])):
            if ev["type"] == "tool_call":
                draw_call(ev)
            elif ev["type"] == "tool_result":
                with st.container(border=True):
                    draw_rep(ev["rep"], f"m{idx}-{n}")
            elif ev["type"] == "notice":
                st.info(ev["text"])
        if m.get("content"):
            st.markdown(m["content"])
        if m.get("engine"):
            st.caption(f"🤖 {m['engine']}")


def diagnose():
    lines = [f"- OpenRouter 열쇠: {'✅ 설정됨' if ROUTER else '❌ 없음 (Secrets에 OPENROUTER_API_KEY 필요)'}"]
    if ROUTER:
        lines.append(f"- 시도할 무료 모델: {', '.join(ROUTER.models)}")
        lines.append(f"- 상태: {'사용 가능' if ROUTER.available else '무료 한도 때문에 잠시 쉬는 중 → 서버 설치 모델 사용'}")
        if ROUTER.last_model:
            lines.append(f"- 마지막으로 답한 모델: {ROUTER.last_model}")
    lines.append(f"- 예비 모델: 서버 설치 {L.LOCAL_MODEL['file']} (llama.cpp)")
    lines.append(f"- 연결된 프로그램: 과제 1·2 대시보드 모델 {len(P.dashboard()['ALL_MODELS'])}종, 과제 3 분석 도구, 도구 {len(P.TOOLS)}개")
    lines.append(f"- 강의 지식 조각 {len(KB.LECTURE) + len(K.NOTES)}개, 기억 {len(T.recall_all())}개 (사용자 정보 {len(C.facts())}개)")
    lines.append("- 오래가는 기억: " + ("✅ GitHub memory 브랜치에 저장" + (f" (⚠️ {MEMORY.error})" if MEMORY.error else "")
                                     if MEMORY.enabled else "❌ 꺼짐 (Secrets에 GITHUB_TOKEN이 없어 서버가 다시 켜지면 기억이 지워짐)"))
    return "**🩺 진단 결과**\n\n" + "\n".join(lines)


st.title(f"🤖 {TITLE}")
engine_now = (f"OpenRouter 무료 오픈소스 모델 (도구 호출)" if ROUTER and ROUTER.available
              else "서버에 설치한 오픈소스 모델 Qwen2.5-1.5B (예비 모드)")
st.caption(f"오픈소스 LLM이 대화로 과제 1·2·3 프로그램을 운용해요 · 지금 쓰는 LLM: {engine_now}")
st.caption("📘 [설명서 (Notion)](https://azure-raven-7e9.notion.site/425361e0fce781e8b3ffcd29c2102751) · "
           "💻 [코드 (GitHub)](https://github.com/csw33400-art/-2/tree/main/agent) · "
           "📚 [과제 모음](https://azure-raven-7e9.notion.site/3f9361e0fce781609556d7190fe57ba1)")

for i, m in enumerate(ss.messages):
    draw_message(m, i)

prompt = st.chat_input("무엇이든 말해 보세요 (📎 CSV·Excel 첨부 가능)", accept_file=True,
                       file_type=["csv", "xlsx", "xls", "tsv", "txt", "json"])
if prompt:
    text = (prompt.text or "").strip()
    files = list(prompt.files or [])
    shown = text + (f"\n\n📎 {', '.join(f.name for f in files)}" if files else "")
    ss.messages.append({"role": "user", "content": shown})
    draw_message(ss.messages[-1], len(ss.messages) - 1)

    t = T.norm(text)
    if any(w in t for w in ("대화지워", "처음부터", "초기화", "리셋")):
        ss.state = T.new_state()
        ss.messages = [{"role": "assistant", "content": "대화를 지우고 처음부터 시작해요. (기억 메모는 그대로예요. 지우려면 \"기억 지워줘\")", "events": []}]
        st.rerun()
    if "기억지워" in t:
        T.forget_all()
        MEMORY.push()
        ss.messages.append({"role": "assistant", "content": "기억 메모를 모두 지웠어요.", "events": []})
        st.rerun()
    if t in ("진단", "상태점검"):
        ss.messages.append({"role": "assistant", "content": diagnose(), "events": []})
        st.rerun()

    msg = {"role": "assistant", "content": "", "events": [], "engine": ""}
    idx = len(ss.messages)
    with st.chat_message("assistant"):
        for f in files:                                   # 첨부 파일은 바로 불러온다
            ev = {"type": "tool_call", "name": "load_file", "args": {"file": f.name}}
            draw_call(ev)
            rep = T.load_file(ss.state, f)
            msg["events"] += [ev, {"type": "tool_result", "name": "load_file", "rep": rep}]
            with st.container(border=True):
                draw_rep(rep, f"m{idx}-file")
        if not text:
            text = "첨부한 데이터를 살펴보고 어떤 분석을 하면 좋을지 알려줘."
        history = [m for m in ss.messages[1:-1] if m.get("content")]     # 첫 안내문은 빼고 보냄
        with st.spinner("LLM이 생각하고 프로그램을 실행하는 중…"):
            for n, ev in enumerate(C.run_turn(ss.state, history, text, ROUTER, LOCAL)):
                if ev["type"] == "tool_call":
                    draw_call(ev)
                elif ev["type"] == "tool_result":
                    with st.container(border=True):
                        draw_rep(ev["rep"], f"m{idx}-{n}")
                elif ev["type"] == "notice":
                    st.info(ev["text"])
                elif ev["type"] == "answer":
                    msg["content"], msg["engine"] = ev["text"], ev["engine"]
                    if ev["text"]:
                        st.markdown(ev["text"])
                    st.caption(f"🤖 {ev['engine']}")
                    continue
                msg["events"].append(ev)
    ss.messages.append(msg)
    MEMORY.push()                      # 이번 대화에서 늘어난 기억을 GitHub에 저장 (열쇠가 있을 때만)
