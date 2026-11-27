"""LLM 에이전트: 대화를 듣고 프로그램을 운용하는 반복 구조.

한 번의 대화(turn)는 이렇게 돈다.

    사용자 말 ─▶ LLM: "어떤 도구를 어떤 값으로 실행할까?" 판단 (명령)
                  │
                  ├─ 도구 호출 ─▶ programs.execute(): 실제 프로그램 실행 ─▶ 결과를 LLM에게 돌려줌 (피드백)
                  │                     └─ (반복: 결과를 보고 다음 도구를 또 고를 수 있음, 최대 MAX_STEPS번)
                  └─ 도구 없이 답 ─▶ 사용자에게 최종 답변

이 함수는 화면(Streamlit)과 분리되어 있어서, 진행 상황을 '이벤트'로 하나씩 내보낸다.
화면은 이벤트를 받아 도구 실행 과정과 그래프·표를 그린다.
"""
import json
import re

import programs as P
import llm as L
import knowledge_base as KB
from programs import T

MAX_STEPS = 5           # 한 번의 대화에서 LLM을 부르는 최대 횟수 (무료 사용 횟수 절약)
HISTORY = 8             # LLM에게 함께 보여 줄 지난 대화 수

SYSTEM = """너는 '{title}'의 AI 분석가다. 사용자는 클릭 없이 대화로만 프로그램을 운용한다.
너는 도구(함수)를 호출해 사용자가 만든 프로그램을 직접 실행한다:
- 과제 1·2 회귀·인공신경망 대시보드 (선형, 릿지, 라쏘, 다항, 결정 트리, 랜덤 포레스트, 부스팅, SVR, KNN, 인공신경망, 곡선 회귀)
- 과제 3 채팅 분석 도구 (데이터 불러오기, 변수 설정, 선형·비선형 회귀, 인공신경망, 딥러닝, 예측, 보고서)

규칙:
1. 분석·학습·예측·그래프·데이터 요청은 반드시 도구를 호출해 실제로 실행한다. 숫자를 지어내지 않는다.
2. 여러 단계가 필요하면 도구를 차례로 부른다. 예: 데이터 불러오기 → 변수 정하기 → 모델 학습 → 예측.
3. 열 이름은 [현재 상태]의 열 목록에 있는 이름을 그대로 쓴다.
4. 개념 질문(역전파, 과적합, 활성화 함수, LLM 등)은 search_lecture로 강의 내용을 먼저 찾아 근거로 설명하고,
   내 코드에서 어디에 쓰였는지 물으면 search_my_code를 쓴다.
5. 도구 결과를 받으면 한국어로 3~6문장으로 핵심 수치와 의미를 설명한다. 표·그래프는 화면에 이미 나오므로 다시 옮겨 적지 않는다.
6. 사용자가 기억해 달라고 하거나, 자기 이름·소속·관심사·선호처럼 다음에도 알아야 할 정보를 말하면 remember로 적는다.
   지난 작업이나 기억을 물으면 recall_memory를 쓰거나 아래 [기억하고 있는 정보]를 보고 답한다.
7. [기억하고 있는 정보]에 사용자 이름이 있으면 그 이름으로 부르고, 이미 아는 것을 다시 묻지 않는다.

[현재 상태] {state}
[기억하고 있는 정보] {facts}
[최근 작업 기록] {memory}"""

FACT_PREFIX = ("사용자 정보:", "메모:")
NAME_PAT = re.compile(r"(?:내|제|나의|저의)\s*이름은\s*([가-힣A-Za-z]{1,12}?)(?:이야|야|이에요|예요|입니다|이다|라고|이라고|$|\s|[.!,])")
CALL_PAT = re.compile(r"(?:나를|날|저를)\s*([가-힣A-Za-z]{1,12}?)(?:이라고|라고)\s*불러")


def facts():
    """기억 메모 중 사용자 정보와 '기억해 줘'로 적은 메모만 모은다 (작업 기록과 구분)."""
    return [m["summary"] for m in T.recall_all() if str(m["summary"]).startswith(FACT_PREFIX)]


def capture_facts(text):
    """LLM이 놓치더라도 이름은 확실히 기억하도록, '내 이름은 ○○' 같은 말을 직접 적는다."""
    for pat in (NAME_PAT, CALL_PAT):
        m = pat.search(text)
        if m and m.group(1) not in facts_names():
            old = [x for x in T.recall_all() if "사용자의 이름" not in str(x["summary"])]   # 예전 이름은 지우고 새 이름만 남김
            T.MEMORY_FILE.write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")
            T.remember(f"사용자 정보: 사용자의 이름(부를 이름)은 {m.group(1)}")
            return m.group(1)
    return None


def facts_names():
    return " ".join(facts())


def system_prompt(state, title="LLM 기반 자동 데이터분석 시스템"):
    work = [m["summary"] for m in T.recall_all() if not str(m["summary"]).startswith(FACT_PREFIX)]
    mem = "; ".join(work[-5:]) or "없음"
    fx = "; ".join(facts()[-15:]) or "없음"
    return SYSTEM.format(title=title, state=P._state_line(state), memory=mem[:600], facts=fx[:900])


def _history(history):
    out = []
    for m in history[-HISTORY:]:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            out.append({"role": m["role"], "content": str(m["content"])[:700]})
    return out


def _args(raw):
    if isinstance(raw, dict):
        return raw
    try:
        v = json.loads(raw or "{}")
        return v if isinstance(v, dict) else {}
    except ValueError:
        return {}


# =============================================================
# 1. OpenRouter 모델이 도구 호출 기능으로 운용 (1순위)
# =============================================================
def _loop_native(state, history, text, router):
    msgs = [{"role": "system", "content": system_prompt(state)}] + _history(history) + [{"role": "user", "content": text}]
    for step in range(MAX_STEPS):
        tools = P.TOOLS if step < MAX_STEPS - 1 else None        # 마지막 번에는 답만 하게 함
        if not tools:
            msgs.append({"role": "user", "content": "지금까지의 도구 결과로 최종 답변을 해 줘."})
        try:
            msg, model = router.chat(msgs, tools=tools)
        except L.LLMError as e:
            if step == 0:
                raise                       # 아직 아무것도 안 했으면 다음 순위 방식으로
            yield {"type": "answer", "engine": "OpenRouter (중간에 연결 끊김)",
                   "text": f"실행 결과는 위와 같아요. 언어모델 연결이 중간에 끊겨 설명은 생략했어요. ({str(e)[:100]})"}
            return
        calls = msg.get("tool_calls") or []
        if not calls:
            yield {"type": "answer", "text": (msg.get("content") or "").strip(), "engine": f"OpenRouter · {model}"}
            return
        msgs.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for c in calls[:4]:
            name, args = c["function"]["name"], _args(c["function"].get("arguments"))
            yield {"type": "tool_call", "name": name, "args": args, "engine": model}
            rep, result = P.execute(state, name, args)
            yield {"type": "tool_result", "name": name, "rep": rep}
            msgs.append({"role": "tool", "tool_call_id": c.get("id", name), "name": name, "content": result})
            msgs[0]["content"] = system_prompt(state)                  # 상태가 바뀌었으니 갱신


# =============================================================
# 2. 글로 도구를 고르게 하는 방식 (2순위: 도구 호출이 막힌 무료 모델)
# =============================================================
def _loop_text(state, history, text, ask, engine):
    msgs = [{"role": "system", "content": system_prompt(state) + L.text_protocol(P.TOOLS)}] + _history(history) + \
           [{"role": "user", "content": text}]
    for step in range(MAX_STEPS):
        reply = ask(msgs)
        call = L.parse_text_tool(reply) if step < MAX_STEPS - 1 else None
        if not call:
            yield {"type": "answer", "text": L.TOOL_TAG.sub("", reply).strip(), "engine": engine}
            return
        name, args = call
        yield {"type": "tool_call", "name": name, "args": args, "engine": engine}
        rep, result = P.execute(state, name, args)
        yield {"type": "tool_result", "name": name, "rep": rep}
        msgs.append({"role": "assistant", "content": reply})
        msgs.append({"role": "user", "content": f"[도구 결과: {name}]\n{result}\n\n필요하면 다음 도구를, 아니면 최종 답변을 해 줘."})
        msgs[0]["content"] = system_prompt(state) + L.text_protocol(P.TOOLS)


# =============================================================
# 3. 예비: 낱말 규칙으로 도구를 고르고, 서버 설치 모델이 결과를 설명 (3순위)
# =============================================================
def rule_plan(state, text):
    """인터넷 LLM을 못 쓸 때 쓰는 규칙. 말 속의 낱말로 실행할 도구 목록을 정한다."""
    t = T.norm(text)
    if NAME_PAT.search(text) or CALL_PAT.search(text):
        return []                                   # 이름 알려 주기: capture_facts가 이미 적어 둠
    if re.search(r"내이름|나누구|내가누구|나를뭐라고", t):
        return [("recall_memory", {})]
    plan = []
    key = T.match_dataset(t)
    if key and (state["key"] != key or re.search(r"불러|가져|로드|데이터로", t)):
        plan.append(("load_dataset", {"name": key}))
    m = re.search(r"y\s*=\s*([^,\n가-힣]+)", text, re.I)
    if m and re.search(r"만들|생성", t):
        plan.append(("make_data", {"formula": m.group(1).strip()}))
    explain = re.search(r"뭐야|무엇|설명|왜|차이|원리|개념|어떻게|뜻|알려줘$", t) and not re.search(r"해줘|돌려|학습시|실행|그려", t)
    if explain:
        return plan + [("search_lecture", {"query": text})]
    if re.search(r"코드|어디에", t):
        return plan + [("search_my_code", {"query": text})]
    if re.search(r"기억해|메모해", t):
        return [("remember", {"note": text})]
    if re.search(r"내이름|나누구|내가누구", t):
        return [("recall_memory", {})]
    if re.search(r"지난|기억|전에뭐|뭐했", t):
        return [("recall_memory", {})]
    if state["df"] is not None and (T.has_var_marker(t) or re.search(r"빼고|제외|으로.*예측|로.*예측", t)):
        msg = T.parse_vars(state, text)
        if msg:
            plan.append(("current_status", {}))
    sweep = T.parse_sweep(state, text) if state["df"] is not None else None
    if sweep:                                   # "물이 늘어나면 강도는?", "시멘트를 100에서 500까지 바꾸면?"
        return plan + [("sweep_variable", {"variable": sweep[0], "start": sweep[1], "end": sweep[2]})]
    vals = T.parse_inputs(state, text) if state["df"] is not None else {}
    rules = [(r"대시보드|모든모델|13|랜덤포레스트|부스팅|svr|knn|릿지|라쏘|결정트리", ("run_dashboard_models", {})),
             (r"비교|전부|다돌려|모두", ("compare_models", {})),
             (r"딥러닝|심층", ("run_deep_learning", {})),
             (r"신경망|ann|mlp", ("run_neural_network", {})),
             (r"비선형|다항|곡선", ("run_nonlinear_regression", {})),
             (r"선형|회귀", ("run_linear_regression", {})),
             (r"상관", ("correlation", {})),
             (r"요약|통계|미리보기", ("data_summary", {})),
             (r"보고서|리포트", ("make_report", {})),
             (r"데이터목록|무슨데이터|어떤데이터", ("list_datasets", {}))]
    for pat, step in rules:
        if re.search(pat, t):
            plan.append(step)
            break
    if vals:
        plan.append(("predict", {"values": {c: v for c, v in vals.items()}}))
    return plan


def _loop_rules(state, history, text, local):
    plan = rule_plan(state, text)
    results = []
    for name, args in plan:
        yield {"type": "tool_call", "name": name, "args": args, "engine": "낱말 규칙"}
        rep, result = P.execute(state, name, args)
        yield {"type": "tool_result", "name": name, "rep": rep}
        results.append(f"[{name}] {result[:1200]}")
    hits = KB.lecture_hits(text, k=2) if not plan else []
    ref = "\n".join(f"- [{s}] {t}" for s, t, _ in hits)
    prompt = ("다음 도구 결과를 한국어 3문장으로 쉽게 설명해 줘. 숫자는 결과에 있는 것만 써.\n\n" + "\n\n".join(results)) if results \
        else (f"질문: {text}\n\n참고 자료:\n{ref}\n\n참고 자료를 근거로 한국어로 쉽게 답해 줘." if ref else text)
    try:
        known = "; ".join(facts()[-10:])
        answer = local.chat([{"role": "system", "content": "너는 데이터 분석을 돕는 한국어 AI 조교다. 짧고 정확하게 답한다."
                              + (f" 기억하고 있는 사용자 정보: {known}" if known else "")},
                             {"role": "user", "content": prompt[:3500]}])
        engine = "서버 설치 모델 Qwen2.5-1.5B (예비 모드)"
    except L.LLMError as e:
        answer = "" if results else ("(언어모델을 쓸 수 없어 강의 자료 검색 결과만 보여 드려요.)\n\n" + ref if ref else
                                     f"지금은 언어모델에 연결할 수 없어요 ({e}). \"선형회귀 해줘\"처럼 분석 명령은 계속 쓸 수 있어요.")
        engine = "낱말 규칙 (언어모델 연결 안 됨)"
    yield {"type": "answer", "text": answer.strip(), "engine": engine}


# =============================================================
# 4. 한 번의 대화: 1순위 → 2순위 → 3순위 순서로 시도
# =============================================================
def run_turn(state, history, text, router, local):
    name = capture_facts(text)
    if name:
        yield {"type": "notice", "text": f"📝 '{name}'(이)라는 이름을 기억해 둘게요."}
    if router is not None and router.available:
        try:
            yield from _loop_native(state, history, text, router)
            return
        except L.RateLimited as e:
            yield {"type": "notice", "text": f"무료 사용 한도에 걸려 서버 설치 모델로 바꿔요. ({str(e)[:120]})"}
        except L.LLMError as e:
            yield {"type": "notice", "text": f"도구 호출 방식이 안 돼서 글로 고르는 방식으로 바꿔요. ({str(e)[:120]})"}
            try:
                def ask(msgs):
                    msg, _ = router.chat(msgs, tools=None)
                    return msg.get("content") or ""
                yield from _loop_text(state, history, text, ask, f"OpenRouter · {router.last_model or router.models[0]} (글 방식)")
                return
            except L.LLMError as e2:
                yield {"type": "notice", "text": f"OpenRouter 연결이 안 돼서 서버 설치 모델로 바꿔요. ({str(e2)[:120]})"}
    yield from _loop_rules(state, history, text, local)
