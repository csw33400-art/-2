# 코드 설명서: LLM 기반 자동 데이터분석 시스템

`agent/` 폴더의 코드를 파일별로 설명합니다. 기능과 사용법은 [PROGRAM.md](PROGRAM.md)에 있습니다.

## 1. 파일 구성

| 파일 | 줄 수 | 역할 |
| --- | --- | --- |
| [`app.py`](../app.py) | 약 200 | 채팅 화면. 도구 실행 과정과 그래프·표를 보여 줌 |
| [`core.py`](../core.py) | 약 210 | 에이전트 반복 구조(명령→실행→결과 회신), 예비 방식 전환 |
| [`llm.py`](../llm.py) | 약 220 | OpenRouter 연결, 서버 설치 Qwen 연결, 글로 도구 고르기 |
| [`programs.py`](../programs.py) | 약 420 | 프로그램 연결층(API): 도구 설명서 24개와 실행기, 대시보드 코드 불러오기 |
| [`knowledge_base.py`](../knowledge_base.py) | 약 130 | 강의 지식 RAG, 내 코드 검색 |
| `requirements.txt`, `packages.txt` | | 서버 설치 패키지 (과제 3과 같은 구성) |

```
사용자 ─▶ app.py ─▶ core.run_turn() ─▶ llm.OpenRouter.chat()  (LLM이 도구 선택)
                         │
                         └─▶ programs.execute() ─┬─▶ 과제 1·2 대시보드 app.py의 build_estimator
                                                  ├─▶ 과제 3 llm_chat/tools.py 분석 함수
                                                  └─▶ knowledge_base 검색, 메모
```

## 2. `programs.py`: 프로그램 연결층 (API)

### 2.1 도구 설명서

LLM은 함수를 직접 실행할 수 없으므로, "이런 이름의 기능이 있고 이런 값을 받는다"는 설명서(JSON 스키마)를 보고 고릅니다.

```python
_fn("run_neural_network", "인공신경망(MLP)을 학습한다. layers=은닉층별 노드 수(예: [64, 32]), activation=relu|tanh|logistic, "
    "epochs=최대 학습 횟수, learning_rate=학습률. 학습 곡선과 Dense 구조 코드를 보여 준다.",
    {"layers": {"type": "array", "items": {"type": "integer"}}, "activation": {"type": "string", "enum": ["relu", "tanh", "logistic"]},
     "epochs": I, "learning_rate": N}),
```

LLM이 고르면 이런 형태로 돌아옵니다: `{"name": "run_neural_network", "arguments": "{\"layers\": [32, 16], \"activation\": \"tanh\"}"}`

### 2.2 실행기

```python
def execute(state, name, args):
    """도구 하나를 실행한다. (화면용 결과, LLM에게 돌려줄 글)"""
    try:
        if name in DATA_TOOLS:
            _ensure_data(state, notes)          # 데이터가 없으면 콘크리트 데이터를 먼저 불러옴
        rep = _run(state, name, args, notes)    # 이름에 맞는 실제 프로그램 함수 실행
    except Exception as e:                      # 실패해도 이유를 LLM에게 알려 다른 방법을 찾게 함
        rep = T.reply(f"⚠️ `{name}` 실행 중 오류: {type(e).__name__}: {e}")
    if rep.get("summary"):
        T.remember(rep["summary"])              # 실행 기록을 메모에 적음 (기억)
    return rep, for_llm(rep) + f"\n\n[현재 상태] {_state_line(state)}"
```

- 화면에는 그래프와 표가 그대로 나가고, LLM에게는 글과 표(마크다운)만 짧게 돌려줍니다(`for_llm`). 그래프는 LLM이 읽을 수 없고, 글이 길면 무료 모델의 처리 한도를 넘기 때문입니다.
- `[현재 상태]`에 데이터·열 이름·X·Y·학습한 모델의 R²를 붙여, LLM이 다음 도구를 고를 때 정확한 열 이름을 쓰게 합니다.
- `_cols`: LLM이 "water", "강도"처럼 이름을 다르게 불러도 실제 열(`물(kg/m3)`, `압축강도(MPa)`)을 찾습니다.

### 2.3 과제 1·2 대시보드 코드 불러오기

```python
@lru_cache(maxsize=1)
def dashboard():
    tree = ast.parse(DASHBOARD_FILE.read_text(encoding="utf-8"))      # app.py를 문법 트리로 읽음
    body = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):            # import 줄 (streamlit·plotly 제외)
            ...
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in KEEP_DEFS:
            body.append(node)                                          # build_estimator, CurveFitRegressor, safe_metrics
        elif isinstance(node, ast.Assign) and ...:                     # ALL_MODELS, CURVE_KIND
            body.append(node)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(DASHBOARD_FILE), "exec"), ns)
```

대시보드는 화면 코드와 모델 코드가 한 파일에 섞여 있어서 그냥 `import`하면 화면이 실행됩니다. 그래서 파일을 문법 트리로 읽어 모델 코드만 골라 실행합니다. 대시보드 파일은 고치지 않으므로 기존 과제 1·2 프로그램은 그대로 동작합니다.

`run_dashboard_models`는 대시보드와 같은 방식으로 학습·평가합니다: 80/20 분할, `build_estimator(name, cfg, 42)`로 모델 생성, `safe_metrics`로 R²·RMSE·MAE, `KFold(5)` + `cross_val_predict`로 교차검증.

## 3. `core.py`: 에이전트 반복 구조

```python
def _loop_native(state, history, text, router):
    msgs = [{"role": "system", "content": system_prompt(state)}] + _history(history) + [{"role": "user", "content": text}]
    for step in range(MAX_STEPS):                          # 최대 5번
        tools = P.TOOLS if step < MAX_STEPS - 1 else None  # 마지막 번에는 답만 하게 함
        msg, model = router.chat(msgs, tools=tools)        # ① LLM이 판단 (명령)
        calls = msg.get("tool_calls") or []
        if not calls:
            yield {"type": "answer", "text": msg["content"], ...}   # 도구가 필요 없으면 최종 답
            return
        msgs.append({"role": "assistant", "content": ..., "tool_calls": calls})
        for c in calls:
            rep, result = P.execute(state, name, args)     # ② 실제 프로그램 실행
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": result})   # ③ 결과 회신
            msgs[0]["content"] = system_prompt(state)      # 바뀐 상태를 다시 알려 줌
```

- `yield`로 진행 상황을 하나씩 내보내서, 화면이 "도구 실행 → 결과 → 다음 도구"를 실시간으로 그립니다.
- 시스템 프롬프트(`SYSTEM`)에 운용할 프로그램 목록, 규칙(숫자를 지어내지 말 것, 개념 질문은 강의 검색 먼저), 현재 상태, 최근 기억을 넣습니다.
- `run_turn`은 1순위(도구 호출) → 2순위(글로 고르기) → 3순위(낱말 규칙 + 서버 모델) 순서로 시도합니다. 중간에 연결이 끊기면 이미 실행한 결과는 보여 주고 설명만 생략합니다.

## 4. `llm.py`: 언어모델 연결

```python
def _post(self, payload):
    req = urllib.request.Request(OPENROUTER_URL, data=json.dumps(payload).encode("utf-8"), method="POST",
                                 headers={"Authorization": f"Bearer {self.key}", ...})
    ...
    except urllib.error.HTTPError as e:
        if e.code == 429:
            raise RateLimited(...)         # 무료 사용 한도 → 예비 모델로
        if "tool" in body.lower():
            raise ToolsUnsupported(body)   # 도구 호출 미지원 → 글 방식으로
```

- OpenRouter는 OpenAI와 같은 형식(`/chat/completions`, `tools`, `tool_calls`)을 씁니다. 별도 라이브러리 없이 파이썬 기본 `urllib`로 보냅니다.
- `chat()`은 무료 모델 목록을 앞에서부터 시도하고, 응답하지 않은 모델은 `self.bad`에 넣어 다음부터 뒤로 미룹니다.
- `LocalQwen`은 과제 3에서 만든 `llm_chat/llm_worker.py`를 별도 프로세스로 실행해 서버 설치 모델과 대화합니다. 모델이 멈춰도 웹 앱은 죽지 않습니다.
- `parse_text_tool`: 도구 호출 기능이 없는 모델이 `<tool>{"name": ..., "arguments": ...}</tool>`로 쓴 글에서 도구를 꺼냅니다.

## 5. `knowledge_base.py`: 강의 RAG와 코드 검색

```python
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
mat = vec.fit_transform([f"{s} {t}" for s, t in docs])               # 조각마다 벡터로
sims = cosine_similarity(vec.transform([query]), mat).ravel()        # 질문과의 코사인 유사도
order = [i for i in np.argsort(-sims)[:k] if sims[i] > 0.05]         # 비슷한 조각 k개
```

- 강의 조각 32개(2장 통계 ~ 8장 CNN, LLM 수업) + 과제 3 메모 20개를 씁니다.
- 한국어는 조사·띄어쓰기가 다양해서 낱말 대신 글자 2~4개 묶음(n-gram)으로 벡터를 만듭니다. "역전파가", "역전파는"이 같은 조각을 찾습니다.
- 코드 검색은 `ast`로 파일을 함수 단위로 잘라 같은 방법으로 찾습니다. 대시보드는 화면 코드가 많아 40줄씩도 잘라 넣습니다.

## 6. `app.py`: 채팅 화면

- `engines()`: Secrets의 `OPENROUTER_API_KEY`로 OpenRouter 연결을 만들고 `@st.cache_resource`로 모든 접속자가 공유합니다(무료 한도도 열쇠 하나 기준이라 같이 관리).
- 도구 실행은 `🔧 프로그램 실행 · 📈 선형 회귀 run_linear_regression() · 과제 1 회귀`처럼 한 줄로 보여 주고, 결과는 테두리 상자 안에 그립니다.
- 첨부 파일은 LLM을 거치지 않고 바로 `T.load_file`로 불러온 뒤 LLM에게 "첨부한 데이터를 살펴보고 분석을 제안해 줘"라고 넘깁니다.
- 코드를 새로 올리면 서버가 `app.py`만 다시 읽기 때문에, 다른 파일의 수정 시각이 바뀌면 `importlib.reload`로 다시 불러옵니다.

## 7. 시험해 본 방법

실제 무료 모델은 개발 환경에서 접속할 수 없어서 아래처럼 시험했습니다.

| 시험 | 방법 | 확인한 것 |
| --- | --- | --- |
| 도구 24개 | 콘크리트·합성 데이터로 하나씩 실행 | 결과·오류 처리, 영어 열 이름 인식 |
| 에이전트 반복 | 정해 둔 순서로 도구를 부르는 가짜 모델 | 여러 단계 연속 실행, 결과 회신, 중간 끊김, 없는 도구 이름 |
| OpenRouter 형식 | OpenRouter처럼 응답하는 테스트 서버 | 요청 형식(tools, tool 역할 메시지), 실패 모델 건너뛰기, 429 시 예비 전환 |
| 화면 | 가짜 streamlit으로 app.py 실행 | 진단, 분석, 예측, 대화 지우기 흐름 |
