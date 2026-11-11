# 코드 설명서: 오픈소스 LLM 챗봇

`llm_chat/` 폴더의 코드를 파일별로 설명합니다. 프로그램 기능과 사용법은 [PROGRAM.md](PROGRAM.md)에 있습니다.

## 1. 파일 구성

| 파일 | 줄 수 | 역할 |
| --- | --- | --- |
| [`app.py`](../app.py) | 약 680 | 채팅 화면, 말 해석(`route`), 언어모델 관리 |
| [`tools.py`](../tools.py) | 약 1,200 | 데이터 읽기·정리, 변수 인식, 회귀·신경망·예측·보고서 |
| [`knowledge.py`](../knowledge.py) | 약 90 | 자유 대화용 참고 메모 20개와 검색 (작은 RAG) |
| [`llm_worker.py`](../llm_worker.py) | 약 55 | 언어모델을 별도 프로세스에서 실행 |
| [`requirements.txt`](../requirements.txt) | | 서버가 설치할 패키지 (`llama-cpp-python` 등) |
| [`packages.txt`](../packages.txt) | | 서버가 설치할 시스템 라이브러리 (`libgomp1`) |

기존 과제 1·2의 대시보드(`app.py`, 저장소 맨 위)와는 파일을 공유하지 않습니다. 데이터 파일 `data/concrete.csv`만 함께 씁니다.

## 2. 한 번의 채팅이 처리되는 순서

1. `st.chat_input`이 글과 첨부 파일을 받습니다.
2. `route(text, files)`가 말을 해석해 실행할 도구를 고릅니다.
3. 도구(`tools.py`의 함수)가 계산하고 `{"text", "items", "summary"}`를 돌려줍니다.
4. `render`가 글, 그래프, 표를 채팅 안에 그립니다.
5. 분석 결과에는 언어모델이 두세 문장 해설을 덧붙입니다.
6. 어떤 도구에도 해당하지 않으면 언어모델이 자유롭게 답합니다.

## 3. `llm_worker.py`: 언어모델 설치와 실행

```python
from llama_cpp import Llama

llm = Llama.from_pretrained(repo_id=repo, filename=filename, n_ctx=4096,
                            n_threads=threads, n_batch=128, verbose=False)
send({"ready": True})

for line in stdin:                       # 부모(app.py)가 보낸 질문을 한 줄씩 읽음
    req = json.loads(line)
    for chunk in llm.create_chat_completion(messages=req["messages"], stream=True, ...):
        text = chunk["choices"][0].get("delta", {}).get("content")
        if text:
            send({"t": text})            # 토큰이 나올 때마다 부모에게 전달
    send({"done": True})
```

- `Llama.from_pretrained`가 Hugging Face에서 GGUF 파일을 내려받아(설치) 메모리에 올립니다.
- 이 파일은 `app.py`가 `subprocess.Popen`으로 실행하는 **자식 프로세스**입니다. 부모와는 한 줄에 JSON 하나씩 주고받습니다.
- 프로세스를 나눈 이유: 언어모델이 메모리 부족 등으로 죽어도 웹 앱 전체가 죽지 않게 하기 위해서입니다. 처음에는 같은 프로세스에서 돌렸는데, 모델이 멈추면 사이트 접속이 안 되는 문제가 있어 바꿨습니다.

## 4. `app.py`: 채팅 화면과 말 해석

### 4.1 언어모델 관리 (`LLMWorker`)

| 함수 | 하는 일 |
| --- | --- |
| `cpu_threads()` | 컨테이너의 CPU 제한(`/sys/fs/cgroup/cpu.max`)을 읽어 스레드 수를 정함 |
| `LLMWorker.ensure(name)` | 모델 프로세스를 켬. 다른 모델이 떠 있으면 먼저 꺼서 메모리를 비움 |
| `LLMWorker.chat(...)` | 질문을 보내고 답을 토큰 단위로 받음. 잠금으로 한 번에 한 질문만 처리 |
| `get_llm()` | 준비된 모델을 돌려줌. 실패하면 `None`을 돌려 분석 기능은 계속 동작 |
| `ask_llm(...)` | `st.write_stream`으로 답을 한 글자씩 화면에 표시하고 속도를 측정 |
| `diagnose()` | "진단" 입력 시 파이썬·패키지·모델 상태를 점검해 표시 |

### 4.2 말 해석 (`route`)

`route`는 위에서부터 차례로 검사해 처음 맞는 도구를 실행합니다.

| 순서 | 검사 | 실행 |
| --- | --- | --- |
| 1 | 도움말, 대화·기억 지우기, 기억 조회, 설정, 진단 | 해당 안내 |
| 2 | 식으로 데이터 만들기, 붙여넣은 표·숫자 | `make_data`, `parse_pasted` |
| 3 | 첨부 파일, 주소, 공개 데이터 이름 | `load_file`, `load_url`, `load_dataset` |
| 4 | "이름 = 식 … 만들어줘" | `add_variable` |
| 5 | 변수 이름 + 숫자, 범위 표현 | `run_predict`, `run_sweep` |
| 6 | 모델 이름 + 실행 표현 | `run_linear`, `run_nonlinear`, `run_ann`, `run_compare` |
| 7 | 보고서, 열 목록, 그래프, 통계, 상관, 요약, 변수 지정 | 해당 도구 |
| 8 | 어디에도 해당하지 않음 | 언어모델 자유 대화 |

질문과 명령은 낱말로 구분합니다.

```python
explain_q, strong, generic = has(t, *EXPLAIN), has(t, *STRONG), has(t, *GENERIC)
intents = detect_intents(t, text)            # 선형 / 비선형 / 신경망 / 딥러닝 / 비교
run = ((strong or generic) and not explain_q) or (bool(intents) and len(t) <= 12 and not explain_q)
```

- `EXPLAIN`: "설명", "뭐야", "왜", "차이", "어떻게" 등. 이 낱말이 있으면 실행하지 않고 언어모델에 넘깁니다.
- `STRONG`/`GENERIC`: "돌려", "학습해", "해줘" 등 실행 표현입니다.

### 4.3 자유 대화에 넣어 주는 정보

```python
notes = K.search(text) or K.search(asked)          # 질문과 관련된 참고 메모 (RAG)
ref = ("\n\n[참고 자료]\n" + "\n".join(f"- {n}" for n in notes)) if notes else ""
messages = [{"role": "system", "content": SYSTEM_PROMPT + ref + llm_context()}] + history
```

언어모델은 (1) 역할 지시문, (2) 검색된 참고 메모, (3) 현재 데이터·학습 결과·작업 기록(`llm_context`), (4) 최근 대화 8개를 함께 읽고 답합니다.

## 5. `knowledge.py`: 참고 메모 검색

```python
def search(text, k=2):
    low = str(text).lower()
    flat = low.replace(" ", "")
    scored = []
    for i, (keys, note) in enumerate(NOTES):
        score = sum(len(key) for key in keys if key in low or key.replace(" ", "") in flat)
        if score:
            scored.append((-score, i, note))
    return [note for _, _, note in sorted(scored)[:k]]
```

`NOTES`는 (낱말 묶음, 설명문) 20쌍입니다. 질문에 들어 있는 낱말의 길이를 점수로 더해 점수가 높은 메모 2개를 고릅니다. 주제는 선형·비선형 회귀, 인공신경망, 활성화 함수, 경사하강법, 오류역전파, 딥러닝, CNN, 과적합, 정규화, 평가지표, LLM, 오픈소스, 양자화, 파인튜닝·RAG 등입니다.

## 6. `tools.py`: 분석 도구

모든 도구는 같은 모양의 결과를 돌려줍니다.

```python
{"text": 설명글(마크다운), "items": [("fig", 그래프) | ("table", 표)], "summary": 기억에 적을 한 줄}
```

분석 상태는 사전 하나에 담습니다: `{"df", "name", "x", "y", "results", "inputs"}`.

### 6.1 데이터 읽기와 정리

| 함수 | 하는 일 |
| --- | --- |
| `load_dataset` | 공개 데이터 9종 (콘크리트, 당뇨병, 자동차 연비, 팁, 펭귄, 집값, 붓꽃, 와인, 합성) |
| `read_table` | CSV·TSV·TXT·Excel·JSON. 인코딩(UTF-8, CP949)과 구분자를 자동으로 맞춤 |
| `load_url` | 구글 스프레드시트·GitHub 링크를 내려받기 주소로 바꿔 읽음 |
| `parse_pasted` | 붙여넣은 표와 `x: 1, 2, 3 / y: …` 형태를 표로 바꿈 |
| `make_data` | `y = 2x + 3` 같은 식에 잡음을 더해 연습용 데이터를 만듦 |
| `_clean` | "1,234"·"12%"를 숫자로, 글자 범주를 0/1 열로 바꾸고 2만 행 초과는 표본 추출 |

### 6.2 변수 인식 (`_find`, `parse_vars`)

`_find`는 세 단계로 문장에서 열을 찾습니다.

1. 열 이름이 그대로 나온 경우 (긴 이름부터, 영어는 단어 경계 확인)
2. 일부만 말했거나("슬래그" → 고로슬래그), 다른 언어로 말했거나("몸무게" → weight, `ALIASES` 사전), 오타가 있는 경우(`difflib`)
3. 순서로 말한 경우 ("첫 번째 열", "마지막 열")

`parse_vars`는 찾은 열을 X와 Y로 나눕니다. "Y는 ○○, X는 △△", "마력으로 연비 예측"(마지막에 말한 것이 Y), "물 빼고"(X에서 제외)를 처리합니다.

### 6.3 모델 학습

```python
def _split(state):                                   # 80% 학습 / 20% 테스트, 난수 고정(42)
    ...
    return train_test_split(X, y, test_size=0.2, random_state=SEED)

def run_linear(state):                               # 선형회귀: 최소제곱법
    m = LinearRegression().fit(Xtr, ytr)

def run_ann(state, deep=False, layers=None, act="relu", epochs=None, lr=0.001):
    layers = tuple(layers or ((128, 64, 32, 16) if deep else (64, 32)))
    mlp = MLPRegressor(hidden_layer_sizes=layers, activation=act, max_iter=epochs, learning_rate_init=lr,
                       random_state=SEED, early_stopping=use_val, validation_fraction=0.2)
    model = TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), mlp), transformer=StandardScaler())
```

- `run_nonlinear`: 변수가 1개면 `_curve_candidates`가 다항·거듭제곱·지수·로그 곡선을 `np.polyfit`과 `scipy.optimize.curve_fit`으로 적합해 테스트 R²로 비교합니다. 여러 개면 `PolynomialFeatures`로 2~3차 다항 회귀를 합니다.
- `run_ann`: 입력과 출력을 `StandardScaler`로 표준화하고 `MLPRegressor`(Adam)로 학습합니다. 딥러닝 모드는 학습 데이터의 20%를 검증용으로 떼어 조기 종료합니다. 결과에는 같은 구조의 Keras `Dense` 코드를 함께 보여 줍니다.
- `_store`: 학습·테스트 R², RMSE, MAE를 계산해 `state["results"]`에 저장합니다.

### 6.4 입력값에 따른 예측

| 함수 | 하는 일 |
| --- | --- |
| `parse_inputs` | "시멘트 300, 물 180"에서 {열: [값]}을 읽음. 값 여러 개도 읽음 |
| `run_predict` | 말한 값으로 모든 학습 모델의 예측을 계산. 말하지 않은 변수는 평균(또는 직전 입력) |
| `parse_sweep`, `run_sweep` | 한 변수를 범위로 바꿀 때 예측값의 변화 곡선 |
| `add_variable` | "이름 = 식"으로 새 변수 생성. 숫자·연산자·열만 허용한 뒤 계산 |
| `_add_inputs` | 입력 변수에 없던 변수를 말하면 추가하고 같은 모델을 다시 학습 |

### 6.5 기억 (메모 방식)

```python
MEMORY_FILE = Path(tempfile.gettempdir()) / "llm_chat_memory.json"

def remember(summary):
    log = recall_all()
    log.append({"time": datetime.now().strftime("%m-%d %H:%M"), "summary": summary})
    MEMORY_FILE.write_text(json.dumps(log[-50:], ensure_ascii=False), encoding="utf-8")
```

분석을 실행할 때마다 한 줄 요약을 파일에 적고(최근 50개), "지난번에 뭐 했지?"에 답하거나 언어모델에게 함께 보여 줍니다.

## 7. 개발 중 고친 문제

| 문제 | 원인 | 해결 |
| --- | --- | --- |
| 채팅 중 오류가 나면 사이트에 다시 접속되지 않음 | 언어모델이 웹 앱과 같은 프로세스에서 돌아, 모델이 멈추면 앱도 멈춤 | 언어모델을 자식 프로세스로 분리 (`llm_worker.py`) |
| `module 'tools' has no attribute 'make_data'` | 코드를 새로 올리면 서버가 `app.py`만 다시 읽고 `tools`는 예전 것을 계속 씀 | 파일 수정 시각이 바뀌면 `importlib.reload` |
| "3차 다항 회귀 해줘"가 다른 곡선을 고름 | 가장 잘 맞는 곡선을 무조건 선택 | 차수를 말하면 그 다항식을 사용 |
| 영어 열 이름이 다른 단어 속에서 잘못 인식됨 | 부분 문자열 비교 | 영어 이름은 단어 경계 확인 |

## 8. 로컬에서 실행

```bash
pip install -r llm_chat/requirements.txt
streamlit run llm_chat/app.py
```
