"""강의 지식 검색(RAG)과 내 코드 검색.

RAG(검색 증강 생성): 문서를 작은 조각으로 나눠 두고, 질문과 비슷한 조각을 찾아 LLM에게 함께 건네서
LLM이 근거를 보고 답하게 하는 방법이다. 모델의 가중치를 바꾸는 파인튜닝과 달리, 조각만 바꾸면 지식이 바뀐다.

- 강의 조각: 데이터처리개론 강의(2~8장, LLM 수업)의 핵심을 장별로 정리한 조각 (내가 정리한 요약)
- 코드 조각: 내가 만든 프로그램 파일을 함수 단위로 자른 조각
- 검색: 글자 2~4개 묶음(n-gram)의 TF-IDF 벡터로 바꾼 뒤 코사인 유사도가 큰 조각을 고른다.
  한국어는 띄어쓰기·조사가 다양해서 낱말 대신 글자 묶음을 쓰면 더 잘 찾는다.
"""
import ast
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "llm_chat"))
import knowledge as K  # noqa: E402  과제 3의 주제별 메모
import tools as T  # noqa: E402

# (출처, 내용) — 강의 장별 정리
LECTURE = [
    ("2장 통계 복습", "평균은 값들의 합을 개수로 나눈 중심값, 중앙값은 크기 순으로 세웠을 때 가운데 값이다. 이상치가 있으면 평균보다 중앙값이 안정적이다."),
    ("2장 통계 복습", "분산은 각 값이 평균에서 떨어진 거리의 제곱을 평균한 값이고, 표준편차는 분산의 제곱근이다. 표준편차는 단위가 원래 값과 같아 퍼진 정도를 읽기 쉽다."),
    ("2장 통계 복습", "정규분포는 평균을 중심으로 좌우 대칭인 종 모양 분포다. 평균±1σ 안에 약 68%, ±2σ 안에 약 95%, ±3σ 안에 약 99.7%가 들어간다. 표준화 z=(x−μ)/σ로 평균 0, 표준편차 1인 표준정규분포로 바꾼다."),
    ("2장 통계 복습", "공분산은 두 변수가 함께 변하는 방향을, 상관계수는 공분산을 두 표준편차로 나눠 −1~1로 맞춘 값이다. 상관이 크다고 원인이라는 뜻은 아니다."),
    ("3장 선형회귀", "선형회귀는 y = ax + b(다중이면 y = a1x1 + … + anxn + b)로 입력과 출력의 관계를 직선(평면)으로 표현한다. 잔차(실제값−예측값)의 제곱합이 최소가 되는 a, b를 구하는 것이 최소제곱법이다."),
    ("3장 선형회귀", "선형회귀의 평가: 결정계수 R²는 1에 가까울수록 데이터 변동을 잘 설명한다. RMSE(평균제곱근오차)=√(Σ(실제−예측)²/n)로 Y와 같은 단위의 오차 크기다. MAE는 오차 절댓값의 평균이다."),
    ("3장 선형회귀", "선형회귀의 한계는 비선형성(휘어진 관계)과 다차원성(변수가 많고 서로 얽힌 관계)을 잘 표현하지 못한다는 점이다. 다항 회귀·곡선 회귀나 인공신경망으로 보완한다."),
    ("3장 선형회귀", "비선형 회귀는 다항식 y=a+bx+cx², 거듭제곱 y=a·x^b, 지수 y=a·e^(bx), 로그 y=a+b·ln x 같은 곡선을 적합한다. 거듭제곱·지수는 로그를 씌우면 직선이 되어 초기값을 구하기 쉽다."),
    ("4장 인공신경망 기초", "인공신경망은 생물의 뉴런을 흉내 낸 모델이다. 뉴런 하나는 입력 x에 가중치 ω를 곱해 더하고 편향 b를 더한 뒤 활성화 함수 φ를 통과시킨다: y = φ(Σωx + b)."),
    ("4장 인공신경망 기초", "신경망은 입력층(입력 변수 수만큼 노드), 은닉층(1개 이상, 노드 수는 설계자가 정함), 출력층(회귀면 노드 1개)으로 이루어진다. Keras로는 Dense(노드 수, activation=…) 층을 차례로 쌓는다."),
    ("4장 인공신경망 기초", "퍼셉트론 하나로는 XOR 같은 비선형 문제를 풀 수 없다. 은닉층을 넣은 다층 퍼셉트론(MLP)과 비선형 활성화 함수가 있어야 복잡한 관계를 배울 수 있다."),
    ("4장 인공신경망 기초", "가중치와 편향의 개수(매개변수 수)는 층마다 (입력 노드 수 × 출력 노드 수 + 출력 노드 수)를 더해 구한다. 예: 입력 8 → 64 → 32 → 1이면 8×64+64 + 64×32+32 + 32×1+1 = 2,689개."),
    ("5장 활성화 함수와 기본 연산", "시그모이드 σ(z)=1/(1+e^−z)는 0~1 값을 내며 확률처럼 해석한다. 입력이 크거나 작으면 기울기가 0에 가까워 깊은 망에서 기울기 소실이 생긴다."),
    ("5장 활성화 함수와 기본 연산", "tanh는 −1~1 값을 내고 0을 중심으로 대칭이다. ReLU는 max(0, z)로 음수는 0, 양수는 그대로 내보내며 계산이 빠르고 기울기 소실이 적어 은닉층에 가장 많이 쓴다. 회귀의 출력층은 활성화 없이(선형) 둔다."),
    ("5장 활성화 함수와 기본 연산", "신경망의 기본 연산은 행렬 곱이다. 입력 벡터 X와 가중치 행렬 W를 곱하고 편향 b를 더한 뒤(XW+b) 활성화 함수를 적용하는 것을 층마다 반복하는 것이 순전파다."),
    ("6장 오류역전파", "학습의 목표는 오차 E(회귀에서는 보통 평균제곱오차, E=(t−o)²)를 줄이는 것이다. 경사하강법은 오차가 줄어드는 방향(기울기의 반대)으로 가중치를 조금씩 고친다: ω ← ω − η·∂E/∂ω, η는 학습률."),
    ("6장 오류역전파", "오류역전파는 출력층의 오차를 입력층 쪽으로 거꾸로 전달하며 각 가중치의 기울기 ∂E/∂ω를 구하는 방법이다. 합성함수 미분법(연쇄법칙): ∂E/∂ω = ∂E/∂o · ∂o/∂z · ∂z/∂ω."),
    ("6장 오류역전파", "학습 한 번의 순서: 순전파로 출력 계산 → 오차 계산 → 역전파로 기울기 계산 → 경사하강법으로 가중치 갱신. 전체 학습 데이터를 한 번 다 쓰는 것을 1 에포크(epoch)라고 한다."),
    ("6장 오류역전파", "학습률이 너무 크면 오차가 줄지 않고 튀거나 발산하고, 너무 작으면 학습이 느리다. Adam은 기울기의 평균과 크기를 이용해 가중치마다 학습률을 자동으로 조절하는 경사하강법이다."),
    ("7장 과적합 방지를 위한 데이터 분할", "과적합은 학습 데이터는 잘 맞히지만 처음 보는 데이터는 못 맞히는 현상이다. 학습 R²가 테스트 R²보다 훨씬 높으면 과적합을 의심한다. 반대로 둘 다 낮으면 과소적합이다."),
    ("7장 과적합 방지를 위한 데이터 분할", "데이터는 학습·검증·테스트로 나눈다(예: 6:2:2, 또는 학습 80%·테스트 20%에 학습의 일부를 검증용으로). 학습 데이터로 가중치를 배우고, 검증 데이터로 에포크·노드 수 같은 설정과 학습을 멈출 시점을 고르고, 테스트 데이터로 최종 성능을 한 번만 잰다."),
    ("7장 과적합 방지를 위한 데이터 분할", "과적합을 막는 방법: 데이터 늘리기, 모델 크기(층·노드) 줄이기, 조기 종료(검증 오차가 더 이상 줄지 않으면 멈춤), L2 규제(가중치가 커지지 않게 벌점), 드롭아웃, k-겹 교차검증으로 평가 안정화."),
    ("7장 과적합 방지를 위한 데이터 분할", "정규화(스케일링)는 변수마다 다른 크기를 맞춘다. 표준화 z=(x−평균)/표준편차, MinMax=(x−최소)/(최대−최소). 크기가 큰 변수가 학습을 지배하는 것을 막고 경사하강법을 안정시킨다. 기준값은 학습 데이터에서만 구한다."),
    ("8장 딥러닝과 CNN 기초", "딥러닝은 은닉층이 여러 개인 깊은 신경망이다. 인공지능 ⊃ 머신러닝 ⊃ 딥러닝. 머신러닝은 사람이 특징을 골라 주지만, 딥러닝은 층을 거치며 특징을 스스로 뽑는다. 데이터와 계산(GPU)이 많이 필요하다."),
    ("8장 딥러닝과 CNN 기초", "CNN(합성곱 신경망)은 이미지를 숫자 행렬로 보고 작은 필터(커널)를 옮겨 가며 곱해 더하는 합성곱으로 윤곽선 같은 특징을 뽑는다. 풀링으로 크기를 줄이고, 마지막에 Dense 층으로 분류한다. 흑백 이미지는 2차원, 컬러는 RGB 3채널 행렬이다."),
    ("8장 딥러닝과 CNN 기초", "딥러닝은 숫자를 예측하는 회귀보다 종류를 고르는 분류에 많이 쓴다. 분류의 출력층은 클래스 수만큼 노드를 두고 소프트맥스로 확률을 낸다."),
    ("LLM 수업", "LLM(거대 언어 모델)은 문장을 토큰으로 잘라 숫자 벡터로 바꾸고, 앞의 토큰들을 보고 다음 토큰을 수만 개 후보 중에서 고르는 분류를 반복해 글을 만든다. 그래서 답이 한 글자씩 나온다."),
    ("LLM 수업", "순서가 있는 데이터를 다루는 신경망으로 RNN(이전 상태를 다음 계산에 넘김)과 LSTM(긴 문장에서도 기억이 사라지지 않게 게이트를 둔 RNN)이 있었고, 지금의 GPT는 문장 전체를 한꺼번에 보고 어느 낱말에 주목할지 계산하는 어텐션 기반 트랜스포머(Transformer)로 만든다."),
    ("LLM 수업", "오픈소스 LLM(Llama, Qwen, DeepSeek, EXAONE, Gemma 등)은 가중치를 공개해 누구나 내려받아 실행·수정할 수 있다. 큰 모델은 비싼 GPU 서버가 필요해서, 회사는 서버 한 대에 설치해 두고 여러 사람이 접속해 쓴다. 작은 모델은 양자화(4비트 등)로 줄여 CPU에서도 돌린다."),
    ("LLM 수업", "LLM이 기억하는 방법은 두 가지다. 가중치를 다시 학습시키는 파인튜닝(작은 모델에 원하는 분야만 학습)과, 적어 둔 메모·문서를 찾아 함께 읽히는 방법(RAG)이다. RAG는 문서를 조각내 근거를 찾게 하므로 헛소리(환각)를 줄인다."),
    ("LLM 수업", "LLM이 프로그램을 움직이려면 API로 연결한다. LLM이 실행할 기능과 값을 고르면(명령) 프로그램이 실행하고, 실행 결과를 다시 LLM에게 돌려줘야(피드백) 다음 행동을 정할 수 있다. CAD 같은 설계 프로그램, 폴더 정리, 메일 정리도 같은 방식으로 말로 반복 작업을 시킬 수 있다."),
    ("LLM 수업", "Streamlit은 파이썬으로 웹 화면을 쉽게 만드는 도구라 학생 프로젝트에 많이 쓰지만, 접속자가 많아지면(예: 만 명) 서버가 버티지 못해 실무 서비스에는 잘 쓰지 않는다. 실무에서는 별도 웹 서버와 프레임워크, 유료 호스팅을 쓴다."),
]


@lru_cache(maxsize=1)
def _lecture_index():
    docs = [(src, txt) for src, txt in LECTURE] + [("보충 메모", note) for _, note in K.NOTES]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
    mat = vec.fit_transform([f"{s} {t}" for s, t in docs])
    return docs, vec, mat


def lecture_hits(query, k=4):
    docs, vec, mat = _lecture_index()
    sims = cosine_similarity(vec.transform([str(query)]), mat).ravel()
    order = [i for i in np.argsort(-sims)[:k] if sims[i] > 0.05]
    return [(docs[i][0], docs[i][1], float(sims[i])) for i in order]


def search_lecture(query):
    hits = lecture_hits(query)
    if not hits:
        return T.reply(f"'{query}'와 관련된 강의 내용을 찾지 못했어요.")
    lines = "\n".join(f"- **[{src}]** {txt}" for src, txt, _ in hits)
    return T.reply(f"📚 강의 자료에서 '{query}'와 관련된 부분 {len(hits)}개를 찾았어요.\n\n{lines}")


# --- 내 코드 검색 ---
CODE_FILES = ["app.py", "llm_chat/tools.py", "llm_chat/app.py", "llm_chat/llm_worker.py",
              "agent/programs.py", "agent/llm.py", "agent/core.py", "agent/knowledge_base.py", "agent/app.py"]
PROGRAM_OF = {"app.py": "과제 1·2 대시보드", "llm_chat": "과제 3 챗봇", "agent": "마지막 과제 시스템"}


@lru_cache(maxsize=1)
def _code_index():
    chunks = []
    for rel in CODE_FILES:
        path = ROOT / rel
        if not path.exists():
            continue
        src = path.read_text(encoding="utf-8")
        lines = src.splitlines()
        try:
            nodes = [n for n in ast.parse(src).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        except SyntaxError:
            nodes = []
        for n in nodes:
            body = "\n".join(lines[n.lineno - 1:n.end_lineno])
            chunks.append((rel, n.name, n.lineno, body))
        if rel == "app.py":       # 대시보드는 화면 코드가 대부분이라 40줄씩 잘라 함께 넣음
            for s in range(0, len(lines), 40):
                chunks.append((rel, f"{s + 1}~{min(s + 40, len(lines))}줄", s + 1, "\n".join(lines[s:s + 40])))
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, max_features=60000)
    mat = vec.fit_transform([f"{c[1]} {c[3]}" for c in chunks])
    return chunks, vec, mat


def search_code(query, k=2):
    chunks, vec, mat = _code_index()
    sims = cosine_similarity(vec.transform([str(query)]), mat).ravel()
    order = [i for i in np.argsort(-sims)[:k] if sims[i] > 0.03]
    if not order:
        return T.reply(f"'{query}'와 관련된 코드를 찾지 못했어요.")
    parts = []
    for i in order:
        rel, name, line, body = chunks[i]
        prog = PROGRAM_OF.get(rel.split("/")[0], "")
        snippet = "\n".join(body.splitlines()[:30])
        parts.append(f"**{prog} · `{rel}` {line}번째 줄 `{name}`**\n```python\n{snippet}\n```")
    return T.reply(f"💻 내 코드에서 '{query}'와 관련된 부분이에요.\n\n" + "\n\n".join(parts))
