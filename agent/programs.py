"""프로그램 연결층 (API).

LLM이 지금까지 만든 프로그램을 직접 움직일 수 있도록, 각 프로그램의 기능을 '도구(함수)'로 묶어 둔 파일이다.

    과제 1·2  회귀·인공신경망 대시보드 (저장소 맨 위 app.py)
              → 모델 정의(build_estimator, CurveFitRegressor, safe_metrics)를 파일에서 그대로 읽어 와 실행
    과제 3    채팅 분석 챗봇 (llm_chat/tools.py)
              → 데이터 불러오기·변수 인식·선형/비선형 회귀·인공신경망·딥러닝·예측·보고서 함수를 그대로 불러 씀

LLM은 TOOLS(도구 설명서, JSON 스키마)를 보고 무엇을 실행할지 고르고,
execute()가 실제 프로그램을 실행한 뒤 결과를 글(LLM에게 돌려줄 값)과 그래프·표(화면에 보여 줄 것)로 돌려준다.
"""
import ast
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import KFold, cross_val_predict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "llm_chat"))
import tools as T  # noqa: E402  과제 3 분석 도구

import knowledge_base as KB  # noqa: E402  강의 지식·코드 검색 (RAG)

SEED = 42


# =============================================================
# 1. 과제 1·2 대시보드의 모델 코드를 그대로 읽어 오기
# =============================================================
DASHBOARD_FILE = ROOT / "app.py"
KEEP_DEFS = {"CurveFitRegressor", "safe_metrics", "build_estimator"}
KEEP_VARS = {"ALL_MODELS", "CURVE_KIND"}

# 대시보드 사이드바의 기본값과 같은 하이퍼파라미터
DEFAULT_CFG = {
    "ridge_alpha": 1.0, "lasso_alpha": 0.1, "poly_degree": 2, "tree_depth": 5, "rf_trees": 200, "rf_depth": 0,
    "gb_trees": 200, "gb_lr": 0.05, "gb_depth": 3, "svr_kernel": "rbf", "svr_c": 10.0, "svr_eps": 0.1,
    "knn_k": 5, "knn_weights": "distance", "ann_layers": [64, 32], "ann_act": "relu", "ann_lr": 0.001,
    "ann_alpha": 0.0001, "ann_iter": 1000,
}


@lru_cache(maxsize=1)
def dashboard():
    """대시보드 파일(app.py)을 문법 트리로 읽어 모델 정의 부분만 골라 실행한다.

    화면을 그리는 Streamlit 코드는 건너뛰고, 모델을 만드는 코드만 가져오므로
    대시보드와 똑같은 모델(13종)과 똑같은 평가 함수를 쓴다. 대시보드 파일은 고치지 않는다.
    """
    tree = ast.parse(DASHBOARD_FILE.read_text(encoding="utf-8"))
    body = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] + [getattr(node, "module", "") or ""]
            if any(m.startswith(("streamlit", "plotly")) for m in mods):
                continue
            body.append(node)
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in KEEP_DEFS:
            body.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in KEEP_VARS for t in node.targets):
            body.append(node)
    ns = {"__name__": "dashboard_core"}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(DASHBOARD_FILE), "exec"), ns)
    return ns


MODEL_ALIASES = {
    "선형": "선형 회귀", "linear": "선형 회귀", "릿지": "릿지 회귀 (Ridge)", "ridge": "릿지 회귀 (Ridge)",
    "라쏘": "라쏘 회귀 (Lasso)", "lasso": "라쏘 회귀 (Lasso)", "다항": "다항 회귀", "poly": "다항 회귀",
    "결정": "결정 트리", "트리": "결정 트리", "tree": "결정 트리", "랜덤": "랜덤 포레스트", "포레스트": "랜덤 포레스트",
    "forest": "랜덤 포레스트", "부스팅": "그래디언트 부스팅", "boost": "그래디언트 부스팅", "svr": "서포트 벡터 회귀 (SVR)",
    "서포트": "서포트 벡터 회귀 (SVR)", "knn": "K-최근접 이웃 (KNN)", "최근접": "K-최근접 이웃 (KNN)",
    "신경망": "인공신경망 (ANN)", "ann": "인공신경망 (ANN)", "mlp": "인공신경망 (ANN)", "거듭제곱": "거듭제곱 회귀 (y=a·x^b)",
    "power": "거듭제곱 회귀 (y=a·x^b)", "지수": "지수 회귀 (y=a·e^(bx))", "exp": "지수 회귀 (y=a·e^(bx))",
    "로그": "로그 회귀 (y=a+b·ln x)", "log": "로그 회귀 (y=a+b·ln x)",
}


def _resolve_models(names):
    allm = dashboard()["ALL_MODELS"]
    if not names:
        return list(allm)
    out = []
    for n in names:
        key = T.norm(n)
        hit = [m for m in allm if T.norm(m) == key or key in T.norm(m)]
        hit = hit or [v for k, v in MODEL_ALIASES.items() if k in key]
        for m in hit[:1]:
            if m not in out:
                out.append(m)
    return out or list(allm)


def run_dashboard_models(state, models=None, cv=True):
    """과제 1·2 대시보드의 모델들을 같은 설정으로 한꺼번에 학습·평가한다."""
    D = dashboard()
    names = _resolve_models(models)
    Xtr, Xte, ytr, yte = T._split(state)
    skipped = {}
    for name in [n for n in names if n in D["CURVE_KIND"]]:
        if Xtr.shape[1] != 1:
            skipped[name] = "X가 1개일 때만"
        elif name.startswith("거듭제곱") and not (np.all(Xtr > 0) and np.all(ytr > 0)):
            skipped[name] = "X, Y가 모두 양수여야 함"
        elif name.startswith("지수") and not np.all(ytr > 0):
            skipped[name] = "Y가 모두 양수여야 함"
        elif name.startswith("로그") and not np.all(Xtr > 0):
            skipped[name] = "X가 모두 양수여야 함"
    names = [n for n in names if n not in skipped]
    n_splits = 5 if cv and 10 <= len(Xtr) <= 5000 else 0
    cfg = dict(DEFAULT_CFG)
    cfg["n_fit_min"] = len(Xtr) - int(np.ceil(len(Xtr) / n_splits)) if n_splits else len(Xtr)
    rows = []
    for name in names:
        try:
            est = D["build_estimator"](name, cfg, SEED)
            t0 = time.perf_counter()
            est.fit(Xtr, ytr)
            sec = time.perf_counter() - t0
            te, tr = D["safe_metrics"](yte, est.predict(Xte)), D["safe_metrics"](ytr, est.predict(Xtr))
            cvm = D["safe_metrics"]([], [])
            if n_splits:
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
                cvm = D["safe_metrics"](ytr, cross_val_predict(D["build_estimator"](name, cfg, SEED), Xtr, ytr, cv=kf))
            if np.isfinite(te["R2"]):
                T._store(state, f"dash:{name}", f"[대시보드] {name}", est.predict, Xtr, Xte, ytr, yte)
            rows.append((name, te["R2"], te["RMSE"], te["MAE"], cvm["R2"], tr["R2"], tr["R2"] - te["R2"], sec))
        except Exception as e:  # 한 모델이 실패해도 나머지는 계속
            skipped[name] = f"학습 실패: {e}"
    if not rows:
        return T.reply("⚠️ 학습에 성공한 모델이 없어요. " + "; ".join(f"{k}: {v}" for k, v in skipped.items()))
    table = pd.DataFrame(rows, columns=["모델", "테스트 R²", "RMSE", "MAE", "5-겹 CV R²", "학습 R²", "과적합 격차", "학습 시간(초)"])
    table = table.sort_values("테스트 R²", ascending=False, na_position="last").round(4).reset_index(drop=True)
    if not n_splits:
        table = table.drop(columns=["5-겹 CV R²"])
    fig = go.Figure(go.Bar(x=table["모델"], y=table["테스트 R²"], marker_color=T.COLORS[1],
                           text=[f"{v:.3f}" for v in table["테스트 R²"]], textposition="outside"))
    fig.update_layout(title="과제 1·2 대시보드 모델 비교 (테스트 R²)", height=440, margin=dict(t=50, b=20),
                      yaxis=dict(range=[min(0, float(table["테스트 R²"].min())) - 0.05, 1.1]))
    best = table.iloc[0]
    note = ("\n\n- 제외: " + "; ".join(f"{k}({v})" for k, v in skipped.items())) if skipped else ""
    text = (f"### 🗂️ 과제 1·2 대시보드 모델 {len(table)}종 학습 결과\n{T._vars_line(state)} · 학습 {len(ytr)}행 / 테스트 {len(yte)}행"
            f"{' · 5-겹 교차검증' if n_splits else ''}\n\n가장 잘 맞는 모델은 **{best['모델']}** (테스트 R² {best['테스트 R²']:.3f})이에요.{note}")
    summary = "대시보드 모델 비교: " + ", ".join(f"{r[0]} {r[1]:.3f}" for r in rows[:6])
    return T.reply(text, [("fig", fig), ("table", table)], summary)


# =============================================================
# 2. LLM에게 보여 줄 도구 설명서 (OpenAI·OpenRouter 함수 호출 형식)
# =============================================================
def _fn(name, desc, props=None, required=None):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props or {}, "required": required or []}}}


S, N, I, B = {"type": "string"}, {"type": "number"}, {"type": "integer"}, {"type": "boolean"}
STRS = {"type": "array", "items": {"type": "string"}}

TOOLS = [
    _fn("list_datasets", "바로 쓸 수 있는 공개 데이터 목록을 보여 준다."),
    _fn("load_dataset", "공개 데이터를 불러온다. 이름: concrete(콘크리트 압축강도), diabetes(당뇨병), mpg(자동차 연비), "
        "tips(식당 팁), penguins(펭귄), california(캘리포니아 집값), iris(붓꽃), wine(와인), synthetic(비선형 합성).",
        {"name": {"type": "string", "enum": list(T.DATASETS)}}, ["name"]),
    _fn("load_data_url", "CSV·Excel 파일 주소, 구글 스프레드시트 공유 링크, GitHub 파일 링크에서 데이터를 불러온다.",
        {"url": S}, ["url"]),
    _fn("make_data", "식으로 연습용 데이터를 만든다. 예: formula='2*x + 3', 'sin(x) + 0.5*x'. x 범위는 0.5~10.",
        {"formula": S, "n": I, "noise": B}, ["formula"]),
    _fn("data_summary", "현재 데이터의 처음 5행, 열 목록, 변수별 평균·표준편차·최솟값·최댓값을 보여 준다."),
    _fn("set_variables", "예측할 값 Y와 입력 변수 X를 정한다. 열 이름은 일부만 써도 되고 한국어·영어 모두 된다. "
        "x를 비우면 Y를 뺀 나머지 전부를 쓴다.",
        {"y": S, "x": STRS, "exclude": STRS}),
    _fn("run_linear_regression", "선형 회귀(최소제곱법)를 학습하고 회귀식·R²·RMSE·MAE·그래프를 보여 준다."),
    _fn("run_nonlinear_regression", "비선형 회귀. 변수 1개면 다항·거듭제곱·지수·로그 곡선을 비교, 여러 개면 2~3차 다항 회귀. "
        "degree를 주면 그 차수의 다항식을 쓴다.", {"degree": I}),
    _fn("run_neural_network", "인공신경망(MLP)을 학습한다. layers=은닉층별 노드 수(예: [64, 32]), activation=relu|tanh|logistic, "
        "epochs=최대 학습 횟수, learning_rate=학습률. 학습 곡선과 Dense 구조 코드를 보여 준다.",
        {"layers": {"type": "array", "items": {"type": "integer"}}, "activation": {"type": "string", "enum": ["relu", "tanh", "logistic"]},
         "epochs": I, "learning_rate": N}),
    _fn("run_deep_learning", "은닉층 4개(128-64-32-16)의 딥러닝 모델을 검증 데이터 조기 종료로 학습한다."),
    _fn("compare_models", "선형 회귀·비선형 회귀·인공신경망·딥러닝 네 가지를 모두 학습해 R²를 비교한다."),
    _fn("run_dashboard_models", "과제 1·2 대시보드의 모델(선형, 릿지, 라쏘, 다항, 결정 트리, 랜덤 포레스트, 그래디언트 부스팅, SVR, "
        "KNN, 인공신경망, 거듭제곱·지수·로그 회귀)을 대시보드와 같은 설정으로 학습하고 5-겹 교차검증까지 비교한다. "
        "models를 비우면 전부.", {"models": STRS}),
    _fn("predict", "학습한 모델로 새 입력값의 결과를 예측한다. values는 {변수 이름: 값} 또는 {변수 이름: [값1, 값2]}. "
        "말하지 않은 변수는 평균을 쓰고, keep_previous=true면 직전 입력값을 이어서 쓴다.",
        {"values": {"type": "object", "additionalProperties": {}}, "keep_previous": B}, ["values"]),
    _fn("sweep_variable", "한 변수를 start부터 end까지 바꿀 때 예측값이 어떻게 변하는지 곡선으로 보여 준다. "
        "start·end를 비우면 데이터 범위 전체.", {"variable": S, "start": N, "end": N}, ["variable"]),
    _fn("correlation", "Y와 입력 변수들의 상관계수 히트맵을 보여 준다."),
    _fn("plot", "그래프를 그린다. kind=scatter면 columns의 마지막을 세로축으로 하는 산점도, histogram이면 첫 열의 분포.",
        {"kind": {"type": "string", "enum": ["scatter", "histogram"]}, "columns": STRS}, ["kind", "columns"]),
    _fn("column_stats", "열의 통계(평균, 중앙값, 최댓값, 최솟값, 표준편차, 분산, 합계, 개수)를 계산한다.",
        {"columns": STRS, "stats": STRS}, ["columns"]),
    _fn("add_variable", "식으로 새 변수를 만들어 입력 변수에 넣는다. 예: name='물시멘트비', expression='물 / 시멘트'.",
        {"name": S, "expression": S}, ["name", "expression"]),
    _fn("make_report", "지금까지 학습한 결과로 분석 보고서를 만든다."),
    _fn("current_status", "현재 데이터, X·Y, 학습한 모델과 R²를 알려 준다."),
    _fn("search_lecture", "데이터처리개론 강의 내용(통계, 선형회귀, 인공신경망, 활성화 함수, 오류역전파, 과적합, 딥러닝·CNN, LLM)에서 "
        "질문과 관련된 부분을 찾는다. 개념 설명 전에 먼저 호출해 근거로 쓴다.", {"query": S}, ["query"]),
    _fn("search_my_code", "내가 만든 프로그램 코드에서 관련 부분을 찾아 보여 준다. 예: 'Dense', 'train_test_split', '학습률'.",
        {"query": S}, ["query"]),
    _fn("remember", "사용자가 기억해 달라고 한 내용이나 중요한 결론을 메모에 적는다.", {"note": S}, ["note"]),
    _fn("recall_memory", "메모에 적어 둔 지난 작업과 기억을 읽는다."),
]
TOOL_NAMES = [t["function"]["name"] for t in TOOLS]
DATA_TOOLS = {"data_summary", "set_variables", "run_linear_regression", "run_nonlinear_regression", "run_neural_network",
              "run_deep_learning", "compare_models", "run_dashboard_models", "predict", "sweep_variable", "correlation",
              "plot", "column_stats", "add_variable"}


# =============================================================
# 3. 실행기: LLM이 고른 도구를 실제 프로그램으로 실행
# =============================================================
# 콘크리트 데이터의 한글 열 이름을 영어로 불러도 찾도록 (LLM이 영어로 부르는 경우 대비)
EN_KO = {"cement": "시멘트", "slag": "고로슬래그", "fly": "플라이애시", "ash": "플라이애시", "water": "물",
         "superplasticizer": "고성능감수제", "plasticizer": "고성능감수제", "coarse": "굵은골재", "fine": "잔골재",
         "age": "재령", "strength": "압축강도", "compressive": "압축강도"}


def _cols(state, names, pool=None):
    """사용자가 말한 이름 목록을 실제 열 이름으로 바꾼다. (찾은 열, 못 찾은 이름)"""
    pool = pool if pool is not None else T.numeric_cols(state)
    found, missing = [], []
    for n in names or []:
        n = str(n).strip()
        alt = " ".join(v for k, v in EN_KO.items() if k in n.lower()) or n
        hit = ([c for c in pool if str(c) == n] or [c for _, c in T.find_cols(n, pool)]
               or [c for _, c in T.find_cols(alt, pool)])
        if hit:
            found += [c for c in hit[:1] if c not in found]
        else:
            missing.append(str(n))
    return found, missing


def _ensure_data(state, notes):
    if state["df"] is None:
        T.load_dataset(state, "concrete")
        notes.append("데이터가 없어서 기본 데이터(콘크리트 압축강도)를 불러왔어요.")


def _table_md(df, rows=10):
    df = df.head(rows).copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].round(4)
    head = "| " + " | ".join(map(str, df.columns)) + " |\n| " + " | ".join("---" for _ in df.columns) + " |\n"
    return head + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False))


def for_llm(rep, limit=1800):
    """화면용 결과를 LLM에게 돌려줄 짧은 글로 바꾼다. (그래프는 빼고 글과 표만)"""
    text = rep.get("text", "")
    parts = [text[:limit] + (" …(줄임)" if len(text) > limit else "")]
    tables = [obj for kind, obj in rep.get("items", []) if kind == "table"][:2]
    parts += [_table_md(t) for t in tables]
    figs = sum(1 for kind, _ in rep.get("items", []) if kind == "fig")
    if figs:
        parts.append(f"(화면에 그래프 {figs}개를 보여 줌)")
    return "\n\n".join(p for p in parts if p)


def _state_line(state):
    if state["df"] is None:
        return "데이터 없음"
    rs = sorted(state["results"].values(), key=lambda r: -r["test_r2"])
    done = ", ".join(f"{r['label']} R²={r['test_r2']:.3f}" for r in rs[:5]) or "없음"
    more = f" 외 {len(rs) - 5}개" if len(rs) > 5 else ""
    cols = ", ".join(T.short(c) for c in state["df"].columns[:25])
    return (f"데이터={state['name']}({len(state['df'])}행), 열=[{cols}], Y={T.short(state['y'])}, "
            f"X=[{', '.join(T.short(c) for c in state['x'][:12])}], 학습한 모델(R² 높은 순): {done}{more}")


def _fit_axes(rep):
    """'실제값 vs 예측값' 그래프의 두 축을 전체 점이 다 보이는 범위로 고정한다.
    (과제 3 파일은 고치지 않고, 마지막 과제에서 받은 그래프만 손본다.)"""
    for kind, fig in rep.get("items", []):
        if kind != "fig":
            continue
        try:
            title = fig.layout.title.text or ""
            if "실제값" not in title:
                continue
            vals = np.concatenate([np.asarray(v, float).ravel() for t in fig.data for v in (t.x, t.y) if v is not None])
            vals = vals[np.isfinite(vals)]
            if vals.size:
                pad = (vals.max() - vals.min()) * 0.05 or 1.0
                rng = [float(vals.min() - pad), float(vals.max() + pad)]
                fig.update_layout(xaxis=dict(range=rng), yaxis=dict(range=rng))
        except Exception:      # 그래프 모양이 달라도 실행 결과에는 영향 없게
            continue


def execute(state, name, args):
    """도구 하나를 실행한다. (화면용 결과, LLM에게 돌려줄 글)"""
    args = args or {}
    notes = []
    try:
        if name in DATA_TOOLS:
            _ensure_data(state, notes)
        rep = _run(state, name, args, notes)
    except Exception as e:  # 도구가 실패해도 LLM에게 이유를 알려 다른 방법을 찾게 함
        rep = T.reply(f"⚠️ `{name}` 실행 중 오류: {type(e).__name__}: {e}")
    _fit_axes(rep)
    if notes:
        rep["text"] = "\n".join(f"ℹ️ {n}" for n in notes) + "\n\n" + rep.get("text", "")
    if rep.get("summary"):
        T.remember(rep["summary"])
    return rep, for_llm(rep) + f"\n\n[현재 상태] {_state_line(state)}"


def _run(state, name, a, notes):
    if name == "list_datasets":
        return T.dataset_list()
    if name == "load_dataset":
        return T.load_dataset(state, a.get("name", "concrete")) if a.get("name") in T.DATASETS else \
            T.reply(f"⚠️ 데이터 이름은 {', '.join(T.DATASETS)} 중 하나여야 해요.")
    if name == "load_data_url":
        return T.load_url(state, a["url"])
    if name == "make_data":
        txt = f"y = {a.get('formula', '2*x + 1')} 데이터 {int(a.get('n') or 100)}개 만들어줘" + ("" if a.get("noise", True) else " 잡음없이")
        made = T.make_data(txt)
        if made is None:
            return T.reply("⚠️ 식을 이해하지 못했어요. x와 숫자, + - * / ^, sin·cos·exp·log·sqrt만 쓸 수 있어요.")
        return T._set_data(state, made[0], made[1])
    if name == "data_summary":
        rep = T.describe(state)
        rep["text"] += "\n\n열: " + ", ".join(map(str, state["df"].columns))
        return rep
    if name == "set_variables":
        return _set_variables(state, a)
    if name == "run_linear_regression":
        return T.run_linear(state)
    if name == "run_nonlinear_regression":
        return T.run_nonlinear(state, degree=a.get("degree"))
    if name == "run_neural_network":
        layers = [int(v) for v in (a.get("layers") or []) if int(v) > 0][:6] or None
        return T.run_ann(state, deep=False, layers=layers, act=a.get("activation") or "relu",
                         epochs=a.get("epochs"), lr=float(a.get("learning_rate") or 0.001))
    if name == "run_deep_learning":
        return T.run_ann(state, deep=True)
    if name == "compare_models":
        return T.run_compare(state)
    if name == "run_dashboard_models":
        return run_dashboard_models(state, a.get("models"))
    if name == "predict":
        return _predict(state, a)
    if name == "sweep_variable":
        cols, miss = _cols(state, [a.get("variable", "")], [c for c in T.numeric_cols(state) if c != state["y"]])
        if not cols:
            return T.reply(f"⚠️ '{a.get('variable')}' 변수를 찾지 못했어요. 열: {', '.join(map(str, state['df'].columns))}")
        d = state["df"][cols[0]]
        lo = float(a["start"]) if a.get("start") is not None else float(d.min())
        hi = float(a["end"]) if a.get("end") is not None else float(d.max())
        return T.run_sweep(state, cols[0], lo, hi, vals={}, text="")
    if name == "correlation":
        return T.correlation(state)
    if name == "plot":
        cols, _ = _cols(state, a.get("columns"))
        words = " ".join(T.short(c) for c in cols) or T.short(state["y"])
        return T.plot_cols(state, words + (" 히스토그램" if a.get("kind") == "histogram" else " 산점도"))
    if name == "column_stats":
        cols, _ = _cols(state, a.get("columns"))
        return T.column_stats(state, " ".join(T.short(c) for c in cols) + " " + " ".join(a.get("stats") or []))
    if name == "add_variable":
        return T.add_variable(state, a["name"], a["expression"])
    if name == "make_report":
        return T.report(state)
    if name == "current_status":
        return T.status(state)
    if name == "search_lecture":
        return KB.search_lecture(a.get("query", ""))
    if name == "search_my_code":
        return KB.search_code(a.get("query", ""))
    if name == "remember":
        T.remember("메모: " + str(a.get("note", ""))[:300])
        return T.reply(f"📝 기억해 둘게요: {a.get('note', '')}")
    if name == "recall_memory":
        return T.recall()
    return T.reply(f"⚠️ 없는 도구예요: {name}. 쓸 수 있는 도구: {', '.join(TOOL_NAMES)}")


def _set_variables(state, a):
    num = [c for c in T.numeric_cols(state) if state["df"][c].nunique(dropna=True) > 1]
    y = state["y"]
    msgs = []
    if a.get("y"):
        ys, miss = _cols(state, [a["y"]], num)
        if ys:
            y = ys[0]
        else:
            msgs.append(f"Y '{a['y']}'를 찾지 못해 그대로 둠")
    xs, miss = _cols(state, a.get("x"), [c for c in num if c != y])
    if miss:
        msgs.append(f"못 찾은 X: {', '.join(miss)}")
    ex, _ = _cols(state, a.get("exclude"), num)
    xs = xs or [c for c in num if c != y]
    xs = [c for c in xs if c != y and c not in ex]
    if not xs:
        return T.reply("⚠️ 입력 변수(X)가 하나도 남지 않았어요. 열: " + ", ".join(map(str, num)))
    state.update(x=xs, y=y, results={})
    state["inputs"] = {}
    note = ("\n\n- " + "\n- ".join(msgs) + f"\n- 쓸 수 있는 열: {', '.join(T.short(c) for c in num)}") if msgs else ""
    return T.reply(f"🎯 변수를 정했어요. Y = **{T.short(y)}**, X = {', '.join(T.short(c) for c in xs)}"
                   + (f" (제외: {', '.join(T.short(c) for c in ex)})" if ex else "") + note,
                   summary=f"변수 설정: Y={T.short(y)}, X={len(xs)}개")


def _predict(state, a):
    vals = a.get("values") or {}
    if isinstance(vals, str):
        try:
            vals = json.loads(vals)
        except ValueError:
            vals = {}
    pool = [c for c in T.numeric_cols(state) if c != state["y"]]
    parsed, miss = {}, []
    for k, v in vals.items():
        cols, _ = _cols(state, [k], pool)
        nums = [float(x) for x in (v if isinstance(v, list) else [v]) if str(x).replace(".", "", 1).lstrip("-").isdigit()]
        if cols and nums:
            parsed[cols[0]] = nums[:12]
        else:
            miss.append(str(k))
    if not parsed:
        return T.reply(f"⚠️ 예측에 쓸 변수 값을 찾지 못했어요({', '.join(miss)}). 입력 변수: {', '.join(T.short(c) for c in state['x'])}")
    rep = T.run_predict(state, parsed, "그대로" if a.get("keep_previous") else "")
    if miss:
        rep["text"] += f"\n\n- 찾지 못한 변수: {', '.join(miss)}"
    return rep
