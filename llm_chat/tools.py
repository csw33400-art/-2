"""채팅으로 실행하는 분석 도구 모음 (선형회귀 · 비선형회귀 · 인공신경망 · 딥러닝)

app.py의 채팅창이 사용자의 말을 해석해서 이 파일의 함수를 실행합니다.
모든 함수는 {"text": 설명글, "items": [("fig", 그래프) | ("table", 표)], "summary": 한 줄 요약}을 돌려줍니다.
"""
import io
import json
import re
import tempfile
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from sklearn.compose import TransformedTargetRegressor
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MEMORY_FILE = Path(tempfile.gettempdir()) / "llm_chat_memory.json"
SEED = 42
COLORS = ["#EF553B", "#636EFA", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3"]

DATASETS = {
    "concrete": {"label": "콘크리트 압축강도 (UCI 공개 데이터, 1,030행)", "words": ("콘크리트", "concrete")},
    "diabetes": {"label": "당뇨병 진행도 (scikit-learn 공개 데이터, 442행)", "words": ("당뇨", "diabetes")},
    "synthetic": {"label": "비선형 합성 데이터 (150행)", "words": ("합성", "synthetic")},
}


def new_state():
    return {"df": None, "key": None, "name": None, "x": [], "y": None, "results": {}}


def reply(text, items=None, summary=None):
    return {"text": text, "items": items or [], "summary": summary}


def norm(text):
    return re.sub(r"\s+", "", str(text).lower())


def short(col):
    return str(col).split("(")[0].strip()


# =============================================================
# 1. 기억 (메모 방식): 한 일을 파일에 적어 두고, 나중에 다시 읽는다
# =============================================================
def remember(summary):
    try:
        log = recall_all()
        log.append({"time": datetime.now().strftime("%m-%d %H:%M"), "summary": summary})
        MEMORY_FILE.write_text(json.dumps(log[-50:], ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def recall_all():
    try:
        return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def forget_all():
    try:
        MEMORY_FILE.unlink()
    except OSError:
        pass


def recall():
    log = recall_all()
    if not log:
        return reply("아직 기억해 둔 작업이 없어요. 분석을 실행하면 자동으로 적어 둘게요.")
    lines = "\n".join(f"- `{m['time']}` {m['summary']}" for m in log[-10:])
    return reply(f"**기억하고 있는 최근 작업 ({min(len(log), 10)}개)**\n\n{lines}\n\n"
                 "가중치를 바꾸는 방식이 아니라, 한 일을 파일에 적어 두고 다시 읽는 '메모 방식'이에요.")


# =============================================================
# 2. 데이터 불러오기와 변수 설정
# =============================================================
def match_dataset(t):
    for key, d in DATASETS.items():
        if any(w in t for w in d["words"]):
            return key
    return None


def _set_data(state, df, name, key=None):
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num) < 2:
        return reply(f"⚠️ '{name}'에는 숫자형 열이 2개 이상 있어야 해요. (X 1개 이상 + Y 1개)")
    state.update(df=df, name=name, key=key, y=num[-1], x=num[:-1], results={})
    remember(f"데이터 '{name}' 불러옴 ({len(df):,}행)")
    return reply(
        f"📂 **{name}** 데이터를 불러왔어요. {len(df):,}행 × {df.shape[1]}열이에요.\n\n"
        f"- 예측할 값(Y): **{short(state['y'])}**\n- 입력 변수(X): {', '.join(short(c) for c in state['x'])}\n\n"
        "바꾸려면 \"Y는 ○○, X는 △△\"처럼 말하세요. 이제 \"선형회귀 해줘\"처럼 시켜 보세요.",
        [("table", df.head(5))],
        f"데이터 {name} {len(df)}행, Y={short(state['y'])}")


def load_dataset(state, key):
    if key == "concrete":
        df = pd.read_csv(DATA_DIR / "concrete.csv")
    elif key == "diabetes":
        df = load_diabetes(as_frame=True).frame
    else:
        rng = np.random.default_rng(0)
        x = np.linspace(0.5, 10, 150)
        df = pd.DataFrame({"x": x, "y": 3 * np.sin(x) + 0.8 * x + 5 + rng.normal(0, 0.6, len(x))})
    return _set_data(state, df, DATASETS[key]["label"], key)


def load_url(state, url):
    try:
        df = pd.read_excel(url) if re.search(r"\.xlsx?($|\?)", url.lower()) else pd.read_csv(url)
    except Exception as e:
        return reply(f"⚠️ 주소에서 데이터를 읽지 못했어요: {e}\n\nCSV나 Excel 파일로 바로 연결되는 주소인지 확인해 주세요.")
    return _set_data(state, df, url.split("/")[-1][:40] or "웹 데이터")


def load_file(state, f):
    try:
        raw = f.getvalue()
        if f.name.lower().endswith(".csv"):
            df = None
            for enc in ("utf-8-sig", "cp949", "latin-1"):
                try:
                    df = pd.read_csv(io.BytesIO(raw), encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
        else:
            df = pd.read_excel(io.BytesIO(raw))
    except Exception as e:
        return reply(f"⚠️ 파일을 읽지 못했어요: {e}")
    return _set_data(state, df, f.name)


def find_cols(text, cols):
    """문장에 나온 열 이름을 나온 순서대로 찾는다. (위치, 열) 목록."""
    t = norm(text)
    found = []
    for c in cols:
        s = norm(short(c))
        if not s:
            continue
        if s.isascii():   # 영문 이름(x, y, age 등)은 다른 단어 속에 섞이지 않게 단어 경계가 있을 때만
            m = re.search(rf"(?<![a-z0-9]){re.escape(s)}(?![a-z0-9])", str(text).lower())
            pos = len(norm(str(text)[:m.start()])) if m else -1
        else:
            pos = t.find(s)
        if pos >= 0:
            found.append((pos, c))
    return sorted(found)


X_MARK = ("x는", "x은", "x를", "x=", "x축", "x로", "독립변수", "입력변수", "입력은", "입력으로")
Y_MARK = ("y는", "y은", "y를", "y=", "y축", "y로", "종속변수", "목표변수", "출력변수", "출력은", "예측대상", "타깃", "타겟")


def has_var_marker(t):
    return any(m in t for m in X_MARK + Y_MARK)


def parse_vars(state, text):
    """문장에서 X·Y 변수 지정을 읽어 반영한다. 바뀌었으면 안내 문장을, 아니면 None을 돌려준다."""
    if state["df"] is None:
        return None
    num = state["df"].select_dtypes(include=[np.number]).columns.tolist()
    t = norm(text)
    mentioned = find_cols(text, num)
    new_x, new_y = list(state["x"]), state["y"]

    marks = sorted([(t.find(m), "x") for m in X_MARK if m in t] + [(t.find(m), "y") for m in Y_MARK if m in t])
    if marks:
        xs, ys = [], []
        for pos, col in mentioned:
            before = [k for p, k in marks if p <= pos]
            if before:
                (xs if before[-1] == "x" else ys).append(col)
        if ys:
            new_y = ys[0]
        if xs:
            new_x = xs
    elif mentioned:
        cand = [c for _, c in mentioned if c != new_y]
        if cand:
            new_x = cand
    if any(w in t for w in ("모든변수", "전체변수", "전부다", "변수전부", "변수전체", "변수다", "모든입력")):
        new_x = [c for c in num if c != new_y]

    new_x = [c for c in new_x if c != new_y] or [c for c in num if c != new_y]
    if new_x == state["x"] and new_y == state["y"]:
        return None
    state.update(x=new_x, y=new_y, results={})
    return f"🎯 변수를 바꿨어요. Y = **{short(new_y)}**, X = {', '.join(short(c) for c in new_x)}"


def status(state):
    if state["df"] is None:
        return reply("아직 데이터를 불러오지 않았어요. \"콘크리트 데이터 불러와\"라고 말해 보세요.")
    done = ", ".join(f"{r['label']}(R² {r['test_r2']:.3f})" for r in state["results"].values()) or "아직 없음"
    return reply(f"**현재 상태**\n\n- 데이터: {state['name']} ({len(state['df']):,}행)\n- Y: {short(state['y'])}\n"
                 f"- X ({len(state['x'])}개): {', '.join(short(c) for c in state['x'])}\n- 학습한 모델: {done}")


def describe(state):
    df = state["df"]
    num = df.select_dtypes(include=[np.number])
    desc = num.describe().T[["mean", "std", "min", "max"]].round(3).reset_index()
    desc.columns = ["변수", "평균", "표준편차", "최솟값", "최댓값"]
    return reply(f"**{state['name']}** 요약이에요. {len(df):,}행 × {df.shape[1]}열, 결측치 {int(df.isna().sum().sum())}개예요.\n\n"
                 "위 표는 처음 5행, 아래 표는 변수별 기술 통계예요.",
                 [("table", df.head(5)), ("table", desc)])


def correlation(state):
    cols = state["x"] + [state["y"]]
    corr = state["df"][cols].corr()
    names = [short(c) for c in cols]
    fig = go.Figure(go.Heatmap(z=corr.values, x=names, y=names, zmin=-1, zmax=1, colorscale="RdBu_r",
                               text=np.round(corr.values, 2), texttemplate="%{text}"))
    fig.update_layout(title="상관계수 히트맵", height=480, margin=dict(t=50, b=20))
    top = corr[state["y"]].drop(state["y"]).sort_values(key=np.abs, ascending=False).head(3)
    tops = ", ".join(f"{short(k)} ({v:+.2f})" for k, v in top.items())
    return reply(f"**{short(state['y'])}**와 상관이 큰 변수는 {tops} 순서예요. +는 같이 커지고, −는 반대로 움직여요.",
                 [("fig", fig)], f"{short(state['y'])}와 상관 큰 변수: {tops}")


# =============================================================
# 3. 공통: 데이터 분할, 평가, 그래프
# =============================================================
def _split(state):
    d = state["df"][state["x"] + [state["y"]]].dropna()
    X, y = d[state["x"]].values.astype(float), d[state["y"]].values.astype(float)
    if len(d) < 8:
        return X, X, y, y
    return train_test_split(X, y, test_size=0.2, random_state=SEED)


def _store(state, key, label, predict, Xtr, Xte, ytr, yte, **extra):
    ptr, pte = np.asarray(predict(Xtr), float), np.asarray(predict(Xte), float)
    r = dict(label=label, predict=predict, train_r2=float(r2_score(ytr, ptr)), test_r2=float(r2_score(yte, pte)),
             rmse=float(np.sqrt(mean_squared_error(yte, pte))), mae=float(mean_absolute_error(yte, pte)),
             pred_test=pte, y_test=yte, n_train=len(ytr), n_test=len(yte), **extra)
    state["results"][key] = r
    return r


def _metrics_line(r):
    return (f"| 테스트 R² | RMSE | MAE | 학습 R² |\n| --- | --- | --- | --- |\n"
            f"| **{r['test_r2']:.3f}** | {r['rmse']:.3f} | {r['mae']:.3f} | {r['train_r2']:.3f} |")


def _interpret(r):
    q = r["test_r2"]
    level = ("아주 잘 맞아요" if q >= 0.9 else "꽤 잘 맞아요" if q >= 0.75 else "어느 정도 맞아요" if q >= 0.5
             else "잘 맞지 않아요")
    out = f"테스트 R² {q:.3f}로 처음 보는 데이터에도 {level}."
    gap = r["train_r2"] - q
    if gap > 0.1:
        out += f" 다만 학습 R²({r['train_r2']:.3f})보다 {gap:.2f} 낮아서 **과적합** 기미가 있어요."
    return out


def _vars_line(state):
    x = ", ".join(short(c) for c in state["x"][:6]) + (f" 외 {len(state['x']) - 6}개" if len(state["x"]) > 6 else "")
    return f"Y = {short(state['y'])}, X = {x}"


def _fig_fit_1d(state, Xtr, Xte, ytr, yte, curves, title):
    lo, hi = min(Xtr.min(), Xte.min()), max(Xtr.max(), Xte.max())
    grid = np.linspace(lo, hi, 300).reshape(-1, 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Xtr.ravel(), y=ytr, mode="markers", name="학습 데이터",
                             marker=dict(color="#555", size=6, opacity=0.6)))
    fig.add_trace(go.Scatter(x=Xte.ravel(), y=yte, mode="markers", name="테스트 데이터",
                             marker=dict(color="white", size=7, line=dict(color="#222", width=1.5))))
    for i, (name, fn) in enumerate(curves.items()):
        with np.errstate(all="ignore"):
            yy = np.asarray(fn(grid), float)
        fig.add_trace(go.Scatter(x=grid.ravel(), y=np.where(np.isfinite(yy), yy, np.nan), mode="lines", name=name,
                                 line=dict(color=COLORS[i % len(COLORS)], width=3)))
    fig.update_layout(title=title, xaxis_title=short(state["x"][0]), yaxis_title=short(state["y"]),
                      height=460, margin=dict(t=50, b=20), legend=dict(orientation="h", y=-0.2))
    return fig


def _fig_actual_pred(state, results, title):
    fig = go.Figure()
    lo, hi = np.inf, -np.inf
    for i, r in enumerate(results):
        fig.add_trace(go.Scatter(x=r["y_test"], y=r["pred_test"], mode="markers", name=r["label"],
                                 marker=dict(color=COLORS[i % len(COLORS)], size=7, opacity=0.7)))
        lo, hi = min(lo, r["y_test"].min()), max(hi, r["y_test"].max())
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="완벽한 예측 (y=x)",
                             line=dict(color="#222", dash="dash")))
    fig.update_layout(title=title, xaxis_title=f"실제 {short(state['y'])}", yaxis_title=f"예측 {short(state['y'])}",
                      height=460, margin=dict(t=50, b=20), legend=dict(orientation="h", y=-0.2))
    return fig


# =============================================================
# 4. 선형회귀
# =============================================================
def run_linear(state):
    Xtr, Xte, ytr, yte = _split(state)
    m = LinearRegression().fit(Xtr, ytr)
    r = _store(state, "linear", "선형 회귀", m.predict, Xtr, Xte, ytr, yte)
    names = [short(c) for c in state["x"]]
    items = []
    if len(names) == 1:
        eq = f"{short(state['y'])} = {m.coef_[0]:.4g} × {names[0]} {'+' if m.intercept_ >= 0 else '−'} {abs(m.intercept_):.4g}"
        items.append(("fig", _fig_fit_1d(state, Xtr, Xte, ytr, yte, {"선형 회귀": m.predict}, "선형 회귀 적합 결과")))
    else:
        eq = f"{short(state['y'])} = {m.intercept_:.4g} + Σ(계수 × 변수)  ← 계수는 아래 표"
        coef = pd.DataFrame({"변수": names, "계수": np.round(m.coef_, 4)})
        coef = coef.reindex(coef["계수"].abs().sort_values(ascending=False).index)
        items += [("fig", _fig_actual_pred(state, [r], "선형 회귀: 실제값 vs 예측값 (테스트 데이터)")), ("table", coef)]
    text = (f"### 📈 선형 회귀 결과\n{_vars_line(state)} · 학습 {r['n_train']}행 / 테스트 {r['n_test']}행\n\n"
            f"`{eq}`\n\n{_metrics_line(r)}\n\n{_interpret(r)} "
            "잔차 제곱합을 최소로 하는 계수를 최소제곱법으로 구했어요.")
    return reply(text, items, f"선형 회귀: 테스트 R² {r['test_r2']:.3f}, RMSE {r['rmse']:.2f}, 학습 R² {r['train_r2']:.3f} ({_vars_line(state)})")


# =============================================================
# 5. 비선형회귀
# =============================================================
def _curve_candidates(x, y, degrees):
    cands = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for d in degrees:
            if len(np.unique(x)) > d:
                p = np.polyfit(x, y, d)
                cands[f"{d}차 다항"] = (lambda X, p=p: np.polyval(p, np.asarray(X, float).ravel()),
                                     "y = " + " + ".join(f"{c:.4g}·x^{d - i}" if d - i > 1 else (f"{c:.4g}·x" if d - i == 1 else f"{c:.4g}")
                                                         for i, c in enumerate(p)).replace("+ -", "- "))
        try:
            if np.all(x > 0) and np.all(y > 0):
                b, la = np.polyfit(np.log(x), np.log(y), 1)
                (a, b), _ = curve_fit(lambda x, a, b: a * np.power(x, b), x, y, p0=[np.exp(la), b], maxfev=20000)
                cands["거듭제곱"] = (lambda X, a=a, b=b: a * np.power(np.asarray(X, float).ravel(), b), f"y = {a:.4g}·x^{b:.4g}")
        except (RuntimeError, ValueError, OverflowError):
            pass
        try:
            if np.all(y > 0):
                b, la = np.polyfit(x, np.log(y), 1)
                (a, b), _ = curve_fit(lambda x, a, b: a * np.exp(b * x), x, y, p0=[np.exp(la), b], maxfev=20000)
                cands["지수"] = (lambda X, a=a, b=b: a * np.exp(b * np.asarray(X, float).ravel()), f"y = {a:.4g}·e^({b:.4g}·x)")
        except (RuntimeError, ValueError, OverflowError):
            pass
        if np.all(x > 0):
            b, a = np.polyfit(np.log(x), y, 1)
            cands["로그"] = (lambda X, a=a, b=b: a + b * np.log(np.asarray(X, float).ravel()), f"y = {a:.4g} + {b:.4g}·ln(x)")
    return cands


def run_nonlinear(state, degree=None):
    Xtr, Xte, ytr, yte = _split(state)
    lin = LinearRegression().fit(Xtr, ytr)
    lin_r2 = float(r2_score(yte, lin.predict(Xte)))
    items = []
    if len(state["x"]) == 1:
        degrees = [min(max(degree, 2), 6)] if degree else [2, 3]
        cands = _curve_candidates(Xtr.ravel(), ytr, degrees)
        rows = []
        for name, (fn, eq) in cands.items():
            with np.errstate(all="ignore"):
                pte, ptr = fn(Xte), fn(Xtr)
            if np.all(np.isfinite(pte)) and np.all(np.isfinite(ptr)):
                rows.append((name, fn, eq, float(r2_score(yte, pte)), float(r2_score(ytr, ptr))))
        rows.sort(key=lambda t: -t[3])
        if not rows:
            return reply("⚠️ 이 데이터에는 맞는 비선형 곡선을 찾지 못했어요.")
        best = rows[0]
        if degree:                                  # 차수를 직접 말했으면 그 다항식을 결과로 사용
            best = next((r for r in rows if r[0] == f"{degrees[0]}차 다항"), best)
        r = _store(state, "nonlinear", f"비선형 회귀 ({best[0]})", best[1], Xtr, Xte, ytr, yte)
        table = pd.DataFrame([(n, round(te, 4), round(tr, 4), eq) for n, _, eq, te, tr in rows],
                             columns=["곡선", "테스트 R²", "학습 R²", "회귀식 (x = " + short(state["x"][0]) + ")"])
        curves = {"선형 (비교)": lin.predict, **{n: fn for n, fn, *_ in rows[:4]}}
        items = [("fig", _fig_fit_1d(state, Xtr, Xte, ytr, yte, curves, "비선형 회귀 곡선 비교")), ("table", table)]
        detail = (f"요청한 **{best[0]}** 곡선으로 적합했어요." if degree else
                  f"{len(rows)}가지 곡선 중 **{best[0]}** 곡선이 가장 잘 맞았어요.") + f"\n\n`{best[2]}`"
    else:
        d = min(max(degree or 2, 2), 3)
        model = make_pipeline(StandardScaler(), PolynomialFeatures(d, include_bias=False), LinearRegression()).fit(Xtr, ytr)
        n_terms = model[1].n_output_features_
        r = _store(state, "nonlinear", f"비선형 회귀 ({d}차 다항)", model.predict, Xtr, Xte, ytr, yte)
        items = [("fig", _fig_actual_pred(state, [r], f"{d}차 다항 회귀: 실제값 vs 예측값 (테스트 데이터)"))]
        detail = (f"변수가 {len(state['x'])}개라서 변수들의 제곱과 곱(상호작용)을 추가한 **{d}차 다항 회귀**를 썼어요. "
                  f"항이 {n_terms}개로 늘어나요.")
    diff = r["test_r2"] - lin_r2
    text = (f"### 📉 비선형 회귀 결과\n{_vars_line(state)} · 학습 {r['n_train']}행 / 테스트 {r['n_test']}행\n\n{detail}\n\n"
            f"{_metrics_line(r)}\n\n{_interpret(r)} 같은 조건의 선형 회귀(R² {lin_r2:.3f})보다 "
            f"{'**' + format(diff, '.3f') + ' 높아요**' if diff > 0.005 else '나아지지 않았어요'}.")
    return reply(text, items, f"{r['label']}: 테스트 R² {r['test_r2']:.3f} (선형 {lin_r2:.3f}), 학습 R² {r['train_r2']:.3f}")


# =============================================================
# 6. 인공신경망 · 딥러닝
# =============================================================
def parse_ann_options(text):
    """'은닉층 64-32', '에포크 500', 'tanh', '학습률 0.01' 같은 말을 읽는다."""
    low = str(text).lower()
    opt = {}
    m = (re.search(r"\d+(?:\s*-\s*\d+)+", low)
         or re.search(r"(?:은닉층|레이어|노드|구조)\D{0,4}(\d+(?:\s*[, ]\s*\d+)+)", low))
    if m:
        seq = m.group(1) if m.lastindex else m.group()
        opt["layers"] = tuple(min(max(int(v), 1), 512) for v in re.findall(r"\d+", seq))[:6]
    else:
        m = re.search(r"(?:은닉층|레이어|층)\D{0,3}(\d)\s*개|(\d)\s*층", low)
        if m:
            n = int(m.group(1) or m.group(2))
            opt["layers"] = tuple(max(8, 128 // 2 ** i) for i in range(min(max(n, 1), 6)))
        m2 = re.search(r"노드\D{0,4}(\d+)", low)
        if m2:
            n = min(max(int(m2.group(1)), 1), 512)
            opt["layers"] = tuple(n for _ in opt.get("layers", (0,)))
    m = re.search(r"(?:에포크|에폭|epochs?|반복)\D{0,4}(\d+)|(\d+)\s*(?:에포크|에폭|epochs?)", low)
    if m:
        opt["epochs"] = min(max(int(m.group(1) or m.group(2)), 10), 5000)
    m = re.search(r"(?:학습률|러닝레이트|learning\s*rate|lr)\D{0,4}(\d*\.?\d+)", low)
    if m:
        opt["lr"] = min(max(float(m.group(1)), 1e-5), 1.0)
    for word, act in (("tanh", "tanh"), ("시그모이드", "logistic"), ("sigmoid", "logistic"), ("logistic", "logistic"),
                      ("relu", "relu"), ("렐루", "relu")):
        if word in low:
            opt["act"] = act
            break
    return opt


def run_ann(state, deep=False, layers=None, act="relu", epochs=None, lr=0.001):
    Xtr, Xte, ytr, yte = _split(state)
    layers = tuple(layers or ((128, 64, 32, 16) if deep else (64, 32)))
    epochs = epochs or (2000 if deep else 1000)
    use_val = deep and len(Xtr) >= 50          # 딥러닝: 학습 데이터의 20%를 검증용으로 떼어 조기 종료
    mlp = MLPRegressor(hidden_layer_sizes=layers, activation=act, max_iter=epochs, learning_rate_init=lr,
                       random_state=SEED, early_stopping=use_val, validation_fraction=0.2,
                       n_iter_no_change=20 if deep else 10)
    model = TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), mlp), transformer=StandardScaler())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(Xtr, ytr)
    net = model.regressor_[-1]
    n_params = int(sum(w.size for w in net.coefs_) + sum(b.size for b in net.intercepts_))
    key, name = ("deep", "딥러닝 (심층 신경망)") if deep else ("ann", "인공신경망")
    label = f"{name} {'-'.join(map(str, layers))}"
    r = _store(state, key, label, model.predict, Xtr, Xte, ytr, yte, n_params=n_params, epochs_run=int(net.n_iter_))

    fig = go.Figure(go.Scatter(y=net.loss_curve_, mode="lines", name="학습 손실", line=dict(color=COLORS[1], width=3)))
    if use_val and getattr(net, "validation_scores_", None):
        fig.add_trace(go.Scatter(y=net.validation_scores_, mode="lines", name="검증 R²", yaxis="y2",
                                 line=dict(color=COLORS[0], width=2, dash="dot")))
        fig.update_layout(yaxis2=dict(title="검증 R²", overlaying="y", side="right", range=[0, 1]))
    fig.update_layout(title="학습 곡선 (에포크별 손실)", xaxis_title="에포크", yaxis_title="손실", yaxis_type="log",
                      height=380, margin=dict(t=50, b=20), legend=dict(orientation="h", y=-0.25))
    act_k = {"logistic": "sigmoid"}.get(act, act)
    dense = "\n".join(f"    Dense({n}, activation='{act_k}'{', input_shape=(' + str(len(state['x'])) + ',)' if i == 0 else ''}),"
                      for i, n in enumerate(layers))
    stop = (f"{net.n_iter_}에포크에서 학습이 끝났어요" +
            (" (검증 점수가 20번 연속 좋아지지 않아 조기 종료)." if use_val and net.n_iter_ < epochs else "."))
    text = (f"### {'🧬' if deep else '🧠'} {name} 결과\n{_vars_line(state)} · 학습 {r['n_train']}행 / 테스트 {r['n_test']}행\n\n"
            f"구조: 입력 {len(state['x'])} → " + " → ".join(f"{n}" for n in layers) + f" → 출력 1 · 활성화 {act_k} · "
            f"가중치·편향 **{n_params:,}개** · {stop}\n\n"
            f"```python\n# 같은 구조를 Keras로 쓰면\nmodel = Sequential([\n{dense}\n    Dense(1),   # 출력: {short(state['y'])}\n])\n```\n"
            f"{_metrics_line(r)}\n\n{_interpret(r)} "
            + ("은닉층이 3개 이상인 깊은 신경망이라 딥러닝이라고 불러요. 입력을 표준화하고, 순전파 → 오차 계산 → 오류역전파 → Adam으로 가중치 갱신을 반복했어요."
               if len(layers) >= 3 else "입력을 표준화하고, 순전파 → 오차 계산 → 오류역전파 → Adam으로 가중치 갱신을 반복했어요."))
    items = [("fig", fig), ("fig", _fig_actual_pred(state, [r], f"{name}: 실제값 vs 예측값 (테스트 데이터)"))]
    return reply(text, items, f"{label}: 테스트 R² {r['test_r2']:.3f}, RMSE {r['rmse']:.2f}, 학습 R² {r['train_r2']:.3f}, "
                              f"매개변수 {n_params}개, {net.n_iter_}에포크")


# =============================================================
# 7. 전체 비교 · 예측 · 보고서
# =============================================================
RUNNERS = {"linear": run_linear, "nonlinear": run_nonlinear,
           "ann": lambda s: run_ann(s, deep=False), "deep": lambda s: run_ann(s, deep=True)}


def run_compare(state):
    for key, fn in RUNNERS.items():
        if key not in state["results"]:
            fn(state)
    rs = sorted(state["results"].values(), key=lambda r: -r["test_r2"])
    table = pd.DataFrame([(r["label"], round(r["test_r2"], 4), round(r["rmse"], 3), round(r["mae"], 3),
                           round(r["train_r2"], 4), round(r["train_r2"] - r["test_r2"], 3)) for r in rs],
                         columns=["모델", "테스트 R²", "RMSE", "MAE", "학습 R²", "과적합 격차"])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[r["label"] for r in rs], y=[r["test_r2"] for r in rs], name="테스트 R²", marker_color=COLORS[1],
                         text=[f"{r['test_r2']:.3f}" for r in rs], textposition="outside"))
    fig.add_trace(go.Bar(x=[r["label"] for r in rs], y=[r["train_r2"] for r in rs], name="학습 R²", marker_color="#C9CED6"))
    fig.update_layout(title="모델별 R² 비교", barmode="group", height=440, margin=dict(t=50, b=20),
                      yaxis=dict(range=[min(0, min(r["test_r2"] for r in rs)) - 0.05, 1.1]),
                      legend=dict(orientation="h", y=-0.3))
    best, worst = rs[0], rs[-1]
    text = (f"### 🏆 모델 비교\n{_vars_line(state)}\n\n가장 잘 맞는 모델은 **{best['label']}** (테스트 R² {best['test_r2']:.3f})이에요. "
            f"가장 낮은 {worst['label']}({worst['test_r2']:.3f})보다 {best['test_r2'] - worst['test_r2']:.3f} 높아요.")
    summary = "모델 비교: " + ", ".join(f"{r['label']} R² {r['test_r2']:.3f}" for r in rs)
    return reply(text, [("fig", fig), ("table", table), ("fig", _fig_actual_pred(state, rs[:3], "실제값 vs 예측값 (상위 3개 모델)"))], summary)


def parse_inputs(state, text):
    t = norm(text)
    vals = {}
    for c in state["x"]:
        m = re.search(re.escape(norm(short(c))) + r"[^\d\-]{0,6}(-?\d+(?:\.\d+)?)", t)
        if m:
            vals[c] = float(m.group(1))
    return vals


def run_predict(state, vals):
    notes = []
    if not state["results"]:
        run_linear(state)
        run_ann(state)
        notes.append("학습된 모델이 없어서 선형 회귀와 인공신경망을 먼저 학습했어요.")
    d = state["df"][state["x"]].dropna()
    row, filled, outside = [], [], []
    for c in state["x"]:
        if c in vals:
            row.append(vals[c])
            if not d[c].min() <= vals[c] <= d[c].max():
                outside.append(short(c))
        else:
            row.append(float(d[c].mean()))
            filled.append(short(c))
    X = np.array([row])
    rs = sorted(state["results"].values(), key=lambda r: -r["test_r2"])
    preds = []
    for r in rs:
        with np.errstate(all="ignore"):
            p = float(np.asarray(r["predict"](X), float).ravel()[0])
        preds.append((r["label"], round(p, 3) if np.isfinite(p) else None, round(r["test_r2"], 3)))
    table = pd.DataFrame(preds, columns=["모델", f"예측 {short(state['y'])}", "모델의 테스트 R²"])
    inp = pd.DataFrame({"변수": [short(c) for c in state["x"]], "입력값": np.round(row, 3)})
    if filled:
        notes.append(f"말하지 않은 변수({', '.join(filled)})는 평균값으로 채웠어요.")
    if outside:
        notes.append(f"⚠️ {', '.join(outside)} 값이 학습 데이터 범위를 벗어나서 예측을 믿기 어려워요.")
    best = preds[0]
    text = (f"### 🔮 예측 결과\n가장 정확한 모델인 **{best[0]}**의 예측은 **{short(state['y'])} = {best[1]}**이에요.\n\n"
            + "\n".join(f"- {n}" for n in notes))
    return reply(text, [("table", table), ("table", inp)], f"예측: {best[0]} → {short(state['y'])} {best[1]}")


def report(state):
    if state["df"] is None or not state["results"]:
        return reply("아직 보고서에 쓸 분석 결과가 없어요. \"전부 비교해줘\"라고 하면 네 가지 모델을 모두 돌려요.")
    rs = sorted(state["results"].values(), key=lambda r: -r["test_r2"])
    rows = "\n".join(f"| {r['label']} | {r['test_r2']:.3f} | {r['rmse']:.3f} | {r['mae']:.3f} | {r['train_r2']:.3f} |" for r in rs)
    best = rs[0]
    lin = state["results"].get("linear")
    vs = (f" 선형 회귀(R² {lin['test_r2']:.3f})보다 {best['test_r2'] - lin['test_r2']:.3f} 높아요."
          if lin and best is not lin else "")
    over = [r["label"] for r in rs if r["train_r2"] - r["test_r2"] > 0.1]
    text = f"""## 📄 분석 보고서

**1. 데이터**
- {state['name']} · {len(state['df']):,}행
- 예측 대상(Y): {short(state['y'])}
- 입력 변수(X, {len(state['x'])}개): {', '.join(short(c) for c in state['x'])}

**2. 방법**
- 데이터의 80%로 학습하고, 학습에 쓰지 않은 20%(테스트)로 평가했어요.
- 평가지표: R²(1에 가까울수록 좋음), RMSE·MAE(오차, 작을수록 좋음)

**3. 결과**

| 모델 | 테스트 R² | RMSE | MAE | 학습 R² |
| --- | --- | --- | --- | --- |
{rows}

**4. 해석**
- 가장 잘 맞는 모델은 **{best['label']}**(테스트 R² {best['test_r2']:.3f})이에요.{vs}
- {('과적합 기미가 있는 모델: ' + ', '.join(over) + ' (학습 R²가 테스트 R²보다 0.1 이상 높음)') if over else '학습 R²와 테스트 R²의 차이가 큰 모델은 없어요 (과적합 징후 없음).'}

**5. 기록**
{chr(10).join(f"- `{m['time']}` {m['summary']}" for m in recall_all()[-6:])}"""
    return reply(text, [("fig", _fig_actual_pred(state, rs[:3], "실제값 vs 예측값 (상위 3개 모델)"))], "보고서 작성")
