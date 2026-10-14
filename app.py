import io
import json
import pickle
import time
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore", category=ConvergenceWarning)

st.set_page_config(page_title="통합 데이터 분석 대시보드", layout="wide")
st.title("🚀 범용 회귀 분석 및 머신러닝 예측 대시보드")
st.markdown(
    "데이터를 업로드하면 **13가지 회귀·머신러닝 모델**을 한 번에 학습하고, "
    "성능 비교 · 잔차 진단 · 변수 영향 해석 · 일괄 예측까지 할 수 있습니다."
)

MIN_ROWS_FOR_SPLIT = 8   # 이보다 적으면 테스트 세트를 나누지 않음
MIN_ROWS_FOR_CV = 6      # 이보다 적으면 교차검증 생략
PALETTE = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692",
           "#B6E880", "#FF97FF", "#FECB52", "#1F77B4", "#8C564B", "#7F7F7F"]


# =============================================================
# 0. 공통 함수
# =============================================================
class CurveFitRegressor(RegressorMixin, BaseEstimator):
    """1변수 비선형 곡선(거듭제곱·지수·로그)을 scikit-learn 모델처럼 쓰기 위한 래퍼."""

    def __init__(self, kind="power"):
        self.kind = kind

    @staticmethod
    def _func(kind):
        if kind == "power":
            return lambda x, a, b: a * np.power(x, b)
        if kind == "exp":
            return lambda x, a, b: a * np.exp(b * x)
        return lambda x, a, b: a + b * np.log(x)

    def fit(self, X, y):
        x = np.asarray(X, dtype=float).ravel()
        y = np.asarray(y, dtype=float)
        if self.kind == "log":
            b, a = np.polyfit(np.log(x), y, 1)  # 로그 회귀는 ln x 에 대해 선형이므로 바로 풂
            self.params_ = np.array([a, b])
            return self
        # 로그 변환 선형회귀로 초기값을 구한 뒤 비선형 최소제곱으로 다듬음
        if self.kind == "power":
            b, ln_a = np.polyfit(np.log(x), np.log(y), 1)
        else:  # exp
            b, ln_a = np.polyfit(x, np.log(y), 1)
        p0 = np.array([np.exp(ln_a), b])
        try:
            with np.errstate(all="ignore"):
                popt, _ = curve_fit(self._func(self.kind), x, y, p0=p0, maxfev=20000)
            self.params_ = popt if np.all(np.isfinite(popt)) else p0
        except (RuntimeError, ValueError, OverflowError):
            self.params_ = p0
        return self

    def predict(self, X):
        x = np.asarray(X, dtype=float).ravel()
        with np.errstate(all="ignore"):
            return self._func(self.kind)(x, *self.params_)


def safe_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if len(y_true) == 0 or not np.all(np.isfinite(y_pred)):
        return dict(R2=np.nan, RMSE=np.nan, MAE=np.nan, MAPE=np.nan)
    return dict(
        R2=r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan,
        RMSE=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        MAE=float(mean_absolute_error(y_true, y_pred)),
        MAPE=float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100) if np.all(y_true != 0) else np.nan,
    )


def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")  # Excel에서 한글이 깨지지 않도록 BOM 포함


@st.cache_data(show_spinner=False)
def read_uploaded(name, raw, sheet):
    if name.lower().endswith(".csv"):
        for enc in ("utf-8-sig", "cp949", "euc-kr", "latin-1"):
            try:
                return pd.read_csv(io.BytesIO(raw), encoding=enc)
            except UnicodeDecodeError:
                continue
    return pd.read_excel(io.BytesIO(raw), sheet_name=sheet or 0)


@st.cache_data(show_spinner=False)
def load_sample(name):
    if name.startswith("콘크리트"):
        return pd.DataFrame({
            'Cement': [540, 540, 332.5, 332.5, 198.6, 266, 380, 380, 266, 475],
            'Water': [162, 162, 228, 228, 192, 228, 228, 228, 228, 228],
            'Age': [28, 28, 270, 365, 360, 90, 365, 28, 28, 28],
            'Strength': [79.99, 61.89, 40.27, 41.05, 44.30, 47.03, 43.70, 36.45, 45.85, 39.29],
        })
    if name.startswith("당뇨"):
        return load_diabetes(as_frame=True).frame
    rng = np.random.default_rng(0)
    x = np.linspace(0.5, 10, 150)
    return pd.DataFrame({
        "x": x,
        "noise_feature": rng.uniform(0, 1, len(x)),
        "y": 3 * np.sin(x) + 0.8 * x + 5 + rng.normal(0, 0.6, len(x)),
    })


# =============================================================
# 1. 데이터 불러오기
# =============================================================
st.sidebar.header("📁 1. 데이터")
uploaded_file = st.sidebar.file_uploader("CSV 또는 Excel 파일 업로드", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    raw = uploaded_file.getvalue()
    sheet = None
    if not uploaded_file.name.lower().endswith(".csv"):
        try:
            sheets = pd.ExcelFile(io.BytesIO(raw)).sheet_names
        except Exception as e:
            st.error(f"Excel 파일을 열지 못했습니다: {e}")
            st.stop()
        if len(sheets) > 1:
            sheet = st.sidebar.selectbox("시트 선택:", sheets)
    try:
        df = read_uploaded(uploaded_file.name, raw, sheet)
    except Exception as e:
        st.error(f"파일을 읽지 못했습니다: {e}")
        st.stop()
    data_name = uploaded_file.name
else:
    data_name = st.sidebar.selectbox(
        "샘플 데이터 선택:",
        ["콘크리트 압축강도 (10행)", "당뇨병 진행도 (442행, sklearn)", "비선형 합성 데이터 (150행)"],
    )
    df = load_sample(data_name)

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(num_cols) < 2:
    st.error("숫자형 열이 2개 이상 필요합니다. (X 1개 이상 + Y 1개)")
    st.stop()

# =============================================================
# 2. 변수 선택 및 전처리
# =============================================================
st.sidebar.header("🎯 2. 변수 선택")
y_col = st.sidebar.selectbox("종속변수 (Y):", num_cols, index=len(num_cols) - 1)
x_options = [c for c in num_cols if c != y_col]
x_cols = st.sidebar.multiselect("독립변수 (X, 다중 선택 가능):", x_options, default=[x_options[0]])
if not x_cols:
    st.warning("분석을 위해 최소 1개의 X변수를 선택해주세요.")
    st.stop()

st.sidebar.header("🧹 3. 전처리")
outlier_method = st.sidebar.radio("이상치 제거:", ["사용 안 함", "IQR (1.5배 범위 밖)", "Z-score (|z| > 3)"])

df_clean = df[x_cols + [y_col]].dropna()
n_missing = len(df) - len(df_clean)
if outlier_method != "사용 안 함" and len(df_clean) > 0:
    vals = df_clean.values.astype(float)
    if outlier_method.startswith("IQR"):
        q1, q3 = np.percentile(vals, 25, axis=0), np.percentile(vals, 75, axis=0)
        iqr = q3 - q1
        keep = np.all((vals >= q1 - 1.5 * iqr) & (vals <= q3 + 1.5 * iqr), axis=1)
    else:
        sd = vals.std(axis=0)
        z = np.abs((vals - vals.mean(axis=0)) / np.where(sd == 0, 1, sd))
        keep = np.all(z <= 3, axis=1)
    n_outliers = int((~keep).sum())
    df_clean = df_clean[keep]
else:
    n_outliers = 0
st.sidebar.caption(f"결측치 제거 {n_missing}행 · 이상치 제거 {n_outliers}행 → 분석 {len(df_clean)}행")

if len(df_clean) < 3:
    st.error("전처리 후 데이터가 3행 미만이라 분석할 수 없습니다. 이상치 제거 옵션을 꺼보세요.")
    st.stop()

X = df_clean[x_cols].values.astype(float)
Y = df_clean[y_col].values.astype(float)
is_1d = len(x_cols) == 1

st.sidebar.header("✂️ 4. 학습/테스트 분할")
seed = int(st.sidebar.number_input("랜덤 시드:", value=42, step=1))
if len(df_clean) >= MIN_ROWS_FOR_SPLIT:
    test_pct = st.sidebar.slider("테스트 데이터 비율 (%):", 10, 50, 20, step=5)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_pct / 100, random_state=seed)
    has_test = True
    st.sidebar.caption(f"학습 {len(X_train)}행 / 테스트 {len(X_test)}행")
else:
    X_train, X_test, Y_train, Y_test = X, X, Y, Y
    has_test = False
    st.sidebar.warning(f"데이터가 {MIN_ROWS_FOR_SPLIT}행 미만이라 테스트 세트 없이 전체 데이터로 평가합니다.")
use_cv = st.sidebar.checkbox("5-겹 교차검증(CV) 함께 계산", value=True)
eval_name = "테스트" if has_test else "전체"

# =============================================================
# 3. 모델 선택 및 하이퍼파라미터
# =============================================================
ALL_MODELS = [
    "선형 회귀", "릿지 회귀 (Ridge)", "라쏘 회귀 (Lasso)", "다항 회귀",
    "결정 트리", "랜덤 포레스트", "그래디언트 부스팅",
    "서포트 벡터 회귀 (SVR)", "K-최근접 이웃 (KNN)", "인공신경망 (ANN)",
    "거듭제곱 회귀 (y=a·x^b)", "지수 회귀 (y=a·e^(bx))", "로그 회귀 (y=a+b·ln x)",
]
CURVE_KIND = {"거듭제곱 회귀 (y=a·x^b)": "power", "지수 회귀 (y=a·e^(bx))": "exp", "로그 회귀 (y=a+b·ln x)": "log"}

excluded = {}
for name in CURVE_KIND:
    if not is_1d:
        excluded[name] = "X가 1개일 때만 사용 가능"
if is_1d:
    if not (np.all(X > 0) and np.all(Y > 0)):
        excluded["거듭제곱 회귀 (y=a·x^b)"] = "X, Y가 모두 양수여야 함"
    if not np.all(Y > 0):
        excluded["지수 회귀 (y=a·e^(bx))"] = "Y가 모두 양수여야 함"
    if not np.all(X > 0):
        excluded["로그 회귀 (y=a+b·ln x)"] = "X가 모두 양수여야 함"
available = [m for m in ALL_MODELS if m not in excluded]

st.sidebar.header("🤖 5. 모델")
selected = st.sidebar.multiselect("학습할 모델:", available, default=available)
if excluded:
    st.sidebar.caption("제외된 모델: " + ", ".join(f"{k.split(' (')[0]}({v})" for k, v in excluded.items()))
if not selected:
    st.warning("학습할 모델을 1개 이상 선택해주세요.")
    st.stop()

cfg = {}
with st.sidebar.expander("⚙️ 하이퍼파라미터 설정"):
    if "릿지 회귀 (Ridge)" in selected:
        cfg["ridge_alpha"] = st.number_input("Ridge 규제 강도 α", 0.0001, 1000.0, 1.0, format="%.4f")
    if "라쏘 회귀 (Lasso)" in selected:
        cfg["lasso_alpha"] = st.number_input("Lasso 규제 강도 α", 0.0001, 1000.0, 0.1, format="%.4f")
    if "다항 회귀" in selected:
        cfg["poly_degree"] = st.slider("다항 회귀 차수", 2, 6, 2)
    if "결정 트리" in selected:
        cfg["tree_depth"] = st.slider("결정 트리 최대 깊이 (0=제한 없음)", 0, 20, 5)
    if "랜덤 포레스트" in selected:
        cfg["rf_trees"] = st.slider("랜덤 포레스트 트리 개수", 10, 500, 200, step=10)
        cfg["rf_depth"] = st.slider("랜덤 포레스트 최대 깊이 (0=제한 없음)", 0, 30, 0)
    if "그래디언트 부스팅" in selected:
        cfg["gb_trees"] = st.slider("부스팅 단계 수", 10, 500, 200, step=10)
        cfg["gb_lr"] = st.number_input("부스팅 학습률", 0.001, 1.0, 0.05, format="%.3f")
        cfg["gb_depth"] = st.slider("부스팅 트리 깊이", 1, 10, 3)
    if "서포트 벡터 회귀 (SVR)" in selected:
        cfg["svr_kernel"] = st.selectbox("SVR 커널", ["rbf", "linear", "poly"])
        cfg["svr_c"] = st.number_input("SVR C (규제)", 0.01, 1000.0, 10.0)
        cfg["svr_eps"] = st.number_input("SVR ε", 0.0, 5.0, 0.1)
    if "K-최근접 이웃 (KNN)" in selected:
        cfg["knn_k"] = st.slider("KNN 이웃 수 k", 1, 30, 5)
        cfg["knn_weights"] = st.selectbox("KNN 가중치", ["distance", "uniform"])
    if "인공신경망 (ANN)" in selected:
        n_layers = st.slider("ANN 은닉층 개수", 1, 3, 2)
        nodes = st.slider("ANN 첫 은닉층 노드 수 (다음 층은 절반)", 8, 256, 64, step=8)
        cfg["ann_layers"] = [max(4, nodes // (2 ** i)) for i in range(n_layers)]
        cfg["ann_act"] = st.selectbox("ANN 활성화 함수", ["relu", "tanh", "logistic"])
        cfg["ann_lr"] = st.number_input("ANN 학습률", 0.0001, 0.1, 0.001, format="%.4f")
        cfg["ann_alpha"] = st.number_input("ANN L2 규제 α", 0.0, 10.0, 0.0001, format="%.4f")
        cfg["ann_iter"] = st.slider("ANN 학습 횟수 (Epochs)", 200, 5000, 1000, step=100)

n_splits = min(5, len(X_train)) if use_cv and len(X_train) >= MIN_ROWS_FOR_CV else 0
# KNN 이웃 수는 교차검증 각 폴드의 학습 데이터 수보다 클 수 없으므로 미리 제한
cfg["n_fit_min"] = len(X_train) - int(np.ceil(len(X_train) / n_splits)) if n_splits else len(X_train)


def build_estimator(name, cfg, seed):
    if name == "선형 회귀":
        return LinearRegression()
    if name == "릿지 회귀 (Ridge)":
        return make_pipeline(StandardScaler(), Ridge(alpha=cfg["ridge_alpha"]))
    if name == "라쏘 회귀 (Lasso)":
        return make_pipeline(StandardScaler(), Lasso(alpha=cfg["lasso_alpha"], max_iter=50000))
    if name == "다항 회귀":
        return make_pipeline(PolynomialFeatures(cfg["poly_degree"], include_bias=False), StandardScaler(),
                             LinearRegression())
    if name == "결정 트리":
        return DecisionTreeRegressor(max_depth=cfg["tree_depth"] or None, random_state=seed)
    if name == "랜덤 포레스트":
        return RandomForestRegressor(n_estimators=cfg["rf_trees"], max_depth=cfg["rf_depth"] or None,
                                     random_state=seed, n_jobs=-1)
    if name == "그래디언트 부스팅":
        return GradientBoostingRegressor(n_estimators=cfg["gb_trees"], learning_rate=cfg["gb_lr"],
                                         max_depth=cfg["gb_depth"], random_state=seed)
    if name == "서포트 벡터 회귀 (SVR)":
        return TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), SVR(kernel=cfg["svr_kernel"], C=cfg["svr_c"],
                                                          epsilon=cfg["svr_eps"])),
            transformer=StandardScaler())
    if name == "K-최근접 이웃 (KNN)":
        k = max(1, min(cfg["knn_k"], cfg["n_fit_min"]))
        return make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=k, weights=cfg["knn_weights"]))
    if name == "인공신경망 (ANN)":
        return TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), MLPRegressor(
                hidden_layer_sizes=tuple(cfg["ann_layers"]), activation=cfg["ann_act"],
                learning_rate_init=cfg["ann_lr"], alpha=cfg["ann_alpha"], max_iter=cfg["ann_iter"],
                random_state=seed)),
            transformer=StandardScaler())
    return CurveFitRegressor(kind=CURVE_KIND[name])


@st.cache_resource(show_spinner="모델 학습 중...")
def train_all(X_train, Y_train, X_test, names, cfg_json, n_splits, seed):
    cfg = json.loads(cfg_json)
    out = {}
    for name in names:
        try:
            est = build_estimator(name, cfg, seed)
            t0 = time.perf_counter()
            est.fit(X_train, Y_train)
            fit_time = time.perf_counter() - t0
            res = {"model": est, "time": fit_time,
                   "pred_train": est.predict(X_train), "pred_test": est.predict(X_test), "cv_pred": None}
            if n_splits:
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
                res["cv_pred"] = cross_val_predict(build_estimator(name, cfg, seed), X_train, Y_train, cv=kf)
            out[name] = res
        except Exception as e:  # 한 모델이 실패해도 나머지는 계속 학습
            out[name] = {"error": str(e)}
    return out


trained = train_all(X_train, Y_train, X_test, tuple(selected), json.dumps(cfg, sort_keys=True), n_splits, seed)
failed = {k: v["error"] for k, v in trained.items() if "error" in v}
for k, v in failed.items():
    st.sidebar.error(f"{k} 학습 실패: {v}")
trained = {k: v for k, v in trained.items() if "error" not in v}
if not trained:
    st.error("학습에 성공한 모델이 없습니다.")
    st.stop()
model_names = list(trained)

rows = []
for name, r in trained.items():
    te, tr = safe_metrics(Y_test, r["pred_test"]), safe_metrics(Y_train, r["pred_train"])
    cv = safe_metrics(Y_train, r["cv_pred"]) if r["cv_pred"] is not None else safe_metrics([], [])
    rows.append({
        "모델": name,
        f"{eval_name} R²": te["R2"], f"{eval_name} RMSE": te["RMSE"], f"{eval_name} MAE": te["MAE"],
        f"{eval_name} MAPE(%)": te["MAPE"],
        "CV R²": cv["R2"], "CV RMSE": cv["RMSE"],
        "학습 R²": tr["R2"], "과적합 격차(학습-평가 R²)": tr["R2"] - te["R2"],
        "학습 시간(초)": r["time"],
    })
res_df = pd.DataFrame(rows).sort_values(by=f"{eval_name} R²", ascending=False, na_position="last").reset_index(drop=True)
if not use_cv or not n_splits:
    res_df = res_df.drop(columns=["CV R²", "CV RMSE"])
best_name = res_df.iloc[0]["모델"]


def predict_safe(name, Xn):
    with np.errstate(all="ignore"):
        p = np.asarray(trained[name]["model"].predict(np.asarray(Xn, float)), dtype=float).ravel()
    return np.where(np.isfinite(p), p, np.nan)


# =============================================================
# 4. 메인 화면
# =============================================================
tabs = st.tabs(["📋 데이터 탐색", "📊 성능 비교", "📉 잔차 진단", "🔍 모델 해석", "🔮 예측", "💾 내보내기"])

# --- TAB 1: 데이터 탐색 ---
with tabs[0]:
    st.subheader(f"📋 데이터 개요 — {data_name}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("전체 행", f"{len(df):,}")
    c2.metric("전체 열", f"{df.shape[1]:,}")
    c3.metric("숫자형 열", f"{len(num_cols):,}")
    c4.metric("분석 사용 행", f"{len(df_clean):,}")

    with st.expander("원본 데이터 미리보기", expanded=True):
        st.dataframe(df.head(100), width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown("**기술 통계량**")
        st.dataframe(df[num_cols].describe().T.style.format(precision=3), width="stretch")
    with right:
        st.markdown("**결측치 현황**")
        miss = pd.DataFrame({"결측 개수": df.isna().sum(), "결측 비율(%)": df.isna().mean() * 100})
        st.dataframe(miss.style.format({"결측 비율(%)": "{:.1f}"}), width="stretch")

    st.markdown("**상관계수 히트맵 (피어슨)**")
    corr = df[num_cols].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    st.plotly_chart(fig, width="stretch")
    y_corr = corr[y_col].drop(y_col).sort_values(key=np.abs, ascending=False)
    st.caption(f"{y_col}와 상관이 가장 큰 변수: " + ", ".join(f"{k} ({v:+.2f})" for k, v in y_corr.head(3).items()))

    left, right = st.columns(2)
    with left:
        hist_col = st.selectbox("분포를 볼 변수:", num_cols, index=num_cols.index(y_col))
        fig = px.histogram(df, x=hist_col, nbins=30, marginal="box")
        st.plotly_chart(fig, width="stretch")
    with right:
        sc_x = st.selectbox("산점도 X:", x_options, index=x_options.index(x_cols[0]))
        fig = px.scatter(df_clean.assign(**{sc_x: df.loc[df_clean.index, sc_x]}), x=sc_x, y=y_col,
                         trendline=None, opacity=0.7)
        st.plotly_chart(fig, width="stretch")

    if st.checkbox("산점도 행렬 보기 (선택한 X + Y, 최대 6개)"):
        mat_cols = (x_cols + [y_col])[-6:]
        fig = px.scatter_matrix(df_clean, dimensions=mat_cols, opacity=0.6)
        fig.update_traces(diagonal_visible=False, marker=dict(size=4))
        fig.update_layout(height=650)
        st.plotly_chart(fig, width="stretch")

# --- TAB 2: 성능 비교 ---
with tabs[1]:
    st.subheader("🏆 모델별 평가지표 순위")
    if has_test:
        st.caption("순위는 학습에 쓰지 않은 **테스트 데이터** 기준입니다. CV는 학습 데이터 안에서 5번 나눠 검증한 값으로, "
                   "데이터가 적을 때 더 안정적인 지표입니다. 과적합 격차가 크면 외워버린 모델일 수 있습니다.")
    else:
        st.caption("⚠️ 테스트 세트 없이 전체 데이터로 평가한 값이라 실제 성능보다 좋게 보일 수 있습니다. CV 값을 참고하세요.")

    best_r2 = res_df.iloc[0][f"{eval_name} R²"]
    st.success(f"🥇 최고 성능 모델: **{best_name}** ({eval_name} R² = {best_r2:.4f})")

    hi_max = [c for c in res_df.columns if c.endswith("R²") and c != "학습 R²"]
    hi_min = [c for c in res_df.columns if any(s in c for s in ("RMSE", "MAE", "MAPE"))]
    st.dataframe(
        res_df.style.format(precision=4)
        .highlight_max(subset=hi_max, color="#d4edda").highlight_min(subset=hi_min, color="#d4edda"),
        width="stretch",
    )

    bar_cols = [c for c in [f"{eval_name} R²", "CV R²", "학습 R²"] if c in res_df.columns]
    bar_df = res_df.melt(id_vars="모델", value_vars=bar_cols, var_name="지표", value_name="R²")
    fig = px.bar(bar_df, x="모델", y="R²", color="지표", barmode="group", title="모델별 R² 비교")
    fig.update_yaxes(range=[min(-0.5, np.nanmin(bar_df["R²"].values)), 1.05])
    st.plotly_chart(fig, width="stretch")

    if is_1d:
        st.subheader("📈 X–Y 적합 곡선")
        x_grid = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X_train.ravel(), y=Y_train, mode="markers", name="학습 데이터",
                                 marker=dict(color="black", size=7)))
        if has_test:
            fig.add_trace(go.Scatter(x=X_test.ravel(), y=Y_test, mode="markers", name="테스트 데이터",
                                     marker=dict(color="white", size=9, line=dict(color="black", width=2))))
        for i, name in enumerate(model_names):
            fig.add_trace(go.Scatter(x=x_grid.ravel(), y=predict_safe(name, x_grid), mode="lines", name=name,
                                     line=dict(color=PALETTE[i % len(PALETTE)], width=2.5 if name == best_name else 1.5)))
        fig.update_layout(xaxis_title=x_cols[0], yaxis_title=y_col, hovermode="x unified", height=550)
        st.plotly_chart(fig, width="stretch")
    elif len(x_cols) == 2:
        st.subheader("🗻 3D 예측 곡면")
        surf_model = st.selectbox("곡면을 그릴 모델:", model_names, index=model_names.index(best_name))
        g1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 40)
        g2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 40)
        G1, G2 = np.meshgrid(g1, g2)
        Z = predict_safe(surf_model, np.column_stack([G1.ravel(), G2.ravel()])).reshape(G1.shape)
        fig = go.Figure([
            go.Surface(x=G1, y=G2, z=Z, opacity=0.75, colorscale="Viridis", showscale=False),
            go.Scatter3d(x=X[:, 0], y=X[:, 1], z=Y, mode="markers", marker=dict(size=3, color="black"), name="실제 데이터"),
        ])
        fig.update_layout(scene=dict(xaxis_title=x_cols[0], yaxis_title=x_cols[1], zaxis_title=y_col), height=650)
        st.plotly_chart(fig, width="stretch")

    st.subheader(f"🎯 실제값 vs 예측값 ({eval_name} 데이터)")
    avp_models = st.multiselect("표시할 모델:", model_names, default=[best_name])
    fig = go.Figure()
    for i, name in enumerate(avp_models):
        fig.add_trace(go.Scatter(x=Y_test, y=trained[name]["pred_test"], mode="markers", name=name,
                                 marker=dict(size=8, color=PALETTE[model_names.index(name) % len(PALETTE)], opacity=0.75)))
    lo, hi = float(np.min(Y_test)), float(np.max(Y_test))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="완벽한 예측 (y=x)",
                             line=dict(color="red", dash="dash")))
    fig.update_layout(xaxis_title=f"실제 {y_col}", yaxis_title=f"예측 {y_col}", height=500)
    st.plotly_chart(fig, width="stretch")

# --- TAB 3: 잔차 진단 ---
with tabs[2]:
    st.subheader("🔬 잔차(Residual) 진단")
    st.markdown("잔차(실제값 − 예측값)가 0 주변에 **패턴 없이 고르게** 흩어지고, **정규분포**에 가까우면 좋은 모델입니다.")
    res_model = st.selectbox("진단할 모델:", model_names, index=model_names.index(best_name), key="res_model")
    pred = trained[res_model]["pred_test"]
    resid = Y_test - pred

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("잔차 평균", f"{np.mean(resid):.4f}")
    c2.metric("잔차 표준편차", f"{np.std(resid, ddof=1) if len(resid) > 1 else 0:.4f}")
    if 3 <= len(resid) <= 5000:
        sw_p = stats.shapiro(resid).pvalue
        c3.metric("정규성 검정 p값 (Shapiro)", f"{sw_p:.4f}", help="0.05보다 크면 정규분포를 따른다고 볼 수 있음")
    dw = np.sum(np.diff(resid) ** 2) / np.sum(resid ** 2) if np.sum(resid ** 2) > 0 else np.nan
    c4.metric("Durbin-Watson", f"{dw:.3f}", help="2에 가까우면 잔차끼리 독립적 (0~4)")

    left, right = st.columns(2)
    with left:
        fig = go.Figure(go.Scatter(x=pred, y=resid, mode="markers", marker=dict(color="purple", size=8)))
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(title="예측값 vs 잔차", xaxis_title="예측값", yaxis_title="잔차")
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = px.histogram(x=resid, nbins=min(30, max(5, len(resid) // 2)), title="잔차 분포",
                           labels={"x": "잔차"})
        st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        if len(resid) >= 3:
            (osm, osr), (slope, intercept, _) = stats.probplot(resid, dist="norm")
            fig = go.Figure([
                go.Scatter(x=osm, y=osr, mode="markers", name="잔차", marker=dict(size=7)),
                go.Scatter(x=osm, y=slope * np.asarray(osm) + intercept, mode="lines", name="정규분포 기준선",
                           line=dict(color="red", dash="dash")),
            ])
            fig.update_layout(title="정규 Q-Q 플롯", xaxis_title="이론적 분위수", yaxis_title="잔차 분위수")
            st.plotly_chart(fig, width="stretch")
    with right:
        res_x = st.selectbox("잔차와 비교할 X변수:", x_cols)
        fig = go.Figure(go.Scatter(x=X_test[:, x_cols.index(res_x)], y=resid, mode="markers",
                                   marker=dict(color="teal", size=8)))
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(title=f"{res_x} vs 잔차", xaxis_title=res_x, yaxis_title="잔차")
        st.plotly_chart(fig, width="stretch")

# --- TAB 4: 모델 해석 ---
with tabs[3]:
    st.subheader("📐 선형 회귀 계수 및 유의성 (학습 데이터, 최소제곱법)")
    A = np.column_stack([np.ones(len(X_train)), X_train])
    beta, *_ = np.linalg.lstsq(A, Y_train, rcond=None)
    dof = len(X_train) - A.shape[1]
    sse = float(np.sum((Y_train - A @ beta) ** 2))
    sst = float(np.sum((Y_train - Y_train.mean()) ** 2))
    coef_df = pd.DataFrame({"변수": ["절편"] + x_cols, "계수": beta})
    if dof > 0:
        sigma2 = sse / dof
        se = np.sqrt(np.clip(np.diag(sigma2 * np.linalg.pinv(A.T @ A)), 0, None))
        with np.errstate(divide="ignore", invalid="ignore"):
            t_vals = beta / se
        coef_df["표준오차"] = se
        coef_df["t값"] = t_vals
        coef_df["p값"] = 2 * stats.t.sf(np.abs(t_vals), dof)
        coef_df["유의(p<0.05)"] = np.where(coef_df["p값"] < 0.05, "✅", "")
    r2_ols = 1 - sse / sst if sst > 0 else np.nan
    adj_r2 = 1 - (1 - r2_ols) * (len(X_train) - 1) / dof if dof > 0 else np.nan
    st.dataframe(coef_df.style.format(precision=4, subset=[c for c in coef_df.columns if c not in ("변수", "유의(p<0.05)")]),
                 width="stretch")
    eq = f"{y_col} = {beta[0]:.4f} " + " ".join(f"{'+' if b >= 0 else '−'} {abs(b):.4f}·{c}" for b, c in zip(beta[1:], x_cols))
    st.code(eq, language=None)
    st.caption(f"학습 R² = {r2_ols:.4f} · 수정 R² = {adj_r2:.4f}" + ("" if dof > 0 else " · 데이터가 너무 적어 p값을 계산할 수 없습니다."))

    curve_models = [m for m in model_names if m in CURVE_KIND]
    if curve_models:
        st.markdown("**비선형 곡선 회귀식**")
        for m in curve_models:
            a, b = trained[m]["model"].params_
            kind = CURVE_KIND[m]
            formula = {"power": f"{y_col} = {a:.4f} · {x_cols[0]}^{b:.4f}",
                       "exp": f"{y_col} = {a:.4f} · e^({b:.4f}·{x_cols[0]})",
                       "log": f"{y_col} = {a:.4f} + {b:.4f} · ln({x_cols[0]})"}[kind]
            st.code(f"[{m.split(' (')[0]}] {formula}", language=None)

    st.divider()
    st.subheader("🔑 변수 중요도 (순열 중요도)")
    st.caption(f"변수 하나의 값을 무작위로 섞었을 때 {eval_name} R²가 얼마나 떨어지는지로 중요도를 잽니다. 모든 모델에 쓸 수 있습니다.")
    imp_model = st.selectbox("해석할 모델:", model_names, index=model_names.index(best_name), key="imp_model")

    @st.cache_resource(show_spinner="변수 중요도 계산 중...")
    def compute_importance(name, _model, X_eval, Y_eval, cfg_json, seed):
        r = permutation_importance(_model, X_eval, Y_eval, n_repeats=15, random_state=seed, scoring="r2")
        return r.importances_mean, r.importances_std

    X_eval, Y_eval = (X_test, Y_test) if has_test and len(X_test) >= 2 else (X_train, Y_train)
    try:
        imp_mean, imp_std = compute_importance(imp_model, trained[imp_model]["model"], X_eval, Y_eval,
                                               json.dumps(cfg, sort_keys=True), seed)
        imp_df = pd.DataFrame({"변수": x_cols, "중요도": imp_mean, "표준편차": imp_std}).sort_values("중요도")
        fig = px.bar(imp_df, x="중요도", y="변수", error_x="표준편차", orientation="h",
                     title=f"{imp_model} — 순열 중요도 (R² 감소량)")
        st.plotly_chart(fig, width="stretch")
    except Exception as e:
        st.info(f"변수 중요도를 계산할 수 없습니다: {e}")

    est = trained[imp_model]["model"]
    if hasattr(est, "feature_importances_"):
        fi = pd.DataFrame({"변수": x_cols, "중요도": est.feature_importances_}).sort_values("중요도")
        fig = px.bar(fi, x="중요도", y="변수", orientation="h", title=f"{imp_model} — 트리 기반 불순도 중요도")
        st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("📈 부분 의존도 (변수 하나만 바꿨을 때 예측 변화)")
    pd_var = st.selectbox("변화시킬 변수:", x_cols, key="pd_var")
    j = x_cols.index(pd_var)
    grid = np.linspace(X[:, j].min(), X[:, j].max(), 50)
    pd_models = st.multiselect("비교할 모델:", model_names, default=[best_name], key="pd_models")
    fig = go.Figure()
    base = X_train.copy()
    for name in pd_models:
        curve = []
        for v in grid:
            tmp = base.copy()
            tmp[:, j] = v
            curve.append(np.nanmean(predict_safe(name, tmp)))
        fig.add_trace(go.Scatter(x=grid, y=curve, mode="lines", name=name,
                                 line=dict(color=PALETTE[model_names.index(name) % len(PALETTE)])))
    fig.update_layout(xaxis_title=pd_var, yaxis_title=f"평균 예측 {y_col}", height=450)
    st.plotly_chart(fig, width="stretch")
    st.caption("다른 변수는 학습 데이터 값을 그대로 둔 채, 선택한 변수만 바꿔가며 평균 예측값을 그렸습니다.")

    if "인공신경망 (ANN)" in trained:
        st.divider()
        st.subheader("🧠 인공신경망 학습 곡선")
        mlp = trained["인공신경망 (ANN)"]["model"].regressor_[-1]
        fig = go.Figure(go.Scatter(y=mlp.loss_curve_, mode="lines"))
        fig.update_layout(xaxis_title="Epoch", yaxis_title="손실 (Loss)", height=350)
        st.plotly_chart(fig, width="stretch")
        st.caption(f"총 {mlp.n_iter_}회 반복 후 종료" + (" (최대 반복 도달 — Epoch를 늘려보세요)" if mlp.n_iter_ >= cfg["ann_iter"] else " (수렴)"))

# --- TAB 5: 예측 ---
with tabs[4]:
    st.subheader("🔮 새로운 값 입력 → 실시간 예측")
    cols = st.columns(min(len(x_cols), 4))
    input_vals = []
    for i, col in enumerate(x_cols):
        with cols[i % len(cols)]:
            input_vals.append(st.number_input(f"{col}", value=float(np.mean(X[:, i])), key=f"in_{col}",
                                              help=f"학습 범위: {X[:, i].min():.4g} ~ {X[:, i].max():.4g}"))
    new_X = np.array([input_vals], dtype=float)
    if np.any(new_X < X.min(axis=0)) or np.any(new_X > X.max(axis=0)):
        st.warning("⚠️ 입력값이 학습 데이터 범위를 벗어났습니다. 외삽 예측은 신뢰도가 낮습니다.")

    preds = pd.DataFrame({"모델": model_names, f"예측 {y_col}": [predict_safe(m, new_X)[0] for m in model_names]})
    preds = preds.set_index("모델").loc[res_df["모델"]].reset_index()  # 성능 순으로 정렬
    valid = preds[f"예측 {y_col}"].dropna()
    top3 = preds.dropna().head(3)[f"예측 {y_col}"]

    c1, c2, c3 = st.columns(3)
    c1.metric(f"🥇 최고 모델 ({best_name.split(' (')[0]})", f"{preds.iloc[0][f'예측 {y_col}']:.4f}")
    if len(top3):
        c2.metric("상위 3개 모델 평균", f"{top3.mean():.4f}")
    if len(valid):
        c3.metric("전체 모델 예측 범위", f"{valid.min():.3f} ~ {valid.max():.3f}")

    left, right = st.columns([1, 1])
    with left:
        st.dataframe(preds.style.format(precision=4, na_rep="계산 불가"), width="stretch")
    with right:
        fig = px.bar(preds.dropna(), x=f"예측 {y_col}", y="모델", orientation="h")
        fig.update_layout(yaxis=dict(autorange="reversed"), height=max(300, 30 * len(preds)))
        st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("📦 파일로 일괄 예측")
    st.caption(f"다음 열이 들어있는 CSV/Excel 파일을 올리면 모든 행을 한 번에 예측합니다: {', '.join(x_cols)}")
    batch_file = st.file_uploader("예측할 데이터 파일", type=["csv", "xlsx", "xls"], key="batch")
    if batch_file is not None:
        try:
            bdf = read_uploaded(batch_file.name, batch_file.getvalue(), None)
            missing = [c for c in x_cols if c not in bdf.columns]
            if missing:
                st.error(f"파일에 다음 열이 없습니다: {', '.join(missing)}")
            else:
                bX = bdf[x_cols].apply(pd.to_numeric, errors="coerce")
                ok = bX.notna().all(axis=1).values
                out = bdf.copy()
                for m in model_names:
                    p = np.full(len(bdf), np.nan)
                    if ok.any():
                        p[ok] = predict_safe(m, bX.values[ok])
                    out[f"예측_{m}"] = p
                st.dataframe(out.head(200), width="stretch")
                if (~ok).any():
                    st.warning(f"X값이 비어있거나 숫자가 아닌 {(~ok).sum()}행은 예측하지 않았습니다.")
                st.download_button("⬇️ 예측 결과 CSV 다운로드", to_csv_bytes(out), "batch_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"파일을 처리하지 못했습니다: {e}")

# --- TAB 6: 내보내기 ---
with tabs[5]:
    st.subheader("💾 결과 내보내기")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**평가지표 표**")
        st.download_button("⬇️ 평가지표 CSV", to_csv_bytes(res_df), "model_metrics.csv", "text/csv")
        st.markdown(f"**{eval_name} 데이터 예측값**")
        pred_out = pd.DataFrame(X_test, columns=x_cols)
        pred_out[f"실제 {y_col}"] = Y_test
        for m in model_names:
            pred_out[f"예측_{m}"] = trained[m]["pred_test"]
        st.download_button("⬇️ 예측값 CSV", to_csv_bytes(pred_out), "test_predictions.csv", "text/csv")
        st.markdown("**전처리된 분석 데이터**")
        st.download_button("⬇️ 분석 데이터 CSV", to_csv_bytes(df_clean), "clean_data.csv", "text/csv")
    with c2:
        st.markdown("**학습된 모델 파일 (pickle)**")
        dl_model = st.selectbox("저장할 모델:", model_names, index=model_names.index(best_name), key="dl_model")
        bundle = {"model": trained[dl_model]["model"], "x_cols": x_cols, "y_col": y_col, "settings": cfg}
        st.download_button("⬇️ 모델 다운로드 (.pkl)", pickle.dumps(bundle), "model.pkl", "application/octet-stream")
        st.caption("파이썬에서 `pickle.load(open('model.pkl','rb'))['model'].predict(X)`로 다시 쓸 수 있습니다. "
                   "곡선 회귀 모델은 불러올 때 이 app.py의 CurveFitRegressor 정의가 필요합니다.")
        st.markdown("**분석 설정 요약**")
        st.json({"데이터": data_name, "X": x_cols, "Y": y_col, "이상치 제거": outlier_method,
                 "테스트 분할": has_test, "랜덤 시드": seed, "교차검증 폴드": n_splits,
                 "모델": model_names, "하이퍼파라미터": cfg})
