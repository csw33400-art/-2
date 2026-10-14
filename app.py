import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

st.set_page_config(page_title="통합 데이터 분석 대시보드", layout="wide")
st.title("🚀 범용 회귀 분석 및 인공신경망(ANN) 예측 대시보드")
st.markdown("임의의 데이터를 업로드하여 **전통적 회귀 모델**과 **다변수 인공신경망**의 성능을 실시간으로 비교하고 예측합니다.")

MIN_ROWS_FOR_SPLIT = 8  # 이보다 적으면 테스트 세트를 나누지 않고 전체 데이터로 평가


def metrics(y_true, y_pred):
    return {
        "R²": r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan,
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


# -------------------------------------------------------------
# 1. 데이터 로더 모듈 (파일 업로드 및 전처리)
# -------------------------------------------------------------
st.sidebar.header("📁 1. 데이터 업로드 및 변수 선택")
uploaded_file = st.sidebar.file_uploader("CSV 또는 Excel 파일 업로드", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"파일을 읽지 못했습니다: {e}")
        st.stop()
else:
    st.sidebar.info("기본 샘플 데이터(콘크리트 압축강도)를 불러옵니다.")
    data = {
        'Cement': [540, 540, 332.5, 332.5, 198.6, 266, 380, 380, 266, 475],
        'Water': [162, 162, 228, 228, 192, 228, 228, 228, 228, 228],
        'Age': [28, 28, 270, 365, 360, 90, 365, 28, 28, 28],
        'Strength': [79.99, 61.89, 40.27, 41.05, 44.30, 47.03, 43.70, 36.45, 45.85, 39.29]
    }
    df = pd.DataFrame(data)

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(num_cols) < 2:
    st.error("숫자형 열이 2개 이상 필요합니다. (X 1개 이상 + Y 1개)")
    st.stop()

# Y를 먼저 고르고, X 후보에서는 Y를 제외해 같은 열이 중복 선택되지 않게 함
y_col = st.sidebar.selectbox("종속변수 (Y축):", num_cols, index=len(num_cols) - 1)
x_options = [c for c in num_cols if c != y_col]
x_cols = st.sidebar.multiselect("독립변수 (X축, 다중 선택 가능):", x_options, default=[x_options[0]])

if not x_cols:
    st.warning("분석을 위해 최소 1개의 X변수를 선택해주세요.")
    st.stop()

df_clean = df[x_cols + [y_col]].dropna()
if len(df_clean) < 3:
    st.error("결측치를 제외한 데이터가 3행 미만이라 분석할 수 없습니다.")
    st.stop()

X = df_clean[x_cols].values.astype(float)
Y = df_clean[y_col].values.astype(float)
is_1d = len(x_cols) == 1

# 학습/테스트 분할
st.sidebar.markdown("---")
st.sidebar.header("✂️ 학습/테스트 분할")
if len(df_clean) >= MIN_ROWS_FOR_SPLIT:
    test_pct = st.sidebar.slider("테스트 데이터 비율 (%):", 10, 50, 20, step=5)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_pct / 100, random_state=42)
    has_test = True
    st.sidebar.caption(f"학습 {len(X_train)}행 / 테스트 {len(X_test)}행")
else:
    X_train, X_test, Y_train, Y_test = X, X, Y, Y
    has_test = False
    st.sidebar.warning(f"데이터가 {MIN_ROWS_FOR_SPLIT}행 미만이라 테스트 세트를 나누지 않고 전체 데이터로 평가합니다.")

# -------------------------------------------------------------
# 2. 모델링 및 평가지표 계산 모듈
#    각 모델은 학습 데이터로만 학습하고, 예측 함수(predict)를 함께 저장
# -------------------------------------------------------------
models = []  # {"모델": 이름, "predict": callable(X 2D) -> 1D}

# [1] 선형 회귀 (Linear Regression)
lin_model = LinearRegression().fit(X_train, Y_train)
models.append({"모델": "선형 회귀 (Linear)", "predict": lin_model.predict})

# [2] 비선형 회귀 (1변수 입력일 경우에만 곡선 적합)
if is_1d:
    x_tr = X_train.flatten()

    if len(np.unique(x_tr)) >= 3:
        p2 = np.polyfit(x_tr, Y_train, 2)
        models.append({"모델": "2차 다항 회귀", "predict": lambda Xn, p=p2: np.polyval(p, Xn.flatten())})
    else:
        st.sidebar.warning("서로 다른 X값이 3개 미만이라 2차 다항 회귀 제외됨")

    if np.all(X > 0) and np.all(Y > 0):
        def power_func(x, a, b):
            return a * np.power(x, b)
        try:
            popt_pow, _ = curve_fit(power_func, x_tr, Y_train, maxfev=10000)
            models.append({"모델": "거듭제곱 회귀 (Power)",
                           "predict": lambda Xn, p=popt_pow: power_func(Xn.flatten(), *p)})
        except (RuntimeError, ValueError, OverflowError):
            st.sidebar.warning("거듭제곱 모델 수렴 실패로 제외됨")
    else:
        st.sidebar.warning("데이터에 0 또는 음수가 포함되어 거듭제곱 회귀 제외됨")

# [3] 인공신경망 (ANN - 다변수 지원)
st.sidebar.markdown("---")
st.sidebar.header("🧠 ANN 하이퍼파라미터")
hidden_layer = st.sidebar.slider("은닉층 노드 수:", 16, 128, 64, step=16)
epochs = st.sidebar.slider("학습 횟수 (Epochs):", 500, 2000, 1000, step=100)

scaler_X = StandardScaler().fit(X_train)
scaler_Y = StandardScaler().fit(Y_train.reshape(-1, 1))
ann_model = MLPRegressor(hidden_layer_sizes=(hidden_layer, hidden_layer // 2), activation='relu',
                         max_iter=epochs, random_state=42)
ann_model.fit(scaler_X.transform(X_train), scaler_Y.transform(Y_train.reshape(-1, 1)).ravel())


def ann_predict(Xn):
    return scaler_Y.inverse_transform(ann_model.predict(scaler_X.transform(Xn)).reshape(-1, 1)).ravel()


models.append({"모델": "인공신경망 (ANN)", "predict": ann_predict})

# 평가지표 계산 (학습/테스트 각각)
results = []
for m in models:
    pred_train = m["predict"](X_train)
    pred_test = m["predict"](X_test)
    tr, te = metrics(Y_train, pred_train), metrics(Y_test, pred_test)
    results.append({
        "모델": m["모델"],
        "테스트 R²": te["R²"], "테스트 RMSE": te["RMSE"], "테스트 MAE": te["MAE"],
        "학습 R²": tr["R²"],
        "예측값_test": pred_test,
    })

# -------------------------------------------------------------
# 3. 메인 화면 UI 및 탭(Tab) 구성
# -------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 성능 종합 비교", "📉 잔차(Residual) 분석", "🔮 실시간 인터랙티브 예측"])

# --- TAB 1: 성능 종합 비교 ---
with tab1:
    st.subheader("🏆 모델별 평가지표 순위")
    if has_test:
        st.caption("순위는 학습에 쓰지 않은 **테스트 데이터** 기준입니다. 학습 R²가 테스트 R²보다 크게 높으면 과적합을 의심하세요.")
    else:
        st.caption("⚠️ 테스트 세트 없이 전체 데이터로 평가한 값이라 실제 성능보다 좋게 보일 수 있습니다.")
    res_df = (pd.DataFrame(results).drop(columns=['예측값_test'])
              .sort_values(by="테스트 R²", ascending=False).reset_index(drop=True))
    st.dataframe(
        res_df.style.format(precision=4)
        .highlight_max(subset=['테스트 R²'], color='#d4edda')
        .highlight_min(subset=['테스트 RMSE', '테스트 MAE'], color='#d4edda'),
        use_container_width=True,
    )

    if is_1d:
        st.subheader("📈 1D 데이터 시각화 (X vs Y)")
        x_grid = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X_train.flatten(), y=Y_train, mode='markers', name='학습 데이터',
                                 marker=dict(color='black', size=8)))
        if has_test:
            fig.add_trace(go.Scatter(x=X_test.flatten(), y=Y_test, mode='markers', name='테스트 데이터',
                                     marker=dict(color='white', size=9, line=dict(color='black', width=2))))
        colors = ['red', 'blue', 'green', 'orange']
        for i, m in enumerate(models):
            fig.add_trace(go.Scatter(x=x_grid.flatten(), y=m["predict"](x_grid), mode='lines', name=m['모델'],
                                     line=dict(color=colors[i % len(colors)])))
        fig.update_layout(xaxis_title=x_cols[0], yaxis_title=y_col, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("다변수(X가 2개 이상) 환경이므로, 테스트 데이터의 [실제값 vs ANN 예측값] 산점도를 출력합니다.")
        y_pred_ann_test = next(r['예측값_test'] for r in results if r['모델'] == "인공신경망 (ANN)")
        fig = px.scatter(x=Y_test, y=y_pred_ann_test, labels={'x': f'실제 {y_col}', 'y': f'ANN 예측 {y_col}'},
                         title="다변수 ANN 모델 적합도")
        lo, hi = min(Y_test.min(), y_pred_ann_test.min()), max(Y_test.max(), y_pred_ann_test.max())
        fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(color="red", dash="dash"))
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: 잔차 분석 ---
with tab2:
    st.subheader("🔬 잔차(Residual) 분석 그래프")
    st.markdown("잔차(실제값 - 예측값)가 0을 중심으로 특정 패턴(U자형 등)을 보이면, 해당 모델이 데이터의 비선형성을 설명하지 못하고 있음을 의미합니다.")

    selected_model = st.selectbox("잔차를 확인할 모델을 선택하세요:", [r['모델'] for r in results])
    y_pred_target = next(r['예측값_test'] for r in results if r['모델'] == selected_model)
    residuals = Y_test - y_pred_target

    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(x=y_pred_target, y=residuals, mode='markers', marker=dict(color='purple', size=8)))
    fig_res.add_hline(y=0, line_dash="dash", line_color="red")
    fig_res.update_layout(title=f"{selected_model} 잔차 분포 ({'테스트' if has_test else '전체'} 데이터)",
                          xaxis_title="예측값 (Predicted)", yaxis_title="잔차 (Residuals)")
    st.plotly_chart(fig_res, use_container_width=True)

# --- TAB 3: 실시간 예측 (Inference) ---
with tab3:
    st.subheader("🔮 새로운 데이터 입력 및 실시간 Y 예측")
    cols = st.columns(len(x_cols))
    input_vals = []
    for i, col in enumerate(x_cols):
        with cols[i]:
            val = st.number_input(f"입력: {col}", value=float(X[:, i].mean()))
            input_vals.append(val)

    new_X = np.array([input_vals], dtype=float)
    if np.any(new_X < X.min(axis=0)) or np.any(new_X > X.max(axis=0)):
        st.warning("입력값이 학습 데이터 범위를 벗어났습니다. 외삽 예측은 신뢰도가 낮습니다.")

    metric_cols = st.columns(len(models))
    for mc, m in zip(metric_cols, models):
        try:
            pred = m["predict"](new_X)[0]
            value = f"{pred:.4f}" if np.isfinite(pred) else "계산 불가"
        except (ValueError, FloatingPointError):
            value = "계산 불가"
        mc.metric(label=m["모델"], value=value)
