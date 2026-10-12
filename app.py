import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

st.set_page_config(page_title="통합 데이터 분석 대시보드", layout="wide")
st.title("🚀 범용 회귀 분석 및 인공신경망(ANN) 예측 대시보드")
st.markdown("임의의 데이터를 업로드하여 **전통적 회귀 모델**과 **다변수 인공신경망**의 성능을 실시간으로 비교하고 예측합니다.")

# -------------------------------------------------------------
# 1. 데이터 로더 모듈 (파일 업로드 및 전처리)
# -------------------------------------------------------------
st.sidebar.header("📁 1. 데이터 업로드 및 변수 선택")
uploaded_file = st.sidebar.file_uploader("CSV 또는 Excel 파일 업로드", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
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
x_cols = st.sidebar.multiselect("독립변수 (X축, 다중 선택 가능):", num_cols, default=[num_cols[0]])
y_col = st.sidebar.selectbox("종속변수 (Y축):", num_cols, index=len(num_cols)-1)

if not x_cols or not y_col:
    st.warning("분석을 위해 최소 1개의 X변수와 1개의 Y변수를 선택해주세요.")
    st.stop()

df_clean = df[x_cols + [y_col]].dropna()
X = df_clean[x_cols].values
Y = df_clean[y_col].values

# -------------------------------------------------------------
# 2. 모델링 및 평가지표 계산 모듈
# -------------------------------------------------------------
results = []

# [1] 선형 회귀 (Linear Regression)
lin_model = LinearRegression()
lin_model.fit(X, Y)
Y_pred_lin = lin_model.predict(X)
results.append({
    "모델": "선형 회귀 (Linear)",
    "R²": r2_score(Y, Y_pred_lin),
    "RMSE": np.sqrt(mean_squared_error(Y, Y_pred_lin)),
    "예측값": Y_pred_lin
})

# [2] 비선형 회귀 (1변수 입력일 경우에만 곡선 적합 및 예외 처리 수행)
if len(x_cols) == 1:
    X_1d = X.flatten()
    
    # 2차 다항 회귀
    p2 = np.polyfit(X_1d, Y, 2)
    Y_pred_poly = np.polyval(p2, X_1d)
    results.append({"모델": "2차 다항 회귀", "R²": r2_score(Y, Y_pred_poly), "RMSE": np.sqrt(mean_squared_error(Y, Y_pred_poly)), "예측값": Y_pred_poly})
    
    # 거듭제곱 회귀 (y = a*x^b) - 예외 처리 포함
    if np.all(X_1d > 0) and np.all(Y > 0):
        try:
            def power_func(x, a, b): return a * (x ** b)
            popt_pow, _ = curve_fit(power_func, X_1d, Y, maxfev=10000)
            Y_pred_pow = power_func(X_1d, *popt_pow)
            results.append({"모델": "거듭제곱 회귀 (Power)", "R²": r2_score(Y, Y_pred_pow), "RMSE": np.sqrt(mean_squared_error(Y, Y_pred_pow)), "예측값": Y_pred_pow})
        except:
            st.sidebar.warning("거듭제곱 모델 수렴 실패로 제외됨")
    else:
        st.sidebar.warning("데이터에 0 또는 음수가 포함되어 거듭제곱 회귀 제외됨 (예외 처리 완료)")

# [3] 인공신경망 (ANN - 다변수 완벽 지원)
st.sidebar.markdown("---")
st.sidebar.header("🧠 ANN 하이퍼파라미터")
hidden_layer = st.sidebar.slider("은닉층 노드 수:", 16, 128, 64, step=16)
epochs = st.sidebar.slider("학습 횟수 (Epochs):", 500, 2000, 1000, step=100)

scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y.reshape(-1, 1)).flatten()

ann_model = MLPRegressor(hidden_layer_sizes=(hidden_layer, int(hidden_layer/2)), activation='relu', max_iter=epochs, random_state=42)
ann_model.fit(X_scaled, Y_scaled)
Y_pred_ann = scaler_Y.inverse_transform(ann_model.predict(X_scaled).reshape(-1, 1)).flatten()

results.append({
    "모델": "인공신경망 (ANN)",
    "R²": r2_score(Y, Y_pred_ann),
    "RMSE": np.sqrt(mean_squared_error(Y, Y_pred_ann)),
    "예측값": Y_pred_ann
})

# -------------------------------------------------------------
# 3. 메인 화면 UI 및 탭(Tab) 구성
# -------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 성능 종합 비교", "📉 잔차(Residual) 분석", "🔮 실시간 인터랙티브 예측"])

# --- TAB 1: 성능 종합 비교 ---
with tab1:
    st.subheader("🏆 모델별 평가지표 순위")
    res_df = pd.DataFrame(results).drop(columns=['예측값']).sort_values(by="R²", ascending=False).reset_index(drop=True)
    st.dataframe(res_df.style.highlight_max(subset=['R²'], color='#d4edda').highlight_min(subset=['RMSE'], color='#d4edda'), use_container_width=True)

    if len(x_cols) == 1:
        st.subheader("📈 1D 데이터 시각화 (X vs Y)")
        X_sort_idx = np.argsort(X_1d)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X_1d[X_sort_idx], y=Y[X_sort_idx], mode='markers', name='실제 데이터', marker=dict(color='black', size=8)))
        
        colors = ['red', 'blue', 'green', 'orange']
        for i, res in enumerate(results):
            fig.add_trace(go.Scatter(x=X_1d[X_sort_idx], y=res['예측값'][X_sort_idx], mode='lines', name=res['모델'], line=dict(color=colors[i % len(colors)])))
        fig.update_layout(xaxis_title=x_cols[0], yaxis_title=y_col, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("다변수(X가 2개 이상) 환경이므로, [실제값 vs ANN 예측값] 산점도를 출력합니다.")
        fig = px.scatter(x=Y, y=Y_pred_ann, labels={'x': f'실제 {y_col}', 'y': f'ANN 예측 {y_col}'}, title="다변수 ANN 모델 적합도")
        fig.add_shape(type="line", x0=min(Y), y0=min(Y), x1=max(Y), y1=max(Y), line=dict(color="red", dash="dash"))
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: 잔차 분석 (과적합 및 선형 한계 검증) ---
with tab2:
    st.subheader("🔬 잔차(Residual) 분석 그래프")
    st.markdown("잔차(실제값 - 예측값)가 0을 중심으로 특정 패턴(U자형 등)을 보이면, 해당 모델이 데이터의 비선형성을 설명하지 못하고 있음을 의미합니다.")
    
    selected_model = st.selectbox("잔차를 확인할 모델을 선택하세요:", [r['모델'] for r in results])
    y_pred_target = next(r['예측값'] for r in results if r['모델'] == selected_model)
    residuals = Y - y_pred_target
    
    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(x=y_pred_target, y=residuals, mode='markers', marker=dict(color='purple', size=8)))
    fig_res.add_hline(y=0, line_dash="dash", line_color="red")
    fig_res.update_layout(title=f"{selected_model} 잔차 분포", xaxis_title="예측값 (Predicted)", yaxis_title="잔차 (Residuals)")
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
            
    new_X = np.array([input_vals])
    
    # 모델별 예측 연산
    new_Y_lin = lin_model.predict(new_X)[0]
    new_X_scaled = scaler_X.transform(new_X)
    new_Y_ann = scaler_Y.inverse_transform(ann_model.predict(new_X_scaled).reshape(-1, 1))[0][0]
    
    col_a, col_b = st.columns(2)
    col_a.metric(label="📊 선형 회귀 예측결과", value=f"{new_Y_lin:.4f}")
    col_b.metric(label="🧠 인공신경망(ANN) 예측결과", value=f"{new_Y_ann:.4f}")
