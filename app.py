import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

st.set_page_config(page_title="인공신경망(ANN) 기반 데이터 예측 프로그램", layout="wide")
st.title("🧠 인공신경망(ANN) 및 회귀 분석 통합 예측 대시보드")
st.write("공개 데이터(CSV/Excel)를 업로드하고, 인공신경망(ANN)을 활용하여 데이터를 학습 및 예측해보세요.")

# -------------------------------------------------------------
# 1. 엑셀 / CSV 파일 업로드
# -------------------------------------------------------------
st.sidebar.header("📁 1. 데이터 파일 업로드")
uploaded_file = st.sidebar.file_uploader("CSV 또는 Excel 파일을 업로드하세요", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.sidebar.success("성공적으로 데이터를 읽었습니다!")
    except Exception as e:
        st.sidebar.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        st.stop()
else:
    st.sidebar.info("기본 샘플 데이터셋을 사용 중입니다.")
    # 기본 다변수 샘플 데이터셋 생성
    data = {
        'Speed': [20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
        'Weight': [1000, 1100, 1050, 1200, 1250, 1300, 1350, 1400, 1450, 1500],
        'BrakingDistance': [2.1, 4.8, 8.5, 13.3, 19.2, 26.1, 34.2, 43.3, 53.5, 64.8]
    }
    df = pd.DataFrame(data)

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(num_cols) < 2:
    st.error("분석을 위해 최소 2개 이상의 숫자형 데이터 열이 필요합니다.")
    st.stop()

# -------------------------------------------------------------
# 2. 변수 지정 (독립변수 X 다중 선택 가능)
# -------------------------------------------------------------
st.sidebar.header("🎯 2. 변수 선택")
x_cols = st.sidebar.multiselect("독립변수 (X축, 다중 선택 가능):", num_cols, default=[num_cols[0]])
y_col = st.sidebar.selectbox("종속변수 (Y축):", num_cols, index=1 if len(num_cols) > 1 else 0)

if not x_cols:
    st.warning("최소 1개 이상의 독립변수를 선택해야 합니다.")
    st.stop()

df_clean = df[x_cols + [y_col]].dropna()
X = df_clean[x_cols].values
Y = df_clean[y_col].values

# -------------------------------------------------------------
# 3. 탭 구성 (원본 데이터, ANN 학습/예측, 모델 비교)
# -------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 데이터 원본", "🧠 인공신경망(ANN) 학습 및 예측", "📈 모델 비교 (회귀 vs ANN)"])

with tab1:
    st.subheader("업로드된 데이터셋")
    st.dataframe(df_clean)

with tab2:
    st.subheader("🧠 인공신경망 (ANN / MLPRegressor) 설정 및 학습")
    
    col_a, col_b = st.columns(2)
    with col_a:
        hidden_layer_1 = st.slider("첫 번째 은닉층 노드 수:", 8, 128, 64, step=8)
        hidden_layer_2 = st.slider("두 번째 은닉층 노드 수:", 0, 64, 32, step=8)
    with col_b:
        max_iter = st.slider("최대 학습 횟수 (Epochs):", 200, 3000, 1000, step=100)
        activation = st.selectbox("활성화 함수 (Activation Function):", ["relu", "tanh", "logistic"])

    hidden_structure = (hidden_layer_1,) if hidden_layer_2 == 0 else (hidden_layer_1, hidden_layer_2)

    # 데이터 스케일링 (ANN 경사하강법 수렴 필수)
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    Y_scaled = scaler_Y.fit_transform(Y.reshape(-1, 1)).flatten()

    # ANN 모델 학습
    ann_model = MLPRegressor(
        hidden_layer_sizes=hidden_structure,
        activation=activation,
        max_iter=max_iter,
        random_state=42
    )
    ann_model.fit(X_scaled, Y_scaled)

    # 예측 및 역스케일링
    Y_pred_ann_scaled = ann_model.predict(X_scaled)
    Y_pred_ann = scaler_Y.inverse_transform(Y_pred_ann_scaled.reshape(-1, 1)).flatten()

    r2_ann = r2_score(Y, Y_pred_ann)
    rmse_ann = np.sqrt(mean_squared_error(Y, Y_pred_ann))
    mae_ann = mean_absolute_error(Y, Y_pred_ann)

    st.success(f"✅ ANN 학습 완료! | **R² (결정계수):** {r2_ann:.4f} | **RMSE:** {rmse_ann:.4f} | **MAE:** {mae_ann:.4f}")

    # 사용자 실시간 예측 인터페이스
    st.markdown("---")
    st.subheader("🔮 새로운 입력값으로 Y 예측하기")
    input_vals = []
    col_inputs = st.columns(len(x_cols))
    for idx, col_name in enumerate(x_cols):
        with col_inputs[idx]:
            val = st.number_input(f"입력 {col_name}:", value=float(X[:, idx].mean()))
            input_vals.append(val)

    new_X = np.array([input_vals])
    new_X_scaled = scaler_X.transform(new_X)
    new_Y_scaled = ann_model.predict(new_X_scaled)
    new_Y_pred = scaler_Y.inverse_transform(new_Y_scaled.reshape(-1, 1))[0][0]

    st.metric(label=f"🧠 ANN 예측 {y_col} 결과", value=f"{new_Y_pred:.4f}")

with tab3:
    st.subheader("📉 전통적 회귀 모델 vs 인공신경망(ANN) 성능 비교")
    
    if len(x_cols) == 1:
        X_1d = X.flatten()
        lin = LinearRegression().fit(X, Y)
        pred_lin = lin.predict(X)
        
        p2 = np.polyfit(X_1d, Y, 2)
        pred_p2 = np.polyval(p2, X_1d)

        comparison_df = pd.DataFrame([
            {"모델": "선형 회귀", "R²": r2_score(Y, pred_lin), "RMSE": np.sqrt(mean_squared_error(Y, pred_lin))},
            {"모델": "2차 다항 회귀", "R²": r2_score(Y, pred_p2), "RMSE": np.sqrt(mean_squared_error(Y, pred_p2))},
            {"모델": "인공신경망 (ANN)", "R²": r2_ann, "RMSE": rmse_ann}
        ]).sort_values(by="R²", ascending=False)

        st.dataframe(comparison_df.style.highlight_max(subset=['R²'], color='lightgreen'))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X_1d, y=Y, mode='markers', name='실제 데이터', marker=dict(color='black', size=10)))
        fig.add_trace(go.Scatter(x=X_1d, y=pred_lin, mode='lines', name='선형 회귀'))
        fig.add_trace(go.Scatter(x=X_1d, y=pred_p2, mode='lines', name='2차 다항 회귀'))
        fig.add_trace(go.Scatter(x=X_1d, y=Y_pred_ann, mode='lines+markers', name='인공신경망 (ANN)'))
        fig.update_layout(title="모델 적합도 비교 그래프", xaxis_title=x_cols[0], yaxis_title=y_col)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("다변수(2개 이상) 입력 시에는 인공신경망(ANN) 성능 지표만 표시됩니다.")
