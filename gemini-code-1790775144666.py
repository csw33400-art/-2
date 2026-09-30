import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.optimize import curve_fit

st.set_page_config(page_title="범용 회귀 분석 대시보드", layout="wide")
st.title("📈 범용 선형 및 비선형 회귀 분석 대시보드")
st.write("CSV/Excel 파일을 업로드하여 다양한 선형/비선형 회귀 모델을 비교하고 예측해 보세요.")

# -------------------------------------------------------------
# 1. 파일 업로드 및 데이터 불러오기
# -------------------------------------------------------------
st.sidebar.header("📁 1. 데이터 파일 업로드")
uploaded_file = st.sidebar.file_uploader("CSV 또는 Excel 파일을 올려주세요", type=["csv", "xlsx", "xls"])
header_row = st.sidebar.number_input("열 이름이 있는 행 번호 (1부터 시작):", min_value=1, value=1, step=1) - 1

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, header=header_row)
        else:
            df = pd.read_excel(uploaded_file, header=header_row)
        st.sidebar.success("성공적으로 파일을 로드했습니다!")
    except Exception as e:
        st.sidebar.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        st.stop()
else:
    st.sidebar.info("기본 제공 샘플 데이터셋을 사용 중입니다.")
    data = {
        'X_variable': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Y_variable': [2.5, 3.8, 6.1, 8.2, 11.0, 14.5, 19.2, 25.1, 33.0, 43.5]
    }
    df = pd.DataFrame(data)

# 숫자형 열만 자동 필터링
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(num_cols) < 2:
    st.error("분석을 위해 최소 2개 이상의 숫자형 데이터 열이 필요합니다.")
    st.stop()

# -------------------------------------------------------------
# 2. X/Y축 선택 및 데이터 필터링
# -------------------------------------------------------------
st.sidebar.header("🎯 2. 변수 및 필터 설정")
x_col = st.sidebar.selectbox("독립변수 (X축) 선택:", num_cols, index=0)
y_col = st.sidebar.selectbox("종속변수 (Y축) 선택:", num_cols, index=1 if len(num_cols) > 1 else 0)

# 필터링 옵션 (그룹별 선택)
non_num_cols = df.columns.tolist()
filter_col = st.sidebar.selectbox("데이터 필터링 기준 열 (선택사항):", ["사용 안 함"] + non_num_cols)

df_filtered = df.copy()
if filter_col != "사용 안 함":
    unique_vals = df[filter_col].unique().tolist()
    selected_val = st.sidebar.selectbox(f"{filter_col} 선택:", ["전체"] + [str(v) for v in unique_vals])
    if selected_val != "전체":
        df_filtered = df_filtered[df_filtered[filter_col].astype(str) == selected_val]

df_clean = df_filtered[[x_col, y_col]].dropna()
X = df_clean[x_col].values
Y = df_clean[y_col].values

# -------------------------------------------------------------
# 3. 회귀 모델 연산 함수 정의
# -------------------------------------------------------------
models_results = {}

# (1) 선형 회귀 (Linear)
lin_reg = LinearRegression()
lin_reg.fit(X.reshape(-1, 1), Y)
pred_lin = lin_reg.predict(X.reshape(-1, 1))
models_results['선형'] = {
    'func': lambda x: lin_reg.predict(np.array(x).reshape(-1, 1)),
    'eq': f"y = {lin_reg.coef_[0]:.4f}·x + {lin_reg.intercept_:.4f}",
    'pred': pred_lin
}

# (2) 다항 회귀 (2차)
p2 = np.polyfit(X, Y, 2)
models_results['다항(2차)'] = {
    'func': lambda x: np.polyval(p2, x),
    'eq': f"y = {p2[0]:.4f}·x² + {p2[1]:.4f}·x + {p2[2]:.4f}",
    'pred': np.polyval(p2, X)
}

# (3) 거듭제곱 회귀 (Power: y = a * x^b) -> X > 0, Y > 0 조건
if np.all(X > 0) and np.all(Y > 0):
    try:
        def power_func(x, a, b): return a * (x ** b)
        popt_p, _ = curve_fit(power_func, X, Y, maxfev=5000)
        models_results['거듭제곱'] = {
            'func': lambda x: power_func(x, *popt_p),
            'eq': f"y = {popt_p[0]:.4f}·x^{popt_p[1]:.4f}",
            'pred': power_func(X, *popt_p)
        }
    except: pass

# (4) 지수 회귀 (Exponential: y = a * e^(b*x)) -> Y > 0 조건
if np.all(Y > 0):
    try:
        def exp_func(x, a, b): return a * np.exp(b * x)
        popt_e, _ = curve_fit(exp_func, X, Y, p0=[1.0, 0.1], maxfev=5000)
        models_results['지수'] = {
            'func': lambda x: exp_func(x, *popt_e),
            'eq': f"y = {popt_e[0]:.4f}·e^({popt_e[1]:.4f}·x)",
            'pred': exp_func(X, *popt_e)
        }
    except: pass

# -------------------------------------------------------------
# 4. 결과 출력 및 탭 구성
# -------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 데이터 및 산점도", "📈 모델 비교 및 시각화", "📉 잔차 분석", "🔮 새로운 X값 예측"])

with tab1:
    st.subheader("데이터 미리보기")
    st.dataframe(df_clean)
    fig_raw = px.scatter(df_clean, x=x_col, y=y_col, title=f"{x_col} vs {y_col} 산점도")
    st.plotly_chart(fig_raw, use_container_width=True)

with tab2:
    st.subheader("모델 평가지표 비교")
    metrics_list = []
    for name, m in models_results.items():
        r2 = r2_score(Y, m['pred'])
        rmse = np.sqrt(mean_squared_error(Y, m['pred']))
        mae = mean_absolute_error(Y, m['pred'])
        metrics_list.append({'모델': name, '회귀식': m['eq'], 'R²': r2, 'RMSE': rmse, 'MAE': mae})
    
    df_metrics = pd.DataFrame(metrics_list).sort_values(by='R²', ascending=False)
    st.dataframe(df_metrics.style.highlight_max(subset=['R²'], color='lightgreen'))

    # Plotly 시각화
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(x=X, y=Y, mode='markers', name='실제 데이터', marker=dict(color='black', size=8)))
    
    x_dense = np.linspace(min(X), max(X), 200)
    for name, m in models_results.items():
        fig_comp.add_trace(go.Scatter(x=x_dense, y=m['func'](x_dense), mode='lines', name=name))
    
    fig_comp.update_layout(title="회귀 모델 적합 곡선 비교", xaxis_title=x_col, yaxis_title=y_col)
    st.plotly_chart(fig_comp, use_container_width=True)

with tab3:
    st.subheader("선택한 모델의 잔차(Residual) 그래프")
    selected_model_name = st.selectbox("잔차 분석할 모델 선택:", list(models_results.keys()))
    residuals = Y - models_results[selected_model_name]['pred']
    
    fig_res = px.scatter(x=models_results[selected_model_name]['pred'], y=residuals,
                         labels={'x': '예측값 (Predicted)', 'y': '잔차 (Residual)'},
                         title=f"{selected_model_name} 모델 잔차 분포")
    fig_res.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_res, use_container_width=True)

with tab4:
    st.subheader("새로운 X 입력에 대한 Y값 추론")
    pred_x_input = st.number_input("예측하고 싶은 X 값을 입력하세요:", value=float(max(X)) + 0.1)
    
    st.write(f"**X = {pred_x_input:.4f} 일 때 모델별 예측 Y값:**")
    for name, m in models_results.items():
        y_val = m['func'](np.array([pred_x_input]))[0]
        st.write(f"- **{name}**: {y_val:.4f}")