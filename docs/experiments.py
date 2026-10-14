"""보고서용 실험: app.py와 같은 모델·기본 하이퍼파라미터로 3개 데이터셋 비교 + 과적합 분석."""
import json, sys, time, warnings
import numpy as np, pandas as pd
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

warnings.filterwarnings("ignore")
CONCRETE = sys.argv[1]
OUT = sys.argv[2]
SEED = 42


class CurveFitRegressor(RegressorMixin, BaseEstimator):
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
        x = np.asarray(X, float).ravel(); y = np.asarray(y, float)
        if self.kind == "log":
            b, a = np.polyfit(np.log(x), y, 1); self.params_ = np.array([a, b]); return self
        if self.kind == "power":
            b, ln_a = np.polyfit(np.log(x), np.log(y), 1)
        else:
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
        with np.errstate(all="ignore"):
            return self._func(self.kind)(np.asarray(X, float).ravel(), *self.params_)


MULTI = ["선형 회귀", "릿지 회귀", "라쏘 회귀", "다항 회귀(2차)", "결정 트리", "랜덤 포레스트",
         "그래디언트 부스팅", "SVR", "KNN", "인공신경망(ANN)"]
CURVES = ["거듭제곱 회귀", "지수 회귀", "로그 회귀"]


def build(name, seed=SEED, n_fit=10**9, poly=2, ann_nodes=64):
    if name == "선형 회귀": return LinearRegression()
    if name == "릿지 회귀": return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == "라쏘 회귀": return make_pipeline(StandardScaler(), Lasso(alpha=0.1, max_iter=50000))
    if name.startswith("다항 회귀"):
        return make_pipeline(PolynomialFeatures(poly, include_bias=False), StandardScaler(), LinearRegression())
    if name == "결정 트리": return DecisionTreeRegressor(max_depth=5, random_state=seed)
    if name == "랜덤 포레스트": return RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
    if name == "그래디언트 부스팅":
        return GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=seed)
    if name == "SVR":
        return TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), SVR(C=10.0, epsilon=0.1)),
                                          transformer=StandardScaler())
    if name == "KNN":
        return make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=max(1, min(5, n_fit)), weights="distance"))
    if name == "인공신경망(ANN)":
        return TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(ann_nodes, ann_nodes // 2), max_iter=1000, random_state=seed)), transformer=StandardScaler())
    return CurveFitRegressor({"거듭제곱 회귀": "power", "지수 회귀": "exp", "로그 회귀": "log"}[name])


def met(y, p):
    return dict(R2=r2_score(y, p), RMSE=float(np.sqrt(mean_squared_error(y, p))), MAE=float(mean_absolute_error(y, p)),
                MAPE=float(np.mean(np.abs((y - p) / y)) * 100) if np.all(y != 0) else None)


def compare(X, Y, names, seeds=range(10)):
    """seed 42 분할로 상세 지표 + CV, 10개 시드 반복으로 테스트 R² 평균·표준편차."""
    Xtr, Xte, ytr, yte = train_test_split(X, Y, test_size=0.2, random_state=SEED)
    kf = KFold(5, shuffle=True, random_state=SEED)
    rows = []
    fitted = {}
    for n in names:
        m = build(n, n_fit=len(Xtr) - int(np.ceil(len(Xtr) / 5)))
        t0 = time.perf_counter(); m.fit(Xtr, ytr); dt = time.perf_counter() - t0
        fitted[n] = m
        te, tr = met(yte, m.predict(Xte)), met(ytr, m.predict(Xtr))
        cv = met(ytr, cross_val_predict(build(n, n_fit=len(Xtr) - int(np.ceil(len(Xtr) / 5))), Xtr, ytr, cv=kf))
        rep = []
        for s in seeds:
            a, b, c, d = train_test_split(X, Y, test_size=0.2, random_state=s)
            rep.append(r2_score(d, build(n, seed=s).fit(a, c).predict(b)))
        rows.append(dict(model=n, test_R2=te["R2"], test_RMSE=te["RMSE"], test_MAE=te["MAE"], test_MAPE=te["MAPE"],
                         cv_R2=cv["R2"], cv_RMSE=cv["RMSE"], train_R2=tr["R2"], gap=tr["R2"] - te["R2"], fit_sec=dt,
                         rep_mean=float(np.mean(rep)), rep_std=float(np.std(rep, ddof=1)),
                         rep_min=float(np.min(rep)), rep_max=float(np.max(rep))))
        print(f"  {n}: test {te['R2']:.3f} cv {cv['R2']:.3f} rep {np.mean(rep):.3f}±{np.std(rep, ddof=1):.3f}", flush=True)
    return sorted(rows, key=lambda r: -r["test_R2"]), fitted, (Xtr, Xte, ytr, yte)


def ols(X, Y, names):
    A = np.column_stack([np.ones(len(X)), X]); beta, *_ = np.linalg.lstsq(A, Y, rcond=None)
    dof = len(X) - A.shape[1]; sse = float(np.sum((Y - A @ beta) ** 2)); sst = float(np.sum((Y - Y.mean()) ** 2))
    se = np.sqrt(np.diag(sse / dof * np.linalg.pinv(A.T @ A))); t = beta / se; p = 2 * stats.t.sf(np.abs(t), dof)
    r2 = 1 - sse / sst
    return dict(r2=r2, adj_r2=1 - (1 - r2) * (len(X) - 1) / dof, n=len(X),
                coef=[dict(var=v, b=float(b), se=float(s), t=float(tt), p=float(pp))
                      for v, b, s, tt, pp in zip(["절편"] + names, beta, se, t, p)])


def importance(model, X, Y, names):
    r = permutation_importance(model, X, Y, n_repeats=15, random_state=SEED, scoring="r2")
    return sorted([dict(var=v, mean=float(m), std=float(s)) for v, m, s in zip(names, r.importances_mean, r.importances_std)],
                  key=lambda d: -d["mean"])


res = {}

# ---------- 데이터 ----------
con = pd.read_csv(CONCRETE)
con_names = ["시멘트", "고로슬래그", "플라이애시", "물", "고성능감수제", "굵은골재", "잔골재", "재령(일)"]
Xc, Yc = con.iloc[:, :8].values.astype(float), con["strength"].values.astype(float)
dia = load_diabetes(as_frame=True).frame
dia_names = list(dia.columns[:10])
Xd, Yd = dia.iloc[:, :10].values, dia["target"].values
rng = np.random.default_rng(0)
xs = np.linspace(0.5, 10, 150)
syn = pd.DataFrame({"x": xs, "noise_feature": rng.uniform(0, 1, 150), "y": 3 * np.sin(xs) + 0.8 * xs + 5 + rng.normal(0, 0.6, 150)})
Xs, Ys = syn[["x"]].values, syn["y"].values


def describe(df, label):
    d = df.describe().T
    return dict(name=label, rows=len(df), cols=df.shape[1], missing=int(df.isna().sum().sum()),
                y_mean=float(df.iloc[:, -1].mean()), y_std=float(df.iloc[:, -1].std()),
                y_min=float(df.iloc[:, -1].min()), y_max=float(df.iloc[:, -1].max()))


res["datasets"] = [describe(con, "콘크리트"), describe(dia, "당뇨병"), describe(syn[["x", "y"]], "합성")]
res["concrete_corr"] = {n: float(np.corrcoef(Xc[:, i], Yc)[0, 1]) for i, n in enumerate(con_names)}
res["diabetes_corr"] = {n: float(np.corrcoef(Xd[:, i], Yd)[0, 1]) for i, n in enumerate(dia_names)}

# ---------- 실험 1: 콘크리트 ----------
print("EXP1 concrete", flush=True)
rows, fitted, (Xtr, Xte, ytr, yte) = compare(Xc, Yc, MULTI)
res["exp1"] = rows
res["exp1_ols"] = ols(Xtr, ytr, con_names)
best = rows[0]["model"]
res["exp1_imp"] = {"model": best, "items": importance(fitted[best], Xte, yte, con_names)}
res["exp1_imp_linear"] = importance(fitted["선형 회귀"], Xte, yte, con_names)

# ---------- 실험 2: 당뇨병 ----------
print("EXP2 diabetes", flush=True)
rows, fitted, (Xtr, Xte, ytr, yte) = compare(Xd, Yd, MULTI)
res["exp2"] = rows
res["exp2_ols"] = ols(Xtr, ytr, dia_names)
best = rows[0]["model"]
res["exp2_imp"] = {"model": best, "items": importance(fitted[best], Xte, yte, dia_names)}

# ---------- 실험 3: 합성 1D ----------
print("EXP3 synthetic", flush=True)
rows, fitted, (Xtr, Xte, ytr, yte) = compare(Xs, Ys, MULTI + CURVES)
res["exp3"] = rows
diag = []
order = np.argsort(Xte.ravel())
for n in ["선형 회귀", rows[0]["model"], "인공신경망(ANN)", "다항 회귀(2차)"]:
    r = (yte - fitted[n].predict(Xte))[order]
    diag.append(dict(model=n, mean=float(r.mean()), std=float(r.std(ddof=1)), shapiro_p=float(stats.shapiro(r).pvalue),
                     dw=float(np.sum(np.diff(r) ** 2) / np.sum(r ** 2)),
                     runs_sign_changes=int(np.sum(np.diff(np.sign(r)) != 0))))
res["exp3_diag"] = diag
res["exp3_curves"] = {n: list(map(float, fitted[n].params_)) for n in CURVES}

# ---------- 실험 4a: 학습 데이터 크기 (콘크리트) ----------
print("EXP4a learning curve", flush=True)
Xtr_all, Xte4, ytr_all, yte4 = train_test_split(Xc, Yc, test_size=0.2, random_state=SEED)
lc = []
for n in ["선형 회귀", "랜덤 포레스트", "그래디언트 부스팅", "인공신경망(ANN)"]:
    for size in [10, 20, 50, 100, 200, 400, len(Xtr_all)]:
        te_s, tr_s = [], []
        for s in range(5):
            idx = np.random.default_rng(s).choice(len(Xtr_all), size, replace=False)
            m = build(n, seed=s).fit(Xtr_all[idx], ytr_all[idx])
            te_s.append(r2_score(yte4, m.predict(Xte4))); tr_s.append(r2_score(ytr_all[idx], m.predict(Xtr_all[idx])))
        lc.append(dict(model=n, size=int(size), test=float(np.mean(te_s)), test_lo=float(np.min(te_s)),
                       test_hi=float(np.max(te_s)), train=float(np.mean(tr_s))))
        print(f"  {n} {size}: test {np.mean(te_s):.3f} train {np.mean(tr_s):.3f}", flush=True)
res["exp4_lc"] = lc

# ---------- 실험 4b: 다항 차수 (합성) ----------
print("EXP4b poly degree", flush=True)
Xtr, Xte, ytr, yte = train_test_split(Xs, Ys, test_size=0.2, random_state=SEED)
kf = KFold(5, shuffle=True, random_state=SEED)
pdg = []
for d in range(1, 13):
    m = build("다항 회귀", poly=d).fit(Xtr, ytr)
    cvp = cross_val_predict(build("다항 회귀", poly=d), Xtr, ytr, cv=kf)
    pdg.append(dict(degree=d, train=float(r2_score(ytr, m.predict(Xtr))), test=float(r2_score(yte, m.predict(Xte))),
                    cv=float(r2_score(ytr, cvp))))
res["exp4_poly"] = pdg

# ---------- 실험 4c: 앱 샘플 10행 ----------
print("EXP4c small sample", flush=True)
small = pd.DataFrame({'Cement': [540, 540, 332.5, 332.5, 198.6, 266, 380, 380, 266, 475],
                      'Water': [162, 162, 228, 228, 192, 228, 228, 228, 228, 228],
                      'Age': [28, 28, 270, 365, 360, 90, 365, 28, 28, 28],
                      'Strength': [79.99, 61.89, 40.27, 41.05, 44.30, 47.03, 43.70, 36.45, 45.85, 39.29]})
Xsm, Ysm = small[["Cement", "Water", "Age"]].values.astype(float), small["Strength"].values
kf = KFold(5, shuffle=True, random_state=SEED)
sm = []
for n in ["선형 회귀", "랜덤 포레스트", "인공신경망(ANN)"]:
    m = build(n).fit(Xsm, Ysm)
    cvp = cross_val_predict(build(n, n_fit=8), Xsm, Ysm, cv=kf)
    sm.append(dict(model=n, train=float(r2_score(Ysm, m.predict(Xsm))), cv=float(r2_score(Ysm, cvp))))
res["exp4_small"] = sm

json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
print("DONE")
