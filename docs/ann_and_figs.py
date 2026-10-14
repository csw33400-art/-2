import json, sys, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
CON, RES, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
res = json.load(open(RES))
for f in font_manager.findSystemFonts():
    if "NotoSansCJK" in f and "Regular" in f:
        font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
GREY, BLUE, ORANGE = "#9aa0a6", "#1f6feb", "#e8710a"

con = pd.read_csv(CON)
X, Y = con.iloc[:, :8].values.astype(float), con["strength"].values.astype(float)
Xtr, Xte, ytr, yte = train_test_split(X, Y, test_size=0.2, random_state=42)


def ann(layers, act="relu", seed=42, alpha=1e-4):
    return TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), MLPRegressor(
        hidden_layer_sizes=layers, activation=act, alpha=alpha, max_iter=1000, random_state=seed)), transformer=StandardScaler())


def score(layers, act="relu", alpha=1e-4, seeds=range(5)):
    te, tr, it = [], [], []
    for s in seeds:
        m = ann(layers, act, s, alpha).fit(Xtr, ytr)
        te.append(r2_score(yte, m.predict(Xte))); tr.append(r2_score(ytr, m.predict(Xtr))); it.append(m.regressor_[-1].n_iter_)
    return dict(test=float(np.mean(te)), test_sd=float(np.std(te, ddof=1)), train=float(np.mean(tr)), iters=float(np.mean(it)))


ann_exp = {"nodes": [], "layers": [], "act": [], "alpha": []}
SKIP = "ann_exp" in res
if SKIP: ann_exp = res["ann_exp"]
for n in [] if SKIP else [4, 8, 16, 32, 64, 128, 256]:
    ann_exp["nodes"].append(dict(nodes=n, **score((n, max(2, n // 2)))))
for k, L in [] if SKIP else [(1, (64,)), (2, (64, 32)), (3, (64, 32, 16))]:
    ann_exp["layers"].append(dict(layers=k, shape="-".join(map(str, L)), **score(L)))
for a in [] if SKIP else ["relu", "tanh", "logistic"]:
    ann_exp["act"].append(dict(act=a, **score((64, 32), a)))
for al in [] if SKIP else [1e-4, 1e-2, 1, 10]:
    ann_exp["alpha"].append(dict(alpha=al, **score((64, 32), alpha=al)))
for k, v in ann_exp.items():
    print(k, [(list(d.values())[0], round(d["test"], 3), round(d["train"], 3), round(d["iters"])) for d in v])

final = ann((64, 32)).fit(Xtr, ytr)
pred = final.predict(Xte)
mlp = final.regressor_[-1]
lin = LinearRegression().fit(Xtr, ytr)
res["ann_exp"] = ann_exp
res["ann_final"] = dict(test_r2=float(r2_score(yte, pred)), n_iter=int(mlp.n_iter_), final_loss=float(mlp.loss_),
                        n_params=int(sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)),
                        sample=[dict(actual=float(a), ann=float(p), lin=float(l)) for a, p, l in
                                list(zip(yte, pred, lin.predict(Xte)))[:5]])
json.dump(res, open(RES, "w"), ensure_ascii=False, indent=1)

# ---------- 그림 ----------
def save(fig, name):
    fig.tight_layout(); fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight"); plt.close(fig)

# 1. 콘크리트 모델 비교 (10개 시드 평균 ± 범위)
rows = sorted(res["exp1"], key=lambda r: r["rep_mean"])
fig, ax = plt.subplots(figsize=(8, 4.6))
lin_like = {"선형 회귀", "릿지 회귀", "라쏘 회귀"}
for i, r in enumerate(rows):
    c = BLUE if r["model"] == "인공신경망(ANN)" else (ORANGE if r["model"] in lin_like else GREY)
    ax.barh(i, r["rep_mean"], color=c, height=0.6)
    ax.errorbar(r["rep_mean"], i, xerr=[[r["rep_mean"] - r["rep_min"]], [r["rep_max"] - r["rep_mean"]]],
                color="#444", capsize=3, lw=1)
    ax.text(r["rep_max"] + 0.01, i, f'{r["rep_mean"]:.3f}', va="center", fontsize=10)
ax.set_yticks(range(len(rows)), [r["model"] for r in rows])
ax.set_xlim(0, 1.05); ax.set_xlabel("테스트 R² (10개 분할 평균, 막대 끝 선 = 최소~최대)")
ax.set_title("콘크리트 압축강도: ANN(0.895)은 선형 회귀(0.581)보다 R² 0.31 높다", loc="left", fontsize=12)
save(fig, "fig_concrete_models")

# 3. 학습 곡선
lc = pd.DataFrame(res["exp4_lc"])
fig, ax = plt.subplots(figsize=(7.5, 4.3))
style = {"선형 회귀": (ORANGE, "o"), "랜덤 포레스트": (GREY, "s"), "그래디언트 부스팅": ("#5f6368", "^"), "인공신경망(ANN)": (BLUE, "D")}
for m, (c, mk) in style.items():
    s = lc[lc.model == m]
    ax.plot(s["size"], s["test"].clip(lower=-0.2), marker=mk, color=c, label=m, lw=2.2 if m.startswith("인공") else 1.4)
ax.set_xscale("log"); ax.set_xticks([10, 20, 50, 100, 200, 400, 824], ["10", "20", "50", "100", "200", "400", "824"])
ax.axhline(0, color="#ccc", lw=0.8)
ax.set_xlabel("학습 데이터 수 (행, 로그 축)"); ax.set_ylabel("테스트 R² (5회 평균)"); ax.set_ylim(-0.2, 1.0)
ax.legend(frameon=False, loc="lower right")
ax.set_title("ANN은 50행부터 선형 회귀를 앞서고, 데이터가 늘수록 격차가 커진다", loc="left", fontsize=12)
ax.annotate("선형 회귀 10행: R² −24.1 (축 밖)", xy=(10, -0.2), xytext=(24, -0.16), fontsize=9, color=ORANGE)
save(fig, "fig_learning_curve")

# 4. ANN 실제 vs 예측 + 선형
fig, ax = plt.subplots(figsize=(5.2, 5))
ax.scatter(yte, lin.predict(Xte), s=14, color=ORANGE, alpha=0.55, label=f"선형 회귀 (R² {r2_score(yte, lin.predict(Xte)):.3f})")
ax.scatter(yte, pred, s=14, color=BLUE, alpha=0.75, label=f"ANN (R² {r2_score(yte, pred):.3f})")
lo, hi = 0, 85
ax.plot([lo, hi], [lo, hi], "--", color="#d93025", lw=1); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("실제 압축강도 (MPa)"); ax.set_ylabel("예측 압축강도 (MPa)"); ax.legend(frameon=False, loc="upper left")
ax.set_title("테스트 206개: ANN 예측이 y=x 선에 더 붙어 있다", loc="left", fontsize=12)
save(fig, "fig_ann_actual_pred")

# 5. ANN 손실 곡선
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(np.arange(1, len(mlp.loss_curve_) + 1), mlp.loss_curve_, color=BLUE)
ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_ylabel("학습 손실 (로그 축)")
ax.set_title(f"손실이 {mlp.n_iter_}회 반복 뒤 더 줄지 않아 학습이 자동 종료됐다", loc="left", fontsize=12)
save(fig, "fig_ann_loss")

# 6. 노드 수
ne = ann_exp["nodes"]
fig, ax = plt.subplots(figsize=(7.5, 4))
n = [e["nodes"] for e in ne]
ax.plot(n, [e["train"] for e in ne], "o-", color=GREY, label="학습 R²")
ax.errorbar(n, [e["test"] for e in ne], yerr=[e["test_sd"] for e in ne], fmt="o-", color=BLUE, capsize=3, label="테스트 R² (±표준편차)")
ax.set_xscale("log", base=2); ax.set_xticks(n, [str(v) for v in n]); ax.set_ylim(0.6, 1.0)
ax.set_xlabel("첫 은닉층 노드 수 (둘째 층은 절반)"); ax.set_ylabel("R² (5회 평균)"); ax.legend(frameon=False, loc="lower right")
ax.set_title("노드 64개부터 테스트 R²는 0.88~0.89에서 더 오르지 않는다", loc="left", fontsize=12)
save(fig, "fig_ann_nodes")
print("FIGS OK")
