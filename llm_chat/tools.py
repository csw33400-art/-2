"""채팅으로 실행하는 분석 도구 모음 (선형회귀 · 비선형회귀 · 인공신경망 · 딥러닝)

app.py의 채팅창이 사용자의 말을 해석해서 이 파일의 함수를 실행합니다.
모든 함수는 {"text": 설명글, "items": [("fig", 그래프) | ("table", 표)], "summary": 한 줄 요약}을 돌려줍니다.
"""
import difflib
import io
import json
import re
import tempfile
import urllib.parse
import urllib.request
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from sklearn.compose import TransformedTargetRegressor
from sklearn.datasets import fetch_california_housing, load_diabetes, load_iris, load_wine
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

SEABORN = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/"
DATASETS = {
    "concrete": {"label": "콘크리트 압축강도 (UCI 공개 데이터)", "words": ("콘크리트", "concrete"), "y": "압축강도(MPa)"},
    "diabetes": {"label": "당뇨병 진행도 (scikit-learn 공개 데이터)", "words": ("당뇨", "diabetes"), "y": "target"},
    "mpg": {"label": "자동차 연비 (Auto MPG 공개 데이터)", "words": ("연비", "자동차", "mpg"), "y": "mpg", "url": SEABORN + "mpg.csv"},
    "tips": {"label": "식당 팁 (tips 공개 데이터)", "words": ("팁", "tips", "식당"), "y": "tip", "url": SEABORN + "tips.csv"},
    "penguins": {"label": "펭귄 신체 측정 (Palmer Penguins 공개 데이터)", "words": ("펭귄", "penguin"), "y": "body_mass_g",
                 "url": SEABORN + "penguins.csv"},
    "california": {"label": "캘리포니아 집값 (scikit-learn 공개 데이터)", "words": ("캘리포니아", "집값", "주택", "california", "housing"),
                   "y": "MedHouseVal"},
    "iris": {"label": "붓꽃 크기 (Iris 공개 데이터)", "words": ("붓꽃", "아이리스", "iris"), "y": "petal width (cm)"},
    "wine": {"label": "와인 성분 (Wine 공개 데이터)", "words": ("와인", "wine"), "y": "alcohol"},
    "synthetic": {"label": "비선형 합성 데이터", "words": ("합성", "synthetic"), "y": "y"},
}
MAX_ROWS = 20000          # 작은 서버라서 이보다 큰 데이터는 무작위로 줄여서 씀

# 사용자가 한국어로 말해도 영어 열 이름을 찾도록 하는 낱말 사전 (반대 방향도 사용)
ALIASES = {
    "나이": ("age",), "연령": ("age",), "키": ("height",), "신장": ("height",), "몸무게": ("weight", "mass"), "체중": ("weight", "mass"),
    "무게": ("weight", "mass"), "질량": ("mass", "weight"), "가격": ("price", "cost", "charges", "val"), "비용": ("cost", "charges", "price"),
    "집값": ("medhouseval", "price"), "소득": ("income", "medinc"), "수입": ("income",), "연봉": ("salary", "income"), "월급": ("salary",),
    "온도": ("temp",), "기온": ("temp",), "습도": ("humid",), "압력": ("press",), "속도": ("speed", "velocity"), "속력": ("speed",),
    "시간": ("time", "hour"), "거리": ("dist",), "길이": ("length",), "너비": ("width",), "폭": ("width",), "높이": ("height",),
    "면적": ("area",), "넓이": ("area",), "부피": ("volume",), "연비": ("mpg",), "마력": ("horsepower", "hp"), "배기량": ("displacement",),
    "실린더": ("cylinder",), "기통": ("cylinder",), "가속": ("accel",), "연식": ("model_year", "year"), "연도": ("year",), "년도": ("year",),
    "팁": ("tip",), "계산서": ("total_bill", "bill"), "금액": ("total_bill", "amount", "price"), "총액": ("total",), "인원": ("size",),
    "성별": ("sex", "gender"), "흡연": ("smok",), "요일": ("day",), "날짜": ("date",), "점수": ("score",), "성적": ("score", "grade"),
    "매출": ("sales", "revenue"), "판매": ("sales",), "수량": ("quantity", "qty", "count"), "개수": ("count",), "인구": ("popul",),
    "방": ("room",), "침실": ("bedrm", "bedroom"), "위도": ("latitude",), "경도": ("longitude",), "강도": ("strength",),
    "혈압": ("bp", "pressure"), "혈당": ("glucose",), "부리": ("bill",), "날개": ("flipper",), "꽃잎": ("petal",), "꽃받침": ("sepal",),
    "알코올": ("alcohol",), "도수": ("alcohol",), "산도": ("acid", "ph"), "색": ("color",), "밀도": ("density",), "농도": ("concentration",),
    "전압": ("volt",), "전류": ("current",), "전력": ("power",), "출력": ("power", "output"), "하중": ("load",), "응력": ("stress",),
    "변형": ("strain",), "깊이": ("depth",), "두께": ("thick",), "지름": ("diam",), "직경": ("diam",), "각도": ("angle",),
    "주행": ("mileage", "odometer"), "평수": ("area",), "층": ("floor",), "목표": ("target",), "결과": ("target", "result", "output"),
}
PARTICLES = ("으로", "이랑", "하고", "에서", "일때", "로", "랑", "와", "과", "은", "는", "이", "가", "을", "를", "의", "도", "만", "에")
STOPWORDS = {
    "데이터", "자료", "회귀", "선형", "비선형", "신경망", "인공", "딥러닝", "모델", "학습", "예측", "분석", "결과", "비교", "그래프", "변수",
    "입력", "전부", "모든", "모두", "다시", "이번", "그리고", "에포크", "은닉층", "학습률", "다항", "평균", "최대", "최소", "표준편차",
    "산점도", "히스토그램", "분포", "관계", "영향", "파일", "목록", "설명", "사용", "이용", "값", "the", "and", "for", "with", "data",
    "linear", "regression", "model", "run", "plot", "show", "mean", "max", "min", "std", "ann", "mlp", "deep", "relu", "tanh",
}
Y_HINTS = ("target", "label", "price", "output", "result", "score", "가격", "강도", "결과", "목표", "점수", "매출")


def new_state():
    return {"df": None, "key": None, "name": None, "x": [], "y": None, "results": {}, "inputs": {}}


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
# 2. 데이터 불러오기: 어떤 표 데이터든 숫자 분석이 가능하게 정리
# =============================================================
def match_dataset(t):
    for key, d in DATASETS.items():
        if any(w in t for w in d["words"]):
            return key
    return None


def dataset_list():
    rows = [(d["label"], f"\"{d['words'][0]} 데이터 불러와\"") for d in DATASETS.values()]
    return reply("**바로 쓸 수 있는 공개 데이터**예요. 이 밖에도 아래 방법으로 아무 데이터나 쓸 수 있어요.\n\n"
                 "- 입력창의 📎로 파일 첨부 (CSV, Excel, TSV, TXT, JSON)\n"
                 "- 데이터 주소 붙여넣기 (CSV·Excel 링크, 구글 스프레드시트 공유 링크, GitHub 파일 링크)\n"
                 "- 채팅창에 표를 그대로 붙여넣기, 또는 `x: 1, 2, 3, 4 / y: 2.1, 3.9, 6.2, 8.1`처럼 직접 입력",
                 [("table", pd.DataFrame(rows, columns=["데이터", "이렇게 말하세요"]))])


def _clean(df):
    """아무 표나 분석할 수 있게 정리한다. (정리한 표, 안내 문장 목록)"""
    notes = []
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()].dropna(axis=1, how="all").dropna(axis=0, how="all")
    for c in [c for c in df.columns if c.lower().startswith("unnamed")]:
        v = pd.to_numeric(df[c], errors="coerce")
        if v.notna().all() and (v.diff().dropna() == 1).all():      # 0,1,2,… 순번 열은 버림
            df = df.drop(columns=c)
    if len(df) > MAX_ROWS:
        notes.append(f"행이 {len(df):,}개라서 무작위로 {MAX_ROWS:,}행만 골라 써요.")
        df = df.sample(MAX_ROWS, random_state=SEED)
    df = df.reset_index(drop=True)
    converted, encoded, ignored = [], [], []
    for c in list(df.columns):
        col = df[c]
        if col.dtype == bool:
            df[c] = col.astype(int)
            continue
        if pd.api.types.is_numeric_dtype(col):
            continue
        if pd.api.types.is_datetime64_any_dtype(col):
            ignored.append(c)
            continue
        txt = col.astype(str).str.strip()
        as_num = pd.to_numeric(txt.str.replace(r"[,\s%$₩원]", "", regex=True), errors="coerce")
        filled = col.notna().sum()
        if filled and as_num.notna().sum() >= 0.9 * filled:          # "1,234"·"12%"처럼 글자로 저장된 숫자
            df[c] = as_num
            converted.append(c)
            continue
        cats = sorted(col.dropna().astype(str).str.strip().unique())
        if 2 <= len(cats) <= 8 and len(cats) < max(3, len(df) // 2):   # 범주(글자) → 0/1 숫자 열
            for v in cats[1:]:
                df[f"{c}={v}"] = (txt == v).astype(float).where(col.notna())
            encoded.append(f"{c}({', '.join(cats)})")
        else:
            ignored.append(c)
    if converted:
        notes.append(f"글자로 저장된 숫자 열을 숫자로 바꿨어요: {', '.join(converted)}")
    if encoded:
        notes.append(f"글자 범주 열은 0/1 숫자 열로 바꿔 분석에 쓸 수 있게 했어요: {'; '.join(encoded)}")
    if ignored:
        notes.append(f"숫자로 바꿀 수 없는 열은 분석에서 빼요: {', '.join(ignored[:8])}")
    n_missing = int(df.select_dtypes(include=[np.number]).isna().sum().sum())
    if n_missing:
        notes.append(f"빈칸(결측치)이 {n_missing:,}개 있어요. 입력 변수의 빈칸은 중앙값으로 채우고, Y가 빈 행은 빼고 학습해요.")
    return df, notes


def numeric_cols(state):
    return state["df"].select_dtypes(include=[np.number]).columns.tolist()


def _guess_y(df, num):
    plain = [c for c in num if "=" not in c] or num
    for c in plain:
        if norm(short(c)) in ("y", "target", "label", "output"):
            return c
    for c in reversed(plain):
        if any(h in norm(c) for h in Y_HINTS):
            return c
    return plain[-1]


def _set_data(state, df, name, key=None, y=None):
    try:
        df, notes = _clean(df)
    except Exception as e:
        return reply(f"⚠️ '{name}' 데이터를 정리하지 못했어요: {e}")
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    num = [c for c in num if df[c].nunique(dropna=True) > 1]
    if len(num) < 2 or len(df) < 3:
        return reply(f"⚠️ '{name}'에서 분석할 수 있는 숫자 열을 2개 이상 찾지 못했어요. (X 1개 이상 + Y 1개, 3행 이상 필요)\n\n"
                     f"읽은 열: {', '.join(map(str, df.columns[:15]))}")
    y = y if y in num else _guess_y(df, num)
    state.update(df=df, name=name, key=key, y=y, x=[c for c in num if c != y], results={}, inputs={})
    remember(f"데이터 '{name}' 불러옴 ({len(df):,}행)")
    xs = ", ".join(short(c) for c in state["x"][:12]) + (f" 외 {len(state['x']) - 12}개" if len(state["x"]) > 12 else "")
    note = ("\n\n" + "\n".join(f"- ℹ️ {n}" for n in notes)) if notes else ""
    return reply(
        f"📂 **{name}** 데이터를 불러왔어요. {len(df):,}행 × {df.shape[1]}열이에요.\n\n"
        f"- 예측할 값(Y): **{short(y)}**\n- 입력 변수(X, {len(state['x'])}개): {xs}{note}\n\n"
        "바꾸려면 \"Y는 ○○, X는 △△\" 또는 \"○○로 △△ 예측해줘\"처럼 말하세요. 이제 \"선형회귀 해줘\"처럼 시켜 보세요.",
        [("table", df.head(5))],
        f"데이터 {name} {len(df)}행, Y={short(y)}")


def load_dataset(state, key):
    d = DATASETS[key]
    try:
        if key == "concrete":
            df = pd.read_csv(DATA_DIR / "concrete.csv")
        elif key == "diabetes":
            df = load_diabetes(as_frame=True).frame
        elif key == "iris":
            df = load_iris(as_frame=True).frame.drop(columns="target")
        elif key == "wine":
            df = load_wine(as_frame=True).frame.drop(columns="target")
        elif key == "california":
            df = fetch_california_housing(as_frame=True).frame
        elif key == "synthetic":
            rng = np.random.default_rng(0)
            x = np.linspace(0.5, 10, 150)
            df = pd.DataFrame({"x": x, "y": 3 * np.sin(x) + 0.8 * x + 5 + rng.normal(0, 0.6, len(x))})
        else:
            df = pd.read_csv(d["url"])
    except Exception as e:
        return reply(f"⚠️ '{d['label']}' 데이터를 내려받지 못했어요: {e}\n\n\"데이터 목록\"이라고 하면 다른 데이터를 볼 수 있어요.")
    return _set_data(state, df, d["label"], key, d.get("y"))


def read_table(raw, name=""):
    """파일 내용(바이트)을 표로 읽는다. CSV·TSV·TXT·Excel·JSON, 한글 인코딩과 구분자를 자동으로 맞춘다."""
    low = name.lower()
    if low.endswith((".xlsx", ".xls")) or raw[:2] == b"PK" or raw[:4] == b"\xd0\xcf\x11\xe0":
        return pd.read_excel(io.BytesIO(raw))
    if raw.lstrip()[:15].lower().startswith((b"<!doctype", b"<html")):
        raise ValueError("데이터 파일이 아니라 웹 페이지예요. CSV나 Excel 파일로 바로 연결되는 주소가 필요해요.")
    text = None
    for enc in ("utf-8-sig", "cp949", "utf-16", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if low.endswith(".json") or text.lstrip()[:1] in ("[", "{"):
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                inner = [v for v in data.values() if isinstance(v, list) and v and isinstance(v[0], dict)]
                data = inner[0] if inner else data
            return pd.json_normalize(data) if isinstance(data, list) else pd.DataFrame(data)
        except ValueError:
            pass
    df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    if df.shape[1] == 1:                                    # 구분자를 못 찾았으면 공백으로 다시 시도
        alt = pd.read_csv(io.StringIO(text), sep=r"\s+", engine="python")
        df = alt if alt.shape[1] > 1 else df
    if all(re.fullmatch(r"-?\d+(\.\d+)?", str(c).strip()) for c in df.columns):   # 첫 줄이 제목이 아니라 숫자인 파일
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python", header=None)
        df.columns = [f"x{i + 1}" for i in range(df.shape[1] - 1)] + ["y"]
    return df


def _direct_url(url):
    """공유 링크를 파일로 바로 받을 수 있는 주소로 바꾼다."""
    m = re.search(r"docs\.google\.com/spreadsheets/d/([\w-]+)", url)
    if m:
        gid = re.search(r"gid=(\d+)", url)
        return f"https://docs.google.com/spreadsheets/d/{m.group(1)}/export?format=csv" + (f"&gid={gid.group(1)}" if gid else "")
    m = re.match(r"https?://github\.com/([^/]+/[^/]+)/blob/(.+)", url)
    if m:
        return f"https://raw.githubusercontent.com/{m.group(1)}/{m.group(2)}"
    if "dropbox.com" in url:
        return re.sub(r"dl=0", "dl=1", url)
    return url


def load_url(state, url):
    try:
        host = urllib.parse.urlparse(url).hostname or ""
        if host in ("localhost",) or re.match(r"^(127\.|10\.|192\.168\.|169\.254\.|172\.(1[6-9]|2\d|3[01])\.)", host):
            raise ValueError("내부 주소는 열 수 없어요.")
        req = urllib.request.Request(_direct_url(url), headers={"User-Agent": "Mozilla/5.0 (data-chatbot)"})
        with urllib.request.urlopen(req, timeout=20) as r:
            raw = r.read(30 * 1024 * 1024)
        df = read_table(raw, url.split("?")[0])
    except Exception as e:
        return reply(f"⚠️ 주소에서 데이터를 읽지 못했어요: {e}\n\nCSV·Excel 파일 주소나 '링크가 있는 모든 사용자'에게 공개된 구글 스프레드시트 주소인지 확인해 주세요.")
    name = "구글 스프레드시트" if "docs.google.com" in url else (url.split("?")[0].rstrip("/").split("/")[-1][:40] or "웹 데이터")
    return _set_data(state, df, name)


def load_file(state, f):
    try:
        df = read_table(f.getvalue(), f.name)
    except Exception as e:
        return reply(f"⚠️ 파일을 읽지 못했어요: {e}")
    return _set_data(state, df, f.name)


NUM = r"-?\d+(?:\.\d+)?(?:e-?\d+)?"


def parse_pasted(text):
    """채팅창에 직접 적거나 붙여넣은 데이터를 표로 바꾼다. 데이터가 아니면 None."""
    # (1) "x: 1, 2, 3 / y: 2, 4, 6" 형태
    named = re.findall(rf"([A-Za-z가-힣_][\w가-힣()%/ ]{{0,20}}?)\s*[:=]\s*[\[(]?\s*({NUM}(?:\s*[,\s]\s*{NUM}){{2,}})", text)
    if len(named) >= 2:
        cols = {n.strip(): [float(v) for v in re.findall(NUM, vals)] for n, vals in named}
        if len(cols) >= 2 and len({len(v) for v in cols.values()}) == 1:
            return pd.DataFrame(cols)
    # (2) 여러 줄 표 (엑셀에서 복사한 탭 구분, 쉼표, 공백, | 구분)
    lines = [ln.strip().strip("|").strip() for ln in text.splitlines() if ln.strip() and not re.fullmatch(r"[\s|:\-]+", ln)]
    if len(lines) >= 3:
        def is_num(c):
            return bool(re.fullmatch(NUM, c.strip().replace(",", "")))

        for sep in ("\t", ",", ";", "|", r"\s+"):
            parts = [[c for c in re.split(sep if sep == r"\s+" else re.escape(sep), ln)] for ln in lines]
            rows = [i for i, p in enumerate(parts) if len(p) >= 2 and sum(map(is_num, p)) >= 0.6 * len(p)]   # 숫자로 된 줄
            if len(rows) < 2:
                continue
            width = max({len(parts[i]) for i in rows}, key=[len(parts[i]) for i in rows].count)
            rows = [i for i in rows if len(parts[i]) == width]
            if len(rows) < 2 or rows[-1] - rows[0] + 1 != len(rows):          # 숫자 줄이 끊기지 않고 이어져야 함
                continue
            first = rows[0]
            if first > 0 and len(parts[first - 1]) == width and not any(map(is_num, parts[first - 1])):
                first -= 1                                                   # 바로 윗줄이 열 이름
            if rows[-1] - first + 1 < 3:
                continue
            names = ([c.strip() for c in parts[first]] if first < rows[0]
                     else [f"x{i + 1}" for i in range(width - 1)] + ["y"])
            body = [[c.strip() for c in parts[i]] for i in rows]
            return pd.DataFrame(body, columns=names)       # 숫자 변환은 _clean이 맡음
    return None


def load_pasted(state, df):
    return _set_data(state, df, "채팅에 입력한 데이터")


# =============================================================
# 2-1. 변수 인식: 정확한 이름, 일부만 말한 이름, 한국어↔영어, 오타, 순서(첫 번째 열)
# =============================================================
def _keys(col):
    n = norm(short(col))
    return n, re.sub(r"[_\-./=]", "", n)


def _token_match(tok, col):
    """사용자가 말한 낱말 tok이 열 col을 가리키는지."""
    key, flat = _keys(col)
    if not key:
        return False
    if (len(tok) >= (3 if tok.isascii() else 2)) and (tok in key or tok in flat):
        return True
    for ko, ens in ALIASES.items():
        if tok == ko and any(e in key or e in flat for e in ens):       # 한국어로 말함 → 영어 열
            return True
        if tok.isascii() and len(tok) >= 3 and any(e.startswith(tok) or tok.startswith(e) for e in ens) and ko in key:
            return True                                                 # 영어로 말함 → 한국어 열
    if tok.isascii() and len(tok) >= 4:
        parts = [p for p in re.split(r"[^a-z0-9]+", norm(short(col))) if p] + [flat]
        return bool(difflib.get_close_matches(tok, parts, n=1, cutoff=0.82))   # 오타
    return False


def _find(text, cols):
    """문장에서 열을 찾는다. (시작 위치, 끝 위치, 열) 목록. 위치는 공백을 뺀 문장 기준."""
    low = str(text).lower()
    t = norm(text)
    found, used = {}, []

    def npos(i):
        return len(norm(low[:i]))

    # 1단계: 열 이름이 그대로 나온 경우 (긴 이름부터)
    for c in sorted(cols, key=lambda c: -len(_keys(c)[0])):
        key, _ = _keys(c)
        if not key:
            continue
        if key.isascii():
            pat = r"[\s_\-]*".join(re.escape(ch) for ch in key)
            m = re.search(rf"(?<![a-z0-9]){pat}(?![a-z0-9])", low)
            span = (npos(m.start()), npos(m.end())) if m else None
        else:
            i = t.find(key)
            span = (i, i + len(key)) if i >= 0 else None
        if span and not any(s < span[1] and span[0] < e for s, e in used):
            found[c] = span
            used.append(span)
    # 2단계: 일부만 말했거나, 다른 언어로 말했거나, 오타가 있는 경우
    for m in re.finditer(r"[가-힣]+|[a-z][a-z0-9_]*", low):
        span = (npos(m.start()), npos(m.end()))
        if any(s < span[1] and span[0] < e for s, e in used):
            continue
        tok = m.group()
        variants = [tok] + [tok[:-len(p)] for p in PARTICLES if tok.endswith(p) and len(tok) > len(p)]
        for v in variants:
            if v in STOPWORDS or len(v) < (3 if v.isascii() else 1) or (len(v) == 1 and v not in ALIASES):
                continue
            hits = [c for c in cols if c not in found and _token_match(v, c)]
            if 1 <= len(hits) <= 6:
                for c in hits:
                    found[c] = (span[0], span[0] + len(v))
                break
    # 3단계: "첫 번째 열", "3번째 변수", "마지막 열"
    for m in re.finditer(r"(첫|두|세|네|다섯|\d+)번째(?:열|변수|컬럼|칸)|(마지막|끝)(?:열|변수|컬럼|칸)", t):
        idx = -1 if m.group(2) else {"첫": 0, "두": 1, "세": 2, "네": 3, "다섯": 4}.get(m.group(1), None)
        idx = int(m.group(1)) - 1 if idx is None else idx
        if -1 <= idx < len(cols) and cols[idx] not in found:
            found[cols[idx]] = (m.start(), m.end())
    return sorted((s, e, c) for c, (s, e) in found.items())


def find_cols(text, cols):
    """문장에 나온 열을 나온 순서대로 찾는다. (위치, 열) 목록."""
    return [(s, c) for s, _, c in _find(text, cols)]


X_MARK = ("x는", "x은", "x를", "x=", "x축", "x로", "독립변수", "입력변수", "입력은", "입력으로", "설명변수")
Y_MARK = ("y는", "y은", "y를", "y=", "y축", "y로", "종속변수", "목표변수", "출력변수", "출력은", "예측대상", "타깃", "타겟", "반응변수")
EXCLUDE = ("빼고", "제외", "없이", "말고", "빼줘", "빼")
RELATION = ("예측", "맞춰", "추정", "에따른", "에따라", "관계", "영향", "설명")


def has_var_marker(t):
    return any(m in t for m in X_MARK + Y_MARK)


def parse_vars(state, text):
    """문장에서 X·Y 변수 지정을 읽어 반영한다. 바뀌었으면 안내 문장을, 아니면 None을 돌려준다."""
    if state["df"] is None:
        return None
    num = [c for c in numeric_cols(state) if state["df"][c].nunique(dropna=True) > 1]
    t = norm(text)
    found = _find(text, num)
    if {"x", "y"} <= {norm(short(c)) for c in num}:           # 열 이름이 x, y인 데이터에서 "x는", "y는"은 표시어가 아님
        marks = []
    else:
        marks = sorted([(t.find(m), "x") for m in X_MARK if m in t] + [(t.find(m), "y") for m in Y_MARK if m in t])
    new_x, new_y = list(state["x"]), state["y"]

    # "물 빼고", "재령이랑 물 제외" → X에서 제외
    excluded = []
    for w in EXCLUDE:
        for m in re.finditer(w, t):
            edge, last = m.start(), None
            for s, e, c in sorted(found, reverse=True):
                if (e <= edge and edge - e <= 4) or (s, e) == last:
                    if c not in excluded:
                        excluded.append(c)
                    edge, last = s, (s, e)
    found = [f for f in found if f[2] not in excluded]

    if marks:
        xs, ys = [], []
        for s, _, col in found:
            before = [k for p, k in marks if p <= s]
            if before:
                (xs if before[-1] == "x" else ys).append(col)
        if ys:
            new_y = ys[0]
        if xs:
            new_x = xs
    elif found and any(w in t for w in RELATION):
        # "마력으로 연비 예측", "무게에 따른 연비", "키와 몸무게의 관계" → 마지막에 말한 것이 Y
        cut = min([t.find(w) for w in ("예측", "맞춰", "추정") if w in t] or [len(t)])
        before = [f for f in found if f[0] < cut] or found
        as_x = [c for s, e, c in before if t[e:e + 2] == "으로" or t[e:e + 1] == "로"]
        rest = [c for _, _, c in before if c not in as_x]
        if rest and (len(before) >= 2 or not as_x):
            new_y = rest[-1]
            others = [c for _, _, c in found if c != new_y]
            new_x = others or [c for c in num if c != new_y]
        elif as_x:
            new_x = as_x
    elif found:
        cand = [c for _, _, c in found if c != new_y]
        if cand:
            new_x = cand
    if any(w in t for w in ("모든변수", "전체변수", "전부다", "변수전부", "변수전체", "변수다", "모든입력", "나머지전부", "나머지다", "나머지모두")):
        new_x = [c for c in num if c != new_y]

    new_x = [c for c in new_x if c != new_y and c not in excluded] or [c for c in num if c != new_y and c not in excluded]
    if not new_x or (new_x == state["x"] and new_y == state["y"]):
        return None
    state.update(x=new_x, y=new_y, results={})
    state["inputs"] = {c: v for c, v in (state.get("inputs") or {}).items() if c in new_x}
    xs = ", ".join(short(c) for c in new_x[:12]) + (f" 외 {len(new_x) - 12}개" if len(new_x) > 12 else "")
    out = f"🎯 변수를 정했어요. Y = **{short(new_y)}**, X = {xs}"
    if excluded:
        out += f" (제외: {', '.join(short(c) for c in excluded)})"
    return out


def columns(state):
    df = state["df"]
    rows = []
    for c in df.columns:
        is_num = pd.api.types.is_numeric_dtype(df[c])
        role = "Y (예측 대상)" if c == state["y"] else "X (입력)" if c in state["x"] else "사용 안 함" if is_num else "글자 (분석 제외)"
        ex = ", ".join(str(v) for v in df[c].dropna().unique()[:3])
        rows.append((c, "숫자" if is_num else "글자", role, int(df[c].isna().sum()), ex))
    return reply(f"**{state['name']}**의 열 {df.shape[1]}개예요. 열 이름은 일부만 말하거나 한국어로 말해도 찾아요.",
                 [("table", pd.DataFrame(rows, columns=["열 이름", "종류", "역할", "빈칸", "예시 값"]))])


STATS = (("평균", ("평균", "mean", "average")), ("중앙값", ("중앙값", "중간값", "median")), ("최댓값", ("최대", "최댓", "가장큰", "제일큰", "가장높", "제일높", "max")),
         ("최솟값", ("최소", "최솟", "가장작", "제일작", "가장낮", "제일낮", "min")), ("표준편차", ("표준편차", "std")), ("분산", ("분산", "variance")),
         ("합계", ("합계", "총합", "더하면", "sum")), ("개수", ("개수", "갯수", "몇개", "몇행", "몇줄", "count")))


def which_stats(t):
    return [name for name, words in STATS if any(w in t for w in words)]


def column_stats(state, text):
    df, t = state["df"], norm(text)
    num = numeric_cols(state)
    cols = [c for _, c in find_cols(text, num)] or ([state["y"]] if len(num) > 12 else num)
    names = which_stats(t) or ["평균", "표준편차", "최솟값", "최댓값"]
    fn = {"평균": "mean", "중앙값": "median", "최댓값": "max", "최솟값": "min", "표준편차": "std", "분산": "var", "합계": "sum", "개수": "count"}
    table = pd.DataFrame({"변수": [short(c) for c in cols], **{n: [round(float(getattr(df[c], fn[n])()), 4) for c in cols] for n in names}})
    if len(cols) == 1 and len(names) <= 2:
        text_out = f"**{short(cols[0])}**의 " + ", ".join(f"{n}은 **{table[n][0]:g}**" for n in names) + f"이에요. (전체 {len(df):,}행)"
    else:
        text_out = f"요청한 통계예요. (전체 {len(df):,}행)"
    return reply(text_out, [("table", table)])


def plot_cols(state, text):
    df, t = state["df"], norm(text)
    num = numeric_cols(state)
    cols = [c for _, c in find_cols(text, num)]
    hist = any(w in t for w in ("히스토그램", "분포", "histogram"))
    if hist or len(cols) == 1:
        c = cols[0] if cols else state["y"]
        fig = go.Figure(go.Histogram(x=df[c], nbinsx=30, marker_color=COLORS[1]))
        fig.update_layout(title=f"{short(c)}의 분포", xaxis_title=short(c), yaxis_title="개수", height=420, margin=dict(t=50, b=20))
        return reply(f"**{short(c)}**의 분포예요. 평균 {df[c].mean():.4g}, 표준편차 {df[c].std():.4g}, 범위 {df[c].min():.4g} ~ {df[c].max():.4g}.",
                     [("fig", fig)])
    if len(cols) >= 2:
        xs, y = cols[:-1], cols[-1]
    else:
        xs, y = state["x"][:1], state["y"]
    fig = go.Figure()
    for i, c in enumerate(xs[:4]):
        fig.add_trace(go.Scatter(x=df[c], y=df[y], mode="markers", name=short(c), marker=dict(color=COLORS[i], size=6, opacity=0.6)))
    fig.update_layout(title=f"{', '.join(short(c) for c in xs[:4])} 와(과) {short(y)}의 관계", xaxis_title=short(xs[0]) if len(xs) == 1 else "입력 변수",
                      yaxis_title=short(y), height=460, margin=dict(t=50, b=20), legend=dict(orientation="h", y=-0.2))
    r = df[[xs[0], y]].corr().iloc[0, 1]
    return reply(f"**{short(xs[0])}**와 **{short(y)}**의 산점도예요. 상관계수는 {r:+.3f}이에요 "
                 f"({'강한' if abs(r) >= 0.7 else '어느 정도의' if abs(r) >= 0.4 else '약한'} {'양' if r >= 0 else '음'}의 관계).", [("fig", fig)])


def data_brief(state, limit=10):
    """언어모델에게 알려 줄 데이터 요약 (열 이름과 기본 통계)."""
    df = state["df"]
    lines = [f"- 데이터 '{state['name']}': {len(df)}행, 열 = {', '.join(short(c) for c in df.columns[:20])}"]
    for c in ([state["y"]] + state["x"])[:limit]:
        lines.append(f"- {short(c)}: 평균 {df[c].mean():.4g}, 최소 {df[c].min():.4g}, 최대 {df[c].max():.4g}")
    return lines


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
    cols = state["x"][:14] + [state["y"]]
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
    d = state["df"][state["x"] + [state["y"]]].replace([np.inf, -np.inf], np.nan)
    d = d[d[state["y"]].notna()]
    d = d.fillna(d.median(numeric_only=True)).dropna()        # 입력의 빈칸은 중앙값으로 채움
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
    """'시멘트 300, 물 180'처럼 변수 이름 뒤에 온 숫자를 읽는다. {열: [값, …]}.

    이름은 일부·한국어·영어 모두 인식하고, '재령 7, 28, 90'처럼 값을 여러 개 주면 모두 읽는다.
    """
    t = norm(text)
    vals = {}
    cols = [c for c in numeric_cols(state) if c != state["y"]]     # X에 없는 변수도 값을 주면 읽음
    for _, e, c in _find(text, cols):
        m = re.match(r"[^\d\-]{0,6}(-?\d+(?:\.\d+)?)((?:(?:,|와|과|랑|및|/)-?\d+(?:\.\d+)?)*)", t[e:])
        if m and not re.match(r"(에서|부터|~|-)\d", t[e + m.end(1):]):       # "100에서 500까지"는 범위(parse_sweep)
            vals[c] = [float(m.group(1))] + [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", m.group(2))]
    return vals


SWEEP_WORDS = ("늘어나", "늘리", "증가", "커지", "높아", "높이", "올라", "올리", "줄어", "줄이", "감소", "작아", "낮아", "낮추", "내리",
               "변하", "변화", "바꾸면", "바뀌면", "달라지", "민감도")


def parse_sweep(state, text):
    """'시멘트를 100에서 500까지 바꾸면', '물이 늘어나면 강도는?' → (열, 시작, 끝). 아니면 None."""
    t = norm(text)
    for _, e, c in _find(text, state["x"]):
        m = re.match(r"[^\d\-]{0,6}(-?\d+(?:\.\d+)?)[^\d\-]{0,3}(?:에서|부터|~|-)(-?\d+(?:\.\d+)?)", t[e:])
        if m:
            return c, float(m.group(1)), float(m.group(2))
    if any(w in t for w in SWEEP_WORDS) and not re.search(r"\d", t):
        hit = _find(text, state["x"])
        if hit:
            d = state["df"][hit[0][2]]
            return hit[0][2], float(d.min()), float(d.max())
    return None


def _base_row(state, keep):
    """말하지 않은 변수에 넣을 값. keep이면 직전 입력값을 이어 쓰고, 없으면 평균(0/1 변수는 중앙값)."""
    d = state["df"][state["x"]]
    last = state.get("inputs") or {}
    base, kept = {}, []
    for c in state["x"]:
        if keep and c in last:
            base[c] = last[c]
            kept.append(c)
        else:
            base[c] = float(d[c].median() if d[c].nunique() <= 2 else d[c].mean())
    return base, kept


def _ensure_models(state, notes):
    if not state["results"]:
        run_linear(state)
        run_ann(state)
        notes.append("학습된 모델이 없어서 선형 회귀와 인공신경망을 먼저 학습했어요.")
    return sorted(state["results"].values(), key=lambda r: -r["test_r2"])


def _predict(r, X):
    with np.errstate(all="ignore"):
        p = np.asarray(r["predict"](np.asarray(X, float)), float).ravel()
    return np.where(np.isfinite(p), p, np.nan)


KEEP_WORDS = ("바꾸", "바꿔", "바뀌", "올리", "내리", "늘리", "줄이", "그대로", "대신", "이번엔", "이번에는", "나머지는같", "만")


def run_predict(state, vals, text=""):
    """입력값에 따라 예측한다. 값을 여러 개 주면 경우별로, 일부만 바꾸면 직전 입력을 이어서 계산한다."""
    notes = []
    extra = [c for c in vals if c not in state["x"]]
    if extra:                                                # 입력 변수에 없던 변수의 값을 주면 변수에 추가하고 다시 학습
        trained = list(state["results"]) or ["linear", "ann"]
        state.update(x=state["x"] + extra, results={})
        for k in trained:
            RUNNERS[k](state)
        notes.append(f"{', '.join(short(c) for c in extra)}은(는) 입력 변수에 없어서 추가하고 모델을 다시 학습했어요.")
    rs = _ensure_models(state, notes)
    t = norm(text)
    keep = bool(state.get("inputs")) and any(w in t for w in KEEP_WORDS)
    base, kept = _base_row(state, keep)
    d = state["df"][state["x"]]
    cases = [dict(base)]
    for c, vs in vals.items():                               # 여러 값을 준 변수는 경우를 나눔
        cases = [dict(row, **{c: v}) for row in cases for v in vs][:24]
    X = np.array([[row[c] for c in state["x"]] for row in cases])
    outside = [short(c) for c, vs in vals.items() if any(not d[c].min() <= v <= d[c].max() for v in vs)]
    filled = [short(c) for c in state["x"] if c not in vals and c not in kept]
    best = rs[0]
    yname = short(state["y"])
    state["inputs"] = dict(cases[-1])
    if kept and [c for c in kept if c not in vals]:
        notes.append("말하지 않은 변수는 직전 입력값을 그대로 썼어요: " + ", ".join(f"{short(c)} {base[c]:g}" for c in kept if c not in vals))
    if filled:
        notes.append(f"말하지 않은 변수({', '.join(filled[:8])}{' 등' if len(filled) > 8 else ''})는 데이터 평균값으로 채웠어요.")
    if outside:
        notes.append(f"⚠️ {', '.join(outside)} 값이 학습 데이터 범위를 벗어나서 예측을 믿기 어려워요.")

    if len(cases) == 1:
        preds = [(r["label"], round(float(_predict(r, X)[0]), 3), round(r["test_r2"], 3)) for r in rs]
        table = pd.DataFrame(preds, columns=["모델", f"예측 {yname}", "모델의 테스트 R²"])
        inp = pd.DataFrame({"변수": [short(c) for c in state["x"]], "입력값": np.round(X[0], 3),
                            "출처": ["입력" if c in vals else "직전 입력" if c in kept else "평균" for c in state["x"]]})
        said = ", ".join(f"{short(c)} {v[0]:g}" for c, v in vals.items())
        text_out = (f"### 🔮 예측 결과\n입력: {said}\n\n가장 정확한 모델인 **{best['label']}**의 예측은 **{yname} = {preds[0][1]}**이에요.\n\n"
                    + "\n".join(f"- {n}" for n in notes)
                    + "\n\n값을 바꿔 보려면 \"물을 200으로 바꾸면?\"처럼, 변화를 보려면 \"시멘트를 100에서 500까지 바꾸면?\"처럼 말하세요.")
        return reply(text_out, [("table", table), ("table", inp)], f"예측({said}): {best['label']} → {yname} {preds[0][1]}")

    varying = [c for c, vs in vals.items() if len(vs) > 1]
    table = pd.DataFrame({**{short(c): X[:, state["x"].index(c)] for c in vals},
                          **{f"{r['label']}": np.round(_predict(r, X), 3) for r in rs}})
    items = [("table", table)]
    if len(varying) == 1:
        xi = X[:, state["x"].index(varying[0])]
        fig = go.Figure(go.Bar(x=[f"{v:g}" for v in xi], y=_predict(best, X), marker_color=COLORS[1],
                               text=[f"{v:.2f}" for v in _predict(best, X)], textposition="outside"))
        fig.update_layout(title=f"{short(varying[0])} 값에 따른 예측 {yname} ({best['label']})", xaxis_title=short(varying[0]),
                          yaxis_title=f"예측 {yname}", height=400, margin=dict(t=50, b=20))
        items.insert(0, ("fig", fig))
    pb = _predict(best, X)
    text_out = (f"### 🔮 경우별 예측 결과 ({len(cases)}가지)\n가장 정확한 모델 **{best['label']}** 기준으로 {yname}는 "
                f"**{np.nanmin(pb):.3f} ~ {np.nanmax(pb):.3f}** 사이예요. 표에 모델별 예측을 모두 적었어요.\n\n" + "\n".join(f"- {n}" for n in notes))
    return reply(text_out, items, f"경우별 예측 {len(cases)}가지: {best['label']} → {yname} {np.nanmin(pb):.2f}~{np.nanmax(pb):.2f}")


def run_sweep(state, col, lo, hi, vals=None, text=""):
    """한 변수를 lo~hi로 바꿀 때 예측값이 어떻게 변하는지 그린다 (나머지 변수는 고정)."""
    notes = []
    rs = _ensure_models(state, notes)
    lo, hi = min(lo, hi), max(lo, hi)
    if lo == hi:
        return reply("범위의 시작과 끝이 같아요. \"시멘트를 100에서 500까지 바꾸면?\"처럼 말해 주세요.")
    keep = bool(state.get("inputs")) and (any(w in norm(text) for w in KEEP_WORDS) or not vals)
    base, kept = _base_row(state, keep)
    for c, vs in (vals or {}).items():
        if c != col:
            base[c] = vs[0]
    grid = np.linspace(lo, hi, 60)
    X = np.array([[g if c == col else base[c] for c in state["x"]] for g in grid])
    fig = go.Figure()
    for i, r in enumerate(rs[:4]):
        fig.add_trace(go.Scatter(x=grid, y=_predict(r, X), mode="lines", name=f"{r['label']} (R² {r['test_r2']:.2f})",
                                 line=dict(color=COLORS[i % len(COLORS)], width=3 if i == 0 else 2, dash=None if i == 0 else "dot")))
    d = state["df"][col]
    fig.add_vrect(x0=float(d.min()), x1=float(d.max()), fillcolor="#888", opacity=0.08, line_width=0)
    yname = short(state["y"])
    fig.update_layout(title=f"{short(col)}을(를) 바꿀 때 예측 {yname}의 변화", xaxis_title=short(col), yaxis_title=f"예측 {yname}",
                      height=460, margin=dict(t=50, b=20), legend=dict(orientation="h", y=-0.25))
    best = rs[0]
    p = _predict(best, X)
    idx = [0, 15, 30, 45, 59]
    table = pd.DataFrame({short(col): np.round(grid[idx], 3), **{r["label"]: np.round(_predict(r, X)[idx], 3) for r in rs[:4]}})
    change = p[-1] - p[0]
    peak = grid[int(np.nanargmax(p))]
    shape = ("계속 커져요" if np.all(np.diff(p) >= -1e-9) else "계속 작아져요" if np.all(np.diff(p) <= 1e-9)
             else f"{peak:.4g} 근처에서 가장 커요 (직선이 아닌 관계)")
    fixed = ", ".join(f"{short(c)} {base[c]:.4g}" for c in state["x"] if c != col)[:300]
    if lo < d.min() or hi > d.max():
        notes.append(f"⚠️ 회색 영역(학습 데이터 범위 {d.min():g} ~ {d.max():g}) 밖은 예측을 믿기 어려워요.")
    if fixed:
        notes.append(f"나머지 변수는 고정했어요: {fixed}")
    text_out = (f"### 📐 {short(col)} 변화에 따른 {yname}\n{short(col)}을(를) **{lo:g} → {hi:g}**로 바꾸면, **{best['label']}** 기준 예측 {yname}는 "
                f"**{p[0]:.3f} → {p[-1]:.3f}** ({change:+.3f})로 변하고, {shape}.\n\n" + "\n".join(f"- {n}" for n in notes))
    return reply(text_out, [("fig", fig), ("table", table)],
                 f"{short(col)} {lo:g}→{hi:g}일 때 {yname} {p[0]:.2f}→{p[-1]:.2f} ({best['label']})")


FUNCS = {"log": np.log, "ln": np.log, "sqrt": np.sqrt, "exp": np.exp, "abs": np.abs, "로그": np.log, "루트": np.sqrt, "제곱근": np.sqrt}


def add_variable(state, name, expr):
    """'물시멘트비 = 물 / 시멘트'처럼 사용자가 정의한 새 변수를 만든다."""
    df = state["df"]
    num = numeric_cols(state)
    expr = expr.replace("^", "**").replace("×", "*").replace("÷", "/").replace("나누기", "/").replace("곱하기", "*").replace("더하기", "+").replace("빼기", "-")
    env, parts, used = {}, [], []
    for tok in re.findall(r"\d+\.?\d*|\*\*|[-+*/()]|[가-힣A-Za-z_][\w가-힣]*(?:\s*\([^()]*\))?", expr):
        if re.fullmatch(r"\d+\.?\d*|\*\*|[-+*/()]", tok):
            parts.append(tok)
            continue
        word = re.sub(r"\s*\(.*\)$", "", tok)
        if word in FUNCS and tok != word:                   # log(물) 같은 함수
            inner = [c for _, c in find_cols(tok[len(word):], num)]
            if len(inner) != 1:
                return reply(f"⚠️ '{tok}' 안의 변수를 찾지 못했어요.")
            env[f"v{len(env)}"] = FUNCS[word](df[inner[0]].astype(float))
            parts.append(f"v{len(env) - 1}")
            used.append(f"{word}({short(inner[0])})")
            continue
        hits = [c for _, c in find_cols(tok, num)]
        if len(hits) == 1:
            env[f"v{len(env)}"] = df[hits[0]].astype(float)
            parts.append(f"v{len(env) - 1}")
            used.append(short(hits[0]))
        elif parts and parts[-1] not in "+-*/(**":           # 식이 끝난 뒤의 말("변수 만들어줘")은 무시
            break
        else:
            return reply(f"⚠️ '{tok}'에 해당하는 열을 찾지 못했어요. \"열 목록 보여줘\"로 이름을 확인해 보세요.")
    code = " ".join(parts)
    if not env or not re.fullmatch(r"[v\d\s.+\-*/()]+", code):
        return reply("⚠️ 식을 이해하지 못했어요. 예: \"물시멘트비 = 물 / 시멘트 변수 만들어줘\"")
    try:
        with np.errstate(all="ignore"):
            new = pd.Series(eval(code, {"__builtins__": {}}, env), index=df.index).replace([np.inf, -np.inf], np.nan)   # 숫자·연산자·v0…만 통과한 식
    except Exception as e:
        return reply(f"⚠️ 식을 계산하지 못했어요: {e}")
    name = name.strip()[:30]
    df[name] = new
    if name != state["y"] and name not in state["x"]:
        state["x"] = state["x"] + [name]
    state.update(results={}, inputs={})
    remember(f"새 변수 '{name}' 만듦 ({', '.join(used)} 사용)")
    return reply(f"🧮 새 변수 **{name}**을(를) 만들어 입력 변수(X)에 넣었어요. 평균 {new.mean():.4g}, 범위 {new.min():.4g} ~ {new.max():.4g}.\n\n"
                 f"- Y: {short(state['y'])}\n- X ({len(state['x'])}개): {', '.join(short(c) for c in state['x'][:12])}\n\n"
                 "모델을 다시 학습해야 해요. \"전부 비교해줘\"라고 해 보세요.", [("table", df[[name]].describe().T.round(4).reset_index(names="변수"))],
                 f"새 변수 {name} 생성")


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
