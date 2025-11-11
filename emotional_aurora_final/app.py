import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from PIL import Image, ImageFilter, ImageDraw
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import requests
from datetime import date

# =========================================================
#                 STREAMLIT SETUP
# =========================================================
st.set_page_config(
    page_title="Emotional InkPrint — Final",
    page_icon="🖨️",
    layout="wide"
)

st.title("🖨️ Emotional InkPrint — Final Version")

# =========================================================
#                     VADER
# =========================================================
@st.cache_resource(show_spinner=False)
def load_vader():
    try:
        nltk.data.find("sentiment/vader_lexicon")
    except LookupError:
        nltk.download("vader_lexicon")
    return SentimentIntensityAnalyzer()

sia = load_vader()

def analyze_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return {"neg":0.0,"neu":1.0,"pos":0.0,"compound":0.0}
    return sia.polarity_scores(text)

# =========================================================
#               NewsAPI Fetching
# =========================================================
def fetch_news(api_key, keyword="technology", page_size=50):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": keyword,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": api_key,
    }

    try:
        resp = requests.get(url, params=params, timeout=12)
        data = resp.json()
        if data.get("status") != "ok":
            st.warning("NewsAPI error: " + str(data.get("message")))
            return pd.DataFrame()

        rows=[]
        for a in data.get("articles", []):
            txt = (a.get("title") or "") + " — " + (a.get("description") or "")
            rows.append({
                "timestamp": (a.get("publishedAt") or "")[:10],
                "text": txt.strip(" — "),
                "source": (a.get("source") or {}).get("name","")
            })
        return pd.DataFrame(rows)

    except Exception as e:
        st.error(f"Error fetching NewsAPI: {e}")
        return pd.DataFrame()

# =========================================================
#        ★ NEW — Ultra-bright “InkPrint” Emotion Palette
# =========================================================
# 你确认 OK 的颜色（高饱和/不蓝/更区分）
DEFAULT_RGB = {
    "joy":        (255,210,60),     # 金黄
    "love":       (255,75,130),     # 艳粉
    "pride":      (190,100,255),    # 紫罗兰
    "hope":       (255,184,74),     # ★ 亮橙黄
    "curiosity":  (0,185,255),      # 天蓝青
    "calm":       (76,242,163),     # ★ 薄荷亮绿
    "surprise":   (255,160,70),     # 杏橙
    "neutral":    (190,190,200),    # 中灰
    "sadness":    (70,140,230),     # 海蓝
    "anger":      (250,70,60),      # 红
    "fear":       (150,70,200),     # 紫
    "disgust":    (135,185,60),     # 绿橄榄
    "anxiety":    (255,190,50),     # 琥珀黄
    "boredom":    (140,140,150),    # 灰蓝
    "nostalgia":  (255,220,160),    # 奶杏
    "gratitude":  (70,230,230),     # 青绿
    "awe":        (255,75,75),      # ★ 樱桃红
    "trust":      (50,200,165),     # 海松绿
    "confusion":  (255,150,190),    # 粉橘
    "mixed":      (240,200,110),    # 金杏
}

ALL_EMOTIONS = list(DEFAULT_RGB.keys())

COLOR_NAMES = {
    "joy":"Gold",
    "love":"Magenta",
    "pride":"Violet",
    "hope":"Bright Orange",
    "curiosity":"Sky Blue",
    "calm":"Mint Green",
    "surprise":"Peach",
    "neutral":"Gray",
    "sadness":"Ocean",
    "anger":"Red",
    "fear":"Purple",
    "disgust":"Olive",
    "anxiety":"Amber",
    "boredom":"Slate",
    "nostalgia":"Cream",
    "gratitude":"Aqua",
    "awe":"Cherry Red",
    "trust":"Teal Green",
    "confusion":"Pink",
    "mixed":"Amber Gold"
}

# =========================================================
#              Sentiment → Emotion Mapping
# =========================================================
def classify_emotion_expanded(row):
    pos, neu, neg, comp = row["pos"], row["neu"], row["neg"], row["compound"]

    if comp >= 0.7 and pos > 0.5: return "joy"
    if comp >= 0.55 and pos > 0.45: return "love"
    if comp >= 0.45 and pos > 0.40: return "pride"

    if 0.25 <= comp < 0.45 and pos > 0.30: return "hope"      # 橙黄
    if 0.10 <= comp < 0.25 and neu >= 0.5: return "calm"      # 薄荷绿
    if 0.25 <= comp < 0.60 and neu < 0.5: return "surprise"

    if comp <= -0.65 and neg > 0.5: return "anger"
    if -0.65 < comp <= -0.40 and neg > 0.45: return "fear"
    if -0.40 < comp <= -0.15 and neg >= 0.35: return "sadness"
    if neg > 0.5 and neu > 0.3: return "anxiety"
    if neg > 0.45 and pos < 0.1: return "disgust"

    if neu > 0.75 and abs(comp) < 0.1: return "boredom"
    if pos > 0.35 and neu > 0.4 and 0.0 <= comp < 0.25: return "trust"
    if pos > 0.30 and neu > 0.35 and -0.05 <= comp <= 0.05: return "nostalgia"
    if pos > 0.25 and neg > 0.25: return "mixed"
    if pos > 0.20 and neu > 0.50 and comp > 0.05: return "curiosity"
    if neu > 0.6 and 0.05 <= comp <= 0.15: return "awe"       # 樱桃红

    return "neutral"

# =========================
# Palette state
# =========================
def init_palette_state():
    if "use_csv_palette" not in st.session_state:
        st.session_state["use_csv_palette"] = False
    if "custom_palette" not in st.session_state:
        st.session_state["custom_palette"] = {}

def get_active_palette():
    if st.session_state.get("use_csv_palette", False):
        return dict(st.session_state.get("custom_palette", {}))
    merged = dict(DEFAULT_RGB)
    merged.update(st.session_state.get("custom_palette", {}))
    return merged

def add_custom_emotion(name, r, g, b):
    if not name: 
        return
    st.session_state["custom_palette"][name.strip()] = (int(r), int(g), int(b))

def import_palette_csv(file):
    try:
        dfc = pd.read_csv(file)
        need = {"emotion", "r", "g", "b"}
        cols = {c.lower(): c for c in dfc.columns}

        if not need.issubset(cols.keys()):
            st.error("CSV must include emotion, r, g, b columns")
            return

        pal = {}
        for _, row in dfc.iterrows():
            emo = str(row[cols["emotion"]]).strip()
            try:
                r = int(row[cols["r"]])
                g = int(row[cols["g"]])
                b = int(row[cols["b"]])
            except:
                continue
            pal[emo] = (r, g, b)

        st.session_state["custom_palette"] = pal
        st.success(f"Imported {len(pal)} colors from CSV.")

    except Exception as e:
        st.error(f"CSV import error: {e}")

def export_palette_csv(pal):
    """✅ FIXED: This function MUST exist before being called"""
    buf = BytesIO()
    pd.DataFrame([
        {"emotion": k, "r": v[0], "g": v[1], "b": v[2]} 
        for k, v in pal.items()
    ]).to_csv(buf, index=False)
    buf.seek(0)
    return buf

# =========================
# Defaults & Reset
# =========================
DEFAULTS = {
    "keyword": "",
    "ribbons_per_emotion": 14,
    "stroke_width": 4,
    "steps": 420,
    "step_len": 2.2,
    "curve_noise": 0.30,
    "stroke_blur": 0.0,
    "ribbon_alpha": 225,
    # 背景相关：只保留自定义，默认纯黑
    "bg_custom": "#000000",

    # 电影级调色 / 自动亮度
    "auto_bright": True,
    "target_mean": 0.50,
    "abc_strength": 0.90,
    "abc_black": 0.05,
    "abc_white": 0.997,
    "abc_max_gain": 2.6,

    "exp": 0.50,
    "contrast": 1.15,
    "saturation": 1.12,
    "gamma_val": 0.92,
    "roll": 0.40,
    "temp": -0.05,
    "tint": 0.02,
    "sh_r": 0.08, "sh_g": 0.06, "sh_b": 0.16,
    "hi_r": 0.10, "hi_g": 0.08, "hi_b": 0.06,
    "tone_balance": 0.0,
    "vignette_strength": 0.18,
    "bloom_radius": 7.0,
    "bloom_intensity": 0.42,

    "cmp_min": -1.0,
    "cmp_max": 1.0,

    # 自动选择最多三个情绪
    "auto_top3": True,
    # 仅使用 CSV 调色板
    "use_csv_palette": False,
}

def _hex_to_rgb(hex_str: str):
    s = hex_str.lstrip("#")
    return tuple(int(s[i:i+2], 16) for i in (0,2,4))

def reset_all():
    st.session_state.clear()
    st.rerun()

# =========================
# Sidebar — How to Use（可折叠）
# =========================
with st.expander("How to Use", expanded=False):
    st.markdown("""
**Workflow**
1) 输入关键词（NewsAPI）获取英文新闻文本（输入框里有示例）  
2) 自动进行 VADER → 情绪映射，默认**自动选择 Top-3 情绪**  
3) 调整左侧所有参数（丝带数量/宽度/长度、背景自定义颜色、电影级调色、自动亮度）  
4) 右侧可查看数据表并下载 PNG
""")

# =========================
# 1) Data Source
# =========================
st.sidebar.header("1) Data Source (NewsAPI)")
keyword = st.sidebar.text_input(
    "Keyword",
    value=st.session_state.get("keyword", DEFAULTS["keyword"]),
    key="keyword",
    placeholder="e.g., AI, climate, aurora, cinema"
)
fetch_btn = st.sidebar.button("Fetch News")

df = pd.DataFrame()
if fetch_btn:
    key = st.secrets.get("NEWS_API_KEY", "")
    if not key:
        st.sidebar.error("Missing NEWS_API_KEY in Secrets")
    else:
        df = fetch_news(key, keyword if keyword.strip() else "aurora")

# 本地占位数据（无接口或未点击 Fetch 时）
if df.empty:
    df = pd.DataFrame({"text":[
        "A breathtaking aurora illuminated the northern sky last night.",
        "Calm atmospheric conditions create a beautiful environment.",
        "Anxiety spreads among investors during unstable market conditions.",
        "A moment of awe as the sky shines with green light.",
        "Hope arises as scientific discoveries advance our understanding."
    ]})
    df["timestamp"] = str(date.today())

df["text"] = df["text"].fillna("")
sent_df = df["text"].apply(analyze_sentiment).apply(pd.Series)
df = pd.concat([df.reset_index(drop=True), sent_df.reset_index(drop=True)], axis=1)
df["emotion"] = df.apply(classify_emotion_expanded, axis=1)

# =========================
# 2) Emotion Mapping
# =========================
st.sidebar.header("2) Emotion Mapping")
cmp_min = st.sidebar.slider(
    "Compound Min", -1.0, 1.0,
    st.session_state.get("cmp_min", DEFAULTS["cmp_min"]),
    0.01, key="cmp_min"
)
cmp_max = st.sidebar.slider(
    "Compound Max", -1.0, 1.0,
    st.session_state.get("cmp_max", DEFAULTS["cmp_max"]),
    0.01, key="cmp_max"
)

init_palette_state()
base_palette = get_active_palette()

# 自动选择 Top-3
auto_top3 = st.sidebar.checkbox(
    "Auto-select Top-3 emotions after fetch",
    value=st.session_state.get("auto_top3", DEFAULTS["auto_top3"]),
    key="auto_top3"
)
top3 = []
if auto_top3 and len(df):
    vc = df["emotion"].value_counts()
    top3 = vc.head(3).index.tolist()

# 多选情绪（选项固定为已知全部情绪，默认选 Top-3；若 Top-3 为空则选当前数据出现的全部）
def _label_emotion(e: str) -> str:
    if e in COLOR_NAMES:
        return f"{e} ({COLOR_NAMES[e]})"
    r, g, b = base_palette.get(e, (0, 0, 0))
    return f"{e} (Custom {r},{g},{b})"

options_labels = [_label_emotion(e) for e in ALL_EMOTIONS]
default_emos = top3 if top3 else sorted(df["emotion"].unique().tolist())
default_labels = [_label_emotion(e) for e in default_emos]

selected_labels = st.sidebar.multiselect(
    "Selected Emotions",
    options_labels,
    default=default_labels
)
selected_emotions = [lbl.split(" (")[0] for lbl in selected_labels]

# 过滤数据
df = df[(df["emotion"].isin(selected_emotions)) &
        (df["compound"] >= cmp_min) &
        (df["compound"] <= cmp_max)]

# =========================
# 3) Ribbon Engine
# =========================
st.sidebar.header("3) Ribbon Engine")
ribbons_per_emotion = st.sidebar.slider(
    "Ribbons per Emotion", 2, 40,
    st.session_state.get("ribbons_per_emotion", DEFAULTS["ribbons_per_emotion"]),
    1, key="ribbons_per_emotion"
)
stroke_width = st.sidebar.slider(
    "Stroke Width", 1, 20,
    st.session_state.get("stroke_width", DEFAULTS["stroke_width"]),
    1, key="stroke_width"
)
steps = st.sidebar.slider(
    "Ribbon Length (steps)", 120, 1200,
    st.session_state.get("steps", DEFAULTS["steps"]),
    10, key="steps"
)
step_len = st.sidebar.slider(
    "Step Length (px)", 0.5, 8.0,
    st.session_state.get("step_len", DEFAULTS["step_len"]),
    0.1, key="step_len"
)
curve_noise = st.sidebar.slider(
    "Curve Randomness", 0.00, 0.90,
    st.session_state.get("curve_noise", DEFAULTS["curve_noise"]),
    0.01, key="curve_noise"
)
stroke_blur = st.sidebar.slider(
    "Stroke Softness (blur px)", 0.0, 10.0,
    st.session_state.get("stroke_blur", DEFAULTS["stroke_blur"]),
    0.5, key="stroke_blur"
)
ribbon_alpha = st.sidebar.slider(
    "Ribbon Alpha", 60, 255,
    st.session_state.get("ribbon_alpha", DEFAULTS["ribbon_alpha"]),
    5, key="ribbon_alpha"
)

# 背景：仅自定义颜色（默认纯黑）
st.sidebar.subheader("Background (Solid Color)")
bg_custom = st.sidebar.color_picker(
    "Pick custom color", value=st.session_state.get("bg_custom", DEFAULTS["bg_custom"]), key="bg_custom"
)
bg_rgb = _hex_to_rgb(bg_custom)

# =========================
# 4) Cinematic Color System
# =========================
st.sidebar.header("4) Cinematic Color System")
exp = st.sidebar.slider("Exposure (stops)", -0.2, 1.8, st.session_state.get("exp", DEFAULTS["exp"]), 0.01, key="exp")
contrast = st.sidebar.slider("Contrast", 0.70, 1.80, st.session_state.get("contrast", DEFAULTS["contrast"]), 0.01, key="contrast")
saturation = st.sidebar.slider("Saturation", 0.70, 1.90, st.session_state.get("saturation", DEFAULTS["saturation"]), 0.01, key="saturation")
gamma_val = st.sidebar.slider("Gamma", 0.70, 1.40, st.session_state.get("gamma_val", DEFAULTS["gamma_val"]), 0.01, key="gamma_val")
roll = st.sidebar.slider("Highlight Roll-off", 0.00, 1.50, st.session_state.get("roll", DEFAULTS["roll"]), 0.01, key="roll")

st.sidebar.subheader("White Balance")
temp = st.sidebar.slider("Temperature (Blue ↔ Red)", -1.0, 1.0, st.session_state.get("temp", DEFAULTS["temp"]), 0.01, key="temp")
tint = st.sidebar.slider("Tint (Green ↔ Magenta)", -1.0, 1.0, st.session_state.get("tint", DEFAULTS["tint"]), 0.01, key="tint")

st.sidebar.subheader("Split Toning")
sh_r = st.sidebar.slider("Shadows R", 0.0, 1.0, st.session_state.get("sh_r", DEFAULTS["sh_r"]), 0.01, key="sh_r")
sh_g = st.sidebar.slider("Shadows G", 0.0, 1.0, st.session_state.get("sh_g", DEFAULTS["sh_g"]), 0.01, key="sh_g")
sh_b = st.sidebar.slider("Shadows B", 0.0, 1.0, st.session_state.get("sh_b", DEFAULTS["sh_b"]), 0.01, key="sh_b")
hi_r = st.sidebar.slider("Highlights R", 0.0, 1.0, st.session_state.get("hi_r", DEFAULTS["hi_r"]), 0.01, key="hi_r")
hi_g = st.sidebar.slider("Highlights G", 0.0, 1.0, st.session_state.get("hi_g", DEFAULTS["hi_g"]), 0.01, key="hi_g")
hi_b = st.sidebar.slider("Highlights B", 0.0, 1.0, st.session_state.get("hi_b", DEFAULTS["hi_b"]), 0.01, key="hi_b")
tone_balance = st.sidebar.slider("Tone Balance (Shadows ↔ Highlights)", -1.0, 1.0, st.session_state.get("tone_balance", DEFAULTS["tone_balance"]), 0.01, key="tone_balance")

st.sidebar.subheader("Bloom & Vignette")
bloom_radius = st.sidebar.slider("Bloom Radius (px)", 0.0, 20.0, st.session_state.get("bloom_radius", DEFAULTS["bloom_radius"]), 0.5, key="bloom_radius")
bloom_intensity = st.sidebar.slider("Bloom Intensity", 0.0, 1.0, st.session_state.get("bloom_intensity", DEFAULTS["bloom_intensity"]), 0.01, key="bloom_intensity")
vignette_strength = st.sidebar.slider("Vignette Strength", 0.0, 0.8, st.session_state.get("vignette_strength", DEFAULTS["vignette_strength"]), 0.01, key="vignette_strength")

# =========================
# 5) Auto Brightness
# =========================
st.sidebar.header("5) Auto Brightness")
auto_bright = st.sidebar.checkbox("Enable Auto Brightness", value=st.session_state.get("auto_bright", DEFAULTS["auto_bright"]), key="auto_bright")
target_mean = st.sidebar.slider("Target Mean", 0.30, 0.70, st.session_state.get("target_mean", DEFAULTS["target_mean"]), 0.01, key="target_mean")
abc_strength = st.sidebar.slider("Remap Strength", 0.0, 1.0, st.session_state.get("abc_strength", DEFAULTS["abc_strength"]), 0.05, key="abc_strength")
abc_black = st.sidebar.slider("Black Point %", 0.00, 0.20, st.session_state.get("abc_black", DEFAULTS["abc_black"]), 0.01, key="abc_black")
abc_white = st.sidebar.slider("White Point %", 0.80, 1.00, st.session_state.get("abc_white", DEFAULTS["abc_white"]), 0.001, key="abc_white")
abc_max_gain = st.sidebar.slider("Max Gain", 1.0, 3.0, st.session_state.get("abc_max_gain", DEFAULTS["abc_max_gain"]), 0.05, key="abc_max_gain")

# =========================
# 6) Palette（可自定义/CSV）
# =========================
st.sidebar.header("6) Custom Palette (RGB)")
use_csv = st.sidebar.checkbox(
    "Use CSV palette only",
    value=st.session_state.get("use_csv_palette", DEFAULTS["use_csv_palette"]),
    key="use_csv_palette"
)

with st.sidebar.expander("Add Custom Emotion", False):
    col1, col2, col3, col4 = st.columns([1.8,1,1,1])
    name = col1.text_input("Emotion", key="add_emo")
    r = col2.number_input("R", 0, 255, 210, key="add_r")
    g = col3.number_input("G", 0, 255, 190, key="add_g")
    b = col4.number_input("B", 0, 255, 140, key="add_b")
    if st.button("Add", key="btn_add"):
        add_custom_emotion(name, r, g, b)
    show = st.session_state.get("custom_palette", {})
    if show:
        st.dataframe(
            pd.DataFrame([{"emotion":k,"r":v[0],"g":v[1],"b":v[2]} for k,v in show.items()]),
            use_container_width=True, height=150
        )

with st.sidebar.expander("Import / Export Palette CSV", False):
    up = st.file_uploader("Import CSV", type=["csv"], key="up_csv")
    if up is not None:
        import_palette_csv(up)
    pal = dict(DEFAULT_RGB)
    pal.update(st.session_state.get("custom_palette", {}))
    if st.session_state.get("use_csv_palette", False):
        pal = dict(st.session_state.get("custom_palette", {}))
    if pal:
        st.dataframe(
            pd.DataFrame([{"emotion":k,"r":v[0],"g":v[1],"b":v[2]} for k,v in pal.items()]),
            use_container_width=True, height=160
        )
        dl = export_palette_csv(pal)
        st.download_button("Download CSV", data=dl, file_name="palette.csv", mime="text/csv", key="dl_csv")

# =========================
# 7) Output / Reset
# =========================
st.sidebar.header("7) Output")
if st.sidebar.button("Reset All", type="primary"):
    reset_all()

# =========================
# DRAW
# =========================
left, right = st.columns([0.60, 0.40])

with left:
    st.subheader("🎐 Ribbon Flow")

    working_palette = get_active_palette()

    img = render_ribbons(
        df=df, palette=working_palette,
        width=1500, height=850, seed=np.random.randint(0, 999999),
        ribbons_per_emotion=ribbons_per_emotion, steps=steps, step_len=step_len,
        stroke_width=stroke_width, curve_noise=curve_noise,
        bg_color=bg_rgb, ribbon_alpha=ribbon_alpha, stroke_blur=int(stroke_blur)
    )

    # 电影级处理
    arr = np.array(img).astype(np.float32)/255.0
    lin = srgb_to_linear(arr)
    lin *= (2.0 ** exp)
    lin = apply_white_balance(lin, temp, tint)
    lin = highlight_rolloff(lin, roll)
    arr = linear_to_srgb(np.clip(lin, 0, 4))
    arr = np.clip(filmic_tonemap(arr*1.25), 0, 1)
    arr = adjust_contrast(arr, contrast)
    arr = adjust_saturation(arr, saturation)
    arr = gamma_correct(arr, gamma_val)
    arr = split_tone(arr, (sh_r, sh_g, sh_b), (hi_r, hi_g, hi_b), tone_balance)

    if auto_bright:
        arr = auto_brightness_compensation(
            arr,
            target_mean=target_mean,
            strength=abc_strength,
            black_point_pct=abc_black,
            white_point_pct=abc_white,
            max_gain=abc_max_gain
        )
    arr = apply_bloom(arr, radius=bloom_radius, intensity=bloom_intensity)
    arr = apply_vignette(arr, strength=vignette_strength)
    arr = ensure_colorfulness(arr, min_sat=0.16, boost=1.18)

    final_img = Image.fromarray((np.clip(arr,0,1)*255).astype(np.uint8), mode="RGB")
    buf = BytesIO(); final_img.save(buf, format="PNG"); buf.seek(0)
    st.image(buf, use_column_width=True)
    st.download_button("💾 Download PNG", data=buf, file_name="ribbon_flow_final.png", mime="image/png")

with right:
    st.subheader("📊 Data & Emotion")
    df2 = df.copy()
    df2["emotion_display"] = df2["emotion"].apply(lambda e: f"{e} ({COLOR_NAMES.get(e,'Custom')})")
    cols = ["text", "emotion_display", "compound", "pos", "neu", "neg"]
    if "timestamp" in df.columns: cols.insert(1, "timestamp")
    if "source" in df.columns: cols.insert(2, "source")
    st.dataframe(df2[cols], use_container_width=True, height=600)

st.markdown("---")
st.caption("© 2025 Emotional Ribbon — Final")
