# app.py  — Emotional Ribbon · Orbit Rings (Vinyl) Edition
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw, ImageFilter
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import requests
from datetime import date

# -----------------------------
# App setup
# -----------------------------
st.set_page_config(page_title="Emotional Orbit Rings — Final", page_icon="💿", layout="wide")
st.title("💿 Emotional Orbit Rings — Final")

# -----------------------------
# VADER
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_vader():
    try:
        nltk.data.find("sentiment/vader_lexicon")
    except LookupError:
        nltk.download("vader_lexicon")
    return SentimentIntensityAnalyzer()
sia = load_vader()

# -----------------------------
# NewsAPI
# -----------------------------
def fetch_news(api_key, keyword="technology", page_size=50):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": keyword, "language": "en", "sortBy": "publishedAt",
        "pageSize": page_size, "apiKey": api_key,
    }
    try:
        r = requests.get(url, params=params, timeout=12)
        data = r.json()
        if data.get("status") != "ok":
            st.warning("NewsAPI error: " + str(data.get("message")))
            return pd.DataFrame()
        rows = []
        for a in data.get("articles", []):
            txt = (a.get("title") or "") + " - " + (a.get("description") or "")
            rows.append({
                "timestamp": (a.get("publishedAt") or "")[:10],
                "text": txt.strip(" -"),
                "source": (a.get("source") or {}).get("name", "")
            })
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Error fetching NewsAPI: {e}")
        return pd.DataFrame()

# -----------------------------
# Emotion palette（高饱和、亮丽）
# -----------------------------
DEFAULT_RGB = {
    "joy": (255,210,70),      # 金黄
    "love": (255,120,170),    # 亮玫红
    "pride": (200,120,255),   # 荧紫
    "hope": (70,235,200),     # 薄荷青
    "curiosity": (60,200,255),# 天青
    "calm": (70,150,255),     # 靛蓝
    "surprise": (255,170,90), # 橙杏
    "neutral": (190,190,195), # 中灰
    "sadness": (80,120,220),  # 海蓝
    "anger": (245,70,70),     # 朱红
    "fear": (160,90,220),     # 暗紫
    "disgust": (150,200,70),  # 青橄榄
    "anxiety": (255,200,60),  # 沙金
    "boredom": (130,130,140), # 灰蓝
    "nostalgia": (250,220,170), # 奶杏
    "gratitude": (120,230,230), # 青绿
    "awe": (140,245,255),     # 冰蓝
    "trust": (70,200,170),    # 海松
    "confusion": (255,150,180),# 粉橘
    "mixed": (230,200,120),   # 金杏
}
ALL_EMOTIONS = list(DEFAULT_RGB.keys())

COLOR_NAMES = {
    "joy":"Jupiter Gold","love":"Rose","pride":"Violet","hope":"Mint",
    "curiosity":"Azure","calm":"Indigo","surprise":"Peach","neutral":"Gray",
    "sadness":"Ocean","anger":"Vermilion","fear":"Mulberry","disgust":"Olive",
    "anxiety":"Sand","boredom":"Slate","nostalgia":"Cream","gratitude":"Cyan",
    "awe":"Ice","trust":"Teal","confusion":"Blush","mixed":"Amber"
}

# -----------------------------
# Sentiment → Emotion
# -----------------------------
def analyze_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return {"neg":0.0,"neu":1.0,"pos":0.0,"compound":0.0}
    return sia.polarity_scores(text)

def classify_emotion(row):
    pos, neu, neg, comp = row["pos"], row["neu"], row["neg"], row["compound"]
    if comp >= 0.7 and pos > 0.5: return "joy"
    if comp >= 0.55 and pos > 0.45: return "love"
    if comp >= 0.45 and pos > 0.40: return "pride"
    if 0.25 <= comp < 0.45 and pos > 0.30: return "hope"
    if 0.10 <= comp < 0.25 and neu >= 0.5: return "calm"
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
    if neu > 0.6 and 0.05 <= comp <= 0.15: return "awe"
    return "neutral"

# -----------------------------
# Helpers
# -----------------------------
def rgb01(rgb):  # [0,255] -> [0,1]
    return np.clip(np.array(rgb, dtype=np.float32)/255.0, 0, 1)

def ensure_brightness(c, min_luma=0.35, sat_boost=1.20):
    """提升最低亮度和饱和度，避免线条发灰/发黑"""
    c = np.array(c, dtype=np.float32)
    luma = 0.2126*c[0]+0.7152*c[1]+0.0722*c[2]
    if luma < min_luma:
        c = c + (min_luma - luma)
    lum = 0.2126*c[0]+0.7152*c[1]+0.0722*c[2]
    c = lum + (c - lum)*sat_boost
    return np.clip(c, 0, 1)

def auto_brightness(arr, target_mean=0.50, max_gain=2.4):
    """简单自动亮度（sRGB域），保持整体“更亮”但不过曝"""
    Y = 0.2126*arr[:,:,0] + 0.7152*arr[:,:,1] + 0.0722*arr[:,:,2]
    meanY = max(Y.mean(), 1e-5)
    gain = np.clip(target_mean/meanY, 1.0/max_gain, max_gain)
    out = np.clip(arr * gain, 0, 1)
    return out

# -----------------------------
# Orbit Rings Renderer（唱片等高线）
# -----------------------------
def orbit_rings_image(df, palette, width=1200, height=1200, seed=1234,
                      rings=24, ring_gap=14, line_width=3,
                      noise_amp=18.0, noise_detail=3.0,
                      bg_rgb=(0,0,0), alpha=240):
    """
    以中心为圆心画多条等间距圆环，半径沿角度方向加入细噪声扰动。
    每个情绪用自己的高饱和颜色，叠加出彩色唱片效果。
    """
    rng = np.random.default_rng(seed)
    W, H = width, height
    cx, cy = W//2, H//2
    base = Image.new("RGBA", (W, H), (bg_rgb[0], bg_rgb[1], bg_rgb[2], 255))
    canvas = Image.new("RGBA", (W, H), (0,0,0,0))
    draw = ImageDraw.Draw(canvas, "RGBA")

    # 情绪频率决定该情绪分配的环条数
    vc = df["emotion"].value_counts()
    if vc.empty:
        vc = pd.Series({"calm":1,"awe":1,"hope":1})
    emo_list = vc.index.tolist()
    freq = (vc / vc.sum()).values

    # 计算每个环的颜色（依频率循环）
    colors = []
    for i in range(rings):
        emo = emo_list[i % len(emo_list)]
        base_col = rgb01(palette.get(emo, (220, 220, 220)))
        base_col = ensure_brightness(base_col, min_luma=0.38, sat_boost=1.25)
        # 轻微抖动 + 渐变到白色以更“发光”
        jitter = (rng.random(3)-0.5)*0.10
        c = np.clip(base_col + jitter, 0, 1)
        c = np.clip(0.75*c + 0.25*np.array([1,1,1]), 0, 1)
        colors.append(tuple((c*255).astype(int).tolist()))

    # 角度采样
    n_ang = int(900)  # 细腻
    theta = np.linspace(0, 2*np.pi, n_ang, endpoint=False)

    # 多层噪声
    def noise1d(a):
        # 简单合成噪声：几条不同频率的正弦叠加 + 随机相位
        v = np.zeros_like(a)
        for k in range(1, int(noise_detail)+2):
            phase = rng.uniform(0, 2*np.pi)
            v += np.sin(k*a + phase) / (k**1.2)
        return v

    # 逐环绘制
    base_radius = min(W, H) * 0.20  # 内孔
    for i in range(rings):
        r = base_radius + i*ring_gap
        # 半径噪声
        amp = noise_amp * (0.65 + 0.35*np.sin(i*0.45))
        dr = amp * noise1d(theta)
        rr = np.clip(r + dr, 0, 1e9)

        # 生成折线点
        xs = cx + rr * np.cos(theta)
        ys = cy + rr * np.sin(theta)
        pts = list(zip(xs.astype(float), ys.astype(float)))

        # 绘制到单独图层以便柔化
        layer = Image.new("RGBA", (W, H), (0,0,0,0))
        d = ImageDraw.Draw(layer, "RGBA")
        d.line(pts + [pts[0]], fill=colors[i] + (alpha,), width=line_width, joint="curve")
        # 微柔化叠加，增加“丝滑”
        layer = layer.filter(ImageFilter.GaussianBlur(radius=0.7))
        canvas.alpha_composite(layer)

    base.alpha_composite(canvas)
    return base.convert("RGB")

# -----------------------------
# Defaults & Reset
# -----------------------------
DEFAULTS = {
    "keyword": "",
    "auto_top3": True,
    "cmp_min": -1.0, "cmp_max": 1.0,
    "rings": 26, "ring_gap": 14, "line_width": 3,
    "noise_amp": 18.0, "noise_detail": 3.0,
    "alpha": 240,
    "bg_hex": "#000000",  # 纯黑默认
}
def reset_all():
    st.session_state.clear()
    try:
        st.rerun()
    except Exception:
        # 兼容旧版本
        st.experimental_rerun()

# -----------------------------
# Instructions
# -----------------------------
with st.expander("How to Use", expanded=False):
    st.markdown("""
1) **Keyword**（示例：`aurora borealis`, `election`, `stock market`, `AI ethics`）→ NewsAPI 抓取英文新闻  
2) **VADER** 情感分析 → **情绪**（颜色固定但更亮丽）  
3) **Auto-Top-3**：抓取后自动选出现最多的三个情绪（仍可手动改）  
4) 左侧调 **Orbit Rings** 参数 & 纯色背景  
5) 点击 **Download** 保存 PNG
""")

# -----------------------------
# Sidebar — Data
# -----------------------------
st.sidebar.header("1) Data Source (NewsAPI)")
kw_placeholder = "e.g. aurora borealis / AI ethics / stock market / climate change"
keyword = st.sidebar.text_input("Keyword", value=st.session_state.get("keyword", DEFAULTS["keyword"]), placeholder=kw_placeholder, key="keyword")
fetch_btn = st.sidebar.button("Fetch News")

df = pd.DataFrame()
if fetch_btn:
    key = st.secrets.get("NEWS_API_KEY", "")
    if not key:
        st.sidebar.error("Missing NEWS_API_KEY in Secrets")
    else:
        df = fetch_news(key, keyword if keyword.strip() else "aurora")

if df.empty:
    # 本地示例
    df = pd.DataFrame({"text":[
        "A breathtaking aurora illuminated the northern sky last night.",
        "Calm atmospheric conditions create a beautiful environment.",
        "Anxiety spreads among investors during unstable market conditions.",
        "A moment of awe as the sky shines with green light.",
        "Hope arises as scientific discoveries advance our understanding."
    ]})
    df["timestamp"] = str(date.today())

# 分析情绪
df["text"] = df["text"].fillna("")
scores = df["text"].apply(analyze_sentiment).apply(pd.Series)
df = pd.concat([df.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
df["emotion"] = df.apply(classify_emotion, axis=1)

# -----------------------------
# Sidebar — Emotion
# -----------------------------
st.sidebar.header("2) Emotion Mapping")
cmp_min = st.sidebar.slider("Compound Min", -1.0, 1.0, value=st.session_state.get("cmp_min", DEFAULTS["cmp_min"]), step=0.01, key="cmp_min")
cmp_max = st.sidebar.slider("Compound Max", -1.0, 1.0, value=st.session_state.get("cmp_max", DEFAULTS["cmp_max"]), step=0.01, key="cmp_max")

auto_top3 = st.sidebar.checkbox("Auto-select Top-3 after fetch", value=st.session_state.get("auto_top3", DEFAULTS["auto_top3"]), key="auto_top3")

# 依据当前 df 统计 Top-3
vc = df["emotion"].value_counts()
top3 = vc.head(3).index.tolist() if not vc.empty else ["calm","awe","hope"]

# 在选项标签里显示颜色小块（用hex文本近似）
def label_with_color(e):
    r,g,b = DEFAULT_RGB.get(e, (200,200,200))
    hexc = f"#{r:02X}{g:02X}{b:02X}"
    name = COLOR_NAMES.get(e, "Custom")
    return f"{e}  {hexc}  ({name})"

options = [label_with_color(e) for e in ALL_EMOTIONS]
default_opts = [label_with_color(e) for e in (top3 if auto_top3 else ALL_EMOTIONS)]

sel_labels = st.sidebar.multiselect("Selected Emotions", options, default=default_opts)
selected_emotions = [s.split("  ")[0] for s in sel_labels]

# 过滤数据
df = df[(df["emotion"].isin(selected_emotions)) & (df["compound"] >= cmp_min) & (df["compound"] <= cmp_max)]
if df.empty:
    # 确保渲染不会无数据
    df = pd.DataFrame({"emotion": ["calm","awe","hope"]})

# -----------------------------
# Sidebar — Orbit Rings params
# -----------------------------
st.sidebar.header("3) Orbit Rings")
rings = st.sidebar.slider("Rings", 8, 80, value=st.session_state.get("rings", DEFAULTS["rings"]), step=1, key="rings")
ring_gap = st.sidebar.slider("Ring Gap (px)", 6, 40, value=st.session_state.get("ring_gap", DEFAULTS["ring_gap"]), step=1, key="ring_gap")
line_width = st.sidebar.slider("Line Width (px)", 1, 10, value=st.session_state.get("line_width", DEFAULTS["line_width"]), step=1, key="line_width")
noise_amp = st.sidebar.slider("Noise Amplitude", 0.0, 50.0, value=st.session_state.get("noise_amp", DEFAULTS["noise_amp"]), step=0.5, key="noise_amp")
noise_detail = st.sidebar.slider("Noise Detail (harmonics)", 1.0, 6.0, value=st.session_state.get("noise_detail", DEFAULTS["noise_detail"]), step=0.5, key="noise_detail")
alpha = st.sidebar.slider("Line Alpha", 40, 255, value=st.session_state.get("alpha", DEFAULTS["alpha"]), step=5, key="alpha")

st.sidebar.subheader("Background (Solid)")
bg_hex = st.sidebar.color_picker("Pick color", value=st.session_state.get("bg_hex", DEFAULTS["bg_hex"]), key="bg_hex")
def hex_to_rgb(hx): hx = hx.lstrip("#"); return (int(hx[0:2],16), int(hx[2:4],16), int(hx[4:6],16))
bg_rgb = hex_to_rgb(bg_hex)

# -----------------------------
# Sidebar — Reset
# -----------------------------
st.sidebar.header("4) Output")
if st.sidebar.button("Reset All", type="primary"):
    reset_all()

# -----------------------------
# DRAW
# -----------------------------
left, right = st.columns([0.62, 0.38])

with left:
    st.subheader("💿 Orbit Rings (Vinyl / Contour)")
    seed = np.random.randint(0, 10_000_000)

    img = orbit_rings_image(
        df=df, palette=DEFAULT_RGB,
        width=1500, height=900, seed=seed,
        rings=rings, ring_gap=ring_gap, line_width=line_width,
        noise_amp=noise_amp, noise_detail=noise_detail,
        bg_rgb=bg_rgb, alpha=alpha
    )

    # 轻量级自动亮度，保证“更亮”
    arr = np.array(img).astype(np.float32)/255.0
    arr = auto_brightness(arr, target_mean=0.50, max_gain=2.2)
    final_img = Image.fromarray((np.clip(arr,0,1)*255).astype(np.uint8), mode="RGB")

    buf = BytesIO(); final_img.save(buf, "PNG"); buf.seek(0)
    st.image(buf, use_column_width=True)
    st.download_button("💾 Download PNG", data=buf, file_name="emotional_orbit_rings.png", mime="image/png")

with right:
    st.subheader("📊 Data & Emotions")
    if "text" in df.columns:
        df2 = df.copy()
        df2["color"] = df2["emotion"].apply(lambda e: "#{:02X}{:02X}{:02X}".format(*DEFAULT_RGB.get(e,(200,200,200))))
        show_cols = ["text","emotion","color","compound","pos","neu","neg"]
        if "timestamp" in df.columns: show_cols.insert(1,"timestamp")
        if "source" in df.columns: show_cols.insert(2,"source")
        st.dataframe(df2[show_cols], use_container_width=True, height=600)
    else:
        st.write("No text rows (using fallback emotions for rendering).")

st.caption("© 2025 Emotional Orbit Rings — Colorful Edition")
