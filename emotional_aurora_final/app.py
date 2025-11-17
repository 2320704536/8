# ============================================================
# Emotional Crystal — FINAL FIXED VERSION (with Random Seed Fix)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from PIL import Image, ImageFilter, ImageDraw
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import requests
from datetime import date
import math


# =========================
# App setup
# =========================
st.set_page_config(page_title="Emotional Crystal — Final", page_icon="❄️", layout="wide")
st.title("❄️ Emotional Crystal — Final")

# =========================
# VADER Sentiment
# =========================
@st.cache_resource(show_spinner=False)
def load_vader():
    try:
        nltk.data.find("sentiment/vader_lexicon")
    except LookupError:
        nltk.download("vader_lexicon")
    return SentimentIntensityAnalyzer()

sia = load_vader()


# =========================
# NewsAPI
# =========================
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
        r = requests.get(url, params=params, timeout=12)
        data = r.json()

        if data.get("status") != "ok":
            return pd.DataFrame()

        rows = []
        for a in data.get("articles", []):
            txt = (a.get("title") or "") + " - " + (a.get("description") or "")
            rows.append({
                "timestamp": (a.get("publishedAt") or "")[:10],
                "text": txt.strip(" -"),
                "source": (a.get("source") or {}).get("name","")
            })
        return pd.DataFrame(rows)

    except:
        return pd.DataFrame()


# ============================================================
# Default emotion colors
# ============================================================
DEFAULT_RGB = {
    "joy": (255,200,60),
    "love": (255,95,150),
    "pride": (190,100,255),
    "hope": (60,235,190),
    "curiosity": (50,190,255),
    "calm": (70,135,255),
    "surprise": (255,160,70),
    "neutral": (190,190,200),
    "sadness": (80,120,230),
    "anger": (245,60,60),
    "fear": (150,70,200),
    "disgust": (150,200,60),
    "anxiety": (255,200,60),
    "boredom": (135,135,145),
    "nostalgia": (250,210,150),
    "gratitude": (90,230,230),
    "awe": (120,245,255),
    "trust": (60,200,160),
    "confusion": (255,140,180),
    "mixed": (230,190,110),
}


# ============================================================
# Sentiment → Emotion
# ============================================================
def analyze_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return {"neg":0.0,"neu":1.0,"pos":0.0,"compound":0.0}
    return sia.polarity_scores(text)


def classify_emotion_expanded(row):
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


# ============================================================
# Crystal shape
# ============================================================
def crystal_shape(center=(0.5,0.5), r=150, wobble=0.25, sides_min=5, sides_max=10, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    cx, cy = center
    n = int(rng.integers(sides_min, sides_max+1))

    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    rng.shuffle(angles)

    radii = r * (1 + rng.uniform(-wobble, wobble, size=n))

    pts = []
    for a, rr in zip(angles, radii):
        x = cx + rr * math.cos(a)
        y = cy + rr * math.sin(a)
        pts.append((x,y))

    pts.append(pts[0])
    return pts


def draw_polygon_soft(canvas_rgba, pts, color01, fill_alpha=200, blur_px=6, edge_width=0):
    W,H = canvas_rgba.size
    layer = Image.new("RGBA", (W,H), (0,0,0,0))
    d = ImageDraw.Draw(layer)

    col = (
        int(color01[0]*255),
        int(color01[1]*255),
        int(color01[2]*255),
        int(fill_alpha)
    )

    d.polygon(pts, fill=col)

    if blur_px > 0:
        layer = layer.filter(ImageFilter.GaussianBlur(blur_px))

    canvas_rgba.alpha_composite(layer)


# ============================================================
# Color jitter
# ============================================================
def jitter_color(rgb01, rng, amount=0.06):
    return np.clip(np.array(rgb01) + (rng.random(3)-0.5)*2*amount, 0, 1)


# ============================================================
# Crystal Renderer
# ============================================================
def render_crystalmix(
    df, palette,
    width=1500, height=850, seed=12345,
    shapes_per_emotion=10,
    min_size=60, max_size=220,
    fill_alpha=210, blur_px=6,
    bg_color=(0,0,0),
    wobble=0.25,
    layers=5
):
    rng = np.random.default_rng(seed)

    base = Image.new("RGBA", (width,height),(bg_color[0],bg_color[1],bg_color[2],255))
    canvas = Image.new("RGBA",(width,height),(0,0,0,0))

    emotions = df["emotion"].unique().tolist()
    if not emotions:
        emotions = ["joy","love","curiosity"]

    for _layer in range(layers):
        for emo in emotions:
            base_rgb = palette.get(emo, (200,200,200))
            base01 = tuple(np.array(base_rgb)/255.0)

            for _ in range(shapes_per_emotion):
                cx = rng.uniform(0.05*width, 0.95*width)
                cy = rng.uniform(0.05*height, 0.95*height)
                size = int(rng.uniform(min_size, max_size))

                pts = crystal_shape(
                    center=(cx,cy),
                    r=size,
                    wobble=wobble,
                    rng=rng
                )

                col01 = jitter_color(base01, rng, amount=0.08)
                draw_polygon_soft(
                    canvas, pts, col01,
                    fill_alpha=int(fill_alpha*rng.uniform(0.85,1.05)),
                    blur_px=int(blur_px*rng.uniform(0.7,1.3))
                )

    base.alpha_composite(canvas)
    return base.convert("RGB")


# ============================================================
# ---------- Streamlit UI ----------
# ============================================================
DEFAULTS = {"keyword":"", "layers":3, "wobble":0.25}

st.sidebar.header("1) Data Source")
keyword = st.sidebar.text_input("Keyword", value=DEFAULTS["keyword"])

fetch_btn = st.sidebar.button("Fetch News")
random_btn = st.sidebar.button("Random Mode")


# ============================================================
# Data load
# ============================================================
df = pd.DataFrame()

# ========== RANDOM MODE ==========
if random_btn:

    # Generate 2–5 emotions only
    rng = np.random.default_rng()
    k = rng.integers(2,6)

    emos = []
    texts = []

    st.session_state["custom_palette"] = {}

    for i in range(k):
        emo = f"crystal_{i+1}"
        emos.append(emo)
        texts.append(f"Random fragment {i+1}")

        r = int(rng.integers(0,256))
        g = int(rng.integers(0,256))
        b = int(rng.integers(0,256))
        st.session_state["custom_palette"][emo] = (r,g,b)

    df = pd.DataFrame({
        "text": texts,
        "emotion": emos,
        "compound":0,
        "pos":0, "neu":1, "neg":0
    })

    # 🔥 NEW FIX: Random Mode generates a new seed
    st.session_state["seed_control"] = int(rng.integers(10,26))

    st.experimental_rerun()


# ========== NEWSAPI MODE ==========
elif fetch_btn:
    key = st.secrets.get("NEWS_API_KEY","")
    df = fetch_news(key, keyword if keyword.strip() else "aurora")


# Default fallback
if df.empty:
    df = pd.DataFrame({"text":[
        "A breathtaking aurora illuminated the sky.",
        "Calm atmosphere created a beautiful scene.",
        "Anxiety spreads among investors.",
        "A moment of awe washed over the crowd.",
        "Hope rises with science breakthroughs."
    ]})

df["text"] = df["text"].fillna("")


# Sentiment mapping only for non-random mode
if "emotion" not in df.columns or not df["emotion"].iloc[0].startswith("crystal_"):
    sent = df["text"].apply(analyze_sentiment).apply(pd.Series)
    df = pd.concat([df.reset_index(drop=True), sent.reset_index(drop=True)],axis=1)
    df["emotion"] = df.apply(classify_emotion_expanded,axis=1)


# ============================================================
# Controls
# ============================================================
st.sidebar.header("Crystal Controls")

if "seed_control" not in st.session_state:
    st.session_state["seed_control"] = 20  # default seed

seed_control = st.sidebar.slider("Seed", 0, 500, st.session_state["seed_control"])
st.session_state["seed_control"] = seed_control

layer_count = st.sidebar.slider("Layers", 1, 20, 3)
wobble = st.sidebar.slider("Wobble", 0.0, 1.0, 0.25)

size_min = st.sidebar.slider("Min Size", 20, 300, 80)
size_max = st.sidebar.slider("Max Size", 60, 600, 240)

blur_px = st.sidebar.slider("Blur px", 0.0, 20.0, 6.0)
alpha = st.sidebar.slider("Alpha", 40, 255, 200)
count = st.sidebar.slider("Crystals per Emotion", 2, 40, 10)

bg = st.sidebar.color_picker("Background", "#000000")
bg_rgb = tuple(int(bg.lstrip("#")[i:i+2],16) for i in (0,2,4))

palette = dict(DEFAULT_RGB)
palette.update(st.session_state.get("custom_palette",{}))


# ============================================================
# RENDER
# ============================================================
st.subheader("❄️ Emotional Crystal")

img = render_crystalmix(
    df=df,
    palette=palette,
    width=1500, height=850,
    seed=st.session_state["seed_control"],
    shapes_per_emotion=count,
    min_size=size_min,
    max_size=size_max,
    fill_alpha=alpha,
    blur_px=blur_px,
    bg_color=bg_rgb,
    wobble=wobble,
    layers=layer_count
)

buf = BytesIO()
img.save(buf, format="PNG")
buf.seek(0)

st.image(buf, use_column_width=True)
st.download_button("Download PNG", buf, file_name="crystal.png")


# ============================================================
# Right panel
# ============================================================
st.subheader("Data Preview")
st.dataframe(df)
