# ============================================================
# Emotional Crystal — FINAL FULL VERSION (Part 1 / 8)
# (Includes global seed system for Random + Keyword)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from PIL import Image, ImageFilter, ImageDraw, ImageFont
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
# Instructions
# =========================
with st.expander("Instructions", expanded=False):
    st.markdown("""
### How to Use This Project  

**This project transforms emotion data into cinematic ice-crystal generative visuals.**

**1. Fetch or Generate Data**  
- Enter a keyword and fetch news  
- Or click **Random Generate** for fully generative crystal patterns  

**2. Emotion Classification**  
- Text → sentiment → emotion  
- Only emotions actually appearing in df will show in *Selected Emotions*  

**3. Crystal Rendering**  
- Uses multi-layered generative crystal fragments  
- Each emotion = one color  
- With bloom, glow, tonemap, white balance  

**4. Palettes**  
- Add custom RGB  
- Import/export CSV  
- Use only CSV palette if needed  

**5. Cinematic Color Controls**  
- Exposure / contrast / saturation  
- White balance / split toning / bloom / vignette  
- Auto brightness compensation  

**6. Export Image**  
- Download high-resolution PNG  

---
""")

# ============================================================
# GLOBAL SEED SYSTEM
# ============================================================

def new_seed():
    """Return a new random seed each time."""
    return int(np.random.randint(0, 10_000_000))

# Random mode seed
if "random_seed" not in st.session_state:
    st.session_state["random_seed"] = new_seed()

# Keyword mode seed
if "fetch_seed" not in st.session_state:
    st.session_state["fetch_seed"] = new_seed()

# ============================================================
# VADER
# ============================================================
@st.cache_resource(show_spinner=False)
def load_vader():
    try:
        nltk.data.find("sentiment/vader_lexicon")
    except LookupError:
        nltk.download("vader_lexicon")
    return SentimentIntensityAnalyzer()

sia = load_vader()

# ============================================================
# News API
# ============================================================
def fetch_news(api_key, keyword="technology", page_size=50):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": keyword, "language": "en", "sortBy": "publishedAt",
        "pageSize": page_size, "apiKey": api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=12)
        data = resp.json()
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
# ============================================================
# Emotional Crystal — FINAL (Part 2 / 8)
# Emotion Colors + Sentiment Mapping + Palette System
# ============================================================

# =========================
# Emotion colors
# =========================
DEFAULT_RGB = {
    "joy":        (255,200,60),
    "love":       (255,95,150),
    "pride":      (190,100,255),
    "hope":       (60,235,190),
    "curiosity":  (50,190,255),
    "calm":       (70,135,255),
    "surprise":   (255,160,70),
    "neutral":    (190,190,200),
    "sadness":    (80,120,230),
    "anger":      (245,60,60),
    "fear":       (150,70,200),
    "disgust":    (150,200,60),
    "anxiety":    (255,200,60),
    "boredom":    (135,135,145),
    "nostalgia":  (250,210,150),
    "gratitude":  (90,230,230),
    "awe":        (120,245,255),
    "trust":      (60,200,160),
    "confusion":  (255,140,180),
    "mixed":      (230,190,110),
}

COLOR_NAMES = {
    "joy":"Jupiter Gold","love":"Rose","pride":"Violet","hope":"Mint",
    "curiosity":"Azure","calm":"Indigo","surprise":"Peach","neutral":"Gray",
    "sadness":"Ocean","anger":"Vermilion","fear":"Mulberry","disgust":"Olive",
    "anxiety":"Sand","boredom":"Slate","nostalgia":"Cream","gratitude":"Cyan",
    "awe":"Ice","trust":"Teal","confusion":"Blush","mixed":"Amber"
}

# =========================
# Sentiment → Emotion
# =========================
def analyze_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return {"neg":0.0,"neu":1.0,"pos":0.0,"compound":0.0}
    return sia.polarity_scores(text)

def classify_emotion_expanded(row):
    pos, neu, neg, comp = row["pos"], row["neu"], row["neg"], row["compound"]

    # positive
    if comp >= 0.7 and pos > 0.5: return "joy"
    if comp >= 0.55 and pos > 0.45: return "love"
    if comp >= 0.45 and pos > 0.40: return "pride"
    if 0.25 <= comp < 0.45 and pos > 0.30: return "hope"
    if 0.10 <= comp < 0.25 and neu >= 0.5: return "calm"
    if 0.25 <= comp < 0.60 and neu < 0.5: return "surprise"

    # negative
    if comp <= -0.65 and neg > 0.5: return "anger"
    if -0.65 < comp <= -0.40 and neg > 0.45: return "fear"
    if -0.40 < comp <= -0.15 and neg >= 0.35: return "sadness"
    if neg > 0.5 and neu > 0.3: return "anxiety"
    if neg > 0.45 and pos < 0.1: return "disgust"

    # neutral-ish
    if neu > 0.75 and abs(comp) < 0.1: return "boredom"
    if pos > 0.35 and neu > 0.4 and 0.0 <= comp < 0.25: return "trust"
    if pos > 0.30 and neu > 0.35 and -0.05 <= comp <= 0.05: return "nostalgia"
    if pos > 0.25 and neg > 0.25: return "mixed"
    if pos > 0.20 and neu > 0.50 and comp > 0.05: return "curiosity"
    if neu > 0.6 and 0.05 <= comp <= 0.15: return "awe"

    # fallback
    return "neutral"

# =========================
# Palette system (default + CSV + custom)
# =========================
def init_palette_state():
    if "use_csv_palette" not in st.session_state:
        st.session_state["use_csv_palette"] = False
    if "custom_palette" not in st.session_state:
        st.session_state["custom_palette"] = {}

def get_active_palette():
    """Merge default palette + CSV/custom unless CSV-only is selected."""
    if st.session_state.get("use_csv_palette", False):
        return dict(st.session_state.get("custom_palette", {}))

    merged = dict(DEFAULT_RGB)
    merged.update(st.session_state.get("custom_palette", {}))
    return merged

def add_custom_emotion(name, r, g, b):
    """Add RGB to custom palette."""
    if not name:
        return
    st.session_state["custom_palette"][name.strip()] = (int(r), int(g), int(b))
# ============================================================
# Emotional Crystal — FINAL (Part 3 / 8)
# Crystal Shape + Soft Polygon + Render Engine
# ============================================================

import math
from PIL import Image, ImageFilter, ImageDraw

# =========================
# Crystal Shape (irregular convex polygon)
# =========================
def crystal_shape(center=(0.5, 0.5), r=150, wobble=0.25,
                  sides_min=5, sides_max=10, rng=None):

    if rng is None:
        rng = np.random.default_rng()

    cx, cy = center
    n_vertices = int(rng.integers(sides_min, sides_max + 1))

    angles = np.linspace(0, 2*np.pi, n_vertices, endpoint=False)
    rng.shuffle(angles)

    radii = r * (1 + rng.uniform(-wobble, wobble, size=n_vertices))

    pts = []
    for a, rr in zip(angles, radii):
        x = cx + rr * math.cos(a)
        y = cy + rr * math.sin(a)
        pts.append((float(x), float(y)))

    pts.append(pts[0])
    return pts


# =========================
# Soft Polygon Draw
# =========================
def draw_polygon_soft(canvas_rgba, pts, color01,
                      fill_alpha=200, blur_px=6, edge_width=0):

    W, H = canvas_rgba.size
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer, "RGBA")

    col = (
        int(color01[0] * 255),
        int(color01[1] * 255),
        int(color01[2] * 255),
        int(fill_alpha)
    )

    d.polygon(pts, fill=col)

    if edge_width > 0:
        edge = (255, 255, 255, max(80, fill_alpha // 2))
        d.line(pts, fill=edge, width=edge_width, joint="curve")

    if blur_px > 0:
        layer = layer.filter(ImageFilter.GaussianBlur(radius=blur_px))

    canvas_rgba.alpha_composite(layer)


# =========================
# Color helpers
# =========================
def _rgb01(rgb):
    return np.clip(np.array(rgb, dtype=np.float32) / 255.0, 0, 1)

def vibrancy_boost(rgb, sat_boost=1.30, min_luma=0.40):
    c = _rgb01(rgb)
    luma = 0.2126*c[0] + 0.7152*c[1] + 0.0722*c[2]
    if luma < min_luma:
        c = np.clip(c + (min_luma - luma), 0, 1)
    lum = 0.2126*c[0] + 0.7152*c[1] + 0.0722*c[2]
    c = np.clip(lum + (c - lum) * sat_boost, 0, 1)
    return tuple(c)

def jitter_color(rgb01, rng, amount=0.06):
    j = (rng.random(3) - 0.5) * 2 * amount
    return tuple(np.clip(np.array(rgb01) + j, 0, 1).tolist())


# =========================
# Crystal Render Engine
# =========================
def render_crystalmix(
    df,
    palette,
    width=1500, height=850,
    seed=12345,
    shapes_per_emotion=10,
    min_size=60, max_size=220,
    fill_alpha=210, blur_px=6,
    bg_color=(0, 0, 0),
    wobble=0.25,
    layers=10
):

    rng = np.random.default_rng(seed)

    base = Image.new("RGBA", (width, height),
                     (bg_color[0], bg_color[1], bg_color[2], 255))
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    # IMPORTANT:
    # emotion list from df (OR from random mode)
    emotions = df["emotion"].value_counts().index.tolist()
    if not emotions:
        emotions = ["joy", "love", "curiosity"]

    for _layer in range(layers):
        for emo in emotions:

            base_rgb = palette.get(emo, palette.get("mixed", (230, 190, 110)))
            base01 = vibrancy_boost(base_rgb)

            for _ in range(shapes_per_emotion):

                cx = rng.uniform(0.05 * width, 0.95 * width)
                cy = rng.uniform(0.08 * height, 0.92 * height)
                rr = int(rng.uniform(min_size, max_size))

                pts = crystal_shape(
                    center=(cx, cy),
                    r=rr,
                    wobble=wobble,
                    sides_min=5,
                    sides_max=10,
                    rng=rng
                )

                col01 = jitter_color(base01, rng, amount=0.07)
                local_alpha = int(np.clip(
                    fill_alpha * rng.uniform(0.85, 1.05), 40, 255))
                local_blur = max(0, int(blur_px * rng.uniform(0.7, 1.4)))
                edge_w = 0 if rng.random() < 0.6 else max(1, int(rr * 0.02))

                draw_polygon_soft(
                    canvas,
                    pts,
                    col01,
                    fill_alpha=local_alpha,
                    blur_px=local_blur,
                    edge_width=edge_w
                )

    base.alpha_composite(canvas)
    return base.convert("RGB")
# ============================================================
# Emotional Crystal — FINAL (Part 4 / 8)
# Cinematic Color Pipeline
# ============================================================

import numpy as np
from PIL import Image, ImageFilter

# =========================
# sRGB ↔ Linear
# =========================
def srgb_to_linear(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.04045, x/12.92,
                    ((x + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x < 0.0031308,
                    x * 12.92,
                    1.055 * (x ** (1/2.4)) - 0.055)

# =========================
# Filmic Tonemap (Hable Curve)
# =========================
def filmic_tonemap(x):
    A = 0.22
    B = 0.30
    C = 0.10
    D = 0.20
    E = 0.01
    F = 0.30
    return ((x * (A * x + C * B) + D * E) /
            (x * (A * x + B) + D * F)) - E / F


# =========================
# White Balance (Temp & Tint)
# =========================
def apply_white_balance(lin_img, temp, tint):
    """
    lin_img: linear RGB float32
    temp: [-1, 1]
    tint: [-1, 1]
    """

    temp_strength = 0.60
    tint_strength = 0.50

    # Temperature shift: R ↔ B
    wb_temp = np.array([
        1.0 + temp * temp_strength,     # R
        1.0,                             # G
        1.0 - temp * temp_strength       # B
    ])

    # Tint shift: G ↔ Magenta (R+B)
    wb_tint = np.array([
        1.0 + tint * tint_strength,     # R
        1.0 - tint * tint_strength,     # G
        1.0 + tint * tint_strength      # B
    ])

    wb = wb_temp * wb_tint

    out = lin_img * wb.reshape(1, 1, 3)
    return np.clip(out, 0, 4)


# =========================
# Contrast / Saturation / Gamma
# =========================
def adjust_contrast(img, c):
    return np.clip((img - 0.5) * c + 0.5, 0, 1)

def adjust_saturation(img, s):
    lum = 0.2126*img[:, :, 0] + 0.7152*img[:, :, 1] + 0.0722*img[:, :, 2]
    lum = lum[..., None]
    return np.clip(lum + (img - lum) * s, 0, 1)

def gamma_correct(img, g):
    return np.clip(img ** (1.0 / g), 0, 1)


# =========================
# Highlight roll-off
# =========================
def highlight_rolloff(img, roll):
    t = np.clip(roll, 0.0, 1.5)
    threshold = 0.80
    mask = np.clip((img - threshold) /
                   (1e-6 + 1 - threshold), 0, 1)

    out = img * (1 - mask) + (
        threshold + (img - threshold) / (1 + 4.0 * t * mask)
    ) * mask

    return np.clip(out, 0, 1)


# =========================
# Split Toning
# =========================
def split_tone(img, sh_rgb, hi_rgb, balance):
    lum = 0.2126*img[:, :, 0] + 0.7152*img[:, :, 1] + 0.0722*img[:, :, 2]
    lum = (lum - lum.min()) / (lum.max() - lum.min() + 1e-6)

    sh = np.clip(1 - lum + 0.5*(1 - balance), 0, 1)[..., None]
    hi = np.clip(lum + 0.5*(1 + balance) - 0.5, 0, 1)[..., None]

    sh_col = np.array(sh_rgb).reshape(1, 1, 3)
    hi_col = np.array(hi_rgb).reshape(1, 1, 3)

    out = img + sh * sh_col * 0.25 + hi * hi_col * 0.25
    return np.clip(out, 0, 1)


# =========================
# Bloom (Gaussian)
# =========================
def apply_bloom(img, radius=6.0, intensity=0.6):
    pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    if radius > 0:
        blurred = pil.filter(ImageFilter.GaussianBlur(radius=radius))
        b = np.array(blurred).astype(np.float32) / 255.0
        return np.clip(img * (1 - intensity) + b * intensity, 0, 1)
    return img


# =========================
# Vignette
# =========================
def apply_vignette(img, strength=0.25):
    h, w, _ = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    xx = (xx - w/2) / (w/2)
    yy = (yy - h/2) / (h/2)

    r = np.sqrt(xx*xx + yy*yy)
    mask = np.clip(1 - strength * (r ** 1.5), 0, 1)

    return np.clip(img * mask[..., None], 0, 1)


# =========================
# Colorfulness protection
# =========================
def ensure_colorfulness(img, min_sat=0.16, boost=1.18):
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = (mx - mn) / (mx + 1e-6)

    if sat.mean() < min_sat:
        return adjust_saturation(img, boost)
    return img


# =========================
# Auto Brightness Compensation
# =========================
def auto_brightness_compensation(
    img_arr,
    target_mean=0.50,
    strength=0.90,
    black_point_pct=0.05,
    white_point_pct=0.997,
    max_gain=2.6
):

    arr = np.clip(img_arr, 0, 1).astype(np.float32)
    lin = srgb_to_linear(arr)

    Y = 0.2126*lin[:, :, 0] + 0.7152*lin[:, :, 1] + 0.0722*lin[:, :, 2]

    bp = np.quantile(Y, black_point_pct)
    wp = np.quantile(Y, white_point_pct)
    if wp <= bp + 1e-6:
        wp = bp + 1e-3

    Y_remap = np.clip((Y - bp) / (wp - bp), 0, 1)
    remap_gain = np.clip(strength, 0, 1)
    Y_final = (1 - remap_gain)*Y + remap_gain*Y_remap

    meanY = max(Y_final.mean(), 1e-4)
    gain = np.clip(target_mean / meanY,
                   1.0/max_gain, max_gain)

    lin *= gain

    Y2 = 0.2126*lin[:, :, 0] + 0.7152*lin[:, :, 1] + 0.0722*lin[:, :, 2]
    blend = 0.65 * remap_gain
    Y_mix = (1-blend)*Y2 + blend * np.clip(Y_final * gain, 0, 2.5)

    ratio = (Y_mix + 1e-6) / (Y2 + 1e-6)
    lin = np.clip(lin * ratio[..., None], 0, 4)

    out = filmic_tonemap(np.clip(lin, 0, 4))
    out = np.clip(out, 0, 1)
    out = linear_to_srgb(out)

    return np.clip(out, 0, 1)
# ============================================================
# Emotional Crystal — FINAL (Part 5 / 8)
# Data Loading (Random / Keyword) + Emotion Mapping
# ============================================================

# Sidebar ─ Data Source
st.sidebar.header("1) Data Source (NewsAPI)")

keyword = st.sidebar.text_input(
    "Keyword (e.g., AI, aurora, science)",
    value=st.session_state.get("keyword", ""),
    key="keyword",
    placeholder="e.g., AI"
)

fetch_btn = st.sidebar.button("Fetch News")
random_btn = st.sidebar.button("Random Generate (Crystal Mode)")   # ❄️ NEW


# ============================================================
# DataFrame Initialization
# ============================================================
df = pd.DataFrame()


# ============================================================
# MODE 1: RANDOM CRYSTAL GENERATE (完全随机模式)
# ============================================================
if random_btn:

    rng = np.random.default_rng()

    num_items = rng.integers(8, 14)   # 8–13 条随机
    texts = []
    emos = []

    # 清空 palette（随机色需要写入 palette）
    st.session_state["custom_palette"] = {}

    for i in range(num_items):

        # 文本：随机 placeholder
        texts.append(f"Crystal fragment #{i+1}")

        # emotion 名：完全随机唯一 ID
        emo = f"random_emotion_{rng.integers(100000, 999999)}"
        emos.append(emo)

        # 完全随机 RGB
        r = int(rng.integers(0, 256))
        g = int(rng.integers(0, 256))
        b = int(rng.integers(0, 256))

        # 写入 palette（用于 CSV override）
        st.session_state["custom_palette"][emo] = (r, g, b)

    df = pd.DataFrame({
        "text": texts,
        "emotion": emos,
        "timestamp": str(date.today()),
        "compound": 0,
        "pos": 0, "neu": 1, "neg": 0,
        "source": "CrystalGen"
    })

    # 保存随机 seed，使 slider 可控制图案变化
    st.session_state["last_seed"] = int(rng.integers(0, 99999))



# ============================================================
# MODE 2: KEYWORD FETCH (新闻模式)
# ============================================================
elif fetch_btn:

    key = st.secrets.get("NEWS_API_KEY", "")

    if not key:
        st.sidebar.error("Missing NEWS_API_KEY in Secrets")
        df = pd.DataFrame()

    else:
        df_raw = fetch_news(key, keyword if keyword.strip() else "aurora")

        # 你要求：如果 keyword 没新闻 → 自动随机 crystal
        if df_raw.empty:
            st.warning("No news found for this keyword. A random crystal cluster was generated instead.")

            rng = np.random.default_rng()

            num_items = rng.integers(8, 14)
            texts = []
            emos = []

            st.session_state["custom_palette"] = {}

            for i in range(num_items):
                texts.append(f"Crystal fragment #{i+1}")

                emo = f"random_emotion_{rng.integers(100000, 999999)}"
                emos.append(emo)

                r = int(rng.integers(0, 256))
                g = int(rng.integers(0, 256))
                b = int(rng.integers(0, 256))

                st.session_state["custom_palette"][emo] = (r, g, b)

            df = pd.DataFrame({
                "text": texts,
                "emotion": emos,
                "timestamp": str(date.today()),
                "compound": 0,
                "pos": 0, "neu": 1, "neg": 0,
                "source": "CrystalGen"
            })

            st.session_state["last_seed"] = int(rng.integers(0, 99999))

        else:
            # 有新闻时，正常进行 sentiment → emotion → palette
            df = df_raw.copy()
            df["text"] = df["text"].fillna("")

            sent_df = df["text"].apply(analyze_sentiment).apply(pd.Series)
            df = pd.concat([df.reset_index(drop=True),
                            sent_df.reset_index(drop=True)], axis=1)

            df["emotion"] = df.apply(classify_emotion_expanded, axis=1)

            # Keyword 模式：seed 每次更新一次，使图案随机
            st.session_state["last_seed"] = np.random.randint(0, 99999)



# ============================================================
# SAFETY: If df is still empty (should not happen), warn
# ============================================================
if df.empty:
    st.warning("No data available. Please enter a keyword or click Random Generate.")
    st.stop()



# ============================================================
# Emotion Filtering UI (只对 keyword 模式有效)
# ============================================================
st.sidebar.header("2) Emotion Mapping (Keyword Mode Only)")

# keyword 模式 → emotion filter 生效
if df["source"].iloc[0] != "CrystalGen":

    cmp_min = st.sidebar.slider("Compound Min", -1.0, 1.0,
        st.session_state.get("cmp_min", -1.0), 0.01)

    cmp_max = st.sidebar.slider("Compound Max", -1.0, 1.0,
        st.session_state.get("cmp_max", 1.0), 0.01)

    init_palette_state()
    base_palette = get_active_palette()

    available_emotions = sorted(df["emotion"].unique().tolist())

    def _label_emotion(e):
        if e in COLOR_NAMES:
            return f"{e} ({COLOR_NAMES[e]})"
        r, g, b = base_palette.get(e, (0,0,0))
        return f"{e} (Custom {r},{g},{b})"

    auto_top3 = st.sidebar.checkbox(
        "Auto-select Top-3 Emotions",
        value=st.session_state.get("auto_top3", True)
    )

    top3 = []
    if auto_top3:
        vc = df["emotion"].value_counts()
        top3 = vc.head(3).index.tolist()

    option_labels = [_label_emotion(e) for e in available_emotions]
    default_labels = [_label_emotion(e) for e in (top3 if top3 else available_emotions)]

    selected_labels = st.sidebar.multiselect(
        "Selected Emotions",
        option_labels,
        default=default_labels
    )

    selected_emotions = [lbl.split(" (")[0] for lbl in selected_labels]

    # Apply filter
    df = df[(df["emotion"].isin(selected_emotions))
            & (df["compound"] >= cmp_min)
            & (df["compound"] <= cmp_max)]

else:
    # Random 模式 → selected emotions 不生效
    pass
# ============================================================
# Emotional Crystal — FINAL (Part 6 / 8)
# Crystal Engine UI + Background + Cinematic Color Controls
# ============================================================

# ----------------------------------------------
# Crystal Engine Controls
# ----------------------------------------------
st.sidebar.header("3) Crystal Engine")

ribbons_per_emotion = st.sidebar.slider(
    "Crystals per Emotion",
    1, 40,
    st.session_state.get("ribbons_per_emotion", 12),
    help="Controls how many crystal fragments per emotion are drawn."
)

stroke_blur = st.sidebar.slider(
    "Crystal Softness (blur px)",
    0.0, 25.0,
    st.session_state.get("stroke_blur", 7.0),
    help="Controls how soft (blurred) the crystal edges appear."
)

ribbon_alpha = st.sidebar.slider(
    "Crystal Transparency (Alpha)",
    40, 255,
    st.session_state.get("ribbon_alpha", 220),
    help="Higher alpha makes crystals more solid; lower alpha is more transparent."
)

# Crystal size
st.sidebar.subheader("Crystal Size Range")

poly_min_size = st.sidebar.slider(
    "Min Size (px)",
    20, 300,
    st.session_state.get("poly_min_size", 70)
)

poly_max_size = st.sidebar.slider(
    "Max Size (px)",
    60, 600,
    st.session_state.get("poly_max_size", 220)
)

# ----------------------------------------------
# Crystal Shape Variation
# ----------------------------------------------
st.sidebar.subheader("Crystal Layer Controls")

layer_count = st.sidebar.slider(
    "Layers (Global Blend Layers)",
    1, 35,
    st.session_state.get("layer_count", 12),
    help="Number of global layers of crystals. Higher = denser & more atmospheric."
)

wobble_control = st.sidebar.slider(
    "Wobble (Crystal Randomness)",
    0.00, 1.00,
    st.session_state.get("wobble_control", 0.22),
    0.01,
    help="Shape randomness. Higher = more chaotic, jagged crystals."
)

# IMPORTANT:
# Seed is used ONLY to allow the user to 'tune' the current pattern.
# Random/Keyword reload creates a NEW seed so patterns change as you wanted.
seed_control = st.sidebar.slider(
    "Seed (Fine-tune this pattern)",
    0, 500,
    st.session_state.get("last_seed", 123),   # ← from Part 5
    help="Use this to fine-tune the current pattern, without regenerating data."
)

# ----------------------------------------------
# Background Color
# ----------------------------------------------
st.sidebar.subheader("Background Color")

def _hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

bg_custom = st.sidebar.color_picker(
    "Choose Solid Background",
    value=st.session_state.get("bg_custom", "#000000")
)

bg_rgb = _hex_to_rgb(bg_custom)
st.session_state["bg_custom"] = bg_custom

# ----------------------------------------------
# Cinematic Color Controls
# ----------------------------------------------
st.sidebar.header("4) Cinematic Color System")

exp = st.sidebar.slider(
    "Exposure (Stops)",
    -0.3, 2.0,
    st.session_state.get("exp", 0.55),
    0.01
)

contrast = st.sidebar.slider(
    "Contrast",
    0.60, 2.00,
    st.session_state.get("contrast", 1.18)
)

saturation = st.sidebar.slider(
    "Saturation",
    0.60, 2.20,
    st.session_state.get("saturation", 1.18)
)

gamma_val = st.sidebar.slider(
    "Gamma",
    0.60, 1.60,
    st.session_state.get("gamma_val", 0.92)
)

roll = st.sidebar.slider(
    "Highlight Roll-off",
    0.0, 1.80,
    st.session_state.get("roll", 0.40)
)

# ----------------------------------------------
# White Balance (Temperature + Tint)
# ----------------------------------------------
st.sidebar.subheader("White Balance")

temp = st.sidebar.slider(
    "Temperature (Blue ↔ Yellow)",
    -1.2, 1.2,
    st.session_state.get("temp", 0.00)
)

tint = st.sidebar.slider(
    "Tint (Green ↔ Magenta)",
    -1.2, 1.2,
    st.session_state.get("tint", 0.00)
)

# ----------------------------------------------
# Split Toning
# ----------------------------------------------
st.sidebar.subheader("Split Toning")

sh_r = st.sidebar.slider("Shadows R", 0.0, 1.0, st.session_state.get("sh_r", 0.08))
sh_g = st.sidebar.slider("Shadows G", 0.0, 1.0, st.session_state.get("sh_g", 0.06))
sh_b = st.sidebar.slider("Shadows B", 0.0, 1.0, st.session_state.get("sh_b", 0.16))

hi_r = st.sidebar.slider("Highlights R", 0.0, 1.0, st.session_state.get("hi_r", 0.10))
hi_g = st.sidebar.slider("Highlights G", 0.0, 1.0, st.session_state.get("hi_g", 0.08))
hi_b = st.sidebar.slider("Highlights B", 0.0, 1.0, st.session_state.get("hi_b", 0.06))

tone_balance = st.sidebar.slider(
    "Tone Balance (Shadow ↔ Highlight)",
    -1.0, 1.0,
    st.session_state.get("tone_balance", 0.0)
)

# ----------------------------------------------
# Bloom & Vignette
# ----------------------------------------------
st.sidebar.subheader("Bloom & Vignette")

bloom_radius = st.sidebar.slider(
    "Bloom Radius (px)",
    0.0, 25.0,
    st.session_state.get("bloom_radius", 7.0)
)

bloom_intensity = st.sidebar.slider(
    "Bloom Intensity",
    0.0, 1.0,
    st.session_state.get("bloom_intensity", 0.40)
)

vignette_strength = st.sidebar.slider(
    "Vignette Strength",
    0.0, 0.8,
    st.session_state.get("vignette_strength", 0.16)
)

# ----------------------------------------------
# Auto Brightness
# ----------------------------------------------
st.sidebar.header("5) Auto Brightness Compensation")

auto_bright = st.sidebar.checkbox(
    "Enable Auto Brightness",
    value=st.session_state.get("auto_bright", True)
)

target_mean = st.sidebar.slider(
    "Target Mean Brightness",
    0.2, 0.8,
    st.session_state.get("target_mean", 0.52)
)

abc_strength = st.sidebar.slider(
    "Histogram Remap Strength",
    0.0, 1.0,
    st.session_state.get("abc_strength", 0.92)
)

abc_black = st.sidebar.slider(
    "Black Point %",
    0.00, 0.20,
    st.session_state.get("abc_black", 0.05)
)

abc_white = st.sidebar.slider(
    "White Point %",
    0.80, 1.00,
    st.session_state.get("abc_white", 0.997)
)

abc_max_gain = st.sidebar.slider(
    "Max Gain",
    1.0, 3.0,
    st.session_state.get("abc_max_gain", 2.6)
)
# ============================================================
# Emotional Crystal — FINAL (Part 7 / 8)
# Palette System (CSV Import / Export + Custom Colors)
# ============================================================

st.sidebar.header("6) Custom Palette System")

# ------------------------
# Toggle: Use CSV palette only
# ------------------------
use_csv_only = st.sidebar.checkbox(
    "Use CSV Palette Only (ignore default colors)",
    value=st.session_state.get("use_csv_palette", False),
    key="use_csv_palette"
)

# ============================================================
# Custom Color Add
# ============================================================
with st.sidebar.expander("Add Custom Emotion Color", expanded=False):

    colA, colB, colC, colD = st.columns([1.4, 1, 1, 1])
    emo_name = colA.text_input("Emotion Name", placeholder="e.g., calm_blue")

    r_val = colB.number_input("R", 0, 255, 180)
    g_val = colC.number_input("G", 0, 255, 180)
    b_val = colD.number_input("B", 0, 255, 200)

    if st.button("Add Custom Color"):
        if emo_name.strip():
            st.session_state["custom_palette"][emo_name.strip()] = (
                int(r_val), int(g_val), int(b_val)
            )
            st.success(f"Added custom color for: {emo_name}")
        else:
            st.warning("Please enter a valid emotion name.")

# ============================================================
# CSV Palette Import / Export
# ============================================================
with st.sidebar.expander("Import / Export Palette CSV", expanded=False):

    # ----------------------
    # Import CSV
    # ----------------------
    uploaded = st.file_uploader("Import CSV Palette", type=["csv"])

    if uploaded is not None:
        try:
            dfc = pd.read_csv(uploaded)

            # Ensure required columns
            expected_cols = {"emotion", "r", "g", "b"}
            lower_cols = {c.lower(): c for c in dfc.columns}

            if not expected_cols.issubset(lower_cols.keys()):
                st.error("CSV must contain columns: emotion, r, g, b")
            else:
                pal = {}
                for _, row in dfc.iterrows():
                    emo = str(row[lower_cols["emotion"]]).strip()
                    try:
                        r = int(row[lower_cols["r"]])
                        g = int(row[lower_cols["g"]])
                        b = int(row[lower_cols["b"]])
                        pal[emo] = (r, g, b)
                    except:
                        continue

                st.session_state["custom_palette"] = pal
                st.session_state["use_csv_palette"] = True
                st.success(f"Imported {len(pal)} colors from CSV!")

        except Exception as e:
            st.error(f"CSV import failed: {e}")

    # ----------------------
    # Export current palette
    # ----------------------
    st.markdown("### Current Palette Preview")

    # merge default + custom unless "csv only" is active
    preview_palette = (
        dict(st.session_state.get("custom_palette", {}))
        if st.session_state.get("use_csv_palette", False)
        else {**DEFAULT_RGB, **st.session_state.get("custom_palette", {})}
    )

    df_preview = pd.DataFrame([
        {"emotion": k, "r": v[0], "g": v[1], "b": v[2]}
        for k, v in preview_palette.items()
    ])

    st.dataframe(df_preview, use_container_width=True)

    # Export button
    buf = BytesIO()
    df_preview.to_csv(buf, index=False)
    buf.seek(0)
    st.download_button(
        "Download Current Palette as CSV",
        data=buf,
        file_name="palette_export.csv",
        mime="text/csv"
    )
# ============================================================
# Emotional Crystal — FINAL (Part 8 / 8)
# Main Rendering + Cinematic Pipeline + Download
# ============================================================

left_col, right_col = st.columns([0.60, 0.40])

# =========================
# LEFT PANEL — VISUALIZATION
# =========================
with left_col:

    st.subheader("❄️ Crystal Mix Visualization")

    working_palette = get_active_palette()

    # ------------------------------------
    # 🎨 Render Crystal (Random or Keyword)
    # ------------------------------------
    img = render_crystalmix(
        df=df,
        palette=working_palette,
        width=1500,
        height=850,
        seed=seed_control,                    # full control
        shapes_per_emotion=ribbons_per_emotion,
        min_size=poly_min_size,
        max_size=poly_max_size,
        fill_alpha=int(ribbon_alpha),
        blur_px=int(stroke_blur),
        bg_color=bg_rgb,
        wobble=wobble_control,
        layers=layer_count
    )

    # =============================
    # Convert PIL → NumPy
    # =============================
    arr = np.array(img).astype(np.float32) / 255.0
    lin = srgb_to_linear(arr)

    # ===== Exposure =====
    lin = lin * (2.0 ** exp)

    # ===== White Balance =====
    lin = apply_white_balance(lin, temp, tint)

    # ===== Highlight Roll-off =====
    lin = highlight_rolloff(lin, roll)

    # Back to sRGB
    arr = linear_to_srgb(np.clip(lin, 0, 4))

    # ===== Filmic Curve =====
    arr = np.clip(filmic_tonemap(arr * 1.20), 0, 1)

    # ===== Contrast / Saturation / Gamma =====
    arr = adjust_contrast(arr, contrast)
    arr = adjust_saturation(arr, saturation)
    arr = gamma_correct(arr, gamma_val)

    # ===== Split Toning =====
    arr = split_tone(
        arr,
        sh_rgb=(sh_r, sh_g, sh_b),
        hi_rgb=(hi_r, hi_g, hi_b),
        balance=tone_balance
    )

    # ===== Auto Brightness =====
    if auto_bright:
        arr = auto_brightness_compensation(
            arr,
            target_mean=target_mean,
            strength=abc_strength,
            black_point_pct=abc_black,
            white_point_pct=abc_white,
            max_gain=abc_max_gain
        )

    # ===== Bloom + Vignette =====
    arr = apply_bloom(arr, radius=bloom_radius, intensity=bloom_intensity)
    arr = apply_vignette(arr, strength=vignette_strength)

    # ===== Ensure Colorfulness =====
    arr = ensure_colorfulness(arr, min_sat=0.16, boost=1.18)

    # Convert back to PIL
    final_img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="RGB")

    # Save to memory buffer
    buf = BytesIO()
    final_img.save(buf, format="PNG")
    buf.seek(0)

    # ---------------------------
    # Display + Download
    # ---------------------------
    st.image(buf, use_column_width=True)
    st.download_button(
        "💾 Download Crystal PNG",
        data=buf,
        file_name="crystal_mix.png",
        mime="image/png"
    )


# =========================
# RIGHT PANEL — DATA TABLE
# =========================
with right_col:

    st.subheader("📊 Data & Emotion Mapping")

    df2 = df.copy()
    df2["emotion_display"] = df2["emotion"].apply(
        lambda e: f"{e} ({COLOR_NAMES.get(e,'Custom')})"
    )

    cols = ["text", "emotion_display", "compound", "pos", "neu", "neg"]
    if "timestamp" in df.columns:
        cols.insert(1, "timestamp")
    if "source" in df.columns:
        cols.insert(2, "source")

    st.dataframe(df2[cols], use_container_width=True, height=600)

# ============================================================
# END OF FINAL APPLICATION (Part 8 / 8)
# ============================================================
