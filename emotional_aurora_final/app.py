# ============================================================
# Emotional Crystal — FINAL FULL VERSION
# Single-file complete app.py
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
import random

# =========================
# App setup
# =========================
st.set_page_config(page_title="Emotional Crystal — Final", page_icon="❄️", layout="wide")
st.title("❄️ Emotional Crystal — Final")


# ============================================================
# GLOBAL SEED (核心：每次 keyword / random 都刷新 crystal 结构)
# ============================================================

if "global_seed" not in st.session_state:
    st.session_state["global_seed"] = random.randint(0, 999999)

def refresh_global_seed():
    st.session_state["global_seed"] = random.randint(0, 999999)


# ============================================================
# RANDOM EMOTION PREFIX (你指定为 R1)
# ============================================================
RANDOM_PREFIX = "R1"
def make_random_emotion_name(i):
    return f"{RANDOM_PREFIX}_{i}"


# ============================================================
# Instructions
# ============================================================
with st.expander("Instructions", expanded=False):
    st.markdown("""
### How to Use This Project  

This project transforms emotional data into **cinematic ice-crystal generative visuals**.

**1. Keyword Mode**  
- Enter keyword → fetch news  
- If NewsAPI has *no news*, a *random crystal pack* is generated automatically  

**2. Random Mode**  
- Generates fully random crystals  
- Ignores Selected Emotions  
- Still influenced by color settings & CSV palette  

**3. Editing the Crystal**  
After generation, adjusting **layers / wobble / seed / colors**  
will change the current crystal in real time.

**4. CSV Palette**  
Random colors can still be overridden by CSV imports.

**5. Export**  
Download final image as PNG.
""")


# ============================================================
# VADER Sentiment
# ============================================================
@st.cache_resource
def load_vader():
    try:
        nltk.data.find("sentiment/vader_lexicon")
    except LookupError:
        nltk.download("vader_lexicon")
    return SentimentIntensityAnalyzer()

sia = load_vader()


# ============================================================
# NEWS API
# ============================================================
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
            return pd.DataFrame()

        rows = []
        for a in data.get("articles", []):
            t = (a.get("title") or "") + " - " + (a.get("description") or "")
            rows.append({
                "timestamp": (a.get("publishedAt") or "")[:10],
                "text": t.strip(" -"),
                "source": (a.get("source") or {}).get("name", "")
            })
        return pd.DataFrame(rows)

    except:
        return pd.DataFrame()


# ============================================================
# Emotion Colors
# ============================================================
DEFAULT_RGB = {
    "joy":(255,200,60),"love":(255,95,150),"pride":(190,100,255),
    "hope":(60,235,190),"curiosity":(50,190,255),"calm":(70,135,255),
    "surprise":(255,160,70),"neutral":(190,190,200),"sadness":(80,120,230),
    "anger":(245,60,60),"fear":(150,70,200),"disgust":(150,200,60),
    "anxiety":(255,200,60),"boredom":(135,135,145),"nostalgia":(250,210,150),
    "gratitude":(90,230,230),"awe":(120,245,255),"trust":(60,200,160),
    "confusion":(255,140,180),"mixed":(230,190,110),
}

COLOR_NAMES = {
    "joy":"Jupiter Gold","love":"Rose","pride":"Violet","hope":"Mint",
    "curiosity":"Azure","calm":"Indigo","surprise":"Peach","neutral":"Gray",
    "sadness":"Ocean","anger":"Vermilion","fear":"Mulberry","disgust":"Olive",
    "anxiety":"Sand","boredom":"Slate","nostalgia":"Cream","gratitude":"Cyan",
    "awe":"Ice","trust":"Teal","confusion":"Blush","mixed":"Amber",
}


# ============================================================
# Sentiment → Emotion
# ============================================================
def analyze_sentiment(txt):
    if not isinstance(txt, str) or not txt.strip():
        return {"neg":0,"neu":1,"pos":0,"compound":0}
    return sia.polarity_scores(txt)

def classify_emotion(row):
    pos, neu, neg, comp = row["pos"], row["neu"], row["neg"], row["compound"]

    if comp>=0.7 and pos>0.5: return "joy"
    if comp>=0.55 and pos>0.45: return "love"
    if comp>=0.45 and pos>0.40: return "pride"
    if 0.25<=comp<0.45 and pos>0.30: return "hope"
    if 0.10<=comp<0.25 and neu>=0.5: return "calm"
    if 0.25<=comp<0.60 and neu<0.5: return "surprise"
    if comp<=-0.65 and neg>0.5: return "anger"
    if -0.65<comp<=-0.40 and neg>0.45: return "fear"
    if -0.40<comp<=-0.15 and neg>=0.35: return "sadness"
    if neg>0.5 and neu>0.3: return "anxiety"
    if neg>0.45 and pos<0.1: return "disgust"
    if neu>0.75 and abs(comp)<0.1: return "boredom"
    if pos>0.35 and neu>0.4 and 0<comp<0.25: return "trust"
    if pos>0.30 and neu>0.35 and -0.05<=comp<=0.05: return "nostalgia"
    if pos>0.25 and neg>0.25: return "mixed"
    if pos>0.20 and neu>0.50: return "curiosity"
    if neu>0.6 and 0.05<=comp<=0.15: return "awe"
    return "neutral"


# ============================================================
# Palette State
# ============================================================
def init_palette_state():
    if "use_csv_palette" not in st.session_state:
        st.session_state["use_csv_palette"] = False
    if "custom_palette" not in st.session_state:
        st.session_state["custom_palette"] = {}

def get_active_palette():
    if st.session_state.get("use_csv_palette", False):
        return dict(st.session_state.get("custom_palette", {}))
    d = dict(DEFAULT_RGB)
    d.update(st.session_state.get("custom_palette", {}))
    return d

def add_custom_emotion(name, r, g, b):
    if name:
        st.session_state["custom_palette"][name.strip()] = (int(r),int(g),int(b))


# ============================================================
# Crystal Shape + Render
# ============================================================
def crystal_shape(center, r, wobble, rng):
    cx, cy = center
    n = rng.integers(5, 11)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    rng.shuffle(angles)
    radii = r*(1 + rng.uniform(-wobble, wobble, n))
    pts = []
    for a, rr in zip(angles, radii):
        pts.append((cx+rr*math.cos(a), cy+rr*math.sin(a)))
    pts.append(pts[0])
    return pts

def draw_polygon_soft(canvas, pts, rgb01, alpha, blur, edge):
    W,H = canvas.size
    layer = Image.new("RGBA", (W,H), (0,0,0,0))
    d = ImageDraw.Draw(layer, "RGBA")

    col = (
        int(rgb01[0]*255),
        int(rgb01[1]*255),
        int(rgb01[2]*255),
        int(alpha),
    )
    d.polygon(pts, fill=col)

    if edge>0:
        d.line(pts, fill=(255,255,255,100), width=edge)

    if blur>0:
        layer = layer.filter(ImageFilter.GaussianBlur(blur))

    canvas.alpha_composite(layer)
# ============================================================
# Part B — Crystal Mix Renderer
# ============================================================

def _rgb01(rgb):
    return np.clip(np.array(rgb)/255.0, 0, 1)

def jitter_color(rgb01, rng, amount=0.07):
    j = (rng.random(3)-0.5)*2*amount
    return tuple(np.clip(rgb01+j, 0, 1))

def vibrancy_boost(rgb, sat_boost=1.28, min_luma=0.38):
    c = _rgb01(rgb)
    lum = 0.2126*c[0]+0.7152*c[1]+0.0722*c[2]
    if lum < min_luma:
        c = np.clip(c + (min_luma-lum), 0, 1)
    lum = 0.2126*c[0]+0.7152*c[1]+0.0722*c[2]
    return tuple(np.clip(lum + (c-lum)*sat_boost, 0, 1))


def render_crystalmix(
    df, palette,
    width=1500, height=850,
    seed=0,
    shapes_per_emotion=10,
    min_size=60, max_size=220,
    fill_alpha=210, blur_px=6,
    bg_color=(0,0,0),
    wobble=0.25,
    layers=10
):
    rng = np.random.default_rng(seed)

    base = Image.new("RGBA", (width,height), (bg_color[0],bg_color[1],bg_color[2],255))
    canvas = Image.new("RGBA", (width,height), (0,0,0,0))

    emotions = df["emotion"].unique().tolist()
    if not emotions:
        emotions = ["neutral"]

    for _layer in range(layers):
        for emo in emotions:

            rgb = palette.get(emo, (230,190,110))
            rgb01 = vibrancy_boost(rgb)

            for _ in range(shapes_per_emotion):
                cx = rng.uniform(width*0.05, width*0.95)
                cy = rng.uniform(height*0.05, height*0.95)
                rr = rng.integers(min_size, max_size)

                pts = crystal_shape((cx,cy), rr, wobble, rng)
                col = jitter_color(rgb01, rng)

                alpha = int(np.clip(fill_alpha*rng.uniform(0.85,1.05), 40,255))
                blur = int(blur_px*rng.uniform(0.7,1.4))
                edge = 0 if rng.random()<0.6 else max(1,int(rr*0.02))

                draw_polygon_soft(canvas, pts, col, alpha, blur, edge)

    base.alpha_composite(canvas)
    return base.convert("RGB")


# ============================================================
# Cinematic Color Pipeline
# ============================================================

def srgb_to_linear(x):
    x = np.clip(x,0,1)
    return np.where(x<=0.04045, x/12.92, ((x+0.055)/1.055)**2.4)

def linear_to_srgb(x):
    x = np.clip(x,0,1)
    return np.where(x<0.0031308, x*12.92, 1.055*(x**(1/2.4)) - 0.055)

def filmic_tonemap(x):
    A=0.22; B=0.30; C=0.10; D=0.20; E=0.01; F=0.30
    return ((x*(A*x + C*B) + D*E) / (x*(A*x+B)+D*F)) - E/F

def apply_white_balance(lin, temp, tint):
    tS=0.6; tiS=0.5
    wb = np.array([
        1+temp*tS,
        1-tint*tiS,
        1-temp*tS + tint*tiS
    ])
    return np.clip(lin*wb.reshape(1,1,3), 0, 4)

def adjust_contrast(img, c):
    return np.clip((img-0.5)*c + 0.5, 0,1)

def adjust_saturation(img, s):
    lum = 0.2126*img[:,:,0] + 0.7152*img[:,:,1] + 0.0722*img[:,:,2]
    return np.clip(lum[...,None] + (img-lum[...,None])*s, 0,1)

def gamma_correct(img, g):
    return np.clip(img**(1.0/g), 0,1)

def highlight_rolloff(img, roll):
    t=np.clip(roll,0,1.5)
    thr=0.8
    mask=np.clip((img-thr)/(1e-6+1-thr),0,1)
    out=img*(1-mask)+(thr+(img-thr)/(1+4*t*mask))*mask
    return np.clip(out,0,1)

def split_tone(img, sh, hi, balance):
    lum=(0.2126*img[:,:,0]+0.7152*img[:,:,1]+0.0722*img[:,:,2])
    lum=(lum-lum.min())/(lum.max()-lum.min()+1e-6)
    shMask=np.clip(1-lum+0.5*(1-balance),0,1)[...,None]
    hiMask=np.clip(lum+0.5*(1+balance)-0.5,0,1)[...,None]
    shCol=np.array(sh).reshape(1,1,3)
    hiCol=np.array(hi).reshape(1,1,3)
    return np.clip(img+shMask*shCol*0.25 + hiMask*hiCol*0.25, 0,1)

def apply_bloom(img, r=6.0, intensity=0.6):
    pil=Image.fromarray((img*255).astype(np.uint8))
    if r>0:
        blur=np.array(pil.filter(ImageFilter.GaussianBlur(r)))/255.0
        return np.clip(img*(1-intensity)+blur*intensity, 0,1)
    return img

def apply_vignette(img, s=0.20):
    h,w,_=img.shape
    yy,xx=np.mgrid[0:h,0:w]
    xx=(xx-w/2)/(w/2); yy=(yy-h/2)/(h/2)
    r=np.sqrt(xx*xx+yy*yy)
    mask=np.clip(1 - s*(r**1.5), 0,1)
    return np.clip(img*mask[...,None], 0,1)


# ============================================================
# Auto Brightness (ABC)
# ============================================================

def auto_brightness_compensation(
    img,
    target_mean=0.52,
    strength=0.92,
    black_point_pct=0.05,
    white_point_pct=0.997,
    max_gain=2.6
):
    arr=np.clip(img,0,1)
    lin=srgb_to_linear(arr)
    Y=0.2126*lin[:,:,0]+0.7152*lin[:,:,1]+0.0722*lin[:,:,2]

    bp=np.quantile(Y, black_point_pct)
    wp=np.quantile(Y, white_point_pct)
    wp=max(wp, bp+1e-3)

    Y2=np.clip((Y-bp)/(wp-bp),0,1)
    Ymix=(1-strength)*Y + strength*Y2

    meanY=max(Ymix.mean(),1e-4)
    gain=np.clip(target_mean/meanY, 1/max_gain, max_gain)

    out=lin*gain
    out=filmic_tonemap(np.clip(out,0,4))
    out=linear_to_srgb(out)
    return np.clip(out,0,1)


# ============================================================
# CSV Palette
# ============================================================

def export_palette_csv(pal):
    buf=BytesIO()
    df=pd.DataFrame([{"emotion":k,"r":v[0],"g":v[1],"b":v[2]} for k,v in pal.items()])
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf

def import_palette_csv(file):
    try:
        df=pd.read_csv(file)
        if not {"emotion","r","g","b"}.issubset(set(df.columns.str.lower())):
            st.error("CSV must include emotion,r,g,b")
            return
        pal={}
        for _,row in df.iterrows():
            pal[str(row["emotion"])]=(int(row["r"]),int(row["g"]),int(row["b"]))
        st.session_state["custom_palette"]=pal
        st.success(f"Imported {len(pal)} colors")
    except:
        st.error("CSV import error")


# ============================================================
# PART 3 — Sidebar UI + Data Loading
# ============================================================

DEFAULTS = {
    "keyword":"",
    "ribbons_per_emotion":10,
    "stroke_blur":6,
    "ribbon_alpha":210,
    "poly_min_size":70,
    "poly_max_size":220,
    "exp":0.55,
    "contrast":1.18,
    "saturation":1.18,
    "gamma_val":0.92,
    "roll":0.40,
    "temp":0.0,"tint":0.0,
    "sh_r":0.08,"sh_g":0.06,"sh_b":0.16,
    "hi_r":0.10,"hi_g":0.08,"hi_b":0.06,
    "tone_balance":0.0,
    "vignette_strength":0.16,
    "bloom_radius":7.0,
    "bloom_intensity":0.40,
    "auto_bright":True,
    "target_mean":0.52,
    "abc_strength":0.92,
    "abc_black":0.05,
    "abc_white":0.997,
    "abc_max_gain":2.6,
    "cmp_min":-1.0,
    "cmp_max":1.0,
    "bg_custom":"#000000",
}


# ============================================================
# SIDEBAR — DATA SOURCE
# ============================================================

st.sidebar.header("1) Data Source")

keyword = st.sidebar.text_input(
    "Keyword",
    value=st.session_state.get("keyword", DEFAULTS["keyword"]),
    key="keyword",
)

btn_fetch = st.sidebar.button("Fetch News")
btn_random = st.sidebar.button("Random Generate")


# ============================================================
# LOAD DATA
# ============================================================

df = pd.DataFrame()

# RANDOM MODE
if btn_random:
    refresh_global_seed()
    rng=np.random.default_rng(st.session_state["global_seed"])
    st.session_state["custom_palette"]={}
    texts=[]; emos=[]
    for i in range(12):
        texts.append(f"Random crystal #{i+1}")
        emo = make_random_emotion_name(i+1)
        emos.append(emo)
        st.session_state["custom_palette"][emo]=(
            int(rng.integers(0,256)),
            int(rng.integers(0,256)),
            int(rng.integers(0,256))
        )
    df=pd.DataFrame({
        "text":texts,
        "emotion":emos,
        "timestamp":str(date.today()),
        "compound":0,"pos":0,"neu":1,"neg":0,
        "source":"Random"
    })

# KEYWORD MODE
elif btn_fetch:
    refresh_global_seed()
    key=st.secrets.get("NEWS_API_KEY","")
    if key:
        df = fetch_news(key, keyword if keyword.strip() else "aurora")
    else:
        df = pd.DataFrame()

    # 如果没有新闻 → 自动生成随机 crystal
    if df.empty:
        rng=np.random.default_rng(st.session_state["global_seed"])
        st.session_state["custom_palette"]={}
        texts=[]; emos=[]
        for i in range(10):
            texts.append(f"Auto Crystal #{i+1}")
            emo=make_random_emotion_name(i+1)
            emos.append(emo)
            st.session_state["custom_palette"][emo]=(
                int(rng.integers(0,256)),
                int(rng.integers(0,256)),
                int(rng.integers(0,256))
            )

        df=pd.DataFrame({
            "text":texts,
            "emotion":emos,
            "timestamp":str(date.today()),
            "compound":0,"pos":0,"neu":1,"neg":0,
            "source":"AutoRandom"
        })

# DEFAULT
if df.empty:
    df=pd.DataFrame({
        "text":["Demo crystal sample"],
        "timestamp":[str(date.today())],
        "source":["Demo"]
    })


# ============================================================
# SAFE LAYER: guarantee sentiment columns
# ============================================================
df["text"]=df["text"].fillna("")

if "compound" not in df.columns:
    df_s = df["text"].apply(analyze_sentiment).apply(pd.Series)
    df = pd.concat([df.reset_index(drop=True), df_s], axis=1)

if "emotion" not in df.columns:
    df["emotion"]=df.apply(classify_emotion, axis=1)


# ============================================================
# START SIDEBAR — Emotion Mapping
# ============================================================

st.sidebar.header("2) Emotion Mapping (Keyword Only)")

cmp_min = st.sidebar.slider("Compound Min", -1.0, 1.0, DEFAULTS["cmp_min"])
cmp_max = st.sidebar.slider("Compound Max", -1.0, 1.0, DEFAULTS["cmp_max"])


# All palette
init_palette_state()
palette = get_active_palette()

available = sorted(df["emotion"].unique().tolist())

def _lbl(e):
    if e in COLOR_NAMES:
        return f"{e} ({COLOR_NAMES[e]})"
    r,g,b=palette.get(e,(0,0,0))
    return f"{e} (Custom {r},{g},{b})"

option_labels=[_lbl(e) for e in available]

selected_labels=st.sidebar.multiselect(
    "Selected Emotions (ignored in Random Mode)",
    option_labels,
    default=option_labels
)

selected_emos=[lbl.split(" (")[0] for lbl in selected_labels]

if not btn_random:  # Keyword mode only
    df=df[(df["emotion"].isin(selected_emos)) &
          (df["compound"]>=cmp_min) &
          (df["compound"]<=cmp_max)]


# ============================================================
# Sidebar — Crystal Controls
# ============================================================

st.sidebar.header("3) Crystal Engine")

layer_count = st.sidebar.slider("Layers", 1, 30, 10)
wobble = st.sidebar.slider("Wobble", 0.0,1.0,0.25,0.01)
seed_local = st.sidebar.slider("Local Seed", 0, 500, 25)
ribbons = st.sidebar.slider("Crystals per Emotion", 1,40, DEFAULTS["ribbons_per_emotion"])
stroke_blur = st.sidebar.slider("Softness (px)", 0,20, DEFAULTS["stroke_blur"])
alpha_val = st.sidebar.slider("Alpha",40,255,DEFAULTS["ribbon_alpha"])
poly_min = st.sidebar.slider("Min Size",20,300,DEFAULTS["poly_min_size"])
poly_max = st.sidebar.slider("Max Size",60,600,DEFAULTS["poly_max_size"])

# Background
st.sidebar.subheader("Background")
bg_hex = st.sidebar.color_picker("Background Color", DEFAULTS["bg_custom"])
bg_rgb = tuple(int(bg_hex[i:i+2],16) for i in (1,3,5))


# ============================================================
# Sidebar — Cinematic Color
# ============================================================

st.sidebar.header("4) Cinematic Color")

exp = st.sidebar.slider("Exposure", -0.2,1.8,DEFAULTS["exp"])
contrast = st.sidebar.slider("Contrast",0.7,1.8,DEFAULTS["contrast"])
saturation= st.sidebar.slider("Saturation",0.7,1.9,DEFAULTS["saturation"])
gamma_val = st.sidebar.slider("Gamma",0.7,1.4,DEFAULTS["gamma_val"])
roll = st.sidebar.slider("Highlight Roll-off",0,1.5,DEFAULTS["roll"])

st.sidebar.subheader("White Balance")
temp = st.sidebar.slider("Temperature", -1,1, DEFAULTS["temp"])
tint = st.sidebar.slider("Tint", -1,1, DEFAULTS["tint"])

st.sidebar.subheader("Split Toning")
sh_r = st.sidebar.slider("Shadows R",0,1,DEFAULTS["sh_r"])
sh_g = st.sidebar.slider("Shadows G",0,1,DEFAULTS["sh_g"])
sh_b = st.sidebar.slider("Shadows B",0,1,DEFAULTS["sh_b"])
hi_r = st.sidebar.slider("Highlights R",0,1,DEFAULTS["hi_r"])
hi_g = st.sidebar.slider("Highlights G",0,1,DEFAULTS["hi_g"])
hi_b = st.sidebar.slider("Highlights B",0,1,DEFAULTS["hi_b"])
tone_balance = st.sidebar.slider("Balance", -1,1, DEFAULTS["tone_balance"])

st.sidebar.subheader("Bloom & Vignette")
bloom_radius = st.sidebar.slider("Bloom Radius",0,20,DEFAULTS["bloom_radius"])
bloom_intensity = st.sidebar.slider("Bloom Intensity",0,1,DEFAULTS["bloom_intensity"])
vignette_strength = st.sidebar.slider("Vignette Strength",0,0.8,DEFAULTS["vignette_strength"])

st.sidebar.header("5) Auto Brightness")
auto_bright = st.sidebar.checkbox("Enable ABC", DEFAULTS["auto_bright"])
target_mean = st.sidebar.slider("Target Mean",0.3,0.7,DEFAULTS["target_mean"])
abc_strength = st.sidebar.slider("Remap Strength",0,1,DEFAULTS["abc_strength"])
abc_black = st.sidebar.slider("Black Point",0,0.2,DEFAULTS["abc_black"])
abc_white = st.sidebar.slider("White Point",0.8,1,DEFAULTS["abc_white"])
abc_max_gain = st.sidebar.slider("Max Gain",1,3,DEFAULTS["abc_max_gain"])


# ============================================================
# Sidebar — Palette (CSV)
# ============================================================

st.sidebar.header("6) Palette CSV / Custom")

use_csv = st.sidebar.checkbox("Use CSV palette only", st.session_state["use_csv_palette"], key="use_csv_palette")

with st.sidebar.expander("Add Custom Emotion"):
    emo_name = st.text_input("Emotion Name")
    col1,col2,col3 = st.columns(3)
    r=col1.number_input("R",0,255,180)
    g=col2.number_input("G",0,255,180)
    b=col3.number_input("B",0,255,200)
    if st.button("Add Color"):
        add_custom_emotion(emo_name,r,g,b)

with st.sidebar.expander("Import/Export CSV"):
    up = st.file_uploader("Import CSV", type=["csv"])
    if up:
        import_palette_csv(up)

    pal_show = get_active_palette()
    df_pal = pd.DataFrame([{"emotion":k,"r":v[0],"g":v[1],"b":v[2]} for k,v in pal_show.items()])
    st.dataframe(df_pal, height=225)
    dl = export_palette_csv(pal_show)
    st.download_button("Download CSV", dl, "palette.csv", "text/csv")


# ============================================================
# MAIN DISPLAY — Render Crystal + Post-processing
# ============================================================

left, right = st.columns([0.60,0.40])

with left:
    st.subheader("❄️ Crystal Mix Visualization")

    img = render_crystalmix(
        df=df,
        palette=get_active_palette(),
        width=1500,
        height=850,
        seed=st.session_state["global_seed"] + seed_local,
        shapes_per_emotion=ribbons,
        min_size=poly_min,
        max_size=poly_max,
        fill_alpha=alpha_val,
        blur_px=stroke_blur,
        bg_color=bg_rgb,
        wobble=wobble,
        layers=layer_count
    )

    arr = np.array(img).astype(np.float32)/255.0
    lin = srgb_to_linear(arr)

    lin = lin*(2**exp)
    lin = apply_white_balance(lin, temp, tint)
    lin = highlight_rolloff(lin, roll)

    arr = linear_to_srgb(np.clip(lin,0,4))
    arr = np.clip(filmic_tonemap(arr*1.2),0,1)
    arr = adjust_contrast(arr, contrast)
    arr = adjust_saturation(arr, saturation)
    arr = gamma_correct(arr, gamma_val)
    arr = split_tone(arr, (sh_r,sh_g,sh_b), (hi_r,hi_g,hi_b), tone_balance)

    if auto_bright:
        arr = auto_brightness_compensation(
            arr,
            target_mean=target_mean,
            strength=abc_strength,
            black_point_pct=abc_black,
            white_point_pct=abc_white,
            max_gain=abc_max_gain
        )

    arr = apply_bloom(arr, bloom_radius, bloom_intensity)
    arr = apply_vignette(arr, vignette_strength)

    final_img = Image.fromarray((arr*255).astype(np.uint8))
    buf=BytesIO()
    final_img.save(buf, format="PNG")
    buf.seek(0)

    st.image(buf, use_column_width=True)
    st.download_button("💾 Download PNG", buf, "crystal_mix.png")


# ============================================================
# RIGHT: Data Table
# ============================================================

with right:
    st.subheader("📊 Data & Emotion Mapping")
    df2=df.copy()
    def disp(e):
        return f"{e} ({COLOR_NAMES.get(e,'Custom')})"
    df2["emotion_display"]=df2["emotion"].apply(disp)

    cols=["text","emotion_display","compound","pos","neu","neg"]
    if "timestamp" in df.columns: cols.insert(1,"timestamp")
    if "source" in df.columns: cols.insert(2,"source")

    st.dataframe(df2[cols], height=650)
