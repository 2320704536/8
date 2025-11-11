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
# App setup
# =========================================================
st.set_page_config(page_title="Emotional PrintFlow — Final", page_icon="🌀", layout="wide")
st.title("🌀 Emotional PrintFlow — Final Edition")

# =========================================================
# Load VADER
# =========================================================
@st.cache_resource(show_spinner=False)
def load_vader():
    try:
        nltk.data.find("sentiment/vader_lexicon")
    except LookupError:
        nltk.download("vader_lexicon")
    return SentimentIntensityAnalyzer()

sia = load_vader()

# =========================================================
# NewsAPI
# =========================================================
def fetch_news(api_key, keyword="emotion", page_size=50):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": keyword,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data.get("status") != "ok":
            st.warning("NewsAPI error: " + str(data.get("message")))
            return pd.DataFrame()
        rows = []
        for a in data.get("articles", []):
            text = (a.get("title") or "") + " - " + (a.get("description") or "")
            rows.append({
                "timestamp": (a.get("publishedAt") or "")[:10],
                "text": text.strip(" -"),
                "source": (a.get("source") or {}).get("name", "")
            })
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Error fetching NewsAPI: {e}")
        return pd.DataFrame()

# =========================================================
# Emotion colors — vivid
# =========================================================
DEFAULT_RGB = {
    "joy": (255,180,80),
    "love": (255,95,140),
    "pride": (210,110,255),
    "hope": (120,255,210),
    "curiosity": (80,220,255),
    "calm": (90,140,255),
    "surprise": (255,165,95),
    "neutral": (200,200,205),
    "sadness": (70,120,220),
    "anger": (255,75,75),
    "fear": (160,80,210),
    "disgust": (150,190,60),
    "anxiety": (255,210,70),
    "boredom": (130,130,150),
    "nostalgia": (255,225,180),
    "gratitude": (120,240,230),
    "awe": (160,245,255),
    "trust": (80,210,170),
    "confusion": (255,150,190),
    "mixed": (230,200,130),
}

ALL_EMOTIONS = list(DEFAULT_RGB.keys())
COLOR_NAMES = {e: e for e in ALL_EMOTIONS}

# =========================================================
# Sentiment → Emotion
# =========================================================
def analyze_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return {"neg":0.0,"neu":1.0,"pos":0.0,"compound":0.0}
    return sia.polarity_scores(text)

def classify_emotion_expanded(row):
    pos, neu, neg, comp = row["pos"], row["neu"], row["neg"], row["compound"]

    if comp >= 0.70 and pos > 0.50: return "joy"
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
    if pos > 0.35 and neu > 0.4 and 0 <= comp < 0.25: return "trust"
    if pos > 0.30 and neu > 0.35 and -0.05 <= comp <= 0.05: return "nostalgia"

    if pos > 0.25 and neg > 0.25: return "mixed"
    if pos > 0.20 and neu > 0.50 and comp > 0.05: return "curiosity"
    if neu > 0.6 and 0.05 <= comp <= 0.15: return "awe"

    return "neutral"

# =========================================================
# Palette state
# =========================================================
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

# =========================================================
# DEFAULTS + Reset
# =========================================================
DEFAULTS = {
    "keyword": "",
    "cmp_min": -1.0,
    "cmp_max": 1.0,
    "ribbons_per_emotion": 18,
    "stroke_width": 5,
    "steps": 580,
    "step_len": 2.0,
    "curve_noise": 0.25,
    "stroke_blur": 0,
    "ribbon_alpha": 230,
    "bg_custom": "#000000",
    "auto_top3": True,
    "exp": 0.5,
    "contrast": 1.15,
    "saturation": 1.12,
    "gamma_val": 0.92,
    "roll": 0.40,
    "temp": -0.05,
    "tint": 0.02,
    "sh_r": 0.08,"sh_g":0.06,"sh_b":0.16,
    "hi_r":0.10,"hi_g":0.08,"hi_b":0.06,
    "tone_balance":0.0,
    "bloom_radius":7.0,
    "bloom_intensity":0.42,
    "vignette_strength":0.18,
    "auto_bright":True,
    "target_mean":0.50,
    "abc_strength":0.9,
    "abc_black":0.05,
    "abc_white":0.997,
    "abc_max_gain":2.6
}

def reset_all():
    st.session_state.clear()
    st.rerun()

# =========================================================
# Helpers — Color
# =========================================================
def _rgb01(rgb):
    return np.clip(np.array(rgb, dtype=np.float32) / 255.0, 0, 1)

def vibrancy_boost(rgb, sat=1.22, min_luma=0.33):
    c = _rgb01(rgb)
    luma = 0.2126*c[0] + 0.7152*c[1] + 0.0722*c[2]
    if luma < min_luma:
        c += (min_luma - luma)
    c = np.clip(c,0,1)
    lum = 0.2126*c[0] + 0.7152*c[1] + 0.0722*c[2]
    c = lum + (c - lum)*sat
    return np.clip(c,0,1)

def jitter_color(rgb01, rng, amt=0.06):
    return tuple(np.clip(np.array(rgb01)+(rng.random(3)-0.5)*2*amt, 0,1))

# =========================================================
# Flow-field + Draw
# =========================================================
def fbm_noise(h,w,rng,octaves=5,base_scale=200,persistence=0.5,lacunarity=2.0):
    acc = np.zeros((h,w), np.float32)
    amp = 1.0
    scale = base_scale
    for _ in range(octaves):
        gh = max(1,h//max(1,scale))
        gw = max(1,w//max(1,scale))
        g = rng.random((gh,gw)).astype(np.float32)
        layer = np.array(
            Image.fromarray((g*255).astype(np.uint8)).resize((w,h), Image.BICUBIC)
        )/255.0
        acc += layer*amp
        amp *= persistence
        scale = max(1,int(scale/lacunarity))
    acc -= acc.min()
    if acc.max()>1e-6:
        acc /= acc.max()
    return acc

def generate_flow_field(height,width,rng,scale=200):
    noise = fbm_noise(height,width,rng,octaves=6,base_scale=scale)
    return noise * 2*np.pi

def draw_poly(canvas, pts, col01, width=4, alpha=210, blur_px=0):
    w,h = canvas.size
    layer = Image.new("RGBA",(w,h),(0,0,0,0))
    d = ImageDraw.Draw(layer,"RGBA")
    col = (int(col01[0]*255),int(col01[1]*255),int(col01[2]*255),int(alpha))
    if len(pts)>=2:
        d.line(pts, fill=col, width=width, joint="curve")
    if blur_px>0:
        layer = layer.filter(ImageFilter.GaussianBlur(radius=blur_px))
    canvas.alpha_composite(layer)

# =========================================================
# PrintFlow Renderer
# =========================================================
def render_printflow(
    df,palette,width=1500,height=850,seed=12345,
    ribbons_per_emotion=18,steps=580,step_len=2.0,
    curve_noise=0.25,stroke_width=5,ribbon_alpha=230,
    stroke_blur=0,bg_rgb=(0,0,0)
):
    rng = np.random.default_rng(seed)
    bg = Image.new("RGBA",(width,height),(bg_rgb[0],bg_rgb[1],bg_rgb[2],255))
    canvas = Image.new("RGBA",(width,height),(0,0,0,0))
    angle = generate_flow_field(height,width,rng,scale=180)

    emotions = df["emotion"].value_counts().index.tolist()
    if not emotions:
        emotions = ["joy","love","curiosity"]

    per = max(1,ribbons_per_emotion)

    for emo in emotions:
        base_rgb = palette.get(emo,palette.get("mixed"))
        base01 = vibrancy_boost(base_rgb, sat=1.25, min_luma=0.36)

        for _ in range(per):
            col01 = jitter_color(base01, rng, amt=0.08)
            x=rng.uniform(0,width)
            y=rng.uniform(0,height)
            pts=[]
            ang_strength = 1.0 + curve_noise*rng.uniform(0.8,1.3)

            for _s in range(steps):
                ix=int(np.clip(x,0,width-1))
                iy=int(np.clip(y,0,height-1))
                a=angle[iy,ix]*ang_strength
                x+=np.cos(a)*step_len
                y+=np.sin(a)*step_len
                if x<-20 or x>width+20 or y<-20 or y>height+20:
                    break
                if not pts or (abs(pts[-1][0]-x)+abs(pts[-1][1]-y))>1.0:
                    pts.append((float(x),float(y)))

            if len(pts)>=2:
                draw_poly(canvas, pts, col01, width=stroke_width, alpha=ribbon_alpha, blur_px=int(stroke_blur))

                shift=max(1,stroke_width//5)
                pts1=[(p[0],p[1]-shift) for p in pts]
                pts2=[(p[0],p[1]+shift) for p in pts]

                draw_poly(canvas, pts1, (1,1,1), width=max(1,stroke_width//6), alpha=min(160,ribbon_alpha))
                draw_poly(canvas, pts2, (1,1,1), width=max(1,stroke_width//6), alpha=min(130,ribbon_alpha))

    bg.alpha_composite(canvas)
    return bg.convert("RGB")

# =========================================================
# Cinematic Functions
# =========================================================
def srgb_to_linear(x):
    x=np.clip(x,0,1)
    return np.where(x<=0.04045,x/12.92,((x+0.055)/1.055)**2.4)

def linear_to_srgb(x):
    x=np.clip(x,0,1)
    return np.where(x<0.0031308,x*12.92,1.055*(x**(1/2.4))-0.055)

def filmic_tonemap(x):
    A=0.22;B=0.30;C=0.10;D=0.20;E=0.01;F=0.30
    return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F

def apply_white_balance(img,temp,tint):
    r,g,b=img[:,:,0],img[:,:,1],img[:,:,2]
    r*=1.0+0.10*temp
    b*=1.0-0.10*temp
    g*=1.0+0.08*tint
    r*=1.0-0.06*tint
    b*=1.0-0.02*tint
    out=np.stack([r,g,b],axis=-1)
    return np.clip(out,0,1)

def highlight_rolloff(img,roll):
    t=np.clip(roll,0.0,1.5)
    th=0.8
    mask=np.clip((img-th)/(1e-6+1.0-th),0,1)
    return np.clip(img*(1-mask)+(th+(img-th)/(1.0+4.0*t*mask))*mask,0,1)

def adjust_contrast(img,c):
    return np.clip((img-0.5)*c+0.5,0,1)

def adjust_saturation(img,s):
    lum=0.2126*img[:,:,0]+0.7152*img[:,:,1]+0.0722*img[:,:,2]
    lum=lum[...,None]
    return np.clip(lum+(img-lum)*s,0,1)

def gamma_correct(img,g):
    return np.clip(img**(1.0/g),0,1)

def split_tone(img,sh,hi,balance):
    lum=0.2126*img[:,:,0]+0.7152*img[:,:,1]+0.0722*img[:,:,2]
    lum=(lum-lum.min())/(lum.max()-lum.min()+1e-6)
    sh_mask=np.clip(1.0 - lum + 0.5*(1-balance),0,1)[...,None]
    hi_mask=np.clip(lum + 0.5*(1+balance)-0.5,0,1)[...,None]
    sh_col=np.array(sh).reshape(1,1,3)
    hi_col=np.array(hi).reshape(1,1,3)
    out=img+sh_mask*sh_col*0.25+hi_mask*hi_col*0.25
    return np.clip(out,0,1)

def apply_bloom(img,radius,intensity):
    pil=Image.fromarray((np.clip(img,0,1)*255).astype(np.uint8))
    if radius>0:
        blurred=pil.filter(ImageFilter.GaussianBlur(radius=radius))
        b=np.array(blurred)/255.0
        out=np.clip(img*(1-intensity)+b*intensity,0,1)
        return out
    return img

def apply_vignette(img,strength):
    h,w,_ = img.shape
    yy,xx=np.mgrid[0:h,0:w]
    xx=(xx-w/2)/(w/2); yy=(yy-h/2)/(h/2)
    r=np.sqrt(xx*xx+yy*yy)
    mask=np.clip(1-strength*(r**1.5),0,1)
    return np.clip(img*mask[...,None],0,1)

def ensure_colorfulness(img,min_sat=0.16,boost=1.18):
    r,g,b=img[:,:,0],img[:,:,1],img[:,:,2]
    mx=np.maximum(np.maximum(r,g),b)
    mn=np.minimum(np.minimum(r,g),b)
    sat=(mx-mn)/(mx+1e-6)
    if sat.mean()<min_sat:
        return adjust_saturation(img,boost)
    return img

def auto_brightness_compensation(
    img_arr,target_mean=0.50,strength=0.9,
    black_point_pct=0.05,white_point_pct=0.997,max_gain=2.6
):
    arr=np.clip(img_arr,0,1).astype(np.float32)
    lin=srgb_to_linear(arr)
    Y=0.2126*lin[:,:,0]+0.7152*lin[:,:,1]+0.0722*lin[:,:,2]
    bp=np.quantile(Y,black_point_pct)
    wp=np.quantile(Y,white_point_pct)
    if wp<=bp+1e-6: wp=bp+1e-3
    Y_remap=np.clip((Y-bp)/(wp-bp),0,1)
    Y_final=(1-strength)*Y + strength*Y_remap
    gain=np.clip(target_mean/max(Y_final.mean(),1e-4),1.0/max_gain,max_gain)
    lin*=gain
    out=filmic_tonemap(np.clip(lin,0,4))
    out=np.clip(out,0,1)
    out=linear_to_srgb(out)
    return np.clip(out,0,1)

# =========================================================
# UI
# =========================================================
with st.expander("How to Use", expanded=False):
    st.markdown("""
**Workflow**
1) 输入关键词（NewsAPI）  
2) 自动情绪分类  
3) 自动 Top-3 情绪  
4) 生成 PrintFlow 纹理作用图  
""")

# Data Source
st.sidebar.header("1) Data Source (NewsAPI)")
keyword = st.sidebar.text_input(
    "Keyword (例：technology, global warming, finance, AI...)",
    value=st.session_state.get("keyword", DEFAULTS["keyword"]),
    key="keyword"
)
fetch_btn = st.sidebar.button("Fetch News")

df = pd.DataFrame()
if fetch_btn:
    key = st.secrets.get("NEWS_API_KEY", "")
    if key:
        df = fetch_news(key, keyword.strip() if keyword.strip() else "aurora")
    else:
        st.sidebar.error("Missing NEWS_API_KEY")

if df.empty:
    df = pd.DataFrame({"text":[
        "A breathtaking aurora illuminated the northern sky last night.",
        "Investors show strong confidence in market recovery.",
        "Fear and tension spread during global conflicts.",
        "New scientific discovery sparks hope across nations.",
        "Unexpected event triggers mixed reactions worldwide."
    ]})
    df["timestamp"] = str(date.today())

df["text"]=df["text"].fillna("")
sent_df=df["text"].apply(analyze_sentiment).apply(pd.Series)
df=pd.concat([df.reset_index(drop=True),sent_df.reset_index(drop=True)],axis=1)
df["emotion"]=df.apply(classify_emotion_expanded,axis=1)

# Emotion Mapping
st.sidebar.header("2) Emotion Mapping")
cmp_min = st.sidebar.slider("Compound Min", -1.0, 1.0,
    st.session_state.get("cmp_min", DEFAULTS["cmp_min"]), 0.01, key="cmp_min")
cmp_max = st.sidebar.slider("Compound Max", -1.0, 1.0,
    st.session_state.get("cmp_max", DEFAULTS["cmp_max"]), 0.01, key="cmp_max")

init_palette_state()
palette = get_active_palette()

available = sorted(df["emotion"].unique().tolist())

auto_top3 = st.sidebar.checkbox(
    "Auto-select Top-3 emotions",
    value=st.session_state.get("auto_top3", DEFAULTS["auto_top3"]),
    key="auto_top3"
)
top3=[]
if auto_top3 and len(df):
    vc=df["emotion"].value_counts()
    top3=vc.head(3).index.tolist()

def _label(e):
    r,g,b = palette.get(e,(0,0,0))
    return f"{e} RGB({r},{g},{b})"

options=[_label(e) for e in ALL_EMOTIONS]
default=[_label(e) for e in (top3 if top3 else available)]
selected_labels = st.sidebar.multiselect("Selected Emotions", options, default=default)
selected_emotions=[lbl.split(" RGB")[0] for lbl in selected_labels]

df=df[
    (df["emotion"].isin(selected_emotions)) &
    (df["compound"]>=cmp_min) &
    (df["compound"]<=cmp_max)
]

# Ribbon Engine
st.sidebar.header("3) Ribbon Engine")
ribbons_per_emotion = st.sidebar.slider(
    "Ribbons per Emotion",2,40,
    st.session_state.get("ribbons_per_emotion",DEFAULTS["ribbons_per_emotion"]),
    1,key="ribbons_per_emotion")
stroke_width = st.sidebar.slider(
    "Stroke Width",1,20,
    st.session_state.get("stroke_width",DEFAULTS["stroke_width"]),
    1,key="stroke_width")
steps = st.sidebar.slider(
    "Ribbon Length (steps)",120,1200,
    st.session_state.get("steps",DEFAULTS["steps"]),
    10,key="steps")
step_len = st.sidebar.slider(
    "Step Length (px)",0.5,8.0,
    st.session_state.get("step_len",DEFAULTS["step_len"]),
    0.1,key="step_len")
curve_noise = st.sidebar.slider(
    "Curve Randomness",0.00,1.00,
    st.session_state.get("curve_noise",DEFAULTS["curve_noise"]),
    0.01,key="curve_noise")
stroke_blur = st.sidebar.slider(
    "Stroke Softness (blur px)",0,10,
    st.session_state.get("stroke_blur",DEFAULTS["stroke_blur"]),
    0.5,key="stroke_blur")
ribbon_alpha = st.sidebar.slider(
    "Ribbon Alpha",40,255,
    st.session_state.get("ribbon_alpha",DEFAULTS["ribbon_alpha"]),
    5,key="ribbon_alpha")

# Background (custom only)
st.sidebar.subheader("Background Color")
bg_custom = st.sidebar.color_picker(
    "Pick Color",
    value=st.session_state.get("bg_custom", DEFAULTS["bg_custom"]),
    key="bg_custom"
)
def _hex_to_rgb(h):
    h=h.lstrip("#")
    return tuple(int(h[i:i+2],16) for i in (0,2,4))
bg_rgb=_hex_to_rgb(bg_custom)

# Cinematic Color
st.sidebar.header("4) Cinematic Color")
exp = st.sidebar.slider("Exposure", -0.2,1.8,st.session_state.get("exp",DEFAULTS["exp"]),0.01,key="exp")
contrast = st.sidebar.slider("Contrast",0.70,1.80,st.session_state.get("contrast",DEFAULTS["contrast"]),0.01,key="contrast")
saturation = st.sidebar.slider("Saturation",0.70,1.90,st.session_state.get("saturation",DEFAULTS["saturation"]),0.01,key="saturation")
gamma_val = st.sidebar.slider("Gamma",0.70,1.40,st.session_state.get("gamma_val",DEFAULTS["gamma_val"]),0.01,key="gamma_val")
roll = st.sidebar.slider("Highlight Roll-off",0.00,1.50,st.session_state.get("roll",DEFAULTS["roll"]),0.01,key="roll")

st.sidebar.subheader("White Balance")
temp=st.sidebar.slider("Temp",-1,1,st.session_state.get("temp",DEFAULTS["temp"]),0.01,key="temp")
tint=st.sidebar.slider("Tint",-1,1,st.session_state.get("tint",DEFAULTS["tint"]),0.01,key="tint")

st.sidebar.subheader("Split Tone")
sh_r=st.sidebar.slider("Shadow R",0,1,st.session_state.get("sh_r",DEFAULTS["sh_r"]),0.01,key="sh_r")
sh_g=st.sidebar.slider("Shadow G",0,1,st.session_state.get("sh_g",DEFAULTS["sh_g"]),0.01,key="sh_g")
sh_b=st.sidebar.slider("Shadow B",0,1,st.session_state.get("sh_b",DEFAULTS["sh_b"]),0.01,key="sh_b")
hi_r=st.sidebar.slider("Highlight R",0,1,st.session_state.get("hi_r",DEFAULTS["hi_r"]),0.01,key="hi_r")
hi_g=st.sidebar.slider("Highlight G",0,1,st.session_state.get("hi_g",DEFAULTS["hi_g"]),0.01,key="hi_g")
hi_b=st.sidebar.slider("Highlight B",0,1,st.session_state.get("hi_b",DEFAULTS["hi_b"]),0.01,key="hi_b")
tone_balance=st.sidebar.slider("Tone Balance",-1,1,st.session_state.get("tone_balance",DEFAULTS["tone_balance"]),0.01,key="tone_balance")

st.sidebar.subheader("Bloom & Vignette")
bloom_radius=st.sidebar.slider("Bloom Radius",0,25,st.session_state.get("bloom_radius",DEFAULTS["bloom_radius"]),0.5,key="bloom_radius")
bloom_intensity=st.sidebar.slider("Bloom Intensity",0,1,st.session_state.get("bloom_intensity",DEFAULTS["bloom_intensity"]),0.01,key="bloom_intensity")
vignette_strength=st.sidebar.slider("Vignette",0,1,st.session_state.get("vignette_strength",DEFAULTS["vignette_strength"]),0.01,key="vignette_strength")

# Auto Brightness
st.sidebar.header("5) Auto Brightness")
auto_bright=st.sidebar.checkbox(
    "Enable Auto Brightness",
    value=st.session_state.get("auto_bright",DEFAULTS["auto_bright"]),
    key="auto_bright"
)
target_mean=st.sidebar.slider("Target Mean",0.30,0.70,st.session_state.get("target_mean",DEFAULTS["target_mean"]),0.01,key="target_mean")
abc_strength=st.sidebar.slider("Remap Strength",0,1,st.session_state.get("abc_strength",DEFAULTS["abc_strength"]),0.05,key="abc_strength")
abc_black=st.sidebar.slider("Black Point",0,0.20,st.session_state.get("abc_black",DEFAULTS["abc_black"]),0.01,key="abc_black")
abc_white=st.sidebar.slider("White Point",0.80,1.00,st.session_state.get("abc_white",DEFAULTS["abc_white"]),0.001,key="abc_white")
abc_max_gain=st.sidebar.slider("Max Gain",1,3,st.session_state.get("abc_max_gain",DEFAULTS["abc_max_gain"]),0.05,key="abc_max_gain")

# Reset All Button
st.sidebar.header("6) Output")
if st.sidebar.button("Reset All",type="primary"):
    reset_all()

# =========================================================
# Draw
# =========================================================
left,right=st.columns([0.60,0.40])

with left:
    st.subheader("🎐 Ribbon Flow — PrintFlow")

    img = render_printflow(
        df=df, palette=palette,
        width=1500, height=850, seed=np.random.randint(0,999999),
        ribbons_per_emotion=ribbons_per_emotion,
        steps=steps, step_len=step_len,
        curve_noise=curve_noise, stroke_width=stroke_width,
        ribbon_alpha=ribbon_alpha, stroke_blur=stroke_blur,
        bg_rgb=bg_rgb
    )

    arr=np.array(img).astype(np.float32)/255.0
    lin=srgb_to_linear(arr)
    lin*=2.0**exp
    lin=apply_white_balance(lin,temp,tint)
    lin=highlight_rolloff(lin,roll)
    arr=linear_to_srgb(np.clip(lin,0,4))
    arr=np.clip(filmic_tonemap(arr*1.25),0,1)
    arr=adjust_contrast(arr,contrast)
    arr=adjust_saturation(arr,saturation)
    arr=gamma_correct(arr,gamma_val)
    arr=split_tone(arr,(sh_r,sh_g,sh_b),(hi_r,hi_g,hi_b),tone_balance)

    if auto_bright:
        arr = auto_brightness_compensation(
            arr,target_mean=target_mean,strength=abc_strength,
            black_point_pct=abc_black,white_point_pct=abc_white,
            max_gain=abc_max_gain
        )

    arr=apply_bloom(arr,bloom_radius,bloom_intensity)
    arr=apply_vignette(arr,vignette_strength)
    arr=ensure_colorfulness(arr)

    final_img=Image.fromarray((np.clip(arr,0,1)*255).astype(np.uint8))
    buf=BytesIO()
    final_img.save(buf,format="PNG")
    buf.seek(0)

    st.image(buf,use_column_width=True)
    st.download_button("💾 Download PNG", data=buf, file_name="printflow.png", mime="image/png")

with right:
    st.subheader("📊 Data & Emotions")
    df2=df.copy()
    df2["emotion_display"]=df2["emotion"].apply(lambda e:f"{e} — RGB{palette.get(e,(0,0,0))}")
    cols=["text","emotion_display","compound","pos","neu","neg"]
    if "timestamp" in df.columns: cols.insert(1,"timestamp")
    if "source" in df.columns: cols.insert(2,"source")
    st.dataframe(df2[cols],use_container_width=True,height=620)

st.markdown("---")
st.caption("© 2025 Emotional Ribbon — PrintFlow Edition")
