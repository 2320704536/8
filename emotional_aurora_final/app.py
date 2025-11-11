# =========================
# Defaults & Reset
# =========================
DEFAULTS = {
    "keyword": "",
    "cmp_min": -1.0,
    "cmp_max": 1.0,
    "auto_top3": True,

    # Polygon parameters
    "ribbons_per_emotion": 16,     # legacy name but kept for layout
    "stroke_width": 5,
    "steps": 500,
    "step_len": 2.5,
    "curve_noise": 0.3,
    "stroke_blur": 0.0,
    "ribbon_alpha": 200,

    # Background
    "bg_custom": "#000000",

    # Color system
    "exp": 0.55,
    "contrast": 1.18,
    "saturation": 1.18,
    "gamma_val": 0.92,
    "roll": 0.40,
    "temp": 0.0,
    "tint": 0.0,

    "sh_r": 0.08, "sh_g": 0.06, "sh_b": 0.16,
    "hi_r": 0.10, "hi_g": 0.08, "hi_b": 0.06,
    "tone_balance": 0.0,

    "bloom_radius": 7.0,
    "bloom_intensity": 0.40,
    "vignette_strength": 0.16,

    # Auto brightness
    "auto_bright": True,
    "target_mean": 0.52,
    "abc_strength": 0.92,
    "abc_black": 0.05,
    "abc_white": 0.997,
    "abc_max_gain": 2.6,
}

def reset_all():
    st.session_state.clear()
    st.rerun()


# =========================
# How to Use
# =========================
with st.expander("How to Use", expanded=False):
    st.markdown("""
1) 输入关键词（如：AI, psychology, aurora borealis），点击 Fetch News  
2) 自动提取情绪，并自动选中 **Top-3 主情绪**  
3) 左侧可以调节多边形生成与电影级调色  
4) 点击下载 PNG  
""")


# =========================
# Sidebar — Data Source
# =========================
st.sidebar.header("1) Data Source (NewsAPI)")
keyword = st.sidebar.text_input(
    "Keyword (e.g., AI, psychology, science, aurora)",
    value=st.session_state.get("keyword", DEFAULTS["keyword"]),
    key="keyword",
)

fetch_btn = st.sidebar.button("Fetch News")


# --- Fetch or fallback ---
df = pd.DataFrame()
if fetch_btn:
    key = st.secrets.get("NEWS_API_KEY", "")
    if not key:
        st.sidebar.error("Missing NEWS_API_KEY in Streamlit Secrets.")
    else:
        df = fetch_news(key, keyword if keyword.strip() else "emotion")


# If empty, fallback to demo
if df.empty:
    df = pd.DataFrame({
        "text":[
            "Hope rises as scientists make new discoveries.",
            "A moment of awe spreads across social media.",
            "Calm energy fills the air after the storm.",
            "Anxiety grows in the financial markets.",
            "People express gratitude for the clear night sky."
        ],
        "timestamp": str(date.today())
    })

df["text"] = df["text"].fillna("")

# Sentiment
sent_df = df["text"].apply(analyze_sentiment).apply(pd.Series)
df = pd.concat([df.reset_index(drop=True), sent_df.reset_index(drop=True)], axis=1)

df["emotion"] = df.apply(classify_emotion_expanded, axis=1)


# =========================
# Sidebar — Emotion Mapping
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

auto_top3 = st.sidebar.checkbox(
    "Auto-select Top-3 emotions", 
    value=st.session_state.get("auto_top3", DEFAULTS["auto_top3"]),
    key="auto_top3"
)

vc = df["emotion"].value_counts()
top3 = vc.head(3).index.tolist() if len(vc) else []

# Emotion select
selected_emotions = st.sidebar.multiselect(
    "Selected Emotions",
    ALL_EMOTIONS,
    default=top3 if auto_top3 else ALL_EMOTIONS
)

# Filter df
df = df[(df["emotion"].isin(selected_emotions)) &
        (df["compound"] >= cmp_min) &
        (df["compound"] <= cmp_max)]


# =========================
# Sidebar — Polygon Engine
# =========================
st.sidebar.header("3) Polygon Engine (A + 2)")
# These sliders are kept for layout consistency
ribbons_per_emotion = st.sidebar.slider(
    "Polygon Density", 2, 50,
    st.session_state.get("ribbons_per_emotion", DEFAULTS["ribbons_per_emotion"]),
    1, key="ribbons_per_emotion"
)
stroke_width = st.sidebar.slider(
    "Edge Strength", 1, 15,
    st.session_state.get("stroke_width", DEFAULTS["stroke_width"]),
    1, key="stroke_width"
)
stroke_blur = st.sidebar.slider(
    "Softness (blur px)", 0.0, 10.0,
    st.session_state.get("stroke_blur", DEFAULTS["stroke_blur"]),
    0.5, key="stroke_blur"
)

# Background (solid)
st.sidebar.subheader("Background")
def _hex_to_rgb(hx):
    hx = hx.lstrip("#")
    return tuple(int(hx[i:i+2], 16) for i in (0,2,4))

bg_custom = st.sidebar.color_picker(
    "Background Color",
    value=st.session_state.get("bg_custom", DEFAULTS["bg_custom"]),
    key="bg_custom"
)
bg_rgb = _hex_to_rgb(bg_custom)


# =========================
# Sidebar — Cinematic Color System
# =========================
st.sidebar.header("4) Cinematic Color System")

exp = st.sidebar.slider("Exposure", -0.2, 1.8,
    st.session_state.get("exp", DEFAULTS["exp"]), 0.01, key="exp")
contrast = st.sidebar.slider("Contrast", 0.70, 1.80,
    st.session_state.get("contrast", DEFAULTS["contrast"]), 0.01, key="contrast")
saturation = st.sidebar.slider("Saturation", 0.70, 1.90,
    st.session_state.get("saturation", DEFAULTS["saturation"]), 0.01, key="saturation")
gamma_val = st.sidebar.slider("Gamma", 0.70, 1.40,
    st.session_state.get("gamma_val", DEFAULTS["gamma_val"]), 0.01, key="gamma_val")
roll = st.sidebar.slider("Highlight Roll-off", 0.00, 1.50,
    st.session_state.get("roll", DEFAULTS["roll"]), 0.01, key="roll")

st.sidebar.subheader("White Balance")
temp = st.sidebar.slider("Temperature", -1.0, 1.0,
    st.session_state.get("temp", DEFAULTS["temp"]), 0.01, key="temp")
tint = st.sidebar.slider("Tint", -1.0, 1.0,
    st.session_state.get("tint", DEFAULTS["tint"]), 0.01, key="tint")

st.sidebar.subheader("Split Toning")
sh_r = st.sidebar.slider("Shadows R", 0.0, 1.0,
    st.session_state.get("sh_r", DEFAULTS["sh_r"]), 0.01, key="sh_r")
sh_g = st.sidebar.slider("Shadows G", 0.0, 1.0,
    st.session_state.get("sh_g", DEFAULTS["sh_g"]), 0.01, key="sh_g")
sh_b = st.sidebar.slider("Shadows B", 0.0, 1.0,
    st.session_state.get("sh_b", DEFAULTS["sh_b"]), 0.01, key="sh_b")

hi_r = st.sidebar.slider("Highlights R", 0.0, 1.0,
    st.session_state.get("hi_r", DEFAULTS["hi_r"]), 0.01, key="hi_r")
hi_g = st.sidebar.slider("Highlights G", 0.0, 1.0,
    st.session_state.get("hi_g", DEFAULTS["hi_g"]), 0.01, key="hi_g")
hi_b = st.sidebar.slider("Highlights B", 0.0, 1.0,
    st.session_state.get("hi_b", DEFAULTS["hi_b"]), 0.01, key="hi_b")

tone_balance = st.sidebar.slider("Shadow ↔ Highlight Balance", -1.0, 1.0,
    st.session_state.get("tone_balance", DEFAULTS["tone_balance"]), 0.01, key="tone_balance")

st.sidebar.subheader("Bloom & Vignette")
bloom_radius = st.sidebar.slider("Bloom Radius", 0.0, 20.0,
    st.session_state.get("bloom_radius", DEFAULTS["bloom_radius"]), 0.5, key="bloom_radius")
bloom_intensity = st.sidebar.slider("Bloom Intensity", 0.0, 1.0,
    st.session_state.get("bloom_intensity", DEFAULTS["bloom_intensity"]), 0.01, key="bloom_intensity")
vignette_strength = st.sidebar.slider("Vignette Strength", 0.0, 0.8,
    st.session_state.get("vignette_strength", DEFAULTS["vignette_strength"]), 0.01, key="vignette_strength")


# =========================
# Sidebar — Auto Brightness
# =========================
st.sidebar.header("5) Auto Brightness")
auto_bright = st.sidebar.checkbox(
    "Enable Auto Brightness",
    value=st.session_state.get("auto_bright", DEFAULTS["auto_bright"]),
    key="auto_bright"
)

target_mean = st.sidebar.slider("Target Mean", 0.30, 0.70,
    st.session_state.get("target_mean", DEFAULTS["target_mean"]), 0.01, key="target_mean")
abc_strength = st.sidebar.slider("Remap Strength", 0.0, 1.0,
    st.session_state.get("abc_strength", DEFAULTS["abc_strength"]), 0.05, key="abc_strength")
abc_black = st.sidebar.slider("Black Point %", 0.0, 0.20,
    st.session_state.get("abc_black", DEFAULTS["abc_black"]), 0.01, key="abc_black")
abc_white = st.sidebar.slider("White Point %", 0.80, 1.00,
    st.session_state.get("abc_white", DEFAULTS["abc_white"]), 0.001, key="abc_white")
abc_max_gain = st.sidebar.slider("Max Gain", 1.0, 3.0,
    st.session_state.get("abc_max_gain", DEFAULTS["abc_max_gain"]), 0.05, key="abc_max_gain")


# =========================
# Sidebar — Palette
# =========================
st.sidebar.header("6) Custom Palette (RGB)")

if "custom_palette" not in st.session_state:
    st.session_state["custom_palette"] = {}

use_csv = st.sidebar.checkbox(
    "Use CSV palette only",
    value=st.session_state.get("use_csv_palette", False),
    key="use_csv_palette"
)

with st.sidebar.expander("Add Custom Color", False):
    col1,col2,col3,col4 = st.columns([1.7,1,1,1])
    name = col1.text_input("Emotion", key="add_emotion")
    r = col2.number_input("R", 0,255,200, key="add_r")
    g = col3.number_input("G", 0,255,160, key="add_g")
    b = col4.number_input("B", 0,255,140, key="add_b")

    if st.button("Add Color", key="btn_add_color"):
        if name:
            st.session_state["custom_palette"][name] = (int(r),int(g),int(b))

with st.sidebar.expander("Import / Export CSV", False):
    up = st.file_uploader("Import CSV", type=["csv"], key="up_csv")
    if up is not None:
        dfc = pd.read_csv(up)
        if {"emotion","r","g","b"}.issubset([c.lower() for c in dfc.columns]):
            pal={}
            for _,row in dfc.iterrows():
                emo=str(row["emotion"]).strip()
                pal[emo] = (int(row["r"]),int(row["g"]),int(row["b"]))
            st.session_state["custom_palette"] = pal
            st.success("Imported.")
        else:
            st.error("CSV must contain emotion,r,g,b")

    pal = dict(DEFAULT_RGB)
    pal.update(st.session_state["custom_palette"])

    df_pal = pd.DataFrame([{"emotion":k,"r":v[0],"g":v[1],"b":v[2]} for k,v in pal.items()])
    st.dataframe(df_pal, use_container_width=True, height=180)

    buf = BytesIO()
    df_pal.to_csv(buf, index=False)
    buf.seek(0)
    st.download_button("Download CSV", data=buf, file_name="palette.csv")


# =========================
# Sidebar — Reset
# =========================
st.sidebar.header("7) Output")
if st.sidebar.button("Reset All"):
    reset_all()
# =========================
# Defaults & Reset
# =========================
DEFAULTS = {
    "keyword": "",
    "cmp_min": -1.0,
    "cmp_max": 1.0,
    "auto_top3": True,

    # Polygon parameters
    "ribbons_per_emotion": 16,     # legacy name but kept for layout
    "stroke_width": 5,
    "steps": 500,
    "step_len": 2.5,
    "curve_noise": 0.3,
    "stroke_blur": 0.0,
    "ribbon_alpha": 200,

    # Background
    "bg_custom": "#000000",

    # Color system
    "exp": 0.55,
    "contrast": 1.18,
    "saturation": 1.18,
    "gamma_val": 0.92,
    "roll": 0.40,
    "temp": 0.0,
    "tint": 0.0,

    "sh_r": 0.08, "sh_g": 0.06, "sh_b": 0.16,
    "hi_r": 0.10, "hi_g": 0.08, "hi_b": 0.06,
    "tone_balance": 0.0,

    "bloom_radius": 7.0,
    "bloom_intensity": 0.40,
    "vignette_strength": 0.16,

    # Auto brightness
    "auto_bright": True,
    "target_mean": 0.52,
    "abc_strength": 0.92,
    "abc_black": 0.05,
    "abc_white": 0.997,
    "abc_max_gain": 2.6,
}

def reset_all():
    st.session_state.clear()
    st.rerun()


# =========================
# How to Use
# =========================
with st.expander("How to Use", expanded=False):
    st.markdown("""
1) 输入关键词（如：AI, psychology, aurora borealis），点击 Fetch News  
2) 自动提取情绪，并自动选中 **Top-3 主情绪**  
3) 左侧可以调节多边形生成与电影级调色  
4) 点击下载 PNG  
""")


# =========================
# Sidebar — Data Source
# =========================
st.sidebar.header("1) Data Source (NewsAPI)")
keyword = st.sidebar.text_input(
    "Keyword (e.g., AI, psychology, science, aurora)",
    value=st.session_state.get("keyword", DEFAULTS["keyword"]),
    key="keyword",
)

fetch_btn = st.sidebar.button("Fetch News")


# --- Fetch or fallback ---
df = pd.DataFrame()
if fetch_btn:
    key = st.secrets.get("NEWS_API_KEY", "")
    if not key:
        st.sidebar.error("Missing NEWS_API_KEY in Streamlit Secrets.")
    else:
        df = fetch_news(key, keyword if keyword.strip() else "emotion")


# If empty, fallback to demo
if df.empty:
    df = pd.DataFrame({
        "text":[
            "Hope rises as scientists make new discoveries.",
            "A moment of awe spreads across social media.",
            "Calm energy fills the air after the storm.",
            "Anxiety grows in the financial markets.",
            "People express gratitude for the clear night sky."
        ],
        "timestamp": str(date.today())
    })

df["text"] = df["text"].fillna("")

# Sentiment
sent_df = df["text"].apply(analyze_sentiment).apply(pd.Series)
df = pd.concat([df.reset_index(drop=True), sent_df.reset_index(drop=True)], axis=1)

df["emotion"] = df.apply(classify_emotion_expanded, axis=1)


# =========================
# Sidebar — Emotion Mapping
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

auto_top3 = st.sidebar.checkbox(
    "Auto-select Top-3 emotions", 
    value=st.session_state.get("auto_top3", DEFAULTS["auto_top3"]),
    key="auto_top3"
)

vc = df["emotion"].value_counts()
top3 = vc.head(3).index.tolist() if len(vc) else []

# Emotion select
selected_emotions = st.sidebar.multiselect(
    "Selected Emotions",
    ALL_EMOTIONS,
    default=top3 if auto_top3 else ALL_EMOTIONS
)

# Filter df
df = df[(df["emotion"].isin(selected_emotions)) &
        (df["compound"] >= cmp_min) &
        (df["compound"] <= cmp_max)]


# =========================
# Sidebar — Polygon Engine
# =========================
st.sidebar.header("3) Polygon Engine (A + 2)")
# These sliders are kept for layout consistency
ribbons_per_emotion = st.sidebar.slider(
    "Polygon Density", 2, 50,
    st.session_state.get("ribbons_per_emotion", DEFAULTS["ribbons_per_emotion"]),
    1, key="ribbons_per_emotion"
)
stroke_width = st.sidebar.slider(
    "Edge Strength", 1, 15,
    st.session_state.get("stroke_width", DEFAULTS["stroke_width"]),
    1, key="stroke_width"
)
stroke_blur = st.sidebar.slider(
    "Softness (blur px)", 0.0, 10.0,
    st.session_state.get("stroke_blur", DEFAULTS["stroke_blur"]),
    0.5, key="stroke_blur"
)

# Background (solid)
st.sidebar.subheader("Background")
def _hex_to_rgb(hx):
    hx = hx.lstrip("#")
    return tuple(int(hx[i:i+2], 16) for i in (0,2,4))

bg_custom = st.sidebar.color_picker(
    "Background Color",
    value=st.session_state.get("bg_custom", DEFAULTS["bg_custom"]),
    key="bg_custom"
)
bg_rgb = _hex_to_rgb(bg_custom)


# =========================
# Sidebar — Cinematic Color System
# =========================
st.sidebar.header("4) Cinematic Color System")

exp = st.sidebar.slider("Exposure", -0.2, 1.8,
    st.session_state.get("exp", DEFAULTS["exp"]), 0.01, key="exp")
contrast = st.sidebar.slider("Contrast", 0.70, 1.80,
    st.session_state.get("contrast", DEFAULTS["contrast"]), 0.01, key="contrast")
saturation = st.sidebar.slider("Saturation", 0.70, 1.90,
    st.session_state.get("saturation", DEFAULTS["saturation"]), 0.01, key="saturation")
gamma_val = st.sidebar.slider("Gamma", 0.70, 1.40,
    st.session_state.get("gamma_val", DEFAULTS["gamma_val"]), 0.01, key="gamma_val")
roll = st.sidebar.slider("Highlight Roll-off", 0.00, 1.50,
    st.session_state.get("roll", DEFAULTS["roll"]), 0.01, key="roll")

st.sidebar.subheader("White Balance")
temp = st.sidebar.slider("Temperature", -1.0, 1.0,
    st.session_state.get("temp", DEFAULTS["temp"]), 0.01, key="temp")
tint = st.sidebar.slider("Tint", -1.0, 1.0,
    st.session_state.get("tint", DEFAULTS["tint"]), 0.01, key="tint")

st.sidebar.subheader("Split Toning")
sh_r = st.sidebar.slider("Shadows R", 0.0, 1.0,
    st.session_state.get("sh_r", DEFAULTS["sh_r"]), 0.01, key="sh_r")
sh_g = st.sidebar.slider("Shadows G", 0.0, 1.0,
    st.session_state.get("sh_g", DEFAULTS["sh_g"]), 0.01, key="sh_g")
sh_b = st.sidebar.slider("Shadows B", 0.0, 1.0,
    st.session_state.get("sh_b", DEFAULTS["sh_b"]), 0.01, key="sh_b")

hi_r = st.sidebar.slider("Highlights R", 0.0, 1.0,
    st.session_state.get("hi_r", DEFAULTS["hi_r"]), 0.01, key="hi_r")
hi_g = st.sidebar.slider("Highlights G", 0.0, 1.0,
    st.session_state.get("hi_g", DEFAULTS["hi_g"]), 0.01, key="hi_g")
hi_b = st.sidebar.slider("Highlights B", 0.0, 1.0,
    st.session_state.get("hi_b", DEFAULTS["hi_b"]), 0.01, key="hi_b")

tone_balance = st.sidebar.slider("Shadow ↔ Highlight Balance", -1.0, 1.0,
    st.session_state.get("tone_balance", DEFAULTS["tone_balance"]), 0.01, key="tone_balance")

st.sidebar.subheader("Bloom & Vignette")
bloom_radius = st.sidebar.slider("Bloom Radius", 0.0, 20.0,
    st.session_state.get("bloom_radius", DEFAULTS["bloom_radius"]), 0.5, key="bloom_radius")
bloom_intensity = st.sidebar.slider("Bloom Intensity", 0.0, 1.0,
    st.session_state.get("bloom_intensity", DEFAULTS["bloom_intensity"]), 0.01, key="bloom_intensity")
vignette_strength = st.sidebar.slider("Vignette Strength", 0.0, 0.8,
    st.session_state.get("vignette_strength", DEFAULTS["vignette_strength"]), 0.01, key="vignette_strength")


# =========================
# Sidebar — Auto Brightness
# =========================
st.sidebar.header("5) Auto Brightness")
auto_bright = st.sidebar.checkbox(
    "Enable Auto Brightness",
    value=st.session_state.get("auto_bright", DEFAULTS["auto_bright"]),
    key="auto_bright"
)

target_mean = st.sidebar.slider("Target Mean", 0.30, 0.70,
    st.session_state.get("target_mean", DEFAULTS["target_mean"]), 0.01, key="target_mean")
abc_strength = st.sidebar.slider("Remap Strength", 0.0, 1.0,
    st.session_state.get("abc_strength", DEFAULTS["abc_strength"]), 0.05, key="abc_strength")
abc_black = st.sidebar.slider("Black Point %", 0.0, 0.20,
    st.session_state.get("abc_black", DEFAULTS["abc_black"]), 0.01, key="abc_black")
abc_white = st.sidebar.slider("White Point %", 0.80, 1.00,
    st.session_state.get("abc_white", DEFAULTS["abc_white"]), 0.001, key="abc_white")
abc_max_gain = st.sidebar.slider("Max Gain", 1.0, 3.0,
    st.session_state.get("abc_max_gain", DEFAULTS["abc_max_gain"]), 0.05, key="abc_max_gain")


# =========================
# Sidebar — Palette
# =========================
st.sidebar.header("6) Custom Palette (RGB)")

if "custom_palette" not in st.session_state:
    st.session_state["custom_palette"] = {}

use_csv = st.sidebar.checkbox(
    "Use CSV palette only",
    value=st.session_state.get("use_csv_palette", False),
    key="use_csv_palette"
)

with st.sidebar.expander("Add Custom Color", False):
    col1,col2,col3,col4 = st.columns([1.7,1,1,1])
    name = col1.text_input("Emotion", key="add_emotion")
    r = col2.number_input("R", 0,255,200, key="add_r")
    g = col3.number_input("G", 0,255,160, key="add_g")
    b = col4.number_input("B", 0,255,140, key="add_b")

    if st.button("Add Color", key="btn_add_color"):
        if name:
            st.session_state["custom_palette"][name] = (int(r),int(g),int(b))

with st.sidebar.expander("Import / Export CSV", False):
    up = st.file_uploader("Import CSV", type=["csv"], key="up_csv")
    if up is not None:
        dfc = pd.read_csv(up)
        if {"emotion","r","g","b"}.issubset([c.lower() for c in dfc.columns]):
            pal={}
            for _,row in dfc.iterrows():
                emo=str(row["emotion"]).strip()
                pal[emo] = (int(row["r"]),int(row["g"]),int(row["b"]))
            st.session_state["custom_palette"] = pal
            st.success("Imported.")
        else:# =========================
# Defaults & Reset
# =========================
DEFAULTS = {
    "keyword": "",
    "cmp_min": -1.0,
    "cmp_max": 1.0,
    "auto_top3": True,

    # Polygon parameters
    "ribbons_per_emotion": 16,     # legacy name but kept for layout
    "stroke_width": 5,
    "steps": 500,
    "step_len": 2.5,
    "curve_noise": 0.3,
    "stroke_blur": 0.0,
    "ribbon_alpha": 200,

    # Background
    "bg_custom": "#000000",

    # Color system
    "exp": 0.55,
    "contrast": 1.18,
    "saturation": 1.18,
    "gamma_val": 0.92,
    "roll": 0.40,
    "temp": 0.0,
    "tint": 0.0,

    "sh_r": 0.08, "sh_g": 0.06, "sh_b": 0.16,
    "hi_r": 0.10, "hi_g": 0.08, "hi_b": 0.06,
    "tone_balance": 0.0,

    "bloom_radius": 7.0,
    "bloom_intensity": 0.40,
    "vignette_strength": 0.16,

    # Auto brightness
    "auto_bright": True,
    "target_mean": 0.52,
    "abc_strength": 0.92,
    "abc_black": 0.05,
    "abc_white": 0.997,
    "abc_max_gain": 2.6,
}

def reset_all():
    st.session_state.clear()
    st.rerun()


# =========================
# How to Use
# =========================
with st.expander("How to Use", expanded=False):
    st.markdown("""
1) 输入关键词（如：AI, psychology, aurora borealis），点击 Fetch News  
2) 自动提取情绪，并自动选中 **Top-3 主情绪**  
3) 左侧可以调节多边形生成与电影级调色  
4) 点击下载 PNG  
""")


# =========================
# Sidebar — Data Source
# =========================
st.sidebar.header("1) Data Source (NewsAPI)")
keyword = st.sidebar.text_input(
    "Keyword (e.g., AI, psychology, science, aurora)",
    value=st.session_state.get("keyword", DEFAULTS["keyword"]),
    key="keyword",
)

fetch_btn = st.sidebar.button("Fetch News")


# --- Fetch or fallback ---
df = pd.DataFrame()
if fetch_btn:
    key = st.secrets.get("NEWS_API_KEY", "")
    if not key:
        st.sidebar.error("Missing NEWS_API_KEY in Streamlit Secrets.")
    else:
        df = fetch_news(key, keyword if keyword.strip() else "emotion")


# If empty, fallback to demo
if df.empty:
    df = pd.DataFrame({
        "text":[
            "Hope rises as scientists make new discoveries.",
            "A moment of awe spreads across social media.",
            "Calm energy fills the air after the storm.",
            "Anxiety grows in the financial markets.",
            "People express gratitude for the clear night sky."
        ],
        "timestamp": str(date.today())
    })

df["text"] = df["text"].fillna("")

# Sentiment
sent_df = df["text"].apply(analyze_sentiment).apply(pd.Series)
df = pd.concat([df.reset_index(drop=True), sent_df.reset_index(drop=True)], axis=1)

df["emotion"] = df.apply(classify_emotion_expanded, axis=1)


# =========================
# Sidebar — Emotion Mapping
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

auto_top3 = st.sidebar.checkbox(
    "Auto-select Top-3 emotions", 
    value=st.session_state.get("auto_top3", DEFAULTS["auto_top3"]),
    key="auto_top3"
)

vc = df["emotion"].value_counts()
top3 = vc.head(3).index.tolist() if len(vc) else []

# Emotion select
selected_emotions = st.sidebar.multiselect(
    "Selected Emotions",
    ALL_EMOTIONS,
    default=top3 if auto_top3 else ALL_EMOTIONS
)

# Filter df
df = df[(df["emotion"].isin(selected_emotions)) &
        (df["compound"] >= cmp_min) &
        (df["compound"] <= cmp_max)]


# =========================
# Sidebar — Polygon Engine
# =========================
st.sidebar.header("3) Polygon Engine (A + 2)")
# These sliders are kept for layout consistency
ribbons_per_emotion = st.sidebar.slider(
    "Polygon Density", 2, 50,
    st.session_state.get("ribbons_per_emotion", DEFAULTS["ribbons_per_emotion"]),
    1, key="ribbons_per_emotion"
)
stroke_width = st.sidebar.slider(
    "Edge Strength", 1, 15,
    st.session_state.get("stroke_width", DEFAULTS["stroke_width"]),
    1, key="stroke_width"
)
stroke_blur = st.sidebar.slider(
    "Softness (blur px)", 0.0, 10.0,
    st.session_state.get("stroke_blur", DEFAULTS["stroke_blur"]),
    0.5, key="stroke_blur"
)

# Background (solid)
st.sidebar.subheader("Background")
def _hex_to_rgb(hx):
    hx = hx.lstrip("#")
    return tuple(int(hx[i:i+2], 16) for i in (0,2,4))

bg_custom = st.sidebar.color_picker(
    "Background Color",
    value=st.session_state.get("bg_custom", DEFAULTS["bg_custom"]),
    key="bg_custom"
)
bg_rgb = _hex_to_rgb(bg_custom)


# =========================
# Sidebar — Cinematic Color System
# =========================
st.sidebar.header("4) Cinematic Color System")

exp = st.sidebar.slider("Exposure", -0.2, 1.8,
    st.session_state.get("exp", DEFAULTS["exp"]), 0.01, key="exp")
contrast = st.sidebar.slider("Contrast", 0.70, 1.80,
    st.session_state.get("contrast", DEFAULTS["contrast"]), 0.01, key="contrast")
saturation = st.sidebar.slider("Saturation", 0.70, 1.90,
    st.session_state.get("saturation", DEFAULTS["saturation"]), 0.01, key="saturation")
gamma_val = st.sidebar.slider("Gamma", 0.70, 1.40,
    st.session_state.get("gamma_val", DEFAULTS["gamma_val"]), 0.01, key="gamma_val")
roll = st.sidebar.slider("Highlight Roll-off", 0.00, 1.50,
    st.session_state.get("roll", DEFAULTS["roll"]), 0.01, key="roll")

st.sidebar.subheader("White Balance")
temp = st.sidebar.slider("Temperature", -1.0, 1.0,
    st.session_state.get("temp", DEFAULTS["temp"]), 0.01, key="temp")
tint = st.sidebar.slider("Tint", -1.0, 1.0,
    st.session_state.get("tint", DEFAULTS["tint"]), 0.01, key="tint")

st.sidebar.subheader("Split Toning")
sh_r = st.sidebar.slider("Shadows R", 0.0, 1.0,
    st.session_state.get("sh_r", DEFAULTS["sh_r"]), 0.01, key="sh_r")
sh_g = st.sidebar.slider("Shadows G", 0.0, 1.0,
    st.session_state.get("sh_g", DEFAULTS["sh_g"]), 0.01, key="sh_g")
sh_b = st.sidebar.slider("Shadows B", 0.0, 1.0,
    st.session_state.get("sh_b", DEFAULTS["sh_b"]), 0.01, key="sh_b")

hi_r = st.sidebar.slider("Highlights R", 0.0, 1.0,
    st.session_state.get("hi_r", DEFAULTS["hi_r"]), 0.01, key="hi_r")
hi_g = st.sidebar.slider("Highlights G", 0.0, 1.0,
    st.session_state.get("hi_g", DEFAULTS["hi_g"]), 0.01, key="hi_g")
hi_b = st.sidebar.slider("Highlights B", 0.0, 1.0,
    st.session_state.get("hi_b", DEFAULTS["hi_b"]), 0.01, key="hi_b")

tone_balance = st.sidebar.slider("Shadow ↔ Highlight Balance", -1.0, 1.0,
    st.session_state.get("tone_balance", DEFAULTS["tone_balance"]), 0.01, key="tone_balance")

st.sidebar.subheader("Bloom & Vignette")
bloom_radius = st.sidebar.slider("Bloom Radius", 0.0, 20.0,
    st.session_state.get("bloom_radius", DEFAULTS["bloom_radius"]), 0.5, key="bloom_radius")
bloom_intensity = st.sidebar.slider("Bloom Intensity", 0.0, 1.0,
    st.session_state.get("bloom_intensity", DEFAULTS["bloom_intensity"]), 0.01, key="bloom_intensity")
vignette_strength = st.sidebar.slider("Vignette Strength", 0.0, 0.8,
    st.session_state.get("vignette_strength", DEFAULTS["vignette_strength"]), 0.01, key="vignette_strength")


# =========================
# Sidebar — Auto Brightness
# =========================
st.sidebar.header("5) Auto Brightness")
auto_bright = st.sidebar.checkbox(
    "Enable Auto Brightness",
    value=st.session_state.get("auto_bright", DEFAULTS["auto_bright"]),
    key="auto_bright"
)

target_mean = st.sidebar.slider("Target Mean", 0.30, 0.70,
    st.session_state.get("target_mean", DEFAULTS["target_mean"]), 0.01, key="target_mean")
abc_strength = st.sidebar.slider("Remap Strength", 0.0, 1.0,
    st.session_state.get("abc_strength", DEFAULTS["abc_strength"]), 0.05, key="abc_strength")
abc_black = st.sidebar.slider("Black Point %", 0.0, 0.20,
    st.session_state.get("abc_black", DEFAULTS["abc_black"]), 0.01, key="abc_black")
abc_white = st.sidebar.slider("White Point %", 0.80, 1.00,
    st.session_state.get("abc_white", DEFAULTS["abc_white"]), 0.001, key="abc_white")
abc_max_gain = st.sidebar.slider("Max Gain", 1.0, 3.0,
    st.session_state.get("abc_max_gain", DEFAULTS["abc_max_gain"]), 0.05, key="abc_max_gain")


# =========================
# Sidebar — Palette
# =========================
st.sidebar.header("6) Custom Palette (RGB)")

if "custom_palette" not in st.session_state:
    st.session_state["custom_palette"] = {}

use_csv = st.sidebar.checkbox(
    "Use CSV palette only",
    value=st.session_state.get("use_csv_palette", False),
    key="use_csv_palette"
)

with st.sidebar.expander("Add Custom Color", False):
    col1,col2,col3,col4 = st.columns([1.7,1,1,1])
    name = col1.text_input("Emotion", key="add_emotion")
    r = col2.number_input("R", 0,255,200, key="add_r")
    g = col3.number_input("G", 0,255,160, key="add_g")
    b = col4.number_input("B", 0,255,140, key="add_b")

    if st.button("Add Color", key="btn_add_color"):
        if name:
            st.session_state["custom_palette"][name] = (int(r),int(g),int(b))

with st.sidebar.expander("Import / Export CSV", False):
    up = st.file_uploader("Import CSV", type=["csv"], key="up_csv")
    if up is not None:
        dfc = pd.read_csv(up)
        if {"emotion","r","g","b"}.issubset([c.lower() for c in dfc.columns]):
            pal={}
            for _,row in dfc.iterrows():
                emo=str(row["emotion"]).strip()
                pal[emo] = (int(row["r"]),int(row["g"]),int(row["b"]))
            st.session_state["custom_palette"] = pal
            st.success("Imported.")
        else:
            st.error("CSV must contain emotion,r,g,b")

    pal = dict(DEFAULT_RGB)
    pal.update(st.session_state["custom_palette"])

    df_pal = pd.DataFrame([{"emotion":k,"r":v[0],"g":v[1],"b":v[2]} for k,v in pal.items()])
    st.dataframe(df_pal, use_container_width=True, height=180)

    buf = BytesIO()
    df_pal.to_csv(buf, index=False)
    buf.seek(0)
    st.download_button("Download CSV", data=buf, file_name="palette.csv")


# =========================
# Sidebar — Reset
# =========================
st.sidebar.header("7) Output")
if st.sidebar.button("Reset All"):
    reset_all()

            st.error("CSV must contain emotion,r,g,b")

    pal = dict(DEFAULT_RGB)
    pal.update(st.session_state["custom_palette"])

    df_pal = pd.DataFrame([{"emotion":k,"r":v[0],"g":v[1],"b":v[2]} for k,v in pal.items()])
    st.dataframe(df_pal, use_container_width=True, height=180)

    buf = BytesIO()
    df_pal.to_csv(buf, index=False)
    buf.seek(0)
    st.download_button("Download CSV", data=buf, file_name="palette.csv")


# =========================
# Sidebar — Reset
# =========================
st.sidebar.header("7) Output")
if st.sidebar.button("Reset All"):
    reset_all()
