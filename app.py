import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DermaScan AI",
    page_icon="🔬",
    layout="centered",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d0f14;
    color: #e8e8e8;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2.5rem 1.5rem 4rem; max-width: 720px; }

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #f0e6ff 0%, #a78bfa 50%, #7c3aed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.1;
}
.hero-sub {
    font-size: 0.95rem;
    color: #8b8fa8;
    margin-top: 0.5rem;
    font-weight: 300;
    letter-spacing: 0.02em;
}
.upload-hint {
    font-size: 0.78rem;
    color: #555870;
    text-align: center;
    margin-top: 0.4rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.result-card {
    background: linear-gradient(145deg, #16181f, #1c1e28);
    border: 1px solid #2a2d3a;
    border-radius: 16px;
    padding: 1.6rem 2rem;
    margin-top: 1.5rem;
    box-shadow: 0 8px 32px rgba(124,58,237,0.12);
}
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #c4b5fd;
    margin: 0;
    text-transform: capitalize;
}
.result-conf { font-size: 0.85rem; color: #6b7280; margin-top: 0.2rem; }
.conf-bar-bg { background: #1e2030; border-radius: 100px; height: 6px; margin-top: 1rem; overflow: hidden; }
.conf-bar-fill { height: 6px; border-radius: 100px; background: linear-gradient(90deg, #7c3aed, #a78bfa); }
.pred-row { display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0; border-bottom: 1px solid #1e2030; font-size: 0.88rem; }
.pred-row:last-child { border-bottom: none; }
.pred-name { color: #d1d5db; text-transform: capitalize; }
.pred-pct  { color: #a78bfa; font-weight: 700; }
.divider   { border: none; border-top: 1px solid #1e2030; margin: 1.5rem 0; }
.disclaimer { font-size: 0.72rem; color: #4b5063; text-align: center; margin-top: 2rem; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# ── Header renders IMMEDIATELY before any heavy import ───────────────────────
st.markdown('<p class="hero-title">DermaScan AI</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Upload a skin image for instant AI-powered classification</p>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    label_visibility="collapsed"
)
st.markdown('<p class="upload-hint">JPG · PNG · BMP · WEBP</p>', unsafe_allow_html=True)

# ── Model loaded lazily & cached across all sessions ─────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    from ultralytics import YOLO   # deferred — keeps cold start fast
    return YOLO("model/model.pt")

# ── Inference ─────────────────────────────────────────────────────────────────
if uploaded:
    from PIL import Image
    import numpy as np

    image = Image.open(uploaded).convert("RGB")
    st.image(image, use_container_width=True)

    with st.spinner("Loading model…"):
        model = load_model()       # instant on 2nd+ upload (cached)

    with st.spinner("Analyzing…"):
        results = model.predict(
            source=np.array(image),
            imgsz=160,
            device="cpu",
            verbose=False,
        )

    r = results[0]
    if r.probs is None:
        st.error("Prediction failed.")
        st.stop()

    probs     = r.probs.data.cpu().numpy()
    top1_idx  = int(r.probs.top1)
    top1_conf = float(probs[top1_idx])
    names     = model.names
    top5      = [(names[i], float(probs[i])) for i in np.argsort(probs)[::-1][:5]]

    bar_pct = int(top1_conf * 100)
    st.markdown(f"""
    <div class="result-card">
        <p class="result-label">{names[top1_idx].replace('_', ' ')}</p>
        <p class="result-conf">Top prediction · {top1_conf * 100:.1f}% confidence</p>
        <div class="conf-bar-bg">
            <div class="conf-bar-fill" style="width:{bar_pct}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Top Predictions")

    rows_html = "".join(f"""
        <div class="pred-row">
            <span class="pred-name">{name.replace('_', ' ')}</span>
            <span class="pred-pct">{conf * 100:.1f}%</span>
        </div>""" for name, conf in top5)

    st.markdown(f'<div style="margin-top:0.5rem">{rows_html}</div>', unsafe_allow_html=True)

    st.markdown("""
    <p class="disclaimer">
    ⚠️ Research purposes only.<br>
    This tool is not medical advice.<br>
    Consult a qualified dermatologist.
    </p>
    """, unsafe_allow_html=True)