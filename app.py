import streamlit as st
from PIL import Image
import torch
import os
import io
import numpy as np

from model import load_model, colorize

# ─────────────────────────────────────────────
#  Config page
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Convertisseur N&B Pro",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Inter:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d1b2a;
    color: #e0e6f0;
}
.stApp { background: linear-gradient(135deg, #0d1b2a 0%, #1a2a3a 100%); }

h1 {
    font-family: 'Orbitron', sans-serif;
    background: linear-gradient(90deg, #ff4da6, #a855f7, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.4rem !important;
    text-align: center;
    margin-bottom: 0.2rem;
}
.subtitle { text-align:center; color:#94a3b8; font-size:0.95rem; margin-bottom:2rem; }
.stButton>button {
    background: linear-gradient(90deg,#ff4da6,#a855f7);
    color: white; border: none; border-radius: 8px;
    padding: 0.6rem 2rem; font-size:1rem; font-weight:600;
    width:100%; transition: opacity 0.2s;
}
.stButton>button:hover { opacity:0.85; }
.section-title {
    color:#ff4da6; font-weight:600; font-size:0.85rem;
    text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.5rem;
}
.success-box {
    background:#14532d22; border:1px solid #22c55e44;
    border-radius:8px; padding:0.75rem 1rem; color:#4ade80; font-size:0.9rem; margin-bottom:1rem;
}
.warn-box {
    background:#7c2d1222; border:1px solid #f9731644;
    border-radius:8px; padding:0.75rem 1rem; color:#fb923c; font-size:0.9rem; margin-bottom:1rem;
}
.upload-box {
    border:2px dashed #334155; border-radius:12px; background:#111827;
    padding:3rem 2rem; text-align:center; color:#64748b;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  En-tête
# ─────────────────────────────────────────────
st.markdown("# Convertisseur N&B Pro")
st.markdown('<p class="subtitle">Transformez vos images noir et blanc en couleur avec Pix2Pix</p>',
            unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Config modèle (silencieux)
# ─────────────────────────────────────────────
import glob
candidates = (
    glob.glob("*.pt") + glob.glob("*.pth") +
    glob.glob("checkpoints/*.pt") + glob.glob("checkpoints/*.pth")
)
available = sorted(set(candidates))
model_choice = available[0] if available else None
device = "cuda" if torch.cuda.is_available() else "cpu"
proc_size = 256


# ─────────────────────────────────────────────
#  Chargement du modèle (mis en cache)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model(path, dev):
    return load_model(path, device=dev)

model = None
if model_choice:
    with st.spinner(f"Chargement de `{model_choice}`…"):
        try:
            model = get_model(model_choice, device)
        except Exception as e:
            st.markdown(f'<div class="warn-box">Erreur de chargement : {e}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Zone principale
# ─────────────────────────────────────────────
col_in, col_out = st.columns(2, gap="large")

with col_in:
    st.markdown('<p class="section-title">Image d\'entrée (Noir & Blanc)</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["jpg","jpeg","png","bmp","webp"],
                                label_visibility="collapsed")
    if uploaded:
        input_img = Image.open(uploaded).convert("L")
        if "result" not in st.session_state:
            st.image(input_img, caption="Image originale (N&B)", use_container_width=True)
    else:
        st.markdown(
            '<div class="upload-box">Déposez l\'image ici<br>'
            '<span style="color:#475569">— ou —</span><br>Cliquez pour télécharger</div>',
            unsafe_allow_html=True)

with col_out:
    st.markdown('<p class="section-title">Image de sortie (Colorisée)</p>', unsafe_allow_html=True)

    if uploaded and model is not None:
        if st.button("Coloriser l'image"):
            with st.spinner("Colorisation en cours…"):
                try:
                    result = colorize(input_img, model, device=device, size=proc_size)
                    st.session_state["result"] = result
                    st.success("Colorisation réussie !")
                except Exception as e:
                    st.error(f"Erreur : {e}")
        if "result" in st.session_state:
            st.image(st.session_state["result"], use_container_width=True)
            buf = io.BytesIO()
            st.session_state["result"].save(buf, format="PNG")
            buf.seek(0)
            st.download_button("Télécharger", data=buf,
                               file_name="colorisee.png", mime="image/png",
                               use_container_width=True)
    elif uploaded and model is None:
        st.warning("Aucun modèle chargé. Vérifiez le chemin dans la sidebar.")
    else:
        st.markdown(
            '<div class="upload-box" style="color:#475569">Le résultat colorisé apparaîtra ici</div>',
            unsafe_allow_html=True)