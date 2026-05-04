import streamlit as st
import numpy as np
import json
from PIL import Image
import tensorflow as tf
import os
import gdown

st.set_page_config(
    page_title="Deteksi Penyakit Lidah Karya EMT 12",
    page_icon="👅",
    layout="centered"
)

@st.cache_resource
def load_model():
    if not os.path.exists("best_model_final.keras"):
        gdown.download(
            "https://drive.google.com/uc?id=1qfGdMOITkEo2hYPWuETQM7jRdYbL2cJh",
            "best_model_final.keras",
            quiet=False
        )
    return tf.keras.models.load_model("best_model_final.keras")

@st.cache_data
def load_labels():
    with open("class_indices.json") as f:
        idx = json.load(f)
    return {v: k for k, v in idx.items()}

LABEL_MAP = {
    "crenated_tongue": "Crenated Tongue (Lidah Bergerigi)",
    "fissured_tongue": "Fissured Tongue (Lidah Retak)",
    "normal_tongue"  : "Normal Tongue (Lidah Normal)",
}
IMG_SIZE = (224, 224)

model   = load_model()
idx2cls = load_labels()

st.title("Deteksi Penyakit Lidah")
st.write("Gunakan kamera atau upload foto lidah untuk dianalisis.")

tab_kamera, tab_upload = st.tabs(["📷 Kamera", "📁 Upload File"])

image = None

with tab_kamera:
    foto = st.camera_input("Arahkan kamera ke lidah, lalu tekan tombol")
    if foto:
        image = Image.open(foto).convert("RGB")

with tab_upload:
    uploaded = st.file_uploader("Pilih gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")

if image:
    st.image(image, caption="Gambar yang digunakan", use_container_width=True)

    img_resized = image.resize(IMG_SIZE)
    img_array   = np.array(img_resized) / 255.0
    img_batch   = np.expand_dims(img_array, axis=0)

    with st.spinner("Menganalisis..."):
        probs      = model.predict(img_batch, verbose=0)[0]
        pred_idx   = int(np.argmax(probs))
        pred_cls   = idx2cls[pred_idx]
        confidence = float(probs[pred_idx])

    st.success(f"**Hasil:** {LABEL_MAP[pred_cls]}")
    st.metric("Tingkat Kepercayaan", f"{confidence:.1%}")

    st.subheader("Distribusi Probabilitas")
    prob_dict = {
        LABEL_MAP[idx2cls[i]]: float(probs[i])
        for i in range(len(probs))
    }
    st.bar_chart(prob_dict)

    if confidence < 0.7:
        st.warning("Confidence rendah — hasil mungkin kurang akurat.")