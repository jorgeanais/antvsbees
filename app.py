"""
Clasificador de Hormigas vs Abejas (ConvNeXt Base + Transfer Learning)
App Streamlit para inferencia sobre una imagen subida por el usuario.
"""
import os
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as T
from PIL import Image, ImageOps

# ----------------------------- Configuración ------------------------------
MODEL_PATH = Path(os.getenv("MODEL_PATH", "model/modelo_hormigas_abejas.pth"))

# ImageFolder ordena las clases alfabéticamente: ants=0, bees=1
CLASS_NAMES = ["ants", "bees"]
CLASS_LABELS_ES = {"ants": "Hormiga 🐜", "bees": "Abeja 🐝"}

st.set_page_config(page_title="Hormigas vs Abejas", page_icon="🐝", layout="centered")

# Mismo preprocesamiento de validación/prueba usado en el notebook
PREPROCESS = T.Compose([
    T.Resize(232),
    T.CenterCrop(224),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ------------------------------- Modelo -----------------------------------
@st.cache_resource(show_spinner="Cargando modelo...")
def load_model(path: Path):
    """Reconstruye la arquitectura del notebook y carga los pesos guardados."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # weights=None: no descarga pesos de ImageNet, se cargan los propios.
    model = torchvision.models.convnext_base(weights=None)
    model.classifier[2] = nn.Linear(1024, len(CLASS_NAMES))

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, device


def predict(model, device, image: Image.Image):
    x = PREPROCESS(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probs = torch.softmax(model(x), dim=1).squeeze(0).cpu()
    return probs


# --------------------------------- UI -------------------------------------
st.title("🐜 Hormigas vs Abejas 🐝")
st.write(
    "Sube una imagen y el modelo (ConvNeXt Base con Transfer Learning) "
    "indicará la clase más probable y su probabilidad."
)

if not MODEL_PATH.exists():
    st.error(
        f"No se encontró el archivo del modelo en `{MODEL_PATH}`. "
        "Expórtalo desde el notebook y colócalo en la carpeta `model/`."
    )
    st.stop()

model, device = load_model(MODEL_PATH)

uploaded = st.file_uploader(
    "Selecciona una imagen", type=["jpg", "jpeg", "png", "webp", "bmp"]
)

if uploaded is not None:
    try:
        image = Image.open(uploaded)
        image = ImageOps.exif_transpose(image).convert("RGB")
    except Exception:
        st.error("No se pudo leer la imagen. Prueba con otro archivo.")
        st.stop()

    st.image(image, caption="Imagen cargada", width="stretch")

    with st.spinner("Clasificando..."):
        probs = predict(model, device, image)

    top_idx = int(probs.argmax())
    top_class = CLASS_NAMES[top_idx]
    top_prob = float(probs[top_idx])

    st.subheader("Resultado")
    col1, col2 = st.columns(2)
    col1.metric("Clase más probable", CLASS_LABELS_ES[top_class])
    col2.metric("Probabilidad", f"{top_prob * 100:.2f}%")

    st.write("**Probabilidad por clase**")
    for name, p in zip(CLASS_NAMES, probs.tolist()):
        st.write(CLASS_LABELS_ES[name])
        st.progress(p, text=f"{p * 100:.2f}%")

    st.caption(
        "Nota: el modelo solo conoce dos clases (hormiga y abeja). Si subes "
        "otro tipo de imagen, igualmente elegirá una de ellas."
    )
