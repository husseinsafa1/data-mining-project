import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    layout="centered"
)

st.title("🩺 Diabetic Retinopathy Detection")
st.write("Upload a retinal fundus image to check for Diabetic Retinopathy.")

# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("C:/Users/User/Desktop/Rouba uni/3rd_year/sem5/Data_Mining/retinopathy/diabetic_retinopathy_cnn.keras")
    return model

model = load_model()

# -----------------------------
# Image preprocessing function
# -----------------------------
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0

    # Ensure shape (1, 224, 224, 3)
    if image.shape[-1] != 3:
        image = np.stack([image]*3, axis=-1)

    image = np.expand_dims(image, axis=0)
    return image

# -----------------------------
# File uploader
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a retinal image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    processed_image = preprocess_image(image)

    # Predict
    prediction = model.predict(processed_image)[0][0]

    st.subheader("🧠 Prediction Result")

    if prediction > 0.5:
        st.error(f"⚠️ Diabetic Retinopathy Detected\n\nConfidence: {prediction:.2%}")
    else:
        st.success(f"✅ No Diabetic Retinopathy Detected\n\nConfidence: {(1 - prediction):.2%}")

    st.caption(
        "⚠️ This tool is for educational and screening purposes only and does not replace medical diagnosis."
    )
