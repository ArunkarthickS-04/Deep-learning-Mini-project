import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
from PIL import Image, ImageOps

st.set_page_config(
    page_title="Handwritten Digit Recognition using ANN",
    page_icon="✍️",
    layout="wide"
)

# 1. Load Trained Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/digit_ann.keras")

model = load_model()

# 2. Image Preprocessing
def preprocess(img):
    img = img.convert("L")                       # grayscale
    if np.array(img).mean() > 127:               # white background -> invert
        img = ImageOps.invert(img)
    img = ImageOps.autocontrast(img)
    img = img.resize((28, 28))
    arr = np.array(img).astype("float32") / 255.0
    return arr.reshape(1, 28, 28, 1), img

st.title("✍️ Handwritten Digit Recognition using ANN")
st.markdown("Deep learning model trained with **image augmentation** on the MNIST dataset.")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("1. Upload Digit Image")
    uploaded = st.file_uploader("Upload an image (0-9)", type=["jpg", "jpeg", "png"])
    predict_btn = st.button("Predict Digit", type="primary", use_container_width=True)

with col_right:
    st.subheader("2. Prediction Results")
    if uploaded:
        raw_img = Image.open(uploaded)
        st.image(raw_img, caption="Uploaded Image", width=200)
    else:
        raw_img = None
        st.info("Upload a handwritten digit image to begin.")

    if predict_btn:
        if raw_img is None:
            st.warning("Please upload an image first.")
        else:
            with st.spinner("Recognizing digit..."):
                x, processed = preprocess(raw_img)
                probs = model.predict(x, verbose=0)[0]
                digit = int(np.argmax(probs))
                conf = float(probs[digit]) * 100

            st.write(f"### 🔢 Predicted Digit: **{digit}**")
            m1, m2 = st.columns(2)
            m1.metric("Predicted Class", str(digit))
            m2.metric("Confidence", f"{conf:.2f}%")
            st.image(processed.resize((112, 112)), caption="Model Input (28x28)")

            st.write("---")
            st.write("#### Class-wise Probability")
            st.bar_chart(pd.DataFrame({"Probability": probs}, index=range(10)))
