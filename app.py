import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Plant Leaf Disease Detector", layout="centered")

# Load class names
with open('classes.txt') as f:
    class_names = [line.strip() for line in f.readlines()]

@st.cache_resource(show_spinner=False)
def load_model():
    model_path = os.path.join('model', 'plant_village_model.h5')
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

st.title("Plant Leaf Disease Detection")
st.write("Upload a leaf image, and the app will predict its disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    image = np.array(image)
    if image.shape[2] == 4:
        image = image[..., :3]
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    processed_img = preprocess_image(image)
    preds = model.predict(processed_img)
    pred_class = class_names[np.argmax(preds)]

    st.markdown(f"### Prediction: **{pred_class}**")
