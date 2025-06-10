import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Set the page layout
st.set_page_config(page_title="Hazelnut Defect Classifier", layout="centered")

st.title("🥜 Hazelnut Defect Classifier")
st.write("Upload an image of a hazelnut and the model will classify it as one of the following:")
st.markdown("**good, crack, cut, hole, print**")

# Load model
@st.cache_resource
def load_hazelnut_model():
    return load_model("keras_model.h5")

model = load_hazelnut_model()

# Define class labels
class_names = ['good', 'crack', 'cut', 'hole', 'print']

# Image upload
uploaded_file = st.file_uploader("Upload a hazelnut image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((224, 224))  # or whatever your model expects
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalize if your model expects it

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    st.markdown("### 🔍 Prediction")
    st.write(f"**Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")
