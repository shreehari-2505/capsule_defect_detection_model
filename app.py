import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load your .h5 model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('keras_model.h5', compile=False)
    return model

model = load_model()

# Define class names (make sure this matches your training order)
class_names = ['Good Capsule', 'Defective Capsule']

# Streamlit UI
st.title("🧪 Capsule Defect Classifier")
st.write("Upload an image of a capsule to check if it's good or defective.")

uploaded_file = st.file_uploader("Choose a capsule image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((224, 224))  # Resize to match training input
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    predictions = model.predict(img_array)[0]

    # Output prediction
    st.write("### Prediction Probabilities:")
    for i, prob in enumerate(predictions):
        st.write(f"🔹 {class_names[i]}: **{prob:.2%}**")

    predicted_class = class_names[np.argmax(predictions)]
    st.success(f"🧠 Model Prediction: **{predicted_class}**")

