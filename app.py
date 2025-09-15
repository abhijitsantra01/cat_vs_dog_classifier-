import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
from huggingface_hub import hf_hub_download

# --- Hugging Face Hub Configuration ---
# Replace 'your-username' and 'your-model-repo' with your actual values.
# The 'cat_dog_classifier.h5' should be the exact filename you uploaded to the repo.
MODEL_REPO_ID = "abhijitsantra100504/cat_dog_classifier"
MODEL_FILENAME = "cat_dog_classifier.h5"

# --- Load the model from Hugging Face Hub ---
@st.cache_resource
def load_model():
    """
    Downloads the model file from Hugging Face Hub and loads it.
    This function will be cached, so the download only happens once.
    """
    st.info("Downloading model from Hugging Face Hub...")
    try:
        model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
        model = tf.keras.models.load_model(model_path, compile=False)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()  # Stop the app if the model can't be loaded

model = load_model()

# --- Streamlit App UI ---
st.title("Cat vs Dog Classifier")
st.write("Upload an image, and I will predict whether it's a cat or a dog.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
result = None

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = img.resize((150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    result = "Dog" if prediction[0][0] > 0.5 else "Cat"

st.subheader("Prediction:")
if result is not None:
    st.success(f"The model predicts: **{result}**")