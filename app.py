import os
os.system("pip uninstall -y opencv-python")
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.title("AI Image Classifier")

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner("Analyzing..."):
        results = model(image)
        res_plotted = results[0].plot()
        
    st.subheader("Results:")
    st.image(res_plotted, caption="Identified Objects", use_container_width=True)
