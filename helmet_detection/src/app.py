"""
Safety Helmet Detection Web Application
======================================

A Streamlit web application for detecting safety helmets in images using YOLOv8.
This app allows users to upload images and get real-time helmet detection results.

Developed by five students during the Intel AI4MFG Internship Program.

Team: Arbaz Ansari, Ajaykumar Mahato, Shivam Mishra, Rain Mohammad Atik, Sukesh Singh
Date: 2025
"""

import streamlit as st
import numpy as np
from PIL import Image
import os
import importlib
import subprocess

# Set page config
st.set_page_config(
    page_title="Safety Helmet Detection",
    page_icon="🪖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load YOLO model with dynamic install & import
@st.cache_resource
def load_model():
    try:
        try:
            YOLO = importlib.import_module("ultralytics").YOLO
            cv2 = importlib.import_module("cv2")
        except ModuleNotFoundError:
            with st.spinner("📦 Installing dependencies (YOLO, OpenCV, etc)..."):
                subprocess.run([
                    "pip", "install", "ultralytics==8.1.24",
                    "torch", "torchvision",
                    "opencv-python-headless", "pillow", "numpy"
                ], check=True)
            YOLO = importlib.import_module("ultralytics").YOLO
            cv2 = importlib.import_module("cv2")

        # Check multiple possible model paths
        model_paths = [
            "runs/detect/train4/weights/best.pt",
            "runs/detect/train3/weights/best.pt",
            "runs/detect/train2/weights/best.pt",
            "runs/detect/train/weights/best.pt",
            "yolov8n.pt"
        ]

        for path in model_paths:
            if os.path.exists(path):
                st.success(f"✅ Model loaded from: {path}")
                return YOLO(path), cv2

        st.error("❌ No model file found in known paths.")
        return None, None

    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None

# Load the model
model, cv2 = load_model()

st.title("🪖 Helmet Compliance Detection App")
st.write("Upload an image below to detect helmets.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)

        img_array = np.array(image)

        with st.spinner("🔍 Running detection..."):
            results = model.predict(source=img_array, conf=0.5, save=False)

        if results and len(results) > 0:
            result = results[0]
            annotated_img = result.plot()
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            st.image(annotated_img_rgb, caption='Detection Result', use_container_width=True)

            if result.boxes is not None:
                num_detections = len(result.boxes)
                st.success(f"✅ Detection complete! Found {num_detections} object(s)")

                st.subheader("Detection Details:")
                for i, box in enumerate(result.boxes):
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = result.names[cls]
                    st.write(f"Object {i+1}: {class_name} (Confidence: {conf:.2f})")
            else:
                st.warning("⚠️ No objects detected.")
        else:
            st.error("❌ Detection failed. Try another image.")

    except Exception as e:
        st.error(f"❌ Error during detection: {str(e)}")

elif uploaded_file is not None and model is None:
    st.error("❌ Model not loaded. Please check the .pt file location.")

with st.expander("ℹ️ About this app"):
    st.write("""
    This app uses a YOLOv8 model trained to detect helmets in images.

    **How to use:**
    1. Upload an image (JPG, JPEG, or PNG)
    2. The app will detect helmets in real-time
    3. See results with bounding boxes & confidence scores

    **Model Info:**
    - Confidence threshold: 0.5
    - Results show helmet/head detection with bounding boxes
    """)
