"""
YOLOv8 Training Script for Safety Helmet Detection
==================================================

This script trains a YOLOv8 model for detecting safety helmets in images.
The model is trained on a custom dataset with helmet and head classes.

Developed by five students during the Intel AI4MFG Internship Program.

Team: Arbaz Ansari, Ajaykumar Mahato, Shivam Mishra, Rain Mohammad Atik, Sukesh Singh
Date: 2025
"""

from ultralytics import YOLO
import os

def train_model():
    """
    Train a YOLOv8 model for helmet detection.
    
    Returns:
        bool: True if training completed successfully, False otherwise
    """
    try:
        print("🚀 Starting YOLOv8 training for helmet detection...")
        
        # Check if data.yaml exists
        if not os.path.exists("config/data.yaml"):
            print("❌ Error: config/data.yaml file not found!")
            print("Please ensure your dataset is properly configured.")
            return False
        
        # Load model (you can change yolov8n.pt to yolov8s.pt, yolov8m.pt, etc.)
        print("📦 Loading YOLOv8 model...")
        model = YOLO("yolov8n.pt")
        
        # Train the model
        print("🎯 Training model...")
        results = model.train(
            data="config/data.yaml",
            epochs=100,  # Increase epochs for better performance
            imgsz=640,
            batch=16,
            patience=20,  # Early stopping patience
            save=True,
            device='auto'  # Use GPU if available
        )
        
        print("✅ Training completed successfully!")
        print(f"📊 Best model saved at: {results.save_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        return False

if __name__ == "__main__":
    train_model()

