"""
Model Testing Script
===================

This script tests the trained YOLOv8 model for helmet detection.
It loads the model and runs inference on a sample image to verify functionality.

Developed by five students during the Intel AI4MFG Internship Program.

Team: Arbaz Ansari, Ajaykumar Mahato, Shivam Mishra, Rain Mohammad Atik, Sukesh Singh
Date: 2025
"""

import os
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

def find_model():
    """
    Find the trained model file in common locations.
    
    Returns:
        str or None: Path to the model file if found, None otherwise
    """
    model_paths = [
        "runs/detect/train/weights/best.pt",
        "runs/detect/train2/weights/best.pt",
        "runs/detect/train3/weights/best.pt",
        "runs/detect/train4/weights/best.pt",
        "runs/detect/train5/weights/best.pt",
        "archive/runs/detect/train/weights/best.pt",
        "archive/runs/detect/train2/weights/best.pt",
        "archive/runs/detect/train3/weights/best.pt",
        "archive/runs/detect/train4/weights/best.pt",
        "archive/runs/detect/train5/weights/best.pt"
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            print(f"✅ Found model at: {path}")
            return path
    
    print("❌ No model file found in expected locations!")
    return None

def test_model():
    """
    Test the YOLO model with a sample image.
    
    Returns:
        bool: True if test successful, False otherwise
    """
    print("🧪 Testing YOLO model...")
    
    # Find the model file
    model_path = find_model()
    
    if not model_path:
        print("❌ No model file found!")
        return False
    
    try:
        # Load the model
        print("📦 Loading model...")
        model = YOLO(model_path)
        print("✅ Model loaded successfully!")
        
        # Find a sample image for testing
        sample_image_paths = [
            "images/val/hard_hat_workers984.png",
            "images/val/",
            "images/train/",
            "images/"
        ]
        
        sample_image_path = None
        for path in sample_image_paths:
            if os.path.exists(path):
                if os.path.isfile(path):
                    sample_image_path = path
                    break
                elif os.path.isdir(path):
                    # Find first image file in directory
                    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
                    for ext in image_extensions:
                        files = list(Path(path).glob(f'*{ext}'))
                        if files:
                            sample_image_path = str(files[0])
                            break
                    if sample_image_path:
                        break
        
        if sample_image_path:
            print(f"🖼️ Testing with image: {sample_image_path}")
            
            # Run prediction
            results = model.predict(source=sample_image_path, conf=0.5, save=False)
            
            if results and len(results) > 0:
                result = results[0]
                print(f"✅ Prediction successful!")
                print(f"📊 Number of detections: {len(result.boxes) if result.boxes is not None else 0}")
                
                if result.boxes is not None and len(result.boxes) > 0:
                    print("🎯 Detection Results:")
                    for i, box in enumerate(result.boxes):
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        class_name = result.names[cls]
                        print(f"  Object {i+1}: {class_name} (Confidence: {conf:.2f})")
                else:
                    print("⚠️ No objects detected in the image")
                
                return True
            else:
                print("❌ No results from prediction")
                return False
        else:
            print("❌ No sample image found for testing!")
            print("💡 Place an image in the images/ directory to test the model")
            return False
            
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        return False

def main():
    """
    Main function to run the model test.
    """
    print("🪖 Safety Helmet Detection Model Test")
    print("=" * 40)
    
    success = test_model()
    
    if success:
        print("\n🎉 Model test completed successfully!")
        print("✅ The model is working correctly")
    else:
        print("\n❌ Model test failed!")
        print("💡 Please check:")
        print("   - Model file exists in expected locations")
        print("   - Sample images are available")
        print("   - All dependencies are installed")
    
    return success

if __name__ == "__main__":
    main() 