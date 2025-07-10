"""
Dataset Splitting Utility
========================

This script splits the dataset into training and validation sets for YOLO training.
It creates the proper directory structure required by YOLOv8.

Developed by five students during the Intel AI4MFG Internship Program.

Team: Arbaz Ansari, Ajaykumar Mahato, Shivam Mishra, Rain Mohammad Atik, Sukesh Singh
Date: 2025
"""

import os
import random
import shutil
from pathlib import Path

def split_dataset(images_dir='images', labels_dir='labels', train_ratio=0.8):
    """
    Split dataset into training and validation sets.
    
    Args:
        images_dir (str): Directory containing image files
        labels_dir (str): Directory containing label files
        train_ratio (float): Ratio of data to use for training (0.0 to 1.0)
    
    Returns:
        bool: True if splitting successful, False otherwise
    """
    # Paths
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    
    # Check if directories exist
    if not images_path.exists():
        print(f"❌ Error: Images directory '{images_dir}' not found!")
        return False
    
    if not labels_path.exists():
        print(f"❌ Error: Labels directory '{labels_dir}' not found!")
        return False
    
    # Create split folders
    train_img_dir = images_path / 'train'
    val_img_dir = images_path / 'val'
    train_label_dir = labels_path / 'train'
    val_label_dir = labels_path / 'val'
    
    for d in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # List images (support multiple formats)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(images_path.glob(f'*{ext}')))
        image_files.extend(list(images_path.glob(f'*{ext.upper()}')))
    
    if not image_files:
        print(f"❌ No image files found in '{images_dir}' directory!")
        return False
    
    print(f"📊 Found {len(image_files)} images to split...")
    
    # Shuffle and split
    random.shuffle(image_files)
    split_index = int(len(image_files) * train_ratio)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]
    
    print(f"📈 Training set: {len(train_files)} images")
    print(f"📉 Validation set: {len(val_files)} images")
    
    # Move files
    moved_count = 0
    for img_file in train_files:
        label_file = labels_path / f"{img_file.stem}.txt"
        if label_file.exists():
            shutil.move(str(img_file), str(train_img_dir / img_file.name))
            shutil.move(str(label_file), str(train_label_dir / label_file.name))
            moved_count += 1
        else:
            print(f"⚠️ Warning: No label file found for {img_file.name}")
    
    for img_file in val_files:
        label_file = labels_path / f"{img_file.stem}.txt"
        if label_file.exists():
            shutil.move(str(img_file), str(val_img_dir / img_file.name))
            shutil.move(str(label_file), str(val_label_dir / label_file.name))
            moved_count += 1
        else:
            print(f"⚠️ Warning: No label file found for {img_file.name}")
    
    print(f"✅ Successfully moved {moved_count} image-label pairs")
    print("📁 Split complete — images/train, images/val, labels/train, labels/val created.")
    return True

if __name__ == "__main__":
    success = split_dataset()
    if success:
        print("🎉 Dataset splitting completed successfully!")
    else:
        print("❌ Dataset splitting failed!")
