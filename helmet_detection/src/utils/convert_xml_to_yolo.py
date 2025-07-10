"""
XML to YOLO Format Converter
============================

This script converts XML annotation files (Pascal VOC format) to YOLO format
for training the helmet detection model.

Developed by five students during the Intel AI4MFG Internship Program.

Team: Arbaz Ansari, Ajaykumar Mahato, Shivam Mishra, Rain Mohammad Atik, Sukesh Singh
Date: 2025
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path

def convert_xml_to_yolo(annotations_dir='annotations', labels_dir='labels'):
    """
    Convert XML annotation files to YOLO format.
    
    Args:
        annotations_dir (str): Directory containing XML annotation files
        labels_dir (str): Directory to save YOLO format labels
    
    Returns:
        bool: True if conversion successful, False otherwise
    """
    # Path setup
    annotations_path = Path(annotations_dir)
    labels_path = Path(labels_dir)
    
    # Create labels directory if it doesn't exist
    labels_path.mkdir(exist_ok=True)
    
    # Class name to ID mapping
    classes = {'helmet': 0, 'head': 1}
    
    # Check if annotations directory exists
    if not annotations_path.exists():
        print(f"❌ Error: Annotations directory '{annotations_dir}' not found!")
        return False

    def convert_annotation(xml_file):
        """
        Convert a single XML annotation file to YOLO format.
        
        Args:
            xml_file (str): Path to XML annotation file
            
        Returns:
            list: List of YOLO format annotation lines
        """
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            img_w = int(root.find('size/width').text)
            img_h = int(root.find('size/height').text)

            lines = []

            for obj in root.findall('object'):
                cls_name = obj.find('name').text
                if cls_name not in classes:
                    print(f"⚠️ Warning: Unknown class '{cls_name}' in {xml_file}")
                    continue

                cls_id = classes[cls_name]

                xmlbox = obj.find('bndbox')
                xmin = int(xmlbox.find('xmin').text)
                ymin = int(xmlbox.find('ymin').text)
                xmax = int(xmlbox.find('xmax').text)
                ymax = int(xmlbox.find('ymax').text)

                # Convert to YOLO format (normalized coordinates)
                x_center = (xmin + xmax) / 2.0 / img_w
                y_center = (ymin + ymax) / 2.0 / img_h
                width = (xmax - xmin) / img_w
                height = (ymax - ymin) / img_h

                lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            return lines
        except Exception as e:
            print(f"❌ Error converting {xml_file}: {str(e)}")
            return []

    # Convert all annotations
    xml_files = list(annotations_path.glob('*.xml'))
    
    if not xml_files:
        print(f"❌ No XML files found in '{annotations_dir}' directory!")
        return False
    
    print(f"🔄 Converting {len(xml_files)} XML files to YOLO format...")
    
    converted_count = 0
    for xml_file in xml_files:
        label_lines = convert_annotation(str(xml_file))
        if label_lines:  # Only create file if conversion was successful
            label_file = labels_path / f"{xml_file.stem}.txt"
            
            with open(label_file, 'w') as f:
                for line in label_lines:
                    f.write(line + '\n')
            converted_count += 1
    
    print(f"✅ Successfully converted {converted_count}/{len(xml_files)} files to YOLO format in '{labels_dir}/' folder.")
    return True

if __name__ == "__main__":
    success = convert_xml_to_yolo()
    if success:
        print("🎉 Conversion completed successfully!")
    else:
        print("❌ Conversion failed!")
