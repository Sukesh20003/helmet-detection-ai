# 🪖 Safety Helmet Detection System

> **Intel AI4MFG Project** - AI-powered computer vision system for helmet compliance monitoring on industrial shop floors.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Latest-green.svg)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Project Overview

This project develops an AI-powered computer vision system that detects whether workers are wearing helmets in real-time. Using YOLOv8 and a custom-trained model, the system provides instant helmet compliance monitoring with bounding boxes and confidence scores.

### ✨ Key Features

- 🔍 **Real-time Helmet Detection** - Detect helmets and heads in images
- 🌐 **Web Interface** - User-friendly Streamlit app
- ⚡ **Fast Processing** - ~150ms per image on CPU
- 🏭 **Industrial Ready** - Can be integrated with CCTV systems
- 📊 **Confidence Scoring** - Detailed detection results
- 🔧 **Easy Deployment** - Local and cloud deployment options

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/arbazansari/helmet-detection-ai.git
   cd helmet-detection-ai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run src/app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8501`

## 📁 Project Structure

```
helmet-detection-ai/
├── src/                    # Source code
│   ├── app.py             # Streamlit web application
│   └── utils/             # Utility scripts
│       ├── convert_xml_to_yolo.py
│       └── split.py
├── config/                # Configuration files
│   └── data.yaml          # YOLO dataset configuration
├── docs/                  # Documentation
│   ├── README.md
│   ├── QUICKSTART.md
│   └── TEAM.md
├── requirements.txt       # Python dependencies
├── setup.py              # Installation script
├── LICENSE               # MIT License
└── .gitignore           # Git ignore file
```

## 🎯 Usage

### Web Application

1. **Start the app**: `streamlit run src/app.py`
2. **Upload an image** (JPG, JPEG, or PNG)
3. **View results** with detection bounding boxes and confidence scores

### Command Line

```bash
# Test the model on an image
yolo detect predict model=runs/detect/train4/weights/best.pt source=path/to/image.jpg
```

## 🛠️ Technical Details

### Model Information
- **Architecture**: YOLOv8n (nano variant)
- **Classes**: 2 (helmet, head)
- **Input Size**: 640x640 pixels
- **Model Size**: 6.0MB
- **Parameters**: 3,006,038

### Performance Metrics
- **Processing Speed**: ~150ms per image on CPU
- **Detection Accuracy**: Successfully detects helmets and heads
- **Confidence Threshold**: 0.5

### Technologies Used
- **Python 3.11.9** - Primary language
- **YOLOv8** - Object detection framework
- **Streamlit** - Web application framework
- **OpenCV** - Computer vision processing
- **PIL** - Image processing

## 📊 Training Results

The model was trained on a custom dataset with:
- **Dataset Size**: 1001 images
- **Training Epochs**: 5 (optimized for CPU)
- **Classes**: helmet (0), head (1)
- **Format**: YOLO format (converted from Pascal VOC)

### Sample Detection Results
```
- "5 helmets" detected in 93.3ms
- "3 helmets, 4 heads" detected in 103.2ms
- "2 helmets" detected in 169.1ms
```

## 🏭 Industrial Applications

- **Construction Sites** - Worker safety monitoring
- **Manufacturing Plants** - Hazardous area compliance
- **Mining Operations** - Remote safety oversight
- **Oil & Gas Facilities** - Refinery safety protocols

## 🔧 Development

### Training Your Own Model

1. **Prepare dataset** in YOLO format
2. **Update config/data.yaml** with your paths
3. **Train the model**:
   ```bash
   yolo detect train data=config/data.yaml model=yolov8n.pt epochs=50
   ```

### Customizing Detection

Modify confidence threshold in `src/app.py`:
```python
results = model.predict(source=img_array, conf=0.5, save=False)
```

## 📈 Future Enhancements

- 🎥 **Real-time CCTV Integration**
- 📱 **Mobile Application**
- ☁️ **Cloud Deployment**
- 📊 **Violation Logging System**
- 🔔 **Alert Notifications**

## 👥 Team

This project was developed by five students during the Intel AI4MFG Internship Program:

- **Arbaz Ansari Khwaza A** - Project Lead & AI Developer
- **MAHATO Ajaykumar Ramdev** - Machine Learning Engineer
- **MISHRA SHIVAM SANTOSH** - Computer Vision Developer
- **Rain Mohammad Atik** - Frontend Developer
- **SINGH SUKESH SUNIL** - Backend Developer

*For detailed team information, see [docs/TEAM.md](docs/TEAM.md)*

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Intel AI4MFG Program
- Ultralytics YOLOv8 Team
- Streamlit Development Team

---

**Project Status**: ✅ **COMPLETED SUCCESSFULLY**  
**Completion Date**: July 5, 2025  
**Program**: Intel AI4MFG 