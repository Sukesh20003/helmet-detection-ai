# 🚀 Quick Start Guide

Get up and running with Safety Helmet Detection in minutes!

> **Note:** This project was developed by five students during the Intel AI4MFG Internship Program, demonstrating practical application of computer vision and machine learning concepts.

## Prerequisites

- Python 3.8 or higher
- Git
- A modern web browser

## Step 1: Clone the Repository

```bash
git clone https://github.com/arbazansari/helmet-detection-ai.git
cd helmet-detection-ai
```

## Step 2: Set Up Environment

### Option A: Using Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv helmet_env

# Activate virtual environment
# Windows:
helmet_env\Scripts\activate
# Linux/Mac:
source helmet_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Option B: Using Conda

```bash
# Create conda environment
conda create -n helmet-detection python=3.9
conda activate helmet-detection

# Install dependencies
pip install -r requirements.txt
```

## Step 3: Run the Application

```bash
streamlit run src/app.py
```

Open your browser and go to `http://localhost:8501`

## Step 4: Test the Model

```bash
python src/test_model.py
```

## 🎯 What's Next?

- **Train your own model**: Follow the training guide in the main README
- **Customize detection**: Modify `config.yaml` for different settings
- **Add your dataset**: Use the provided scripts to prepare your data

## 🆘 Troubleshooting

### Common Issues

1. **Model not found**: Place your trained model in one of the expected paths
2. **Dependencies error**: Make sure you're using Python 3.8+ and have activated your virtual environment
3. **CUDA errors**: The app will automatically use CPU if GPU is not available

### Getting Help

- Check the [main README](README.md) for detailed documentation
- Open an issue on GitHub for bugs or feature requests
- Review the configuration options in `config.yaml`

## 👥 About the Team

This project was developed by five students during the Intel AI4MFG Internship Program:

- **Arbaz Ansari Khwaza A** - Project Lead & AI Developer
- **MAHATO Ajaykumar Ramdev** - Machine Learning Engineer  
- **MISHRA SHIVAM SANTOSH** - Computer Vision Developer
- **Rain Mohammad Atik** - Frontend Developer
- **SINGH SUKESH SUNIL** - Backend Developer

Showcasing their skills in:
- Computer Vision and Deep Learning
- Web Application Development
- Data Processing and Model Training
- Software Engineering Best Practices

---

**Happy detecting! 🪖** 