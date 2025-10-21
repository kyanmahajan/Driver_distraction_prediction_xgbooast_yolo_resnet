# 🚗 Driver Distraction Prediction (YOLO + ResNet + XGBoost)

This project detects **driver distraction** by combining **deep learning** and **machine learning** models.  
It integrates **YOLO** for object detection, **ResNet** for feature extraction, and **XGBoost** for classification — creating a hybrid pipeline that identifies when a driver is distracted based on camera images.

---

## 🧠 Overview

Distracted driving is one of the leading causes of road accidents worldwide.  
This project aims to automatically analyze in-car footage to detect whether a driver is distracted (e.g., using a phone, looking away, not focused) using visual cues extracted from images.

The pipeline fuses **object detection** + **feature-based classification** for robust and interpretable predictions.

---

## ⚙️ Architecture

1. **Input**: Driver image or video frame from dashboard camera.  
2. **YOLO (Object Detection)**:  
   - Detects key objects like **hands**, **phone**, **face**, **steering wheel**, etc.  
   - Outputs bounding boxes and object categories.  
3. **ResNet (Feature Extraction)**:  
   - Extracts deep embeddings from the cropped image regions or full frames.  
4. **Feature Fusion**:  
   - Combines YOLO outputs + ResNet embeddings + spatial/behavioral cues.  
5. **XGBoost (Classification)**:  
   - Uses the fused feature set to classify whether the driver is **distracted** or **attentive**.  
6. **Output**:  
   - Final class label (`distracted` / `not distracted`) and confidence score.  
   - Optionally visualized with bounding boxes and heatmaps.

---

## 🧩 Tech Stack

- **YOLOv8 / YOLOv5** → Object detection  
- **ResNet50** → Deep visual feature extraction  
- **XGBoost** → Gradient boosting for classification  
- **Python** → Core implementation  
- **PyTorch, OpenCV, NumPy, Pandas, Scikit-learn**

---

---

## 🚀 Usage

### 1️⃣ Training
1. Annotate driver images with bounding boxes for relevant objects.  
2. Train **YOLO** for object detection.  
3. Extract **ResNet** features for detected regions or frames.  
4. Combine YOLO + ResNet outputs into a feature dataset.  
5. Train **XGBoost** using these features and corresponding distraction labels.

### 2️⃣ Inference
Run the trained pipeline on new images or video frames after cloning the repo
```bash
pip install -r requirements.txt
```bash
streamlit run app.py



