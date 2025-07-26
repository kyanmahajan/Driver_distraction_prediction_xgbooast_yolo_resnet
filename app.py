import streamlit as st
import cv2
import os
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np
from torchvision import models, transforms
import tempfile

from xgboost import XGBClassifier

st.title("Driver Distraction Detection from Video (ResNet + XGBoost)")

# Upload video file
video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

# Load models once
@st.cache_resource
def load_models():
    # YOLO object detector
    detector = YOLO("modelll.pt")
    
    # ResNet feature extractor
    resnet = models.resnet50(pretrained=True)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # remove final FC
    resnet.eval()

    # Load XGBoost classifier
    clf = XGBClassifier()
    clf.load_model("xg_drive.json")

    return detector, resnet, clf

detector, resnet, xgb_model = load_models()

# Image transformation (same as used during feature extraction training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# Label mapping
sol_map = {
    0: "safe_drive",         # Both hands on wheel, looking forward
    1: "texting",      # Right hand texting on phone
    2: "talking_phone",# Talking on phone with right hand
    3: "reaching_behind",
    4: "talking_to_passenger"
 
}


# Process video
if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()
    label_box = st.empty()
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = detector(frame)
        boxes = results[0].boxes

        if boxes is not None and boxes.xyxy is not None:
            for box in boxes.xyxy:
                x1, y1, x2, y2 = map(int, box.tolist())
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
                cropped = frame[y1:y2, x1:x2]

                # Convert cropped frame to PIL and apply transforms
                
                image = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

                # Extract features with ResNet
                with torch.no_grad():
                    features = resnet(input_tensor).squeeze().numpy()  # [2048]

                # Classify with XGBoost
                pred = xgb_model.predict([features])[0]
                label = sol_map.get(pred, "unknown")

                # Display label
                label_box.title(f"Detected label: {label}")

                # Draw box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Show frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_container_width=True)

        frame_id += 1
        if frame_id > 100:
            st.warning("Preview limited to 100 frames.")
            break

    cap.release()

