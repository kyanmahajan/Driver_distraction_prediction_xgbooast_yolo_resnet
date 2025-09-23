import streamlit as st
import cv2
import os
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np
from torchvision import models, transforms
import tempfile
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from xgboost import XGBClassifier

from PIL import Image
import numpy as np
import cv2

def get_edge_for_visualization(pil_img):
    # Resize the shortest side to 256
    w, h = pil_img.size
    scale = 256 / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = pil_img.resize((new_w, new_h), Image.BILINEAR)

    # Center crop to 224×224
    left = (resized.width - 224) // 2
    top = (resized.height - 224) // 2
    cropped = resized.crop((left, top, left + 224, top + 224))

    # Convert to grayscale and apply Canny
    gray = cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = 255 - cv2.Canny(blurred, 30, 80)

    # Convert to 3 channels + normalize to [0, 1]
    edge_rgb = np.stack([edges]*3, axis=-1).astype(np.float32) / 255.0

    return edge_rgb


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
    model = resnet;
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # remove final FC
    resnet.eval()

    # Load XGBoost classifier
    clf = XGBClassifier()
    clf.load_model("xg_drive.json")

    return detector, resnet, clf, model

detector, resnet, xgb_model, model = load_models()

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
    0: "safe drive",         # Both hands on wheel, looking forward
    1: "Using phone",      # Right hand texting on phone
    2: "Talking on phone ",# Talking on phone with right hand
    3: "Trying to reach behind",
    4: "Talking to a passenger"
 
}


# Process video
if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()
    stframe2 = st.empty()
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
                classes = [ClassifierOutputTarget(pred)];
                target_layer = [model.layer4[-1]];
                cam = GradCAM(model= model, target_layers=target_layer)

                # Display label
                label_box.title(f"Driver is distacted : {label}")

                # Draw box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                
                heatmap = cam(input_tensor=input_tensor, targets=classes)
                edge_image = get_edge_for_visualization(Image.fromarray(frame_rgb));
                stframe2.image(show_cam_on_image(edge_image, heatmap[0], use_rgb=True));

        # Show frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_container_width=True)
       

        frame_id += 1
        if frame_id > 100:
            st.warning("Preview limited to 100 frames.")
            break

    cap.release()

