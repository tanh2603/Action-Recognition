import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from models import ConvLSTM
from dataset import get_label_names
from PIL import Image

# Thiết lập thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tham số
class Options:
    video_path = 'data/UCF-101/BabyCrawling/v_BabyCrawling_g01_c01.avi'
    dataset_path = 'data/UCF-101-frames'
    image_dim = 112
    channels = 3
    latent_dim = 512
    checkpoint_model = 'model_checkpoints/ConvLSTM_5.pth'

opt = Options()

# Load label
labels = get_label_names()

# Load mô hình
model = ConvLSTM(num_classes=len(labels), latent_dim=opt.latent_dim)
model.load_state_dict(torch.load(opt.checkpoint_model, map_location=device))
model = model.to(device)
model.eval()

# Tiền xử lý khung hình
transform = transforms.Compose([
    transforms.Resize((opt.image_dim, opt.image_dim)),
    transforms.ToTensor()
])

def preprocess_video(video_path, max_frames=40):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = 0

    while True:
        ret, frame = cap.read()
        if not ret or total >= max_frames:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = transform(frame)
        frames.append(frame)
        total += 1

    cap.release()
    # Padding nếu số lượng khung hình < max_frames
    while len(frames) < max_frames:
        frames.append(torch.zeros_like(frames[0]))

    video_tensor = torch.stack(frames)  # [T, C, H, W]
    video_tensor = video_tensor.unsqueeze(0).to(device)  # [1, T, C, H, W]
    return video_tensor

def predict_on_video(video_path):
    input_tensor = preprocess_video(video_path)  # [1, T, C, H, W]
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()
        return labels[prediction]

if __name__ == "__main__":
    result = predict_on_video(opt.video_path)
    print(f"Dự đoán hành động: {result}")
