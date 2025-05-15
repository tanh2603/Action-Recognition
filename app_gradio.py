import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from models import ConvLSTM
from dataset import get_label_names

# Cấu hình
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
labels = get_label_names()
latent_dim = 512
image_dim = 112
max_frames = 40
checkpoint_path = "model_checkpoints/ConvLSTM_5.pth"

# Load mô hình
model = ConvLSTM(num_classes=len(labels), latent_dim=latent_dim)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model = model.to(device)
model.eval()

# Tiền xử lý ảnh
transform = transforms.Compose([
    transforms.Resize((image_dim, image_dim)),
    transforms.ToTensor()
])

def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = transform(frame)
        frames.append(frame)
        count += 1

    cap.release()

    # Padding nếu thiếu
    while len(frames) < max_frames:
        frames.append(torch.zeros_like(frames[0]))

    video_tensor = torch.stack(frames)  # [T, C, H, W]
    video_tensor = video_tensor.unsqueeze(0).to(device)  # [1, T, C, H, W]
    return video_tensor

def predict_action(video_file):
    video_tensor = preprocess_video(video_file)
    with torch.no_grad():
        output = model(video_tensor)
        predicted_index = torch.argmax(output, dim=1).item()
        predicted_label = labels[predicted_index]
    return f"Dự đoán hành động: {predicted_label}"

# Giao diện Gradio
interface = gr.Interface(
    fn=predict_action,
    inputs=gr.Video(label="Tải video lên"),
    outputs="text",
    title="Nhận diện hành động từ video (ConvLSTM)",
    description="Tải lên video ngắn (tối đa ~40 frame) để dự đoán hành động."
)

if __name__ == "__main__":
    interface.launch()
