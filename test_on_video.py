from models import *
from dataset import *
from data.extract_frames import extract_frames
import argparse
import os
import glob
import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image, ImageDraw
import skvideo.io
import torch
from torch.autograd import Variable
import numpy as np
import collections

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path", type=str, default="data/UCF-101/BabyCrawling/v_BabyCrawling_g01_c01.avi", help="Path to video"
    )
    parser.add_argument("--dataset_path", type=str, default="data/UCF-101-frames", help="Path to UCF-101 dataset")
    parser.add_argument("--image_dim", type=int, default=112, help="Height / width dimension")
    parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
    parser.add_argument("--latent_dim", type=int, default=512, help="Dimensionality of the latent representation")
    parser.add_argument("--checkpoint_model", type=str, default="", help="Optional path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    assert opt.checkpoint_model, "Specify path to checkpoint model using arg. '--checkpoint_model'"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_shape = (opt.channels, opt.image_dim, opt.image_dim)

    transform = transforms.Compose(
        [
            transforms.Resize(input_shape[-2:], Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Lấy danh sách nhãn từ thư mục dataset_path (tên các thư mục con)
    labels = sorted(list(set(os.listdir(opt.dataset_path))))

    # Khởi tạo model và load checkpoint
    model = ConvLSTM(num_classes=101, latent_dim=opt.latent_dim)
    model.to(device)
    model.load_state_dict(torch.load(opt.checkpoint_model, map_location=torch.device('cpu')))
    model.eval()

    # Lấy nhãn thật của video từ thư mục chứa video
    true_label = os.path.basename(os.path.dirname(opt.video_path))
    print(f"True label (ground truth): {true_label}")

    predicted_labels = []
    output_frames = []

    for frame in tqdm.tqdm(extract_frames(opt.video_path), desc="Processing frames"):
        image_tensor = Variable(transform(frame)).to(device)
        image_tensor = image_tensor.view(1, 1, *image_tensor.shape)  # shape (batch=1, seq_len=1, c, h, w)

        with torch.no_grad():
            prediction = model(image_tensor)
            predicted_label = labels[prediction.argmax(1).item()]
            predicted_labels.append(predicted_label)

        # Vẽ nhãn dự đoán lên frame
        d = ImageDraw.Draw(frame)
        d.text(xy=(10, 10), text=f"Predicted: {predicted_label}", fill=(255, 255, 255))
        output_frames.append(frame)

    # Tính nhãn dự đoán video dựa trên nhãn frame xuất hiện nhiều nhất
    most_common_label = collections.Counter(predicted_labels).most_common(1)[0][0]

    # Tính accuracy video-level (1 nếu đúng, 0 nếu sai)
    accuracy = 1.0 if most_common_label == true_label else 0.0

    print(f"Predicted label (video-level): {most_common_label}")
    print(f"Accuracy: {accuracy:.4f}")

    # Ghi accuracy lên tất cả các frame
    for frame in output_frames:
        d = ImageDraw.Draw(frame)
        d.text(xy=(10, 30), text=f"Accuracy: {accuracy * 100:.0f}%", fill=(255, 255, 255))

    # Tạo GIF đầu ra bằng Pillow
    output_frames[0].save(
        "output.gif",
        save_all=True,
        append_images=output_frames[1:], 
        duration=40,      # thời gian hiển thị mỗi frame (ms)
        loop=0            # lặp vô hạn
    )
