import gradio as gr
from test_on_video import predict_on_video  # import hàm xử lý từ test_on_video.py

def process_video(video):
    # Lưu video tạm để xử lý
    temp_path = "temp_input_video.mp4"
    with open(temp_path, "wb") as f:
        f.write(video.read())

    # Gọi hàm xử lý video
    output_path = predict_on_video(temp_path)

    return output_path  # trả về video kết quả để hiển thị

gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Tải video hành động vào"),
    outputs=gr.Video(label="Kết quả dự đoán (hiển thị nhãn)"),
    title="Nhận diện hành động bằng ConvLSTM"
).launch()
