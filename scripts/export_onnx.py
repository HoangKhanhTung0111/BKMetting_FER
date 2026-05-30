import os
import urllib.request

# Định nghĩa các model FER+ với các phiên bản khác nhau
models_to_download = {
    "ferplus_float32": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
        "filename": "emotion-ferplus-8.onnx",
        "description": "Bản chuẩn Float32 (Opset 8) - Chạy CPU/GPU tốt"
    },
    "ferplus_int8": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-12-int8.onnx",
        "filename": "emotion-ferplus-12-int8.onnx",
        "description": "Bản nén Quantized INT8 (Opset 12) - Tối ưu cực mạnh cho NPU/Edge"
    }
}

save_dir = "models/"
os.makedirs(save_dir, exist_ok=True)

print("Đang kiểm tra và tải các mô hình FER+ từ ONNX Model Zoo...\n")

for name, info in models_to_download.items():
    filepath = os.path.join(save_dir, info["filename"])
    
    print(f"[{name}] - {info['description']}")
    if not os.path.exists(filepath):
        print(f" -> Đang tải {info['filename']}...")
        try:
            # Tải trực tiếp bằng thư viện chuẩn của Python
            urllib.request.urlretrieve(info["url"], filepath)
            print(f" -> Tải thành công và lưu tại: {filepath}\n")
        except Exception as e:
            print(f" -> Lỗi khi tải {name}: {e}\n")
    else:
        print(f" -> File '{info['filename']}' đã tồn tại, bỏ qua bước tải.\n")

print("Hoàn tất! Hãy kiểm tra thư mục 'models/'.")