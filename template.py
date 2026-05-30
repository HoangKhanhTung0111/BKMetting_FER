import os
from pathlib import Path

# 1. Danh sách các file cần tạo
list_of_files = [
    ".github/copilot_instructions.md",
    "configs/pipeline_cfg.yaml",
    "core/__init__.py",
    "core/video_stream.py",
    "core/face_detector.py",            # Nhận vào tensor, trả về face bbox 
    "core/emotion_recognizer.py",       # Nhận vào face crop 64x64, trả về emotion label
    "core/tracker.py",                  # Gán ID cho từng người dựa trên vị trí khuôn mặt
    "logic/__init__.py",
    "logic/trigger.py",                 # Chỉ chạy pipeline khi phát hiện người đang nói
    "logic/smoothing.py",               # Giữ cho emotion label ổn định trong 30 frames
    "scripts/__init__.py",
    "scripts/benchmark.py",
    "scripts/export_onnx.py",
    "app.py",
    "Dockerfile",
    "requirements.txt"
]

# 2. Danh sách các thư mục chỉ cần tạo vỏ (không chứa file mẫu)
empty_dirs = [
    "assets",
    "models"
]

# Tạo các thư mục rỗng
for dir_path in empty_dirs:
    os.makedirs(dir_path, exist_ok=True)
    print(f"Checked/Created directory: {dir_path}")

# Tạo các file và thư mục cha của chúng
for filepath in list_of_files:
    filepath_obj = Path(filepath)
    filedir, filename = os.path.split(filepath_obj)
    
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        
    if not os.path.exists(filepath_obj) or os.path.getsize(filepath_obj) == 0:
        with open(filepath_obj, "w", encoding="utf-8") as f:
            pass # Tạo file rỗng
        print(f"Created file: {filepath}")
    else:
        print(f"File is already present at: {filepath}")