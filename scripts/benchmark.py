import onnxruntime as ort
import numpy as np
import time
import os

models = [
    {"path": "models/emotion-ferplus-8.onnx", "input_shape": (1, 1, 64, 64), "dtype": np.float32},
    {"path": "models/emotion-ferplus-12-int8.onnx", "input_shape": (1, 1, 64, 64), "dtype": np.float32},
    {"path": "models/yolov8n-face.onnx", "input_shape": (1, 3, 640, 640), "dtype": np.float32} # YOLO mặc định
]

print("BẮT ĐẦU BENCHMARK TỐC ĐỘ (CPU)\n" + "-"*40)

for m in models:
    if not os.path.exists(m["path"]):
        continue
        
    # Tắt cảnh báo của ONNX
    options = ort.SessionOptions()
    options.log_severity_level = 3 
    session = ort.InferenceSession(m["path"], options)
    input_name = session.get_inputs()[0].name
    
    # Tạo dữ liệu giả (Dummy data) để stress-test
    dummy_input = np.random.randn(*m["input_shape"]).astype(m["dtype"])
    
    # Warm-up (Chạy mồi 10 vòng để CPU nóng lên, không tính thời gian)
    for _ in range(10):
        session.run(None, {input_name: dummy_input})
        
    # Đo thời gian thật (100 vòng)
    iterations = 100
    start_time = time.perf_counter()
    for _ in range(iterations):
        session.run(None, {input_name: dummy_input})
    end_time = time.perf_counter()
    
    avg_time_ms = ((end_time - start_time) / iterations) * 1000
    print(f"Model: {os.path.basename(m['path'])}")
    print(f"Thời gian suy luận trung bình: {avg_time_ms:.2f} ms / frame\n")