import os
import cv2
import numpy as np
import onnxruntime as ort
from sklearn.metrics import classification_report
import glob

# Định nghĩa đường dẫn và file
dataset_path = "assets/fer2013_test/validation"
models = ["models/emotion-ferplus-8.onnx", "models/emotion-ferplus-12-int8.onnx"]

# Model FER+ output index: [Neutral, Happiness, Surprise, Sadness, Anger, Disgust, Fear, Contempt]
# Tên thư mục FER2013 thường là: ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
# Ta cần map tên thư mục của dataset sang index đầu ra của model. (Bỏ qua Contempt vì FER2013 không có)
label_map = {
    "neutral": 0,
    "happy": 1,
    "surprise": 2,
    "sad": 3,
    "angry": 4,
    "disgust": 5,
    "fear": 6
}

def load_dataset(dataset_dir):
    image_paths = []
    true_labels = []
    
    for folder_name in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
            
        # Chuẩn hóa tên thư mục về chữ thường để khớp với label_map
        label_name = folder_name.lower()
        if label_name not in label_map:
            continue
            
        true_index = label_map[label_name]
        
        # Tìm tất cả ảnh jpg/png trong thư mục đó
        for img_path in glob.glob(os.path.join(folder_path, "*.*")):
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(img_path)
                true_labels.append(true_index)
                
    return image_paths, true_labels

print(f"Đang quét thư mục dữ liệu: {dataset_path} ...")
image_paths, true_labels = load_dataset(dataset_path)

if len(image_paths) == 0:
    print("LỖI CỰC KỲ QUAN TRỌNG: Không tìm thấy ảnh nào! Hãy kiểm tra lại xem giải nén dataset có đúng đường dẫn không.")
    exit()
    
print(f"Đã tìm thấy {len(image_paths)} bức ảnh. Bắt đầu đánh giá...\n")

for model_path in models:
    if not os.path.exists(model_path):
        print(f"Không tìm thấy model: {model_path}")
        continue
        
    print(f"=====================================")
    print(f"Đánh giá Model: {os.path.basename(model_path)}")
    
    # Khởi tạo ONNX Runtime
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    
    predictions = []
    
    for img_path in image_paths:
        # Tiền xử lý theo chuẩn FER+: Grayscale, 64x64, float32, shape (1, 1, 64, 64)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))
        img = np.array(img, dtype=np.float32)
        img = np.expand_dims(img, axis=0) # shape (1, 64, 64)
        img = np.expand_dims(img, axis=0) # shape (1, 1, 64, 64)
        
        # Inference
        outputs = session.run(None, {input_name: img})
        # outputs[0] có shape (1, 8), dùng argmax lấy index lớn nhất
        pred_idx = np.argmax(outputs[0][0])
        predictions.append(pred_idx)
        
    target_names = [name for name, idx in sorted(label_map.items(), key=lambda item: item[1])]
    
    # Do model có 8 output (Contempt ở index 7), nhưng tập test không có ảnh Contempt
    # Ta bỏ qua label 7 khi in classification_report để tránh warning
    print(classification_report(true_labels, predictions, target_names=target_names, labels=[0,1,2,3,4,5,6]))