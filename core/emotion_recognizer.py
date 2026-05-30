import cv2
import numpy as np
import onnxruntime as ort

class EmotionRecognizer:
    def __init__(self, model_path: str):
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.intra_op_num_threads = 4
        
        self.session = ort.InferenceSession(model_path, so)
        self.input_name = self.session.get_inputs()[0].name
        
        self.labels = ["Neutral", "Happy", "Surprise", "Sad", "Angry", "Disgust", "Fear", "Contempt"]

    def predict(self, face_img):
        """Hàm nhận vào ảnh khuôn mặt và trả về nhãn cảm xúc."""
        if face_img.size == 0:
            return "Unknown"
            
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_NEAREST)
        img = np.array(resized, dtype=np.float32)
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=0)
        
        outputs = self.session.run(None, {self.input_name: img})
        pred_idx = np.argmax(outputs[0][0])
        return self.labels[pred_idx]