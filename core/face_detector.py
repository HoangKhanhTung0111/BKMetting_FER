import cv2
import numpy as np
import onnxruntime as ort

class FaceDetector:
    def __init__(self, model_path: str):
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.intra_op_num_threads = 4 
        
        self.session = ort.InferenceSession(model_path, so)
        self.input_name = self.session.get_inputs()[0].name

    def _process_face(self, frame):
        orig_h, orig_w = frame.shape[:2]
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_NEAREST)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img, orig_w, orig_h

    def _get_face_boxes(self, outputs, orig_w, orig_h):
        output = np.squeeze(outputs[0]) 
        
        if output.shape[0] < output.shape[1]:
            output = output.T 
            
        boxes = output[:, :4]  
        scores = output[:, 4]  
        
        mask = scores > 0.5
        boxes = boxes[mask]
        scores = scores[mask]
        
        if len(boxes) == 0:
            return []
            
        cx, cy = boxes[:, 0], boxes[:, 1]
        w, h = boxes[:, 2], boxes[:, 3]
        
        x_factor = orig_w / 640.0
        y_factor = orig_h / 640.0
        
        x_min = (cx - w / 2) * x_factor
        y_min = (cy - h / 2) * y_factor
        w_scaled = w * x_factor
        h_scaled = h * y_factor
        
        boxes_xywh = np.column_stack((x_min, y_min, w_scaled, h_scaled))
        
        indices = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(), 0.5, 0.4)
        
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w_box, h_box = boxes_xywh[i]
                if w_box < 30 or h_box < 30:
                    continue
                results.append((int(x), int(y), int(x + w_box), int(y + h_box)))
                
        return results

    def detect(self, frame):
        """Hàm duy nhất được gọi từ bên ngoài, trả về danh sách tọa độ khuôn mặt."""
        img_tensor, orig_w, orig_h = self._process_face(frame)
        outputs = self.session.run(None, {self.input_name: img_tensor})
        return self._get_face_boxes(outputs, orig_w, orig_h)