# app.py
import cv2
import time
import yaml
import json
import argparse
import sys

from core.face_detector import FaceDetector
from core.emotion_recognizer import EmotionRecognizer
from core.tracker import SimpleTracker
from core.video_stream import VideoStream
from logic.smoothing import TemporalSmoother

def load_config(cfg_path="configs/pipeline_cfg.yaml"):
    with open(cfg_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def main():
    parser = argparse.ArgumentParser(description="Production Edge FER Pipeline")
    parser.add_argument('--mode', type=str, choices=['gui', 'json'], default='gui', 
                        help='Chế độ xuất kết quả: gui hoặc json')
    # Thêm cờ --source
    parser.add_argument('--source', type=str, default=None, 
                        help='Ghi đè nguồn video (VD: 0 cho webcam laptop, hoặc đường_dẫn.mp4)')
    args = parser.parse_args()

    # 1. Tải cấu hình
    cfg = load_config()
    
    # Ghi đè nguồn video nếu người dùng truyền vào từ terminal
    if args.source is not None:
        # Nếu nhập số "0", chuyển thành int để OpenCV nhận diện là Camera
        cfg['io']['video_source'] = int(args.source) if args.source.isdigit() else args.source
    
    # 2. Khởi tạo Modules
    detector = FaceDetector(cfg['models']['face_detector'])
    recognizer = EmotionRecognizer(cfg['models']['emotion_recognizer'])
    tracker = SimpleTracker()
    smoother = TemporalSmoother(window_size=cfg['params']['smoothing_window'])
    stream = VideoStream(cfg['io']['video_source'])
    
    if args.mode == 'gui':
        cv2.namedWindow("Production Edge FER", cv2.WINDOW_NORMAL)

    frame_count = 0
    skip_frames = cfg['params'].get('skip_frames', 2)

    try:
        while True:
            start_time = time.time()
            ret, frame = stream.get_frame()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % skip_frames != 0:
                continue

            # Core Pipeline
            raw_boxes = detector.detect(frame)
            tracked_faces = tracker.update(raw_boxes)
            
            frame_results = [] # Lưu dữ liệu để xuất JSON

            for face_id, (x1, y1, x2, y2) in tracked_faces:
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
                
                face_crop = frame[y1:y2, x1:x2]
                raw_emotion = recognizer.predict(face_crop)
                stable_emotion = smoother.smooth(face_id, raw_emotion)
                
                # Gom dữ liệu đầu ra
                frame_results.append({
                    "id": face_id,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "emotion": stable_emotion
                })

                if args.mode == 'gui':
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"ID:{face_id} {stable_emotion}"
                    cv2.putText(frame, label, (x1, max(y1 - 10, 0)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Xử lý đầu ra (Interface)
            if args.mode == 'json':
                # Bắn JSON thẳng ra stdout cho Backend chụp lấy
                output_payload = {
                    "timestamp": time.time(),
                    "frame": frame_count,
                    "faces": frame_results
                }
                print(json.dumps(output_payload))
                sys.stdout.flush() # Ép tuôn luồng để tránh nghẽn buffer
            
            elif args.mode == 'gui':
                fps = 1.0 / (time.time() - start_time)
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Production Edge FER', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        stream.release()
        if args.mode == 'gui':
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()