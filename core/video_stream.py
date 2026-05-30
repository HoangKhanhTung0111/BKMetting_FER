# core/video_stream.py
import cv2

class VideoStream:
    def __init__(self, source):
        # Hỗ trợ cả file mp4 (str) hoặc camera (int 0)
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"CRITICAL: Không thể mở luồng video tại {source}")

    def get_frame(self):
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        self.cap.release()