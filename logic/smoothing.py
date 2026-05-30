from collections import deque, Counter

class TemporalSmoother:
    def __init__(self, window_size=15):
        # Dictionary lưu lịch sử theo dạng: {ID_người: [Lịch_sử_cảm_xúc]}
        self.history = {}
        self.window_size = window_size

    def smooth(self, face_id, current_emotion):
        if current_emotion == "Unknown":
            return "Unknown"

        if face_id not in self.history:
            self.history[face_id] = deque(maxlen=self.window_size)

        # Thêm cảm xúc mới nhất vào hàng đợi
        self.history[face_id].append(current_emotion)

        # Lấy cảm xúc xuất hiện nhiều nhất (Majority Voting)
        most_common_emotion = Counter(self.history[face_id]).most_common(1)[0][0]
        
        return most_common_emotion