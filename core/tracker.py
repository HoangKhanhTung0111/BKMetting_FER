import numpy as np
from collections import OrderedDict

class SimpleTracker:
    def __init__(self, max_disappeared=15, distance_threshold=50):
        self.next_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.distance_threshold = distance_threshold

    def register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1
        return self.next_id - 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, boxes):
        if len(boxes) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return []

        # Tính tâm (Centroid) của các bounding box hiện tại
        input_centroids = np.zeros((len(boxes), 2), dtype="int")
        for i, (startX, startY, endX, endY) in enumerate(boxes):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            tracked_boxes = []
            for i in range(len(input_centroids)):
                obj_id = self.register(input_centroids[i])
                tracked_boxes.append((obj_id, boxes[i]))
            return tracked_boxes

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        # Tính ma trận khoảng cách giữa tâm cũ và tâm mới
        D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()
        tracked_boxes = []

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.distance_threshold:
                continue

            object_id = object_ids[row]
            self.objects[object_id] = input_centroids[col]
            self.disappeared[object_id] = 0
            tracked_boxes.append((object_id, boxes[col]))
            used_rows.add(row)
            used_cols.add(col)

        # Xóa các ID bị mất tích
        unused_rows = set(range(0, D.shape[0])).difference(used_rows)
        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        # Đăng ký ID mới cho người mới xuất hiện
        unused_cols = set(range(0, D.shape[1])).difference(used_cols)
        for col in unused_cols:
            obj_id = self.register(input_centroids[col])
            tracked_boxes.append((obj_id, boxes[col]))

        return tracked_boxes