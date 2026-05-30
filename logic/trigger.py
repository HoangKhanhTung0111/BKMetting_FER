from __future__ import annotations

from collections import deque
from typing import Deque, Dict

import numpy as np


class SpeakerTrigger:
	"""Detect active speakers based on mouth motion proxy via Y-center variance."""

	def __init__(self, active_speaker_variance_thresh: float, maxlen: int = 10) -> None:
		"""Initialize the speaker trigger.

		Args:
			active_speaker_variance_thresh: Threshold for variance or std.
			maxlen: Maximum history length per face ID.
		"""
		self._thresh = float(active_speaker_variance_thresh)
		self._history: Dict[int, Deque[float]] = {}
		self._maxlen = int(maxlen)

	def update(self, face_id: int, y_center: float) -> None:
		"""Update history for a specific face ID.

		Args:
			face_id: Unique face identifier.
			y_center: Y-center coordinate of the face bounding box.
		"""
		if face_id not in self._history:
			self._history[face_id] = deque(maxlen=self._maxlen)
		self._history[face_id].append(float(y_center))

	def is_active_speaker(self, face_id: int) -> bool:
		"""Determine whether the face is actively speaking/moving.

		Args:
			face_id: Unique face identifier.

		Returns:
			True if variance or std exceeds the threshold.
		"""
		history = self._history.get(face_id)
		if not history or len(history) < 2:
			return False

		values = np.array(history, dtype=np.float32)
		variance = float(np.var(values))
		std = float(np.std(values))
		return variance > self._thresh or std > self._thresh

