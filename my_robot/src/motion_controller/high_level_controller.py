import numpy as np


class HighLevelController:
    def __init__(self):
        super().__init__()
        self._points = []
        self._current_id = 0

    def add_point(self, point: np.ndarray):
        self._points.append(point)

    def ctrl(self):
        if self._current_id < len(self._points) - 1:
            return self._points[self._current_id], self._points[self._current_id + 1]
        return self._points[-1], self._points[0]

    def update(self):
        self._current_id += 1
        if self._current_id > len(self._points) - 1:
            self._current_id = 0

    @property
    def current_id(self):
        return self._current_id