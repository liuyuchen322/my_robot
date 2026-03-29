class GripperController:
    def __init__(self, ts):
        super().__init__()

        self._ts = ts
        # 根据URDF中左右手指关节的速度限制 (velocity="0.053")
        self._velocity = 0.053  # m/s
        # 根据URDF中左右手指关节的限位范围 (lower="0" upper="0.044")
        self._min_position = 0.0     # 夹爪完全闭合
        self._max_position = 0.044   # 夹爪完全打开
        self._current_position = self._max_position

    def get(self):
        """获取当前夹爪位置"""
        return self._current_position

    def open(self):
        """打开夹爪"""
        increment = self._velocity * self._ts
        self._current_position += increment
        arrive = False
        if self._current_position >= self._max_position:
            self._current_position = self._max_position
            arrive = True
        return arrive, self._current_position

    def close(self):
        """关闭夹爪"""
        increment = self._velocity * self._ts
        self._current_position -= increment
        arrive = False
        if self._current_position <= self._min_position:
            self._current_position = self._min_position
            arrive = True
        return arrive, self._current_position