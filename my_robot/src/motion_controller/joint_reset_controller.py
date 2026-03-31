import numpy as np


class JointResetController:
    """Joint-space quintic return-to-home controller."""

    _QUINTIC_VEL_PEAK = 1.875
    _QUINTIC_ACC_PEAK = 5.773502691896258

    def __init__(
        self,
        ts: float,
        q_home: np.ndarray,
        qd_lim: np.ndarray,
        qdd_lim: np.ndarray | None = None,
        velocity_scale: float = 0.7,
        acceleration_scale: float = 0.6,
        min_duration: float = 0.8,
    ):
        self._ts = float(ts)
        self._q_home = np.asarray(q_home, dtype=float).copy()
        self._qd_lim = velocity_scale * np.asarray(qd_lim, dtype=float).copy()
        if qdd_lim is None:
            base_qdd_lim = np.maximum(2.0, 2.0 * np.asarray(qd_lim, dtype=float))
        else:
            base_qdd_lim = np.asarray(qdd_lim, dtype=float)
        self._qdd_lim = acceleration_scale * base_qdd_lim
        self._min_duration = float(min_duration)

        self._active = False
        self._t = 0.0
        self._tf = self._min_duration
        self._q0 = self._q_home.copy()
        self._q1 = self._q_home.copy()
        self._delta = np.zeros_like(self._q_home)

    def start(self, q_start: np.ndarray, q_goal: np.ndarray | None = None):
        self._q0 = np.asarray(q_start, dtype=float).copy()
        self._q1 = self._q_home.copy() if q_goal is None else np.asarray(q_goal, dtype=float).copy()
        self._delta = self._q1 - self._q0

        if np.max(np.abs(self._delta)) < 1e-6:
            self._active = False
            self._t = 0.0
            self._tf = self._ts
            return

        tf_vel = np.max(
            self._QUINTIC_VEL_PEAK * np.abs(self._delta) / np.maximum(self._qd_lim, 1e-6)
        )
        tf_acc = np.max(
            np.sqrt(
                self._QUINTIC_ACC_PEAK * np.abs(self._delta) / np.maximum(self._qdd_lim, 1e-6)
            )
        )

        self._tf = max(self._ts, self._min_duration, tf_vel, tf_acc)
        self._t = 0.0
        self._active = True

    def stop(self):
        self._active = False

    def sample(self):
        if not self._active:
            return self._q1.copy(), np.zeros_like(self._q1), True

        self._t = min(self._t + self._ts, self._tf)
        tau = self._t / self._tf

        s = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
        ds = (30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4) / self._tf

        q = self._q0 + s * self._delta
        qd = ds * self._delta
        done = self._t >= self._tf - 1e-9
        if done:
            q = self._q1.copy()
            qd = np.zeros_like(qd)
            self._active = False
        return q, qd, done

    @property
    def active(self) -> bool:
        return self._active