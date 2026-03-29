import numpy as np
from scipy.spatial.transform import Rotation
import modern_robotics as mr

from ..robot import Robot


def get_mapping_from_local_angular_velocity_to_rpy_derivative(rpy_angles: np.ndarray):
    sx = np.sin(rpy_angles[0])
    cx = np.cos(rpy_angles[0])
    cy = np.cos(rpy_angles[1])
    ty = np.tan(rpy_angles[1])

    M = np.array([
        [1.0, ty * sx, ty * cx],
        [0.0, cx, -cx],
        [0.0, sx / cy, cx / cy]
    ])
    return M


def get_mapping_from_rpy_derivative_to_local_angular_velocity(rpy_angles: np.ndarray):
    sx = np.sin(rpy_angles[0])
    cx = np.cos(rpy_angles[0])
    sy = np.sin(rpy_angles[1])
    cy = np.cos(rpy_angles[1])

    M = np.array([
        [1.0, 0.0, -sy],
        [0.0, cx, sx * cy],
        [0.0, -sx, cx * cy]
    ])
    return M


class MotMReacher:
    def __init__(self, ts: float, robot: Robot):
        super().__init__()

        self._ka = 0.5
        self._ts = ts

        self._robot: Robot = robot

        self._poses = np.zeros(6)
        self._vels = np.zeros(6)
        self._accs = np.zeros(6)

    def reset(self, T_bg: np.ndarray):
        self._poses[:3] = T_bg[:3, 3]
        self._poses[3:] = Rotation.from_matrix(T_bg[:3, :3]).as_euler('xyz')
        self._vels[:] *= 0
        self._accs[:] *= 0

    def ctrl(self, tf, T_bt, v_base_desired):
        tf = max(float(tf), 0.2)

        poses1 = np.zeros(6)
        poses1[:3] = T_bt[:3, 3]
        poses1[3:] = Rotation.from_matrix(T_bt[:3, :3]).as_euler('xyz')

        for i in range(3):
            if poses1[3 + i] - self._poses[3 + i] > np.pi:
                poses1[3 + i] -= 2 * np.pi
            elif poses1[3 + i] - self._poses[3 + i] < -np.pi:
                poses1[3 + i] += 2 * np.pi

        for i in range(6):
            p, v, a = self.plan(self._poses[i], self._vels[i], self._accs[i], poses1[i], tf)
            self._poses[i] = p
            self._vels[i] = v
            self._accs[i] = a

        V = np.zeros(6)
        V[:3] = Rotation.from_euler('xyz', self._poses[3:]).as_matrix().T @ self._vels[:3]
        V[3:] = get_mapping_from_rpy_derivative_to_local_angular_velocity(self._poses[3:]) @ self._vels[3:]

        dq = np.zeros(self._robot.dof)
        dq[0] = v_base_desired[0]   # vx
        dq[1] = v_base_desired[1]   # vy
        dq[5] = v_base_desired[2]   # wz
        Je = self._robot.jacobe(self._robot.q)
        V[:] += Je @ dq

        return V

    def plan(self, p0, v0, a0, p1, tf):
        A = np.zeros((6, 6))
        A[0, 0] = 1.0
        A[2, 1] = 1.0
        A[4, 2] = 2.0
        for i in range(6):
            A[1, i] = (tf ** i)
            A[3, i] = i * (tf ** (i - 1))
            A[5, i] = (i - 1) * i * (tf ** (i - 2))

        b = np.zeros(6)
        b[0] = p0
        b[1] = p1
        b[2] = v0
        b[4] = a0

        x = np.linalg.inv(A) @ b
        p = 0
        v = 0
        a = 0
        for i in range(6):
            p += x[i] * (self._ts ** i)
            v += x[i] * i * (self._ts ** (i - 1))
            a += x[i] * (i - 1) * i * (self._ts ** (i - 2))

        return p, v, a