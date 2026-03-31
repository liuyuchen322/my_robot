import numpy as np
import modern_robotics as mr


class BaseController:
    def __init__(self):
        super().__init__()

        self._rR = 0.25
        # FR5 side-grasp needs the chassis to stand slightly closer than the
        # top-grasp reference, otherwise state2 never closes the last gap.
        self._rP = 0.28

        self._vF = 1.0

        self._k_alpha = 4
        self._k_beta = -1.5
        self._wz_max = 1.5

        self._rC = self._rR + self._rP

    def ctrl(self, T_target: np.ndarray, T_next: np.ndarray, T_base: np.ndarray, R_closest: np.ndarray):
        t_target = T_target[:2, 3]
        t_next = T_next[:2, 3]
        t_base = T_base[:2, 3]

        t_tb = t_base - t_target
        d_tb = np.linalg.norm(t_tb)

        t_radius = np.min([self._rC, d_tb])

        ct = t_radius / d_tb
        st = np.sqrt(1 - ct * ct)
        Rt = np.array([
            [ct, -st],
            [st, ct]
        ])

        u1 = Rt @ t_tb / d_tb
        u2 = Rt.T @ t_tb / d_tb

        t_tn = t_next - t_target

        if np.dot(t_tn, u1) > np.dot(t_tn, u2):
            u = u1
        else:
            u = u2
        t_closest = t_target + self._rC * u

        t_bc = t_closest - t_base
        d_bc = np.linalg.norm(t_bc)

        if d_bc < 1e-6:
            T_closest = mr.RpToTrans(R_closest, np.append(t_closest, 0))
            return True, np.zeros(3), 0.0, T_closest

        n_base = T_base[:2, 0]
        n_bc = t_bc / d_bc

        sin_alpha = np.cross(n_base, n_bc)
        cos_alpha = np.dot(n_base, n_bc)
        alpha = np.arcsin(sin_alpha)

        n_closest = R_closest[:2, 0]
        sin_beta = np.cross(n_bc, n_closest)
        cos_beta = np.dot(n_bc, n_closest)
        beta = np.arcsin(sin_beta)

        vB = self._vF
        wB = (self._k_alpha * alpha + self._k_beta * beta) * self._vF / d_bc

        vx = vB
        vy = 0.0
        wz = np.clip(wB, -self._wz_max, self._wz_max)

        succeed = False
        if t_radius < self._rC:
            succeed = True
        T_closest = mr.RpToTrans(R_closest, np.append(t_closest, 0))

        time_in = d_bc / self._vF
        return succeed, np.array([vx, vy, wz]), time_in, T_closest