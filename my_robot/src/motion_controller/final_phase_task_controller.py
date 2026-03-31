import numpy as np
import modern_robotics as mr


class FinalPhaseTaskController:
    def __init__(self):
        super().__init__()

        self._kP = 14.0

    def reset(self):
        pass

    def ctrl(self, T_gripper, T_target):
        Vb = np.zeros(6)
        T_err = mr.TransInv(T_gripper) @ T_target
        Vb[:3] = T_err[:3, -1]
        Vb[3:] = mr.so3ToVec(mr.MatrixLog3(T_err[:3, :3]))
        V = self._kP * Vb
        return V