import numpy as np

from .motm_reacher import MotMReacher
from .final_phase_task_controller import FinalPhaseTaskController
from ..robot import Robot


STATE_NAMES = {
    0: "Prepare (wait for motion time)",
    1: "Motion (MotMReacher)",
    2: "Final Phase (FinalPhaseTaskController)",
}


class ArmController:
    def __init__(self, ts: float, robot: Robot):
        super().__init__()

        self._d_task = 0.60
        self._vB = 1.0
        self._kA = 0.5
        self._state2_vx_scale = 0.2

        self._robot = robot

        self._motm_reacher = MotMReacher(ts, self._robot)
        self._final_phase_task_controller = FinalPhaseTaskController()

        self._state = 0
        self._prev_state = -1
        self._last_desired_pose = np.eye(4)

    def reset(self, T_bg):
        self._state = 0
        self._motm_reacher.reset(T_bg)
        self._final_phase_task_controller.reset()
        self._last_desired_pose = np.array(T_bg, copy=True)

    def ctrl(self, T_target, time_in, v_base_desired, T_base, T_bg, T_closest):
        prev_state = self._state

        T_gripper_closest = T_closest @ T_bg
        T_gt_closest = np.linalg.inv(T_gripper_closest) @ T_target
        t_gt_closest = T_gt_closest[:3, 3]
        time_arm = np.linalg.norm(t_gt_closest) / (self._vB * self._kA)

        T_gripper = T_base @ T_bg
        T_gt = np.linalg.inv(T_gripper) @ T_target
        t_gt = T_gt[:3, 3]
        d_gt = np.linalg.norm(t_gt)

        if self._state == 0:
            if time_arm > time_in:
                self._state = 1
                self._motm_reacher.reset(T_bg)

        if self._state != 2:
            if d_gt < self._d_task:
                self._state = 2

        if self._state == 2 and prev_state != 2:
            self._final_phase_task_controller.reset()

        if self._state != self._prev_state:
            print(
                f"[STATE CHANGE] State: {self._prev_state} → {self._state} "
                f"({STATE_NAMES.get(self._prev_state, 'Unknown')} → {STATE_NAMES.get(self._state, 'Unknown')})"
            )
            self._prev_state = self._state

        if self._state == 1:
            T_bt_closest = np.linalg.inv(T_closest) @ T_target
            v_gripper_desired = self._motm_reacher.ctrl(time_in, T_bt_closest, v_base_desired)
            self._last_desired_pose = self._motm_reacher.desired_pose
        elif self._state == 2:
            v_gripper_desired = self._final_phase_task_controller.ctrl(T_gripper, T_target)
            v_base_desired = np.array(v_base_desired, copy=True)
            v_base_desired[0] *= self._state2_vx_scale
            v_base_desired[2] = 0.0
            self._last_desired_pose = np.linalg.inv(T_base) @ T_target
        else:
            dq = np.zeros(self._robot.dof)
            dq[0] = v_base_desired[2]
            dq[1] = v_base_desired[0]
            Je = self._robot.jacobe(self._robot.q)
            v_gripper_desired = Je @ dq
            if T_gripper[2, 3] < T_target[2, 3] + 0.1:
                v_gripper_desired[2] -= 0.2
            self._last_desired_pose = np.array(T_bg, copy=True)

        return v_gripper_desired, v_base_desired, self._state, d_gt

    @property
    def desired_pose(self) -> np.ndarray:
        return np.array(self._last_desired_pose, copy=True)