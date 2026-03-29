import numpy as np
import qpsolvers as qp
import time

from ..robot import Robot


class RedundancyResolutionController:
    def __init__(self, robot: Robot):
        super().__init__()
        self._robot = robot

    def ctrl(self, v_gripper_desired, v_base_desired):
        """Whole-body control via QP.

        Velocity DOF ordering (12-DOF model):
            [0] vx,  [1] vy,  [2] vz,  [3] wx,  [4] wy,  [5] wz,
            [6] dj1, [7] dj2, ..., [11] dj6

        Non-holonomic constraints enforce: vz=0, wx=0, wy=0.
        Commanded base velocity v_base_desired = [vx, vy, wz].

        Args:
            v_gripper_desired: (6,) desired EE velocity [v; omega] in EE frame
            v_base_desired:    (3,) [vx, vy, wz] from base controller (body frame)
            
        Returns:
            qd_solution: (12,) joint velocities
            success: bool, whether QP solver succeeded
            solve_time: float, solver execution time in milliseconds
        """
        n = self._robot.dof  # 12

        t_err = max(np.sum(np.abs(v_gripper_desired[:3])), 1e-6)
        Y = 0.01

        # Cost matrix (n + 6) for [joint velocities | slack]
        Q = np.eye(n + 6)
        Q[:n, :n] *= Y
        # Up-weight free base DOFs (vx=0, vy=1, wz=5) relative to task error
        Q[0, 0] *= 1.0 / t_err
        Q[1, 1] *= 1.0 / t_err
        Q[5, 5] *= 1.0 / t_err
        Q[n:, n:] = (2.0 / t_err) * np.eye(6)

        q = self._robot.q

        # Equality: jacobe * qd + slack = v_gripper_desired
        Aeq = np.c_[self._robot.jacobe(q), np.eye(6)]
        beq = v_gripper_desired.reshape((6,))

        # Inequality: joint limit avoidance
        Ain = np.zeros((n + 6, n + 6))
        bin = np.zeros(n + 6)
        ps = 0.1
        pi = 0.9
        Ain[:n, :n], bin[:n] = self._robot.joint_velocity_damper(q, ps, pi)

        # Gradient term: maximize arm manipulability (skip 6 base DOFs)
        c = np.concatenate((
            np.zeros(6),
            -self._robot.jacobm(q, start=6).reshape((n - 6,)),
            np.zeros(6)
        ))

        # Alignment cost term (kept for reference, not added to c here)
        ke = 0.5
        bTe = self._robot.fkine(q, include_base=False)
        theta_e = np.arctan2(bTe[1, -1], bTe[0, -1])
        e = ke * theta_e

        # Velocity bounds
        lb = -np.r_[self._robot.qd_lim[:n], 10 * np.ones(6)]
        ub = np.r_[self._robot.qd_lim[:n], 10 * np.ones(6)]

        # Commanded base velocities: fix to upper-level planned values
        lb[0] = ub[0] = v_base_desired[0]   # vx
        lb[1] = ub[1] = v_base_desired[1]   # vy
        lb[5] = ub[5] = v_base_desired[2]   # wz

        # Non-holonomic constraints: vz = wx = wy = 0
        lb[2] = ub[2] = 0.0   # vz
        lb[3] = ub[3] = 0.0   # wx
        lb[4] = ub[4] = 0.0   # wy

        # Measure solver execution time
        t_start = time.time()
        qd_solution = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver='osqp')
        t_end = time.time()
        solve_time_ms = (t_end - t_start) * 1000.0
        
        success = qd_solution is not None
        if not success:
            qd_fallback = self._fallback_solution(v_gripper_desired, v_base_desired)
            return qd_fallback, success, solve_time_ms

        return qd_solution[:n], success, solve_time_ms

    def _fallback_solution(self, v_gripper_desired, v_base_desired):
        q = self._robot.q
        jacobe = self._robot.jacobe(q)

        base_cols = [0, 1, 5]
        arm_cols = list(range(6, self._robot.dof))
        base_jacobian = jacobe[:, base_cols]
        arm_jacobian = jacobe[:, arm_cols]

        base_cmd = np.array([v_base_desired[0], v_base_desired[1], v_base_desired[2]])
        base_twist = base_jacobian @ base_cmd
        residual_twist = v_gripper_desired - base_twist

        damp = 1e-4
        arm_solution = arm_jacobian.T @ np.linalg.solve(
            arm_jacobian @ arm_jacobian.T + damp * np.eye(6),
            residual_twist,
        )

        qd = np.zeros(self._robot.dof)
        qd[0] = v_base_desired[0]
        qd[1] = v_base_desired[1]
        qd[5] = v_base_desired[2]
        qd[6:] = arm_solution

        qd = np.clip(qd, -self._robot.qd_lim[: self._robot.dof], self._robot.qd_lim[: self._robot.dof])
        qd[0] = v_base_desired[0]
        qd[1] = v_base_desired[1]
        qd[5] = v_base_desired[2]
        qd[2] = 0.0
        qd[3] = 0.0
        qd[4] = 0.0
        return qd