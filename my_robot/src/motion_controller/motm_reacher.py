import time

import numpy as np
import qpsolvers as qp
from qpsolvers.exceptions import SolverError

from ..robot import Robot


class RedundancyResolutionController:
    def __init__(self, robot: Robot):
        super().__init__()

        self._robot = robot
        self._Y = 0.01
        self._manip_weight = 0.0
        self.success = False

    def ctrl(self, v_gripper_desired, v_base_desired):
        n = self._robot.dof

        t_err = max(np.sum(np.abs(v_gripper_desired[:3])), 1e-6)
        Q = np.eye(n + 6)
        Q[:n, :n] *= self._Y
        Q[0, 0] *= 1.0 / t_err
        Q[1, 1] *= 1.0 / t_err
        Q[n:, n:] = (2.0 / t_err) * np.eye(6)

        q = self._robot.q
        Aeq = np.c_[self._robot.jacobe(q), np.eye(6)]
        beq = v_gripper_desired.reshape((6,))

        Ain = np.zeros((n + 6, n + 6))
        bin = np.zeros(n + 6)
        ps = 0.1
        pi = 0.9
        Ain[:n, :n], bin[:n] = self._robot.joint_velocity_damper(q, ps, pi)

        c = np.zeros(n + 6)
        if self._manip_weight > 0.0:
            c[2:n] = -self._manip_weight * self._robot.jacobm(q, start=2).reshape((n - 2,))

        lb = -np.r_[self._robot.qd_lim[:n], 10 * np.ones(6)]
        ub = np.r_[self._robot.qd_lim[:n], 10 * np.ones(6)]

        lb[0] = ub[0] = v_base_desired[2]
        lb[1] = ub[1] = v_base_desired[0]

        t_start = time.time()
        qd = None
        for solver_name in ('cvxopt', 'highs', 'osqp'):
            try:
                qd = qp.solve_qp(
                    Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver=solver_name
                )
            except (ArithmeticError, ValueError, SolverError):
                qd = None
            if qd is not None:
                break
        t_end = time.time()
        solve_time_ms = (t_end - t_start) * 1000.0

        self.success = qd is not None
        if qd is None:
            qd = self._fallback_solution(v_gripper_desired, v_base_desired)
        else:
            qd = qd[:n]

        return qd, self.success, solve_time_ms

    def _fallback_solution(self, v_gripper_desired, v_base_desired):
        q = self._robot.q
        jacobe = self._robot.jacobe(q)

        base_cols = [0, 1]
        arm_cols = list(range(2, self._robot.dof))
        base_jacobian = jacobe[:, base_cols]
        arm_jacobian = jacobe[:, arm_cols]

        base_cmd = np.array([v_base_desired[2], v_base_desired[0]])
        base_twist = base_jacobian @ base_cmd
        residual_twist = v_gripper_desired - base_twist

        damp = 1e-4
        arm_solution = arm_jacobian.T @ np.linalg.solve(
            arm_jacobian @ arm_jacobian.T + damp * np.eye(6),
            residual_twist,
        )

        qd = np.zeros(self._robot.dof)
        qd[0] = v_base_desired[2]
        qd[1] = v_base_desired[0]
        qd[2:] = arm_solution
        qd = np.clip(qd, -self._robot.qd_lim[:self._robot.dof], self._robot.qd_lim[:self._robot.dof])
        qd[0] = v_base_desired[2]
        qd[1] = v_base_desired[0]
        return qd