import sys
import os

# ============================================================
# �️ 自动环境净化与重启机制 🛡️
# ------------------------------------------------------------
# 目的：确保在 ROS 环境中也能使用 Conda 环境的 Pinocchio
# 原理：检测到 LD_LIBRARY_PATH 中包含 ROS 路径时，自动清理并重启脚本
# ============================================================
def purify_environment_and_restart():
    # 检查是否已经标记为已净化（防止无限递归）
    if os.environ.get("MIBOT_ENV_PURIFIED") == "1":
        return

    needs_restart = False
    new_environ = os.environ.copy()
    
    # 1. 清理 LD_LIBRARY_PATH (解决 undefined symbol: EIGENPY error)
    ld_path = new_environ.get("LD_LIBRARY_PATH", "")
    if "/opt/ros" in ld_path:
        print("⚠️ 检测到 ROS 环境变量干扰，正在自动净化环境...")
        paths = ld_path.split(":")
        # 移除非 Conda 的路径 (特别是 ROS)
        clean_paths = [p for p in paths if "ros" not in p and "/opt/" not in p]
        new_environ["LD_LIBRARY_PATH"] = ":".join(clean_paths)
        needs_restart = True
        print(f"  - 已清理 LD_LIBRARY_PATH (剩余 {len(clean_paths)} 个路径)")

    # 2. 清理 PYTHONPATH (解决 import error)
    python_path = new_environ.get("PYTHONPATH", "")
    if "/opt/ros" in python_path:
        paths = python_path.split(":")
        clean_paths = [p for p in paths if "ros" not in p and "/opt/" not in p]
        new_environ["PYTHONPATH"] = ":".join(clean_paths)
        needs_restart = True
        print(f"  - 已清理 PYTHONPATH (剩余 {len(clean_paths)} 个路径)")

    # 3. 如果发现问题，标记并重启
    if needs_restart:
        new_environ["MIBOT_ENV_PURIFIED"] = "1"
        print("🔄 正在使用净化后的环境重启脚本...\n" + "="*70)
        # 使用当前 Python解释器 重启脚本
        try:
            os.execve(sys.executable, [sys.executable] + sys.argv, new_environ)
        except Exception as e:
            print(f"❌ 自动重启失败: {e}")
            print("请尝试手动运行: export LD_LIBRARY_PATH=\"\" && python3 " + " ".join(sys.argv))
            sys.exit(1)

# 在导入任何敏感库之前执行净化检查
purify_environment_and_restart()

# ============================================================
# 强制路径优先级（双重保险）
# ============================================================
# 移除 sys.path 中的冲突路径（针对当前进程）
sys.path = [p for p in sys.path if "ros" not in p and "/opt/" not in p]

# 调试信息
print("=" * 70)
print("环境检查通过")
print("=" * 70)
print(f"Python: {sys.version.split()[0]}")
# ============================================================
import numpy as np
import pinocchio as pin
# 验证 Pinocchio
print("=" * 70)
print(f"Pinocchio: {pin.__file__}")
try:
    print(f"Version:   {pin.__version__}")
except AttributeError: pass
print("=" * 70 + "\n")
from scipy.spatial.transform import Rotation
from pathlib import Path


def _require_frame_id(model, candidates):
    for candidate in candidates:
        for idx, frame in enumerate(model.frames):
            if frame.name == candidate:
                return idx
    raise ValueError(f"Could not find any frame in {candidates}")


def _skew(p: np.ndarray) -> np.ndarray:
    return np.array([
        [0.0, -p[2], p[1]],
        [p[2], 0.0, -p[0]],
        [-p[1], p[0], 0.0],
    ])


def _adjoint_vw(T: np.ndarray) -> np.ndarray:
    """Adjoint for twists ordered as [v; omega]."""
    R = T[:3, :3]
    p = T[:3, 3]
    A = np.zeros((6, 6))
    A[:3, :3] = R
    A[:3, 3:] = _skew(p) @ R
    A[3:, 3:] = R
    return A


class Robot:
    """
    Mobile manipulator kinematic model using pinocchio.

    State representation:
      - Position (q_pos): [x, y, z, yaw, pitch, roll, j1..j6]  (12,)
        Base orientation as ZYX Euler angles (yaw=rot_z, pitch=rot_y, roll=rot_x).
      - Control velocity (q_vel): [wz, vx, dj1..dj6]   (8,)
        This matches the reference project: only differential-drive base
        yaw/forward velocities are optimized in WBC.

    Integration uses pinocchio's manifold integration (pin.integrate) for
    correct SO3 attitude update.
    """

    def __init__(self, urdf_path: str = None):
        if urdf_path is None:
            urdf_path = str(
                Path(__file__).parent.parent.parent
                / "robot_model" / "urdf" / "robot.urdf"
            )

        # Build kinematics-only model from URDF
        self._model = pin.buildModelFromUrdf(urdf_path)
        self._data = self._model.createData()

        # Joint IDs -------------------------------------------------------
        self._base_jid = self._model.getJointId('world_to_base')
        self._arm_jids = [self._model.getJointId(f'j{i}') for i in range(1, 7)]

        # Mapping: our 12 velocity DOFs → columns in pinocchio v vector
        base_v_start = self._model.joints[self._base_jid].idx_v
        arm_v_cols = [self._model.joints[jid].idx_v for jid in self._arm_jids]
        self._full_v_cols = list(range(base_v_start, base_v_start + 6)) + arm_v_cols
        # Reference-style control space: [wz, vx, j1..j6]
        self._control_cols = [5, 0, 6, 7, 8, 9, 10, 11]
        self._v_cols = [self._full_v_cols[i] for i in self._control_cols]

        # Mapping: arm joint config indices in pinocchio q vector
        self._arm_q_cols = [self._model.joints[jid].idx_q for jid in self._arm_jids]
        self._base_q_start = self._model.joints[self._base_jid].idx_q

        # Tool frame aligned with the actual fingertip center for the
        # horizontal gripper mount.
        lt_fid = _require_frame_id(self._model, ['lt'])
        lt_frame = self._model.frames[lt_fid]
        T_offset = pin.SE3(
            Rotation.from_euler('x', np.pi / 2).as_matrix(),
            np.array([0.0, -0.2306, 0.0]),
        )
        ee_frame = pin.Frame(
            'ee_frame',
            lt_frame.parentJoint,
            lt_fid,
            lt_frame.placement * T_offset,
            pin.FrameType.OP_FRAME
        )
        self._model.addFrame(ee_frame)
        self._data = self._model.createData()
        self._ee_fid = self._model.getFrameId('ee_frame')
        self._mobile_base_fid = _require_frame_id(self._model, ['frankie_base0'])
        self._arm_base_fid = _require_frame_id(self._model, ['base_link', 'base_Link'])

        # WBC control DOF: [wz, vx, j1..j6]
        self._dof = 8

        # Internal pinocchio state
        self._q_pin = pin.neutral(self._model)   # configuration (nq,)
        self._v_pin = np.zeros(self._model.nv)   # velocity     (nv,)

        # Joint position limits in reduced control space [wz, vx, j1..j6]
        self._q_lim = np.zeros((2, self._dof))
        self._q_lim[0, :2] = -1e6
        self._q_lim[1, :2] = 1e6
        for i, jid in enumerate(self._arm_jids):
            qi = self._model.joints[jid].idx_q
            self._q_lim[0, 2 + i] = self._model.lowerPositionLimit[qi]
            self._q_lim[1, 2 + i] = self._model.upperPositionLimit[qi]

        # Joint velocity limits in reduced control space
        self._qd_lim = np.zeros(self._dof)
        self._qd_lim[:2] = 4.0
        for i, jid in enumerate(self._arm_jids):
            vi = self._model.joints[jid].idx_v
            self._qd_lim[2 + i] = self._model.velocityLimit[vi]

    # =====================================================================
    # Internal helpers
    # =====================================================================

    def _q_pos_to_pin(self, q_pos: np.ndarray) -> np.ndarray:
        """Convert 12-DOF position to pinocchio full q vector."""
        q_pin = pin.neutral(self._model)
        # Base: xyz + ZYX euler → xyz + quaternion [qx,qy,qz,qw]
        q_pin[self._base_q_start:self._base_q_start + 3] = q_pos[:3]
        quat = Rotation.from_euler('ZYX', q_pos[3:6]).as_quat()
        q_pin[self._base_q_start + 3:self._base_q_start + 7] = quat
        # Arm joints
        for i, qi in enumerate(self._arm_q_cols):
            q_pin[qi] = q_pos[6 + i]
        return q_pin

    def _pin_to_q_pos(self) -> np.ndarray:
        """Extract 12-DOF position from internal pinocchio q."""
        pos = np.zeros(12)
        pos[:3] = self._q_pin[self._base_q_start:self._base_q_start + 3]
        quat = self._q_pin[self._base_q_start + 3:self._base_q_start + 7]
        pos[3:6] = Rotation.from_quat(quat).as_euler('ZYX')
        for i, qi in enumerate(self._arm_q_cols):
            pos[6 + i] = self._q_pin[qi]
        return pos

    def _control_q_from_full(self, q_pos: np.ndarray) -> np.ndarray:
        q_ctrl = np.zeros(self._dof)
        q_ctrl[2:] = np.asarray(q_pos, dtype=float)[6:12]
        return q_ctrl

    def _reduced_qd_to_full(self, qd: np.ndarray) -> np.ndarray:
        qd = np.asarray(qd, dtype=float)
        if qd.shape[0] == 12:
            return qd
        if qd.shape[0] != self._dof:
            raise ValueError(f"Expected qd length {self._dof} or 12, got {qd.shape[0]}")
        qd_full = np.zeros(12)
        qd_full[0] = qd[1]
        qd_full[5] = qd[0]
        qd_full[6:12] = qd[2:]
        return qd_full

    # =====================================================================
    # State properties
    # =====================================================================

    @property
    def dof(self) -> int:
        return self._dof

    @property
    def q_pos(self) -> np.ndarray:
        """Position: [x, y, z, yaw, pitch, roll, j1..j6]  (12,)"""
        return self._pin_to_q_pos()

    @q_pos.setter
    def q_pos(self, pos):
        self._q_pin[:] = self._q_pos_to_pin(np.asarray(pos, dtype=float))

    @property
    def q_vel(self) -> np.ndarray:
        """Reduced control velocity: [wz, vx, dj1..dj6]  (8,)"""
        full_vel = self._v_pin[self._full_v_cols]
        return full_vel[self._control_cols].copy()

    @q_vel.setter
    def q_vel(self, vel):
        vel_full = self._reduced_qd_to_full(np.asarray(vel, dtype=float))
        self._v_pin[self._full_v_cols] = vel_full

    @property
    def q(self) -> np.ndarray:
        """Alias of q_pos (position representation)."""
        return self.q_pos

    @q.setter
    def q(self, pos):
        self.q_pos = pos

    @property
    def qd_lim(self) -> np.ndarray:
        return self._qd_lim.copy()

    # =====================================================================
    # Integration
    # =====================================================================

    def integrate(self, qd: np.ndarray, dt: float):
        """Integrate reduced WBC velocity via manifold integration.

        Uses pinocchio.integrate for correct SO3 attitude update:
            p_new = p + R * v_body * dt
            R_new = R * Exp(omega_body * dt)

        Args:
            qd: (8,) velocity [wz, vx, dj1..dj6] or full (12,) velocity
            dt: time step [s]
        """
        v = np.zeros(self._model.nv)
        v[self._full_v_cols] = self._reduced_qd_to_full(qd)
        self._q_pin = pin.integrate(self._model, self._q_pin, v * dt)

    # =====================================================================
    # Kinematics
    # =====================================================================

    def fkine(self, q_pos: np.ndarray, include_base: bool = True) -> np.ndarray:
        """Forward kinematics.

        Args:
            q_pos: (12,) [x, y, z, yaw, pitch, roll, j1..j6]
            include_base: True → T_world_ee; False → T_mobile_base_ee

        Returns:
            (4, 4) SE3 homogeneous transform
        """
        q_pin = self._q_pos_to_pin(np.asarray(q_pos, dtype=float))
        pin.forwardKinematics(self._model, self._data, q_pin)
        pin.updateFramePlacements(self._model, self._data)
        T_world_ee = self._data.oMf[self._ee_fid].homogeneous
        if not include_base:
            T_world_mobile_base = self._data.oMf[self._mobile_base_fid].homogeneous
            T_world_ee = np.linalg.inv(T_world_mobile_base) @ T_world_ee
        return T_world_ee

    def arm_base_pose(self, q_pos: np.ndarray, include_mobile_base: bool = True) -> np.ndarray:
        """Return the arm-base pose.

        Args:
            q_pos: (12,) [x, y, z, yaw, pitch, roll, j1..j6]
            include_mobile_base: True → T_world_arm_base; False → T_mobile_arm_base

        Returns:
            (4, 4) SE3 homogeneous transform
        """
        q_pin = self._q_pos_to_pin(np.asarray(q_pos, dtype=float))
        pin.forwardKinematics(self._model, self._data, q_pin)
        pin.updateFramePlacements(self._model, self._data)
        T_world_arm_base = self._data.oMf[self._arm_base_fid].homogeneous
        if include_mobile_base:
            return T_world_arm_base
        T_world_mobile_base = self._data.oMf[self._mobile_base_fid].homogeneous
        return np.linalg.inv(T_world_mobile_base) @ T_world_arm_base

    def jacobe(self, q_pos: np.ndarray) -> np.ndarray:
        """Body Jacobian of the world EE pose in the EE local frame.

        This matches the reference project's convention: the returned Jacobian
        maps [wz, vx, dj1..dj6] to the end-effector body twist
        expressed in the end-effector frame, for the world-to-EE transform.
        """
        q_pin = self._q_pos_to_pin(np.asarray(q_pos, dtype=float))
        pin.computeJointJacobians(self._model, self._data, q_pin)
        pin.forwardKinematics(self._model, self._data, q_pin)
        pin.updateFramePlacements(self._model, self._data)
        return pin.getFrameJacobian(
            self._model, self._data, self._ee_fid, pin.LOCAL
        )[:, self._v_cols]

    def jacob0(self, q_pos: np.ndarray, start: int = 0, end: int = -1) -> np.ndarray:
        """Geometric Jacobian at end-effector in world-aligned frame.

        [v; omega] is expressed at the EE position aligned with world frame.

        Args:
            q_pos: (12,) full configuration
            start: first DOF index to include (0-based)
            end:   last DOF index (exclusive); -1 means all
        """
        q_pin = self._q_pos_to_pin(np.asarray(q_pos, dtype=float))
        pin.computeJointJacobians(self._model, self._data, q_pin)
        pin.updateFramePlacements(self._model, self._data)
        J_full = pin.getFrameJacobian(
            self._model, self._data, self._ee_fid, pin.LOCAL_WORLD_ALIGNED
        )
        J = J_full[:, self._v_cols]
        if end < 0:
            end = self._dof
        return J[:, start:end]

    def hessian0(self, q_pos: np.ndarray, start: int = 0, end: int = -1,
                 J: np.ndarray = None) -> np.ndarray:
        """Hessian of jacob0 w.r.t. joint velocities."""
        n = self._dof - start
        H = np.zeros((n, 6, n))
        if J is None:
            J = self.jacob0(q_pos, start=start, end=end)
        for i in range(n):
            for j in range(n):
                a = min(i, j)
                b = max(i, j)
                H[i, :3, j] = np.cross(J[3:, a], J[:3, b])
                if i < j:
                    H[i, 3:, j] = np.cross(J[3:, i], J[3:, j])
        return H

    def manipulability(self, q_pos: np.ndarray, start: int = 0, end: int = -1,
                       J: np.ndarray = None) -> float:
        """Yoshikawa manipulability measure."""
        if J is None:
            J = self.jacob0(q_pos, start=start, end=end)
        return np.sqrt(np.abs(np.linalg.det(J @ J.T)))

    def jacobm(self, q_pos: np.ndarray, start: int = 0, end: int = -1,
               J: np.ndarray = None, H: np.ndarray = None) -> np.ndarray:
        """Gradient of manipulability w.r.t. joint velocities.

        Returns:
            (n, 1) array, n = dof - start
        """
        if J is None:
            J = self.jacob0(q_pos, start=start, end=end)
        n = J.shape[1]
        if H is None:
            H = self.hessian0(q_pos, start=start, end=end, J=J)
        w = self.manipulability(q_pos, start=start, end=end, J=J)
        b = np.linalg.pinv(J @ J.T)
        Jm = np.zeros((n, 1))
        for i in range(n):
            c = J @ H[i, :, :].T
            Jm[i, 0] = w * c.flatten('F') @ b.flatten('F')
        return Jm

    def joint_velocity_damper(self, q_pos: np.ndarray, ps: float, pi: float,
                               gain: float = 1.0):
        """Joint limit avoidance constraint matrices (Ain, Bin).

        Populates inequality constraint Ain * qd <= Bin for joints near limits.

        Args:
            q_pos: (12,) current configuration
            ps:    safety distance from limit
            pi:    influence distance from limit (> ps)
            gain:  constraint gain

        Returns:
            Ain: (8, 8)
            Bin: (8,)
        """
        q_ctrl = self._control_q_from_full(q_pos)
        n = self._dof
        Ain = np.zeros((n, n))
        Bin = np.zeros(n)
        for i in range(n):
            if q_ctrl[i] - self._q_lim[0, i] <= pi:
                Bin[i] = -gain * ((self._q_lim[0, i] - q_ctrl[i]) + ps) / (pi - ps)
                Ain[i, i] = -1
            if self._q_lim[1, i] - q_ctrl[i] <= pi:
                Bin[i] = gain * ((self._q_lim[1, i] - q_ctrl[i]) - ps) / (pi - ps)
                Ain[i, i] = 1
        return Ain, Bin

    def p_servo(self, T_actual: np.ndarray, T_desired: np.ndarray,
                gain: float = 1.0, threshold: float = 0.1):
        """Proportional SE3 pose servo.

        Returns:
            v:       (6,) velocity [v; omega] in T_actual frame
            arrived: bool
        """
        T_err = np.linalg.inv(T_actual) @ T_desired
        Vb = np.zeros(6)
        Vb[:3] = T_err[:3, 3]
        Vb[3:] = Rotation.from_matrix(T_err[:3, :3]).as_rotvec()
        v = gain * Vb
        arrived = bool(np.sum(np.abs(Vb)) < threshold)
        return v, arrived

    def set_base(self, T_base: np.ndarray):
        """Set base pose from SE3 matrix (e.g. for initialisation).

        Args:
            T_base: (4, 4) world-to-base transform
        """
        self._q_pin[self._base_q_start:self._base_q_start + 3] = T_base[:3, 3]
        quat = Rotation.from_matrix(T_base[:3, :3]).as_quat()
        self._q_pin[self._base_q_start + 3:self._base_q_start + 7] = quat

    def sync_state(self, T_base: np.ndarray, arm_q: np.ndarray):
        """Synchronize the kinematic model with the measured robot state."""
        q_pos = self.q_pos
        q_pos[6:12] = np.asarray(arm_q, dtype=float)
        self.q_pos = q_pos
        self.set_base(T_base)