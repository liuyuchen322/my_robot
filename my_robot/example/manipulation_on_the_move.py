import sys
from pathlib import Path

# Add my_robot/ to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import numpy as np
import mujoco
import mujoco.viewer
import modern_robotics as mr
from scipy.spatial.transform import Rotation

from src.robot import Robot
from src.wheel_controller import DifferentialDriveWheelController
from src.motion_controller import (
    BaseController,
    ArmController,
    RedundancyResolutionController,
    GripperController,
    JointResetController,
)

if __name__ == '__main__':
    # ------------------------------------------------------------------ #
    # MuJoCo model                                                         #
    # ------------------------------------------------------------------ #
    scene_xml_path = (
        Path(__file__).parent.parent / 'robot_model' / 'mjcf' / 'world_scene.xml'
    )
    model = mujoco.MjModel.from_xml_path(str(scene_xml_path))
    data = mujoco.MjData(model)

    # Arm joint IDs and qpos addresses
    arm_joint_names = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6']
    arm_joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        for n in arm_joint_names
    ]
    arm_qaddrs = [model.jnt_qposadr[jid] for jid in arm_joint_ids]
    left_finger_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'fl_joint8')
    right_finger_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'fl_joint7')
    left_finger_qaddr = model.jnt_qposadr[left_finger_joint_id]
    right_finger_qaddr = model.jnt_qposadr[right_finger_joint_id]

    # Body / site IDs
    mobile_base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'frankie_base0')
    base_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'base_site')
    ee_site_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'end_effector_site')
    box_joint_names = ['box1_joint', 'box2_joint']
    box_joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        for n in box_joint_names
    ]
    box_body_names = ['box1', 'box2']
    box_body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        for n in box_body_names
    ]

    box_site_names = ['box_site1', 'box_site2']
    box_site_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
        for n in box_site_names
    ]
    home_site_names = ['grasp_object1', 'grasp_object2']
    home_site_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
        for n in home_site_names
    ]
    drop_site_names = ['drop_location1', 'drop_location2']
    drop_site_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
        for n in drop_site_names
    ]

    # ------------------------------------------------------------------ #
    # Initial MuJoCo state                                                 #
    # ------------------------------------------------------------------ #
    # 0.0, -1.2, 1.8, -1.6, 1.57, 0.0
    init_arm_q = [0.0, -1.57, -1.57, 0, 0, 0.0]
    gripper_max_opening = 0.044

    def gripper_opening_to_joint_targets(opening):
        opening = float(np.clip(opening, 0.0, gripper_max_opening))
        closing_amount = gripper_max_opening - opening
        return -closing_amount, closing_amount

    for addr, q in zip(arm_qaddrs, init_arm_q):
        data.qpos[addr] = q
    left_finger_target, right_finger_target = gripper_opening_to_joint_targets(gripper_max_opening)
    data.qpos[left_finger_qaddr] = left_finger_target
    data.qpos[right_finger_qaddr] = right_finger_target

    data.ctrl[2:8] = init_arm_q   # arm position actuators
    data.ctrl[8]   = left_finger_target
    data.ctrl[9]   = right_finger_target
    mujoco.mj_forward(model, data)

    # ------------------------------------------------------------------ #
    # Pinocchio robot model                                                #
    # ------------------------------------------------------------------ #
    robot = Robot()

    def body_T(body_id):
        R = data.xmat[body_id].reshape(3, 3)
        t = data.xpos[body_id].copy()
        return mr.RpToTrans(R, t)

    def set_freejoint_pose(joint_id, T):
        qadr = model.jnt_qposadr[joint_id]
        vadr = model.jnt_dofadr[joint_id]
        quat_xyzw = Rotation.from_matrix(T[:3, :3]).as_quat()
        data.qpos[qadr:qadr + 3] = T[:3, 3]
        data.qpos[qadr + 3:qadr + 7] = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
        data.qvel[vadr:vadr + 6] = 0.0

    # Initialise robot model from MuJoCo's measured state
    T_mobile_base = body_T(mobile_base_body_id)
    robot.sync_state(T_mobile_base, init_arm_q)

    # ------------------------------------------------------------------ #
    # Task poses (read once; world sites are static)                       #
    # ------------------------------------------------------------------ #
    def site_T(sid):
        R = data.site_xmat[sid].reshape(3, 3)
        t = data.site_xpos[sid].copy()
        return mr.RpToTrans(R, t)

    # ------------------------------------------------------------------ #
    # Controllers                                                          #
    # ------------------------------------------------------------------ #
    # Wheel: r=0.05 m (wheel radius), w=0.2 m (half axle width)
    wheel_controller = DifferentialDriveWheelController(r=0.05, w=0.2)

    control_timestep = 0.01                                  # 100 Hz
    n_steps = round(control_timestep / model.opt.timestep)   # = 10

    base_controller = BaseController()

    arm_controller = ArmController(control_timestep, robot)
    R_gripper = data.site_xmat[ee_site_id].reshape(3, 3)
    t_gripper = data.site_xpos[ee_site_id].copy()
    T_gripper  = mr.RpToTrans(R_gripper, t_gripper)
    T_gb = np.linalg.inv(T_mobile_base) @ T_gripper
    arm_controller.reset(T_gb)

    redundancy_resolution_controller = RedundancyResolutionController(robot)
    gripper_controller = GripperController(control_timestep)
    joint_reset_controller = JointResetController(
        control_timestep,
        np.asarray(init_arm_q, dtype=float),
        robot.qd_lim[2:],
    )

    task_state = {
        'box_idx': 0,
        'mode': 'pick',
        'next_place_site_ids': [drop_site_ids[0], drop_site_ids[1]],
    }

    def task_pose(mode: str, box_idx: int):
        if mode == 'pick':
            T_pick = site_T(home_site_ids[box_idx])
            T_box = site_T(box_site_ids[box_idx])
            T_pick[:3, 3] = T_box[:3, 3]
            return T_pick
        return site_T(task_state['next_place_site_ids'][box_idx])

    def next_task():
        if task_state['mode'] == 'pick':
            return 'place', task_state['box_idx']
        return 'pick', (task_state['box_idx'] + 1) % len(box_site_ids)

    def current_targets():
        target = task_pose(task_state['mode'], task_state['box_idx'])
        next_mode, next_box_idx = next_task()
        return target, task_pose(next_mode, next_box_idx)

    def advance_task():
        if task_state['mode'] == 'pick':
            task_state['mode'] = 'place'
            return

        box_idx = task_state['box_idx']
        if task_state['next_place_site_ids'][box_idx] == drop_site_ids[box_idx]:
            task_state['next_place_site_ids'][box_idx] = home_site_ids[box_idx]
        else:
            task_state['next_place_site_ids'][box_idx] = drop_site_ids[box_idx]
        task_state['box_idx'] = (box_idx + 1) % len(box_site_ids)
        task_state['mode'] = 'pick'

    recovery = {
        'mode': None,            # None | lift | joint | hold
        'lift_target': np.eye(4),
        'lift_cycles': 0,
    }
    recovery_lift_offset = np.array([0.0, 0.0, 0.10])
    recovery_lift_max_cycles = 40
    grasp_trigger_distance = 0.05
    place_trigger_distance = 0.15
    attached_box_index = None
    attached_box_T_go = None

    def start_recovery(T_gb_current):
        recovery['mode'] = 'lift'
        recovery['lift_target'] = np.array(T_gb_current, copy=True)
        recovery['lift_target'][:3, 3] += recovery_lift_offset
        recovery['lift_cycles'] = 0
        joint_reset_controller.stop()

    def stop_recovery():
        recovery['mode'] = None
        recovery['lift_cycles'] = 0
        joint_reset_controller.stop()

    # ------------------------------------------------------------------ #
    # Main loop                                                            #
    # ------------------------------------------------------------------ #
    num = 0
    qd          = np.zeros(robot.dof)   # [wz, vx, j1..j6]
    arm_joint_cmd = np.asarray(init_arm_q, dtype=float).copy()
    direct_arm_control = False
    gripper_ctrl = gripper_controller.get()  # finger opening [m]
    gripper_action = None               # None | "close" | "open"
    print_interval = 10  # Print logs every 50 control cycles (0.5 seconds at 100 Hz)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            start_time = time.time()

            # ---- Control update at 100 Hz --------------------------------
            if num % n_steps == 0:
                # Read base and end-effector poses from MuJoCo
                T_mobile_base = body_T(mobile_base_body_id)
                R_gripper = data.site_xmat[ee_site_id].reshape(3, 3)
                t_gripper = data.site_xpos[ee_site_id].copy()
                T_gripper = mr.RpToTrans(R_gripper, t_gripper)

                arm_q_measured = np.array([data.qpos[addr] for addr in arm_qaddrs])
                robot.sync_state(T_mobile_base, arm_q_measured)

                # Current task target and the next preview target
                T_target, T_next = current_targets()

                # Base trajectory controller → [vx, vy, wz]
                succeed, v_base, time_in, T_closest = base_controller.ctrl(
                    T_target, T_next, T_mobile_base, T_mobile_base[:3, :3]
                )

                T_gb = np.linalg.inv(T_mobile_base) @ T_gripper
                task_v_gripper_desired, task_v_base_desired, task_arm_state, d_gt = arm_controller.ctrl(
                    T_target, time_in, v_base, T_mobile_base, T_gb, T_closest
                )
                task_desired_pose = arm_controller.desired_pose

                if recovery['mode'] is not None and task_arm_state in (1, 2):
                    stop_recovery()

                use_direct_arm_cmd = False
                arm_state = task_arm_state
                v_gripper_desired = task_v_gripper_desired
                v_base_desired = task_v_base_desired
                T_ee_desired = task_desired_pose

                if recovery['mode'] is not None and gripper_action is None and task_arm_state == 0:
                    if recovery['mode'] == 'lift':
                        v_gripper_desired, lift_arrived = robot.p_servo(
                            T_gb, recovery['lift_target'], gain=6.0, threshold=0.02
                        )
                        v_base_desired = v_base
                        qd, wbc_success, wbc_solve_time = redundancy_resolution_controller.ctrl(
                            v_gripper_desired, v_base_desired
                        )
                        arm_joint_cmd = robot.q_pos[6:12]
                        T_ee_desired = recovery['lift_target']
                        arm_state = 3
                        recovery['lift_cycles'] += 1
                        if lift_arrived or recovery['lift_cycles'] >= recovery_lift_max_cycles:
                            joint_reset_controller.start(arm_q_measured)
                            recovery['mode'] = 'joint'
                    elif recovery['mode'] == 'joint':
                        arm_joint_cmd, arm_joint_vel_cmd, reset_done = joint_reset_controller.sample()
                        qd = np.zeros(robot.dof)
                        qd[0] = v_base[2]
                        qd[1] = v_base[0]
                        qd[2:] = arm_joint_vel_cmd
                        v_base_desired = v_base
                        v_gripper_desired = robot.jacobe(robot.q) @ qd
                        q_pos_desired = robot.q_pos.copy()
                        q_pos_desired[6:12] = arm_joint_cmd
                        T_ee_desired = robot.fkine(q_pos_desired, include_base=False)
                        wbc_success = True
                        wbc_solve_time = 0.0
                        use_direct_arm_cmd = True
                        arm_state = 4
                        if reset_done:
                            recovery['mode'] = 'hold'
                    else:
                        arm_joint_cmd = np.asarray(init_arm_q, dtype=float).copy()
                        qd = np.zeros(robot.dof)
                        qd[0] = v_base[2]
                        qd[1] = v_base[0]
                        v_base_desired = v_base
                        v_gripper_desired = robot.jacobe(robot.q) @ qd
                        q_pos_desired = robot.q_pos.copy()
                        q_pos_desired[6:12] = arm_joint_cmd
                        T_ee_desired = robot.fkine(q_pos_desired, include_base=False)
                        wbc_success = True
                        wbc_solve_time = 0.0
                        use_direct_arm_cmd = True
                        arm_state = 5
                else:
                    qd, wbc_success, wbc_solve_time = redundancy_resolution_controller.ctrl(
                        v_gripper_desired, v_base_desired
                    )
                direct_arm_control = use_direct_arm_cmd

                # Gripper logic
                dist_to_target = np.linalg.norm(T_target[:3, 3] - t_gripper)
                if gripper_action == "close":
                    arrive, gripper_ctrl = gripper_controller.close()
                    if arrive:
                        attached_box_index = task_state['box_idx']
                        T_box = body_T(box_body_ids[attached_box_index])
                        attached_box_T_go = np.linalg.inv(T_gripper) @ T_box
                        advance_task()
                        T_gb = np.linalg.inv(T_mobile_base) @ T_gripper
                        arm_controller.reset(T_gb)
                        start_recovery(T_gb)
                        gripper_action = None
                elif gripper_action == "open":
                    arrive, gripper_ctrl = gripper_controller.open()
                    if arrive:
                        if attached_box_index is not None:
                            T_box = np.array(T_target, copy=True)
                            set_freejoint_pose(box_joint_ids[attached_box_index], T_box)
                            attached_box_index = None
                            attached_box_T_go = None
                        advance_task()
                        T_gb = np.linalg.inv(T_mobile_base) @ T_gripper
                        arm_controller.reset(T_gb)
                        start_recovery(T_gb)
                        gripper_action = None
                else:
                    trigger_distance = (
                        grasp_trigger_distance if task_state['mode'] == 'pick' else place_trigger_distance
                    )
                    if dist_to_target < trigger_distance and recovery['mode'] is None:
                        if task_state['mode'] == 'pick':
                            # Pick: close gripper
                            gripper_action = "close"
                            _, gripper_ctrl = gripper_controller.close()
                        else:
                            # Place: open gripper
                            gripper_action = "open"
                            _, gripper_ctrl = gripper_controller.open()
                    else:
                        gripper_ctrl = gripper_controller.get()

                # Print logs at regular intervals
                if num % (n_steps * print_interval) == 0:
                    state_names = {
                        0: "Prepare",
                        1: "Motion",
                        2: "Final Phase",
                        3: "Recovery Lift",
                        4: "Recovery Reset",
                        5: "Recovery Hold",
                    }
                    state_name = state_names.get(arm_state, "Unknown")
                    
                    # Get EE position and target position in base frame
                    ee_pos_in_base = T_gb[:3, 3]
                    ee_desired_pos_in_base = T_ee_desired[:3, 3]
                    target_pos_in_base = np.linalg.inv(T_mobile_base) @ np.r_[T_target[:3, 3], 1]
                    target_pos_in_base = target_pos_in_base[:3]
                    
                    # Format positions
                    ee_pos_str = f"[{ee_pos_in_base[0]:.4f}, {ee_pos_in_base[1]:.4f}, {ee_pos_in_base[2]:.4f}]"
                    ee_desired_pos_str = (
                        f"[{ee_desired_pos_in_base[0]:.4f}, {ee_desired_pos_in_base[1]:.4f}, {ee_desired_pos_in_base[2]:.4f}]"
                    )
                    target_pos_str = f"[{target_pos_in_base[0]:.4f}, {target_pos_in_base[1]:.4f}, {target_pos_in_base[2]:.4f}]"

                    ee_quat_xyzw = Rotation.from_matrix(T_gb[:3, :3]).as_quat()
                    ee_desired_quat_xyzw = Rotation.from_matrix(T_ee_desired[:3, :3]).as_quat()
                    ee_quat_str = f"[{ee_quat_xyzw[0]:.4f}, {ee_quat_xyzw[1]:.4f}, {ee_quat_xyzw[2]:.4f}, {ee_quat_xyzw[3]:.4f}]"
                    ee_desired_quat_str = (
                        f"[{ee_desired_quat_xyzw[0]:.4f}, {ee_desired_quat_xyzw[1]:.4f}, {ee_desired_quat_xyzw[2]:.4f}, {ee_desired_quat_xyzw[3]:.4f}]"
                    )
                    
                    real_base_vel = np.zeros(6)
                    mujoco.mj_objectVelocity(
                        model, data, mujoco.mjtObj.mjOBJ_SITE, base_site_id, real_base_vel, 1
                    )
                    real_ee_vel = np.zeros(6)
                    mujoco.mj_objectVelocity(
                        model, data, mujoco.mjtObj.mjOBJ_SITE, ee_site_id, real_ee_vel, 1
                    )
                    v_cmd_ee = robot.jacobe(robot.q) @ qd

                    base_plan_vel_str = f"[{v_base[0]:.4f}, {v_base[1]:.4f}, {v_base[2]:.4f}]"
                    base_wbc_vel_str = (
                        f"[{v_base_desired[0]:.4f}, {v_base_desired[1]:.4f}, {v_base_desired[2]:.4f}]"
                    )
                    base_actual_vel_str = (
                        f"[{real_base_vel[3]:.4f}, {real_base_vel[4]:.4f}, {real_base_vel[2]:.4f}]"
                    )
                    
                    # Format gripper velocity [vx, vy, vz, wx, wy, wz]
                    gripper_vel_str = f"[{v_gripper_desired[0]:.4f}, {v_gripper_desired[1]:.4f}, {v_gripper_desired[2]:.4f}, {v_gripper_desired[3]:.4f}, {v_gripper_desired[4]:.4f}, {v_gripper_desired[5]:.4f}]"
                    gripper_vel_cmd_str = (
                        f"[{v_cmd_ee[0]:.4f}, {v_cmd_ee[1]:.4f}, {v_cmd_ee[2]:.4f}, "
                        f"{v_cmd_ee[3]:.4f}, {v_cmd_ee[4]:.4f}, {v_cmd_ee[5]:.4f}]"
                    )
                    gripper_vel_actual_str = (
                        f"[{real_ee_vel[3]:.4f}, {real_ee_vel[4]:.4f}, {real_ee_vel[5]:.4f}, "
                        f"{real_ee_vel[0]:.4f}, {real_ee_vel[1]:.4f}, {real_ee_vel[2]:.4f}]"
                    )
                    
                    # Format WBC info
                    wbc_status = "SUCCESS" if wbc_success else "FAILED"
                    task_label = f"{task_state['mode']}_box{task_state['box_idx'] + 1}"
                    
                    print(f"[Control {num//n_steps:05d}] Task: {task_label} | State: {arm_state} ({state_name})")
                    print(f"  EE Pos (base frame): {ee_pos_str} m | Target Pos (base frame): {target_pos_str} m | Distance: {d_gt:.4f} m")
                    print(f"  EE Actual Pose (base): pos={ee_pos_str} m | quat_xyzw={ee_quat_str}")
                    print(f"  EE Desired Pose (base): pos={ee_desired_pos_str} m | quat_xyzw={ee_desired_quat_str}")
                    print(f"  Base Plan: {base_plan_vel_str} | Base WBC: {base_wbc_vel_str} | Base Actual: {base_actual_vel_str}")
                    print(f"  Gripper Vel Desired: {gripper_vel_str} m/s")
                    print(f"  Gripper Vel WBC:     {gripper_vel_cmd_str} m/s")
                    print(f"  Gripper Vel Actual:  {gripper_vel_actual_str} m/s")
                    print(f"  WBC: {wbc_status} | Solve Time: {wbc_solve_time:.2f} ms")

            # ---- Integrate robot model at simulation rate ----------------
            robot.integrate(qd, model.opt.timestep)

            # ---- Apply actuator commands to MuJoCo ----------------------
            # Wheel velocity actuators (angular velocity, rad/s)
            # qd[0]=wz, qd[1]=vx
            wheel_velocity = wheel_controller.ctrl(qd[1], 0.0, qd[0])
            data.ctrl[:2] = wheel_velocity

            # Arm position actuators: track integrated joint positions
            if direct_arm_control:
                data.ctrl[2:8] = arm_joint_cmd
            else:
                data.ctrl[2:8] = robot.q_pos[6:12]

            # Gripper position actuators (meters, 0=closed, 0.044=open)
            left_finger_target, right_finger_target = gripper_opening_to_joint_targets(gripper_ctrl)
            data.ctrl[8] = left_finger_target
            data.ctrl[9] = right_finger_target

            if attached_box_index is not None:
                R_gripper_step = data.site_xmat[ee_site_id].reshape(3, 3)
                t_gripper_step = data.site_xpos[ee_site_id].copy()
                T_gripper_step = mr.RpToTrans(R_gripper_step, t_gripper_step)
                T_box = T_gripper_step @ attached_box_T_go
                set_freejoint_pose(box_joint_ids[attached_box_index], T_box)

            mujoco.mj_step(model, data)
            viewer.sync()

            num += 1

            end_time = time.time()
            elapsed = end_time - start_time
            if model.opt.timestep - elapsed > 0:
                time.sleep(model.opt.timestep - elapsed)