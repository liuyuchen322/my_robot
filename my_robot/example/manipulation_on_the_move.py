import sys
from pathlib import Path

# Add my_robot/ to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import numpy as np
import mujoco
import mujoco.viewer
import modern_robotics as mr

from src.robot import Robot
from src.wheel_controller import DifferentialDriveWheelController
from src.motion_controller import (
    BaseController,
    HighLevelController,
    ArmController,
    RedundancyResolutionController,
    GripperController,
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

    # Body / site IDs
    mobile_base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'frankie_base0')
    ee_site_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'end_effector_site')

    task_site_names = [
        'grasp_object1', 'drop_location1',
        'grasp_object2', 'drop_location2',
        'grasp_object3',
    ]
    task_site_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
        for n in task_site_names
    ]

    # ------------------------------------------------------------------ #
    # Initial MuJoCo state                                                 #
    # ------------------------------------------------------------------ #
    # Match the reference project with a stable FR5 home pose.
    init_arm_q = [0.0, -1.2, 1.8, 0-1.6, 1.57, 0.0]
    init_gripper = 0.044
    for addr, q in zip(arm_qaddrs, init_arm_q):
        data.qpos[addr] = q

    data.ctrl[2:8] = init_arm_q   # arm position actuators
    data.ctrl[8]   = -init_gripper
    data.ctrl[9]   = init_gripper
    mujoco.mj_forward(model, data)

    # ------------------------------------------------------------------ #
    # Pinocchio robot model                                                #
    # ------------------------------------------------------------------ #
    robot = Robot()

    def body_T(body_id):
        R = data.xmat[body_id].reshape(3, 3)
        t = data.xpos[body_id].copy()
        return mr.RpToTrans(R, t)

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

    T_tasks = [site_T(sid) for sid in task_site_ids]

    # ------------------------------------------------------------------ #
    # Controllers                                                          #
    # ------------------------------------------------------------------ #
    # Wheel: r=0.05 m (wheel radius), w=0.2 m (half axle width)
    wheel_controller = DifferentialDriveWheelController(r=0.05, w=0.2)

    high_level_controller = HighLevelController()
    for T in T_tasks:
        high_level_controller.add_point(T)

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

    # ------------------------------------------------------------------ #
    # Main loop                                                            #
    # ------------------------------------------------------------------ #
    num = 0
    qd          = np.zeros(robot.dof)   # 12-DOF velocity command
    gripper_ctrl = gripper_controller.get()  # finger opening [m]
    gripper_action = None               # None | "close" | "open"
    print_interval = 10  # Print logs every 50 control cycles (0.5 seconds at 100 Hz)
    last_state = -1  # Track last printed state

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

                # High-level target
                T_target, T_next = high_level_controller.ctrl()

                # Base trajectory controller → [vx, vy, wz]
                succeed, v_base, time_in, T_closest = base_controller.ctrl(
                    T_target, T_next, T_mobile_base, T_mobile_base[:3, :3]
                )

                T_gb = np.linalg.inv(T_mobile_base) @ T_gripper
                v_gripper_desired, v_base_desired, arm_state, d_gt = arm_controller.ctrl(
                    T_target, time_in, v_base, T_mobile_base, T_gb, T_closest
                )

                # Whole-body controller → 12-DOF qd
                qd, wbc_success, wbc_solve_time = redundancy_resolution_controller.ctrl(
                    v_gripper_desired, v_base_desired
                )

                # Gripper logic
                dist_to_target = np.linalg.norm(T_target[:3, 3] - t_gripper)
                if gripper_action == "close":
                    arrive, gripper_ctrl = gripper_controller.close()
                    if arrive:
                        high_level_controller.update()
                        T_gb = np.linalg.inv(T_mobile_base) @ T_gripper
                        arm_controller.reset(T_gb)
                        gripper_action = None
                elif gripper_action == "open":
                    arrive, gripper_ctrl = gripper_controller.open()
                    if arrive:
                        high_level_controller.update()
                        T_gb = np.linalg.inv(T_mobile_base) @ T_gripper
                        arm_controller.reset(T_gb)
                        gripper_action = None
                elif dist_to_target < 0.03:
                    if high_level_controller.current_id % 2 == 0:
                        # Even index: grasp (close gripper)
                        gripper_action = "close"
                        _, gripper_ctrl = gripper_controller.close()
                    else:
                        # Odd index: place (open gripper)
                        gripper_action = "open"
                        _, gripper_ctrl = gripper_controller.open()
                else:
                    gripper_ctrl = gripper_controller.get()

                # Print logs at regular intervals
                if num % (n_steps * print_interval) == 0:
                    state_names = {0: "Prepare", 1: "Motion", 2: "Final Phase"}
                    state_name = state_names.get(arm_state, "Unknown")
                    
                    # Get EE position and target position in base frame
                    ee_pos_in_base = T_gb[:3, 3]
                    target_pos_in_base = np.linalg.inv(T_mobile_base) @ np.r_[T_target[:3, 3], 1]
                    target_pos_in_base = target_pos_in_base[:3]
                    
                    # Format positions
                    ee_pos_str = f"[{ee_pos_in_base[0]:.4f}, {ee_pos_in_base[1]:.4f}, {ee_pos_in_base[2]:.4f}]"
                    target_pos_str = f"[{target_pos_in_base[0]:.4f}, {target_pos_in_base[1]:.4f}, {target_pos_in_base[2]:.4f}]"
                    
                    # Format base velocity [vx, vy, wz]
                    base_vel_str = f"[{v_base[0]:.4f}, {v_base[1]:.4f}, {v_base[2]:.4f}]"
                    
                    # Format gripper velocity [vx, vy, vz, wx, wy, wz]
                    gripper_vel_str = f"[{v_gripper_desired[0]:.4f}, {v_gripper_desired[1]:.4f}, {v_gripper_desired[2]:.4f}, {v_gripper_desired[3]:.4f}, {v_gripper_desired[4]:.4f}, {v_gripper_desired[5]:.4f}]"
                    
                    # Format WBC info
                    wbc_status = "SUCCESS" if wbc_success else "FAILED"
                    
                    print(f"[Control {num//n_steps:05d}] State: {arm_state} ({state_name})")
                    print(f"  EE Pos (base frame): {ee_pos_str} m | Target Pos (base frame): {target_pos_str} m | Distance: {d_gt:.4f} m")
                    print(f"  Base Vel: {base_vel_str} m/s | Gripper Vel: {gripper_vel_str} m/s")
                    print(f"  WBC: {wbc_status} | Solve Time: {wbc_solve_time:.2f} ms")

            # ---- Integrate robot model at simulation rate ----------------
            robot.integrate(qd, model.opt.timestep)

            # ---- Apply actuator commands to MuJoCo ----------------------
            # Wheel velocity actuators (angular velocity, rad/s)
            # qd[0]=vx (body), qd[1]=vy (body), qd[5]=wz
            wheel_velocity = wheel_controller.ctrl(qd[0], qd[1], qd[5])
            data.ctrl[:2] = wheel_velocity

            # Arm position actuators: track integrated joint positions
            data.ctrl[2:8] = robot.q_pos[6:12]

            # Gripper position actuators (meters, 0=closed, 0.044=open)
            data.ctrl[8] = -gripper_ctrl
            data.ctrl[9] = gripper_ctrl

            mujoco.mj_step(model, data)
            viewer.sync()

            num += 1

            end_time = time.time()
            elapsed = end_time - start_time
            if model.opt.timestep - elapsed > 0:
                time.sleep(model.opt.timestep - elapsed)