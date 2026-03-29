"""
交互式机器人控制脚本
- 显示固定的机器人模型
- 通过GUI滑块调节机械臂关节角度和底盘运动
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import mujoco
import mujoco.viewer
from src.wheel_controller import DifferentialDriveWheelController

# GUI library
try:
    import tkinter as tk
    from tkinter import Scale, Button, Label, Frame, HORIZONTAL
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False
    print("Warning: tkinter not available, GUI will not work")


class RobotInteractiveController:
    """交互式控制机器人"""
    
    def __init__(self):
        # ==================== MuJoCo Setup ====================
        scene_xml_path = (
            Path(__file__).parent.parent / 'robot_model' / 'mjcf' / 'world_scene.xml'
        )
        self.model = mujoco.MjModel.from_xml_path(str(scene_xml_path))
        self.data = mujoco.MjData(self.model)

        # Arm joint IDs
        self.arm_joint_names = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6']
        self.arm_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in self.arm_joint_names
        ]
        self.arm_qaddrs = [self.model.jnt_qposadr[jid] for jid in self.arm_joint_ids]

        # Gripper joint IDs
        self.gripper_joint_names = ['fl_joint8', 'fl_joint7']
        self.gripper_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in self.gripper_joint_names
        ]
        self.gripper_qaddrs = [self.model.jnt_qposadr[jid] for jid in self.gripper_joint_ids]

        # Wheel controller
        self.wheel_controller = DifferentialDriveWheelController(r=0.05, w=0.2)

        # ==================== Control Variables ====================
        # 6个机械臂关节角度 (rad)
        self.arm_q = np.array([0.0, -1.57, -1.57, 0, 0.0, 0.0])
        
        # 底盘运动参数
        self.vx = 0.0  # 前进速度 (m/s)
        self.wz = 0.0  # 转弯角速度 (rad/s)
        
        # 夹爪位置 (m, 0=closed, 0.0475=open)
        self.gripper_pos = 0.0

        # ==================== GUI Setup ====================
        if HAS_TKINTER:
            self.setup_gui()
        else:
            print("tkinter not available. Use keyboard in viewer:")
            print("  j1-j6: 1,2,3,4,5,6 keys to control joints")
            print("  Arrow up/down: vx")
            print("  Arrow left/right: wz")

    def setup_gui(self):
        """设置 Tkinter GUI"""
        self.root = tk.Tk()
        self.root.title("Robot Interactive Control")
        self.root.geometry("500x600")

        # ==================== Joint Controls ====================
        joint_frame = Frame(self.root)
        joint_frame.pack(padx=10, pady=10, fill="both", expand=True)

        Label(joint_frame, text="Arm Joint Angles (rad)", font=("Arial", 12, "bold")).pack()

        self.joint_scales = []
        self.joint_labels = []
        self.joint_entries = []
        
        for i, name in enumerate(self.arm_joint_names):
            # Get joint limits from model
            jid = self.arm_joint_ids[i]
            lower = self.model.jnt_range[jid][0]
            upper = self.model.jnt_range[jid][1]
            
            frame = Frame(joint_frame)
            frame.pack(fill="x", padx=5, pady=5)
            
            label = Label(frame, text=f"{name}: 0.00 rad", width=12)
            label.pack(side="left")
            self.joint_labels.append(label)
            
            scale = Scale(
                frame,
                from_=lower,
                to=upper,
                resolution=0.01,
                orient=HORIZONTAL,
                command=lambda val, idx=i: self.update_joint(idx, float(val))
            )
            scale.set(self.arm_q[i])
            scale.pack(side="left", fill="x", expand=True)
            self.joint_scales.append(scale)
            
            # 输入框用于直接输入角度
            entry = tk.Entry(frame, width=8)
            entry.insert(0, f"{self.arm_q[i]:.2f}")
            entry.pack(side="left", padx=5)
            entry.bind("<Return>", lambda evt, idx=i: self.input_joint(idx, evt))
            self.joint_entries.append(entry)

        # ==================== Base Motion Controls ====================
        base_frame = Frame(self.root)
        base_frame.pack(padx=10, pady=10, fill="both")

        Label(base_frame, text="Base Motion Control", font=("Arial", 12, "bold")).pack()

        # vx control
        vx_frame = Frame(base_frame)
        vx_frame.pack(fill="x", padx=5, pady=5)
        Label(vx_frame, text="vx (m/s):", width=15).pack(side="left")
        self.vx_label = Label(vx_frame, text="0.00")
        self.vx_label.pack(side="right", padx=5)
        self.vx_scale = Scale(
            vx_frame,
            from_=-1.0,
            to=1.0,
            resolution=0.01,
            orient=HORIZONTAL,
            command=self.update_vx
        )
        self.vx_scale.set(0.0)
        self.vx_scale.pack(side="left", fill="x", expand=True)

        # wz control
        wz_frame = Frame(base_frame)
        wz_frame.pack(fill="x", padx=5, pady=5)
        Label(wz_frame, text="wz (rad/s):", width=15).pack(side="left")
        self.wz_label = Label(wz_frame, text="0.00")
        self.wz_label.pack(side="right", padx=5)
        self.wz_scale = Scale(
            wz_frame,
            from_=-2.0,
            to=2.0,
            resolution=0.01,
            orient=HORIZONTAL,
            command=self.update_wz
        )
        self.wz_scale.set(0.0)
        self.wz_scale.pack(side="left", fill="x", expand=True)

        # Zero button for base motion
        Button(base_frame, text="Zero Velocity (vx=0, wz=0)", command=self.zero_base_velocity, width=30).pack(padx=5, pady=5)

        # ==================== Gripper Control ====================
        gripper_frame = Frame(self.root)
        gripper_frame.pack(padx=10, pady=10, fill="x")

        Label(gripper_frame, text="Gripper", font=("Arial", 12, "bold")).pack()

        button_frame = Frame(gripper_frame)
        button_frame.pack(fill="x", padx=5, pady=5)

        Button(button_frame, text="Close (0)", command=lambda: self.set_gripper(0.0), width=15).pack(side="left", padx=5)
        Button(button_frame, text="Open (0.0475)", command=lambda: self.set_gripper(0.0475), width=15).pack(side="left", padx=5)

        # Gripper slider
        gripper_scale_frame = Frame(gripper_frame)
        gripper_scale_frame.pack(fill="x", padx=5, pady=5)
        
        self.gripper_label = Label(gripper_scale_frame, text="Gripper: 0.000 m")
        self.gripper_label.pack(side="left")
        
        self.gripper_scale = Scale(
            gripper_scale_frame,
            from_=0.0,
            to=0.0475,
            resolution=0.001,
            orient=HORIZONTAL,
            command=self.update_gripper
        )
        self.gripper_scale.set(0.0)
        self.gripper_scale.pack(side="left", fill="x", expand=True)

        # ==================== Reset Button ====================
        Button(self.root, text="Reset to Initial Pose", command=self.reset_pose, height=2).pack(padx=10, pady=10, fill="x")

    def update_joint(self, idx, value):
        """更新关节角度"""
        self.arm_q[idx] = value
        if HAS_TKINTER:
            self.joint_labels[idx].config(text=f"{self.arm_joint_names[idx]}: {value:.2f} rad")
            self.joint_entries[idx].delete(0, tk.END)
            self.joint_entries[idx].insert(0, f"{value:.2f}")

    def input_joint(self, idx, event):
        """从输入框读取关节角度"""
        try:
            jid = self.arm_joint_ids[idx]
            lower = self.model.jnt_range[jid][0]
            upper = self.model.jnt_range[jid][1]
            
            value = float(self.joint_entries[idx].get())
            # 限制在范围内
            value = np.clip(value, lower, upper)
            
            self.arm_q[idx] = value
            self.joint_scales[idx].set(value)
            self.joint_labels[idx].config(text=f"{self.arm_joint_names[idx]}: {value:.2f} rad")
        except ValueError:
            self.joint_entries[idx].delete(0, tk.END)
            self.joint_entries[idx].insert(0, f"{self.arm_q[idx]:.2f}")

    def update_vx(self, value):
        """更新前进速度"""
        self.vx = float(value)
        if HAS_TKINTER:
            self.vx_label.config(text=f"{self.vx:.2f}")

    def update_wz(self, value):
        """更新转弯角速度"""
        self.wz = float(value)
        if HAS_TKINTER:
            self.wz_label.config(text=f"{self.wz:.2f}")

    def zero_base_velocity(self):
        """将底盘速度和角速度置零"""
        self.vx = 0.0
        self.wz = 0.0
        if HAS_TKINTER:
            self.vx_scale.set(0.0)
            self.wz_scale.set(0.0)
            self.vx_label.config(text="0.00")
            self.wz_label.config(text="0.00")

    def update_gripper(self, value):
        """更新夹爪位置"""
        self.gripper_pos = float(value)
        if HAS_TKINTER:
            self.gripper_label.config(text=f"Gripper: {self.gripper_pos:.3f} m")

    def set_gripper(self, value):
        """设置夹爪到特定位置"""
        self.gripper_pos = value
        if HAS_TKINTER:
            self.gripper_scale.set(value)
            self.gripper_label.config(text=f"Gripper: {self.gripper_pos:.3f} m")

    def reset_pose(self):
        """重置到初始姿态"""
        self.arm_q = np.array([0.0, -1.57, 1.57, 0.0, 0.0, 0.0])
        self.vx = 0.0
        self.wz = 0.0
        self.gripper_pos = 0.0
        
        if HAS_TKINTER:
            for i, scale in enumerate(self.joint_scales):
                scale.set(self.arm_q[i])
            self.vx_scale.set(0.0)
            self.wz_scale.set(0.0)
            self.gripper_scale.set(0.0)

    def update_mujoco_state(self):
        """更新 MuJoCo 的状态"""
        # 设置关节角度
        for addr, q in zip(self.arm_qaddrs, self.arm_q):
            self.data.qpos[addr] = q

        # 设置夹爪位置 (遵循 equality 约束: q8 = -q7)
        # fl_joint8 (拇指): range [-0.0475, 0]
        # fl_joint7 (食指): range [0, 0.0475]
        self.data.qpos[self.gripper_qaddrs[0]] = -self.gripper_pos  # q8 = -gripper_pos
        self.data.qpos[self.gripper_qaddrs[1]] = self.gripper_pos   # q7 = gripper_pos

        # 计算轮子速度
        wheel_velocity = self.wheel_controller.ctrl(self.vx, 0.0, self.wz)
        
        # 设置执行器命令
        self.data.ctrl[:2] = wheel_velocity      # 轮子速度
        self.data.ctrl[2:8] = self.arm_q         # 机械臂位置目标
        self.data.ctrl[8] = -self.gripper_pos    # left_finger_ctrl (q8)
        self.data.ctrl[9] = self.gripper_pos     # right_finger_ctrl (q7)

    def run(self):
        """运行控制循环"""
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # 启动 GUI
            if HAS_TKINTER:
                def gui_update():
                    try:
                        self.root.update_idletasks()
                        self.root.update()
                    except:
                        pass
                    # 继续更新
                    viewer.user_scn.ngeom = 0

            while viewer.is_running():
                # 更新 MuJoCo 状态
                self.update_mujoco_state()

                # 更新 GUI
                if HAS_TKINTER:
                    gui_update()

                # 前向动力学计算
                mujoco.mj_forward(self.model, self.data)
                mujoco.mj_step(self.model, self.data)
                
                viewer.sync()

        if HAS_TKINTER:
            self.root.quit()


if __name__ == '__main__':
    controller = RobotInteractiveController()
    controller.run()