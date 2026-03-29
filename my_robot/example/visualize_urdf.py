#!/usr/bin/env python3
"""
PyBullet URDF 模型可视化脚本
修改 URDF_FILENAME 变量可视化不同的模型
"""

import pybullet as p
import pybullet_data
import time
import os

# ==================== 用户配置区 ====================
# 修改这个变量来可视化不同的 URDF 模型
URDF_FILENAME = "robot.urdf"  # 可选: "robot.urdf", "chassis.urdf" 等
# ====================================================


def main():
    """主函数"""
    # 连接到 PyBullet GUI
    client = p.connect(p.GUI)
    print("已连接到 PyBullet GUI")
    
    # 设置额外的搜索路径以查找网格文件
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # 获取脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # 构建 URDF 文件的完整路径
    urdf_path = os.path.join(project_root, "robot_model", "urdf", URDF_FILENAME)
    urdf_root_path = os.path.join(project_root, "robot_model", "urdf")
    
    # 检查文件是否存在
    if not os.path.exists(urdf_path):
        print(f"错误: URDF 文件不存在: {urdf_path}")
        print(f"请检查 URDF_FILENAME 变量是否设置正确")
        p.disconnect()
        return
    
    # 加载 URDF 模型，指定根路径以查找mesh文件
    try:
        robot_id = p.loadURDF(urdf_path, useMaximalCoordinates=False)
        print(f"成功加载 URDF 文件: {urdf_path}")
    except Exception as e:
        print(f"错误: 无法加载 URDF 文件")
        print(f"详细信息: {e}")
        p.disconnect()
        return
    
    # 设置相机位置以获得更好的视图
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0]
    )
    
    print("\n=== 控制说明 ===")
    print("鼠标: 旋转视图")
    print("滚轮: 缩放视图")
    print("右键拖动: 平移视图")
    print("按 Ctrl+C 或关闭窗口退出")
    print("=================\n")
    
    # 保持窗口打开
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n已中断")
    finally:
        p.disconnect()
        print("已断开连接")


if __name__ == "__main__":
    main()