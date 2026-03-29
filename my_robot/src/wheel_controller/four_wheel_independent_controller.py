import numpy as np

from .wheel_controller import WheelController


class FourWheelIndependentController(WheelController):
    """
    四轮独立驱动控制器
    
    机器人配置：
    - 左轮：(-lx, ly) 位置
    - 右轮：(-lx, -ly) 位置
    - 前轮：(lx, 0) 位置
    - 后轮：(-lx, 0) 位置
    
    其中 lx 是轴向距离，ly 是横向距离
    """

    def __init__(self, lx: float, ly: float):
        """
        初始化四轮独立驱动控制器
        
        :param lx: 轮子到机器人中心的轴向距离
        :param ly: 左右轮之间的一半距离
        """
        super().__init__()
        self._lx = lx  # 前后轴距的一半
        self._ly = ly  # 左右轮距的一半

    def ctrl(self, vx: float, vy: float, wz: float) -> np.ndarray:
        """
        根据基座速度计算各轮速度
        
        :param vx: 基座纵向速度
        :param vy: 基座横向速度
        :param wz: 基座角速度
        :return: 四个轮子的速度 [v_left, v_right, v_front, v_back]
        """
        # 运动学方程：
        # v_wheel = [vx, vy] + wz * r_perp
        # 其中 r_perp 是轮子相对于机器人中心的垂直向量

        # 左轮速度（位置：前方、左侧）
        v_left = vx - self._ly * wz

        # 右轮速度（位置：前方、右侧）
        v_right = vx + self._ly * wz

        # 前轮速度（位置：前方、中心）
        v_front = vy + self._lx * wz

        # 后轮速度（位置：后方、中心）
        v_back = vy - self._lx * wz

        return np.array([v_left, v_right, v_front, v_back])


class FourWheelMecanumController(WheelController):
    """
    四轮麦克纳姆轮控制器
    
    麦克纳姆轮配置（对称布置）：
    - 前左轮(FL)：位置 (lx, ly)，滚轮方向顺时针45°
    - 前右轮(FR)：位置 (lx, -ly)，滚轮方向逆时针45°
    - 后左轮(BL)：位置 (-lx, ly)，滚轮方向逆时针45°
    - 后右轮(BR)：位置 (-lx, -ly)，滚轮方向顺时针45°
    """

    def __init__(self, lx: float, ly: float, r: float):
        """
        初始化麦克纳姆轮控制器
        
        :param lx: 轮子到机器人中心的轴向距离
        :param ly: 轮子到机器人中心的横向距离
        :param r: 轮子半径
        """
        super().__init__()
        self._lx = lx
        self._ly = ly
        self._r = r
        
        # 计算逆运动学矩阵
        # v_wheel = M * v_base，其中 v_base = [vx, vy, wz]^T
        self._M_inv = np.array([
            [1, 1, self._lx + self._ly],
            [1, -1, -(self._lx + self._ly)],
            [1, -1, self._lx + self._ly],
            [1, 1, -(self._lx + self._ly)]
        ])

    def ctrl(self, vx: float, vy: float, wz: float) -> np.ndarray:
        """
        根据基座速度计算各麦克纳姆轮的速度
        
        :param vx: 基座纵向速度
        :param vy: 基座横向速度
        :param wz: 基座角速度 (rad/s)
        :return: 四个轮子的速度 [v_FL, v_FR, v_BL, v_BR] (m/s)
        """
        v_base = np.array([vx, vy, self._r * wz])
        
        # 计算各轮速度
        wheel_speeds = self._M_inv @ v_base
        
        # 归一化（防止速度超过限制）
        max_speed = np.max(np.abs(wheel_speeds))
        if max_speed > 1.0:
            wheel_speeds = wheel_speeds / max_speed
        
        return wheel_speeds