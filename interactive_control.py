
import pybullet as p
import time
import pybullet_data
import math
import threading
import numpy as np
import torch
import sys
from pathlib import Path

# 添加 lerobot 路径
sys.path.insert(0, str(Path(__file__).parent.parent / "lerobot"))

# 导入 lerobot 机器人控制模块
try:
    from lerobot.common.robot_devices.robots.configs import So101RobotConfig
    from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
    LEROBOT_AVAILABLE = True
except ImportError as e:
    LEROBOT_AVAILABLE = False
    print(f"警告: 无法导入 lerobot 模块，真实机械臂控制功能将不可用")
    print(f"错误详情: {e}")
    print(f"提示: 可能需要安装依赖，运行: cd lerobot && pip install -e .")

# ==================== 初始化 ====================
# 尝试使用 GUI 模式，如果失败则使用 DIRECT 模式
try:
    physicsClient = p.connect(p.GUI)
    print("✓ PyBullet GUI 模式已启动")
except Exception as e:
    print(f"⚠️  GUI 模式启动失败: {e}")
    print("切换到 DIRECT 模式（无可视化窗口）")
    physicsClient = p.connect(p.DIRECT)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)

# 加载场景
planeId = p.loadURDF("plane.urdf")
robotId = p.loadURDF("so101_new_calib.urdf", [0, 0, 0], 
                     p.getQuaternionFromEuler([0, 0, 0]), 
                     useFixedBase=True)

# 找到末端执行器
numJoints = p.getNumJoints(robotId)
endEffectorIndex = None
for i in range(numJoints):
    info = p.getJointInfo(robotId, i)
    linkName = info[12].decode('utf-8')
    if 'gripper_frame' in linkName.lower():
        endEffectorIndex = i
        break

if endEffectorIndex is None:
    endEffectorIndex = numJoints - 1

print("="*80)
print("机械臂交互式控制")
print("="*80)
print(f"末端执行器索引: {endEffectorIndex}")

# 获取当前末端位置
linkState = p.getLinkState(robotId, endEffectorIndex)
currentPos = linkState[0]
print(f"当前末端位置: [{currentPos[0]:.3f}, {currentPos[1]:.3f}, {currentPos[2]:.3f}]")

# 在当前位置放一个标记
sphereId = p.createVisualShape(p.GEOM_SPHERE, radius=0.01, rgbaColor=[0, 1, 0, 0.8])
currentMarker = p.createMultiBody(baseMass=0, baseVisualShapeIndex=sphereId, 
                                  basePosition=currentPos)

print("\n" + "="*80)
print("控制说明:")
print("="*80)
print("1. 输入目标坐标: x y z")
print("   例如: 0.30 0.10 0.25")
print("2. 输入 'circle' 画圆轨迹 ⭐ 新增")
print("   例如: circle 0.30 0.10 0.25 0.08  (半径8cm)")
print("3. 输入 'joint' 显示/设置关节角度")
print("   例如: joint 0 45.0  (设置关节0为45度)")
print("4. 输入 'current' 显示当前位置")
print("5. 输入 'home' 回到初始位置")
print("6. 输入 'connect' 连接真实机械臂")
print("7. 输入 'compare' 对比仿真和真实角度")
print("8. 输入 'help' 显示所有命令")
print("9. 输入 'quit' 退出")
print("="*80)

if LEROBOT_AVAILABLE:
    print("✓ LeRobot 模块已加载，可以连接真实机械臂")
else:
    print("✗ LeRobot 模块未加载，只能使用仿真模式")

# 保存的位置列表
savedPositions = []

# 真实机械臂连接
real_robot = None
use_real_robot = False

# 零点偏移补偿（如果 URDF 零点和真实机械臂零点不对应）
# 单位：度，格式：[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
ZERO_POINT_OFFSET = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# 存储最后一次IK解（用于角度对比）
last_ik_solution_deg = None

# 监控线程控制
monitor_running = False
monitor_thread = None

# 仿真线程
simulationRunning = True

def simulation_loop():
    """后台仿真线程"""
    global currentMarker
    while simulationRunning:
        p.stepSimulation()
        
        # 更新当前位置标记
        linkState = p.getLinkState(robotId, endEffectorIndex)
        currentPos = linkState[0]
        p.resetBasePositionAndOrientation(currentMarker, currentPos, [0, 0, 0, 1])
        
        time.sleep(1./240.)

# 启动仿真线程
sim_thread = threading.Thread(target=simulation_loop, daemon=True)
sim_thread.start()

def compare_angles():
    """
    对比IK解、仿真和真实机械臂的关节角度
    
    Returns:
        tuple: (ik_angles_deg, sim_angles_deg, real_angles_deg, errors_deg, joint_names) 或 None
    """
    global real_robot, use_real_robot, last_ik_solution_deg
    
    if not use_real_robot or real_robot is None:
        print("⚠️  真实机械臂未连接")
        return None
    
    try:
        # 读取仿真关节角度
        sim_angles_rad = []
        joint_names = []
        for i in range(min(6, numJoints)):
            info = p.getJointInfo(robotId, i)
            jointName = info[1].decode('utf-8')
            jointType = info[2]
            if jointType == p.JOINT_REVOLUTE:
                state = p.getJointState(robotId, i)
                sim_angles_rad.append(state[0])
                joint_names.append(jointName)
        
        sim_angles_deg = np.rad2deg(sim_angles_rad)
        
        # 读取真实机械臂角度
        real_angles_deg_all = None
        for name in real_robot.follower_arms:
            real_angles_deg_all = real_robot.follower_arms[name].read("Present_Position")
            break  # 只有一个 follower arm
        
        if real_angles_deg_all is None:
            print("❌ 无法读取真实机械臂角度")
            return None
        
        # 只取前5个关节（排除gripper），与仿真的关节数匹配
        num_joints_to_compare = len(sim_angles_deg)
        real_angles_deg = real_angles_deg_all[:num_joints_to_compare]
        
        # 获取IK解角度（如果有的话）
        ik_angles_deg = None
        if last_ik_solution_deg is not None:
            ik_angles_deg = last_ik_solution_deg[:num_joints_to_compare]
        
        # 计算误差（真实 - 仿真）
        errors_deg = real_angles_deg - sim_angles_deg
        
        return ik_angles_deg, sim_angles_deg, real_angles_deg, errors_deg, joint_names
        
    except Exception as e:
        print(f"❌ 对比失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_angle_comparison(show_header=True):
    """打印角度对比表格"""
    result = compare_angles()
    if result is None:
        return
    
    ik_angles, sim_angles, real_angles, errors, joint_names = result
    
    if show_header:
        print("\n" + "="*100)
        print("关节角度对比（IK解 → 仿真 → 真实）")
        print("="*100)
    
    # 根据是否有IK解决定表头
    if ik_angles is not None:
        print(f"\n{'关节名称':20s} | {'IK解':>10s} | {'仿真角度':>10s} | {'真实角度':>10s} | {'误差(真-仿)':>12s} | 状态")
        print("-" * 100)
        
        for i, name in enumerate(joint_names):
            ik = ik_angles[i]
            sim = sim_angles[i]
            real = real_angles[i]
            error = errors[i]
            
            # 根据误差大小显示状态
            if abs(error) < 2.0:
                status = "✓ 优秀"
            elif abs(error) < 5.0:
                status = "○ 良好"
            elif abs(error) < 15.0:
                status = "△ 可接受"
            else:
                status = "✗ 偏差大"
            
            print(f"{name:20s} | {ik:>9.2f}° | {sim:>9.2f}° | {real:>9.2f}° | {error:>+11.2f}° | {status}")
    else:
        print(f"\n{'关节名称':20s} | {'仿真角度':>10s} | {'真实角度':>10s} | {'误差(真-仿)':>12s} | 状态")
        print("-" * 100)
        print("(暂无IK解数据，请先执行一次移动命令)")
        print("-" * 100)
        
        for i, name in enumerate(joint_names):
            sim = sim_angles[i]
            real = real_angles[i]
            error = errors[i]
            
            # 根据误差大小显示状态
            if abs(error) < 2.0:
                status = "✓ 优秀"
            elif abs(error) < 5.0:
                status = "○ 良好"
            elif abs(error) < 15.0:
                status = "△ 可接受"
            else:
                status = "✗ 偏差大"
            
            print(f"{name:20s} | {sim:>9.2f}° | {real:>9.2f}° | {real:>9.2f}° | {error:>+11.2f}° | {status}")
    
    # 统计信息
    mean_error = np.mean(np.abs(errors))
    max_error = np.max(np.abs(errors))
    rms_error = np.sqrt(np.mean(errors**2))
    
    print("-" * 100)
    print(f"平均误差: {mean_error:6.2f}°  |  最大误差: {max_error:6.2f}°  |  RMS误差: {rms_error:6.2f}°")
    
    if show_header:
        print("="*100)
        print("\n💡 提示:")
        if max_error > 15.0:
            print("  ⚠️  检测到较大偏差，可能需要调整零点偏移")
            print("  建议: 使用 'calibrate' 命令自动计算偏移量")
        elif max_error > 5.0:
            print("  △ 存在一定偏差，可以使用 'set_offset' 命令微调")
        else:
            print("  ✓ 角度对应良好！")
    
    return result


def show_raw_steps():
    """
    实时显示真实机械臂的原始步进值（未校准）
    按回车保存当前值
    """
    global real_robot, use_real_robot
    
    if not use_real_robot or real_robot is None:
        print("⚠️  需要先连接真实机械臂")
        return
    
    import sys
    import json
    from datetime import datetime
    
    # Windows 兼容性检查
    is_windows = sys.platform == 'win32'
    if not is_windows:
        import select
    else:
        # Windows 使用 msvcrt 实现非阻塞输入
        try:
            import msvcrt
        except ImportError:
            print("⚠️  Windows 系统需要 msvcrt 模块")
            return
    
    print("\n" + "="*80)
    print("实时显示原始步进值（未校准）")
    print("="*80)
    print("提示：")
    print("  - 实时刷新显示舵机的原始步进值")
    print("  - 为了方便手动调整，将自动禁用舵机力矩")
    print("  - 按 Enter 保存当前值到文件")
    if is_windows:
        print("  - 按 's' 键保存，按 'q' 键退出")
    else:
        print("  - 按 Ctrl+C 或输入 'q' 退出")
    print("="*80 + "\n")
    
    # 禁用舵机力矩（让机械臂可以手动移动）
    print("正在禁用舵机力矩...")
    torque_states = {}  # 保存原始状态
    try:
        for name in real_robot.follower_arms:
            # 保存当前力矩状态
            torque_states[name] = real_robot.follower_arms[name].read("Torque_Enable")
            # 禁用力矩
            real_robot.follower_arms[name].write("Torque_Enable", 0)
            print(f"  ✓ 已禁用 {name} 力矩 - 现在可以手动移动机械臂")
    except Exception as e:
        print(f"⚠️  禁用力矩时出错: {e}")
    
    print("\n💡 现在可以手动移动机械臂到想要的姿势，然后按键保存")
    input("按 Enter 开始...")
    
    saved_positions = []
    
    try:
        while True:
            try:
                # 清屏
                print("\033[2J\033[H", end="")
                
                print("="*80)
                print("原始步进值实时显示（每 0.3 秒刷新）")
                print("="*80)
                print("\n按 Enter 保存 | 输入 'q' + Enter 退出\n")
                
                # 读取原始步进值（不经过校准）
                for name in real_robot.follower_arms:
                    # 直接读取原始值，不使用校准
                    motor_bus = real_robot.follower_arms[name]
                    
                    # 读取原始步进值
                    raw_values = []
                    for motor_name in motor_bus.motor_names:
                        try:
                            motor_idx, motor_model = motor_bus.motors[motor_name]
                            raw_val = motor_bus.read_with_motor_ids(
                                [motor_model], 
                                [motor_idx], 
                                "Present_Position"
                            )
                            # read_with_motor_ids 返回列表，取第一个元素
                            if isinstance(raw_val, list):
                                raw_values.append(raw_val[0])
                            else:
                                raw_values.append(raw_val)
                        except Exception as e:
                            print(f"⚠️  读取 {motor_name} 失败: {e}")
                            raw_values.append(0)  # 添加默认值
                    
                    # 同时读取校准后的角度值
                    try:
                        calibrated_angles = motor_bus.read("Present_Position")
                    except Exception as e:
                        print(f"⚠️  读取校准后角度失败: {e}")
                        calibrated_angles = [0.0] * len(raw_values)
                    
                    # 打印表头和数据
                    print(f"{'索引':>4s} | {'舵机名称':20s} | {'原始步进值':>12s} | {'校准后角度':>12s}")
                    print("-" * 80)
                    
                    for i, motor_name in enumerate(motor_bus.motor_names):
                        raw_val = raw_values[i]
                        calib_val = calibrated_angles[i]
                        
                        # 确保 raw_val 是整数（不是列表）
                        if isinstance(raw_val, (list, tuple)):
                            raw_val = int(raw_val[0])
                        else:
                            raw_val = int(raw_val)
                        
                        # 判断是角度还是线性（gripper）
                        if motor_name == "gripper":
                            unit = "%"
                        else:
                            unit = "°"
                        
                        print(f"{i:4d} | {motor_name:20s} | {raw_val:12d} | {calib_val:11.2f}{unit}")
                    
                    print("-" * 80)
                    print(f"\n已保存 {len(saved_positions)} 个位置")
                    
                    if is_windows:
                        print("\n💡 提示：按 's' 保存当前位置，按 'q' 退出")
                    else:
                        print("\n💡 提示：按 Enter 保存当前位置，输入 'q' + Enter 退出")
                    
                    # 检查是否有输入（非阻塞）
                    should_save = False
                    should_quit = False
                    
                    if not is_windows:
                        # Unix/Linux/Mac - 使用 select
                        if select.select([sys.stdin], [], [], 0.3)[0]:
                            user_input = sys.stdin.readline().strip()
                            if user_input.lower() == 'q':
                                should_quit = True
                            else:
                                # 按了 Enter，保存当前位置
                                should_save = True
                    else:
                        # Windows - 使用 msvcrt.kbhit()
                        if msvcrt.kbhit():
                            key = msvcrt.getch()
                            if key == b'q' or key == b'Q':
                                should_quit = True
                            elif key == b's' or key == b'S' or key == b'\r':
                                # 's' 键或 Enter 键保存
                                should_save = True
                        time.sleep(0.3)
                    
                    # 处理保存
                    if should_save:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # 确保 raw_values 是整数列表
                        raw_steps_int = []
                        for v in raw_values:
                            if isinstance(v, (list, tuple)):
                                raw_steps_int.append(int(v[0]))
                            else:
                                raw_steps_int.append(int(v))
                        
                        position_data = {
                            "timestamp": timestamp,
                            "raw_steps": raw_steps_int,
                            "calibrated_angles": [float(v) for v in calibrated_angles],
                            "motor_names": motor_bus.motor_names
                        }
                        saved_positions.append(position_data)
                        
                        # 立即保存到文件（增量保存）
                        filename = f"saved_raw_positions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        with open(filename, 'w', encoding='utf-8') as f:
                            json.dump(saved_positions, f, indent=2, ensure_ascii=False)
                        
                        # 显示保存成功消息
                        print("\033[2J\033[H", end="")  # 清屏
                        print("="*80)
                        print(f"✅ 已保存位置 #{len(saved_positions)}")
                        print("="*80)
                        print(f"文件: {filename}")
                        print(f"时间: {timestamp}")
                        print("\n保存的数据:")
                        print(f"  原始步进值: {raw_steps_int}")
                        print(f"  校准后角度: {[f'{v:.2f}' for v in calibrated_angles]}")
                        print("="*80)
                        time.sleep(1.5)
                
                    # 处理退出
                    if should_quit:
                        raise KeyboardInterrupt
            
            except KeyboardInterrupt:
                raise  # 重新抛出以便外层捕获
            except Exception as e:
                print(f"\n⚠️  显示数据时出错: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\n停止显示")
    
    finally:
        # 恢复舵机力矩状态
        print("\n正在恢复舵机力矩状态...")
        try:
            for name in real_robot.follower_arms:
                if name in torque_states:
                    # 恢复原始力矩状态（通常是1-启用）
                    # 但为了安全，默认保持禁用状态
                    # 用户可以通过其他命令重新启用
                    print(f"  ℹ️  {name} 力矩保持禁用状态（安全考虑）")
                    print(f"     如需启用力矩，请手动发送控制命令")
        except Exception as e:
            print(f"⚠️  恢复力矩状态时出错: {e}")
        
        # 如果有保存的位置，显示摘要
        if saved_positions:
            print(f"\n" + "="*80)
            print(f"📊 保存摘要")
            print("="*80)
            print(f"共保存了 {len(saved_positions)} 个位置")
            
            # 最终保存文件
            filename = f"saved_raw_positions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(saved_positions, f, indent=2, ensure_ascii=False)
            
            print(f"文件: {filename}")
            print("\n保存的位置:")
            for i, pos in enumerate(saved_positions, 1):
                print(f"  {i}. {pos['timestamp']}")
            print("="*80)
        else:
            print("\n未保存任何位置")
        
        print("\n按 Enter 继续...")
        input()


def monitor_angles_loop():
    """持续监控角度对比（后台线程）"""
    global monitor_running
    
    print("\n开始实时监控...")
    print("按 Ctrl+C 或输入 'stop_monitor' 停止监控\n")
    
    while monitor_running:
        # 清屏（使用 ANSI 转义码）
        print("\033[2J\033[H", end="")
        
        print("="*100)
        print("实时角度监控（每 0.5 秒刷新）")
        print("="*100)
        
        result = compare_angles()
        if result:
            ik_angles, sim_angles, real_angles, errors, joint_names = result
            
            if ik_angles is not None:
                print(f"\n{'关节名称':20s} | {'IK解':>10s} | {'仿真角度':>10s} | {'真实角度':>10s} | {'误差(真-仿)':>12s} | 状态")
                print("-" * 100)
                
                for i, name in enumerate(joint_names):
                    ik = ik_angles[i]
                    sim = sim_angles[i]
                    real = real_angles[i]
                    error = errors[i]
                    
                    if abs(error) < 2.0:
                        status = "✓ 优秀"
                    elif abs(error) < 5.0:
                        status = "○ 良好"
                    elif abs(error) < 15.0:
                        status = "△ 可接受"
                    else:
                        status = "✗ 偏差大"
                    
                    print(f"{name:20s} | {ik:>9.2f}° | {sim:>9.2f}° | {real:>9.2f}° | {error:>+11.2f}° | {status}")
            else:
                print(f"\n{'关节名称':20s} | {'仿真角度':>10s} | {'真实角度':>10s} | {'误差(真-仿)':>12s} | 状态")
                print("-" * 100)
                
                for i, name in enumerate(joint_names):
                    sim = sim_angles[i]
                    real = real_angles[i]
                    error = errors[i]
                    
                    if abs(error) < 2.0:
                        status = "✓ 优秀"
                    elif abs(error) < 5.0:
                        status = "○ 良好"
                    elif abs(error) < 15.0:
                        status = "△ 可接受"
                    else:
                        status = "✗ 偏差大"
                    
                    print(f"{name:20s} | {sim:>9.2f}° | {real:>9.2f}° | {error:>+11.2f}° | {status}")
            
            mean_error = np.mean(np.abs(errors))
            max_error = np.max(np.abs(errors))
            rms_error = np.sqrt(np.mean(errors**2))
            
            print("-" * 100)
            print(f"平均误差: {mean_error:6.2f}°  |  最大误差: {max_error:6.2f}°  |  RMS误差: {rms_error:6.2f}°")
        
        print("\n按 Ctrl+C 停止监控")
        
        time.sleep(0.5)


def write_to_real_robot(joint_angles_rad):
    """
    将 PyBullet IK 解写入真实机械臂
    参照 manipulator.py 的 teleop_step 方法中的写入逻辑
    
    ============================================================================
    单位转换流程（重要！）：
    ============================================================================
    1. PyBullet IK 输出：弧度（rad），范围约 [-π, π]
    2. 转换为度数：np.rad2deg() → [-180°, 180°]
    3. 输入到 LeRobot：度数（float32）
    4. LeRobot 自动转换：度数 → 舵机步进值（0-4095）
       - 这个转换由 feetech.py 的 revert_calibration() 自动完成
       - 前提：必须有校准文件（robot.connect() 时自动加载）
    5. 发送到舵机：步进值
    
    ============================================================================
    关键点：
    ============================================================================
    - 有校准模式：write("Goal_Position", degrees) → 自动转换为步进值
    - 无校准模式：write("Goal_Position", steps) → 直接发送步进值
    - 当前使用有校准模式，所以输入必须是度数！
    
    Args:
        joint_angles_rad: PyBullet IK 解（弧度），numpy array 或 list
    
    Returns:
        bool: 是否成功写入
    """
    global real_robot, use_real_robot, last_ik_solution_deg
    
    if not use_real_robot or real_robot is None:
        print("⚠️  真实机械臂未连接，跳过写入")
        return False
    
    try:
        # ========================================================================
        # 步骤 1：弧度 → 度数
        # ========================================================================
        # PyBullet IK 输出弧度，LeRobot 有校准模式接受度数
        # 只取前6个关节（不包括夹爪）
        joint_angles_deg = np.rad2deg(joint_angles_rad[:6])
        
        # 应用零点偏移补偿
        joint_angles_deg = joint_angles_deg + ZERO_POINT_OFFSET
        
        # 保存IK解（用于后续角度对比）
        globals()['last_ik_solution_deg'] = joint_angles_deg.copy()
        
        print(f"\n将IK解写入真实机械臂...")
        print(f"IK解（弧度）: {joint_angles_rad[:6]}")
        print(f"IK解（度数，偏移前）: {np.rad2deg(joint_angles_rad[:6])}")
        if np.any(ZERO_POINT_OFFSET != 0):
            print(f"零点偏移: {ZERO_POINT_OFFSET}")
            print(f"IK解（度数，偏移后）: {joint_angles_deg}")
        else:
            print(f"IK解（度数）: {joint_angles_deg}")
        
        # ========================================================================
        # 步骤 2-5：写入舵机（参照 manipulator.py line 509-527）
        # ========================================================================
        for name in real_robot.follower_arms:
            before_write_t = time.perf_counter()
            
            # 步骤 2：转换为 PyTorch tensor（与 manipulator.py 保持一致）
            goal_pos = torch.from_numpy(joint_angles_deg).float()
            
            # 步骤 3：可选的安全检查 - 限制单步移动幅度
            # 如果配置了 max_relative_target，限制移动幅度以避免突然的大幅移动
            if real_robot.config.max_relative_target is not None:
                # 读取当前位置（度数）
                present_pos = real_robot.follower_arms[name].read("Present_Position")
                present_pos = torch.from_numpy(present_pos)
                
                # 使用 manipulator.py 中的安全检查函数
                from lerobot.common.robot_devices.robots.manipulator import ensure_safe_goal_position
                goal_pos = ensure_safe_goal_position(
                    goal_pos, present_pos, real_robot.config.max_relative_target
                )
                print(f"  安全检查：限制单步移动不超过 {real_robot.config.max_relative_target}°")
            
            # 步骤 4：转换为 numpy float32 类型（与 manipulator.py line 526 一致）
            goal_pos_np = goal_pos.numpy().astype(np.float32)
            # 注意：此时仍然是度数，不是步进值！
            
            # 步骤 5：写入目标位置（与 manipulator.py line 527 一致）
            # LeRobot 内部会自动调用 revert_calibration() 将度数转换为步进值
            # 详见 feetech.py line 1181-1182
            real_robot.follower_arms[name].write("Goal_Position", goal_pos_np)
            
            write_time = time.perf_counter() - before_write_t
            print(f"✓ 写入 {name} follower 臂（度数 → 自动转换为步进值）")
            print(f"  耗时: {write_time*1000:.2f}ms")
        
        print("✅ 成功写入真实机械臂")
        print("\n提示：LeRobot 已自动将度数转换为舵机步进值（0-4095）")
        return True
        
    except Exception as e:
        print(f"❌ 写入真实机械臂失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def set_joint_angles(joint_angles_deg, joint_indices=None):
    """
    直接设置关节角度（不使用IK）
    
    Args:
        joint_angles_deg: 关节角度（度数），numpy array 或 list
        joint_indices: 要设置的关节索引列表，如果为None则设置所有关节
    
    Returns:
        bool: 是否成功
    """
    global use_real_robot, last_ik_solution_deg
    
    try:
        # 转换为numpy数组
        joint_angles_deg = np.array(joint_angles_deg, dtype=float)
        
        # 如果没有指定关节索引，设置所有可旋转关节
        if joint_indices is None:
            joint_indices = []
            for i in range(min(6, numJoints)):
                info = p.getJointInfo(robotId, i)
                jointType = info[2]
                if jointType == p.JOINT_REVOLUTE:
                    joint_indices.append(i)
        
        # 确保角度数量匹配
        if len(joint_angles_deg) != len(joint_indices):
            print(f"❌ 角度数量({len(joint_angles_deg)})与关节数量({len(joint_indices)})不匹配")
            return False
        
        # 转换为弧度
        joint_angles_rad = np.deg2rad(joint_angles_deg)
        
        print(f"\n设置关节角度...")
        print(f"关节索引: {joint_indices}")
        print(f"目标角度（度数）: {joint_angles_deg}")
        
        # ========================================================================
        # 写入真实机械臂
        # ========================================================================
        if use_real_robot:
            # 扩展为6个关节的完整角度数组（使用当前角度填充未指定的关节）
            full_angles_deg = np.zeros(6)
            
            # 读取当前角度作为基础
            for name in real_robot.follower_arms:
                current_angles = real_robot.follower_arms[name].read("Present_Position")
                full_angles_deg = current_angles[:6]
                break
            
            # 更新指定的关节角度
            for idx, joint_idx in enumerate(joint_indices):
                if joint_idx < 6:
                    full_angles_deg[joint_idx] = joint_angles_deg[idx]
            
            # 应用零点偏移
            full_angles_deg = full_angles_deg + ZERO_POINT_OFFSET
            
            # 保存IK解（用于后续角度对比）
            globals()['last_ik_solution_deg'] = full_angles_deg.copy()
            
            print(f"\n将角度写入真实机械臂...")
            print(f"完整角度（含偏移）: {full_angles_deg}")
            
            for name in real_robot.follower_arms:
                goal_pos = torch.from_numpy(full_angles_deg).float()
                goal_pos_np = goal_pos.numpy().astype(np.float32)
                real_robot.follower_arms[name].write("Goal_Position", goal_pos_np)
                print(f"✓ 已写入 {name} follower 臂")
            
            # 等待真实机械臂移动
            time.sleep(1.0)
        
        # ========================================================================
        # 设置仿真
        # ========================================================================
        for idx, joint_idx in enumerate(joint_indices):
            p.setJointMotorControl2(
                robotId,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=joint_angles_rad[idx],
                force=500
            )
        
        # 等待仿真到达
        print("移动中", end="", flush=True)
        for _ in range(50):
            time.sleep(0.05)
            print(".", end="", flush=True)
        print(" ✓")
        
        # 显示角度对比
        if use_real_robot:
            print("\n" + "-"*100)
            print("移动后角度对比:")
            print("-"*100)
            print_angle_comparison(show_header=False)
        
        return True
        
    except Exception as e:
        print(f"❌ 设置关节角度失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def draw_circle(center, radius=0.05, steps=50, threshold=0.020, draw_trajectory=True):
    """
    让机械臂画圆
    
    Args:
        center: 圆心位置 [x, y, z]
        radius: 半径（米），默认0.05米=5厘米
        steps: 圆周上的点数，默认50
        threshold: 误差阈值（米），默认0.020米=20毫米
        draw_trajectory: 是否绘制轨迹线，默认True
    
    Returns:
        dict: 统计信息 {'success': 成功点数, 'fail': 失败点数, 'total': 总点数}
    """
    print(f"\n" + "="*80)
    print("开始画圆")
    print("="*80)
    print(f"圆心位置: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
    print(f"半径: {radius:.3f}米 ({radius*100:.1f}厘米)")
    print(f"点数: {steps}")
    print(f"误差阈值: {threshold:.3f}米 ({threshold*1000:.0f}毫米)")
    print("="*80 + "\n")
    
    prevActualPos = None
    successCount = 0
    failCount = 0
    
    # 存储轨迹点（用于后续分析）
    trajectory_points = []
    
    for step in range(steps):
        # 计算圆周上的目标点（在XY平面上画圆）
        angle = 2 * math.pi * step / steps
        targetPos = [
            center[0] + radius * math.cos(angle),
            center[1] + radius * math.sin(angle),
            center[2]  # Z保持不变
        ]
        
        # 在目标点放置标记（红色小球，每5个点放一个）
        if draw_trajectory and step % 5 == 0:
            sphereId = p.createVisualShape(p.GEOM_SPHERE, radius=0.003, 
                                          rgbaColor=[1, 0, 0, 0.5])
            p.createMultiBody(baseMass=0, baseVisualShapeIndex=sphereId, 
                            basePosition=targetPos)
        
        # 计算IK
        jointPoses = p.calculateInverseKinematics(
            robotId, 
            endEffectorIndex, 
            targetPos,
            maxNumIterations=1000,
            residualThreshold=0.0001
        )
        
        # 写入真实机械臂
        if use_real_robot:
            write_to_real_robot(jointPoses)
        
        # 控制仿真机械臂移动
        for i in range(len(jointPoses)):
            p.setJointMotorControl2(
                bodyIndex=robotId,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=jointPoses[i],
                force=500,
                maxVelocity=5
            )
        
        # 等待机械臂到达目标
        maxWaitSteps = 100  # 减少等待步数，加快画圆速度
        reached = False
        
        for waitStep in range(maxWaitSteps):
            time.sleep(0.01)  # 10ms
            
            # 检查是否到达
            linkState = p.getLinkState(robotId, endEffectorIndex)
            actualPos = linkState[0]
            error = math.sqrt(sum((actualPos[i] - targetPos[i])**2 for i in range(3)))
            
            if error < threshold:
                reached = True
                
                # 画实际轨迹（绿色线）
                if draw_trajectory and prevActualPos is not None:
                    p.addUserDebugLine(prevActualPos, actualPos, [0, 1, 0], 5, lifeTime=0)
                prevActualPos = actualPos
                
                successCount += 1
                trajectory_points.append({
                    'target': targetPos,
                    'actual': actualPos,
                    'error': error
                })
                
                if step % 10 == 0:
                    print(f"进度: {step+1}/{steps} ({100*(step+1)/steps:.1f}%) - "
                          f"到达 (误差{error*1000:.1f}mm)")
                break
        
        if not reached:
            linkState = p.getLinkState(robotId, endEffectorIndex)
            actualPos = linkState[0]
            error = math.sqrt(sum((actualPos[i] - targetPos[i])**2 for i in range(3)))
            
            # 即使未完全到达，也画轨迹（黄色线）
            if draw_trajectory and prevActualPos is not None:
                p.addUserDebugLine(prevActualPos, actualPos, [1, 1, 0], 3, lifeTime=0)
            prevActualPos = actualPos
            
            failCount += 1
            trajectory_points.append({
                'target': targetPos,
                'actual': actualPos,
                'error': error
            })
            
            print(f"进度: {step+1}/{steps} - 超时 (误差{error*1000:.1f}mm)")
    
    # 完成统计
    print(f"\n" + "="*80)
    print("画圆完成！")
    print("="*80)
    print(f"成功: {successCount}/{steps} ({100*successCount/steps:.1f}%)")
    print(f"失败: {failCount}/{steps} ({100*failCount/steps:.1f}%)")
    
    # 计算平均误差
    if trajectory_points:
        avg_error = sum(p['error'] for p in trajectory_points) / len(trajectory_points)
        max_error = max(p['error'] for p in trajectory_points)
        print(f"平均误差: {avg_error*1000:.1f}毫米")
        print(f"最大误差: {max_error*1000:.1f}毫米")
    
    if draw_trajectory:
        print("\n图例:")
        print("  🟢 绿色粗线 = 成功到达的轨迹")
        print("  🟡 黄色细线 = 未完全到达的轨迹")
        print("  🔴 红色小球 = 目标点标记")
    print("="*80 + "\n")
    
    return {
        'success': successCount,
        'fail': failCount,
        'total': steps,
        'trajectory': trajectory_points
    }


def move_to_position(targetPos, waitTime=2.0):
    """移动到目标位置（使用迭代式IK提高精度）"""
    print(f"\n目标位置: [{targetPos[0]:.3f}, {targetPos[1]:.3f}, {targetPos[2]:.3f}]")
    
    # 在目标位置放标记
    targetSphereId = p.createVisualShape(p.GEOM_SPHERE, radius=0.008, 
                                         rgbaColor=[1, 0, 0, 0.6])
    targetMarker = p.createMultiBody(baseMass=0, baseVisualShapeIndex=targetSphereId, 
                                     basePosition=targetPos)
    
    # 改进的IK：迭代式求解，逐步逼近目标
    print("计算IK（迭代优化中...）", end="", flush=True)
    
    # 第一次IK
    jointPoses = p.calculateInverseKinematics(
        robotId, 
        endEffectorIndex, 
        targetPos,
        maxNumIterations=1000,      # 增加迭代次数
        residualThreshold=0.0001    # 降低收敛阈值
    )
    
    # 迭代式IK：多次调用，每次用当前结果作为起点
    for iteration in range(5):  # 最多5次迭代
        # 设置当前关节角度
        for i in range(len(jointPoses)):
            p.resetJointState(robotId, i, jointPoses[i])
        
        # 检查当前误差
        linkState = p.getLinkState(robotId, endEffectorIndex)
        currentPos = linkState[0]
        error = math.sqrt(sum((currentPos[i] - targetPos[i])**2 for i in range(3)))
        
        if error < 0.001:  # 误差<1mm，足够好
            print(f" ✓ (迭代{iteration+1}次)")
            break
        
        # 继续优化
        jointPoses = p.calculateInverseKinematics(
            robotId, 
            endEffectorIndex, 
            targetPos,
            maxNumIterations=1000,
            residualThreshold=0.0001
        )
        print(".", end="", flush=True)
    else:
        print(f" (完成{iteration+1}次迭代)")
    
    # 验证IK质量
    for i in range(len(jointPoses)):
        p.resetJointState(robotId, i, jointPoses[i])
    linkState = p.getLinkState(robotId, endEffectorIndex)
    ikPos = linkState[0]
    ikError = math.sqrt(sum((ikPos[i] - targetPos[i])**2 for i in range(3)))
    print(f"IK求解误差: {ikError*1000:.2f}毫米")
    
    print("IK计算的关节角度:")
    for i in range(min(6, len(jointPoses))):
        info = p.getJointInfo(robotId, i)
        jointName = info[1].decode('utf-8')
        print(f"  关节{i} ({jointName:20s}): {math.degrees(jointPoses[i]):7.2f}°")
    
    # ========================================================================
    # 新增：将IK解写入真实机械臂
    # ========================================================================
    if use_real_robot:
        write_to_real_robot(jointPoses)
        
        # 等待机械臂移动
        time.sleep(1.0)
        
        # 自动显示角度对比
        print("\n" + "-"*80)
        print("移动后角度对比:")
        print("-"*80)
        print_angle_comparison(show_header=False)
    
    # 设置仿真控制目标
    for i in range(len(jointPoses)):
        p.setJointMotorControl2(
            bodyIndex=robotId,
            jointIndex=i,
            controlMode=p.POSITION_CONTROL,
            targetPosition=jointPoses[i],
            force=500,
            maxVelocity=5
        )
    
    # 等待到达
    print("移动中", end="", flush=True)
    startTime = time.time()
    reached = False
    
    while time.time() - startTime < waitTime:
        time.sleep(0.2)
        print(".", end="", flush=True)
        
        linkState = p.getLinkState(robotId, endEffectorIndex)
        actualPos = linkState[0]
        error = math.sqrt(sum((actualPos[i] - targetPos[i])**2 for i in range(3)))
        
        if error < 0.015:  # 15mm
            reached = True
            break
    
    # 检查最终位置
    linkState = p.getLinkState(robotId, endEffectorIndex)
    actualPos = linkState[0]
    error = math.sqrt(sum((actualPos[i] - targetPos[i])**2 for i in range(3)))
    
    print()  # 换行
    print(f"仿真实际位置: [{actualPos[0]:.3f}, {actualPos[1]:.3f}, {actualPos[2]:.3f}]")
    print(f"仿真误差: {error*1000:.1f}毫米")
    
    if reached:
        print("✅ 仿真已到达目标位置")
    else:
        if error < 0.030:
            print("⚠️  仿真接近目标位置（误差<30mm）")
        else:
            print("❌ 仿真未能到达目标位置（误差较大）")
            print("提示: 该位置可能超出工作空间")
    
    return actualPos, error

# ==================== 主循环 ====================
print("\n准备就绪！请输入命令：")

try:
    while True:
        try:
            user_input = input("\n> ").strip().lower()
            
            if not user_input:
                continue
            
            if user_input == 'quit' or user_input == 'exit':
                print("退出程序...")
                break
            
            elif user_input == 'current':
                linkState = p.getLinkState(robotId, endEffectorIndex)
                pos = linkState[0]
                orn = linkState[1]
                euler = p.getEulerFromQuaternion(orn)
                print(f"当前末端位置: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                print(f"当前末端姿态(欧拉角): [{math.degrees(euler[0]):.1f}°, "
                      f"{math.degrees(euler[1]):.1f}°, {math.degrees(euler[2]):.1f}°]")
            
            elif user_input == 'home':
                print("回到初始位置...")
                
                # 使用 set_joint_angles 函数，它会同时控制仿真和真实机械臂
                # 所有关节回到 0 度
                home_angles = [0.0, 0.0, 0.0, 0.0, 0.0]
                
                # 调用统一的关节角度设置函数
                set_joint_angles(home_angles)
                
                # 等待机械臂移动到位
                time.sleep(2.0)
                
                # 如果连接了真实机械臂，显示对比信息
                if use_real_robot:
                    print("\n" + "="*80)
                    print("初始位置角度对比:")
                    print("="*80)
                    print_angle_comparison(show_header=False)
                else:
                    print("✅ 仿真机械臂已回到初始位置")
            
            elif user_input == 'save':
                linkState = p.getLinkState(robotId, endEffectorIndex)
                pos = linkState[0]
                savedPositions.append(list(pos))
                print(f"✅ 已保存位置 #{len(savedPositions)}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
            
            elif user_input == 'list':
                if not savedPositions:
                    print("没有保存的位置")
                else:
                    print(f"已保存 {len(savedPositions)} 个位置:")
                    for i, pos in enumerate(savedPositions):
                        print(f"  {i+1}. [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
            
            elif user_input.startswith('goto'):
                parts = user_input.split()
                if len(parts) != 2:
                    print("用法: goto N (N为位置编号)")
                    continue
                try:
                    idx = int(parts[1]) - 1
                    if 0 <= idx < len(savedPositions):
                        move_to_position(savedPositions[idx])
                    else:
                        print(f"错误: 位置编号必须在 1-{len(savedPositions)} 之间")
                except ValueError:
                    print("错误: 请输入有效的数字")
            
            elif user_input == 'load':
                if not savedPositions:
                    print("没有保存的位置")
                else:
                    move_to_position(savedPositions[-1])
            
            elif user_input == 'connect':
                if not LEROBOT_AVAILABLE:
                    print("❌ LeRobot 模块未加载，无法连接真实机械臂")
                    continue
                
                if use_real_robot:
                    print("⚠️  真实机械臂已连接")
                    continue
                
                try:
                    print("正在连接真实机械臂（仅 follower 臂）...")
                    
                    # 创建自定义配置：只连接 follower，设置正确的校准路径
                    config = So101RobotConfig(
                        # 设置校准文件所在的目录（注意：是目录，不是文件）
                        # LeRobot 会自动在这个目录下查找 main_follower.json
                        calibration_dir="/home/martina/Y3_Project/learn_PyBullet",
                        # 不连接 leader 臂（设置为空字典）
                        leader_arms={},
                        # 不连接相机（设置为空字典）
                        cameras={}
                    )
                    
                    real_robot = ManipulatorRobot(config)
                    real_robot.connect()
                    use_real_robot = True
                    print("✅ 真实机械臂已连接（follower 臂）")
                except Exception as e:
                    print(f"❌ 连接失败: {e}")
                    import traceback
                    traceback.print_exc()
                    real_robot = None
                    use_real_robot = False
            
            elif user_input == 'disconnect':
                if not use_real_robot:
                    print("⚠️  真实机械臂未连接")
                    continue
                
                try:
                    print("正在断开真实机械臂...")
                    if real_robot:
                        # 重要：退出前禁用所有舵机的力矩，让机械臂可以手动移动
                        print("  禁用舵机力矩...")
                        for name in real_robot.follower_arms:
                            try:
                                real_robot.follower_arms[name].write("Torque_Enable", 0)
                                print(f"  ✓ 已禁用 {name} follower 臂力矩")
                            except Exception as e:
                                print(f"  ⚠️  禁用 {name} 力矩失败: {e}")
                        
                        # 断开连接
                        real_robot.disconnect()
                    
                    use_real_robot = False
                    real_robot = None
                    print("✅ 真实机械臂已断开（力矩已禁用，可以手动移动）")
                except Exception as e:
                    print(f"❌ 断开失败: {e}")
            
            elif user_input == 'compare':
                if not use_real_robot:
                    print("⚠️  需要先连接真实机械臂")
                    continue
                
                print_angle_comparison()
            
            elif user_input == 'monitor':
                if not use_real_robot:
                    print("⚠️  需要先连接真实机械臂")
                    continue
                
                if monitor_running:
                    print("⚠️  监控已在运行")
                    continue
                
                # 使用 global 声明来修改全局变量
                globals()['monitor_running'] = True
                globals()['monitor_thread'] = threading.Thread(target=monitor_angles_loop, daemon=True)
                globals()['monitor_thread'].start()
            
            elif user_input == 'stop_monitor':
                if not monitor_running:
                    print("⚠️  监控未运行")
                    continue
                
                # 停止监控
                globals()['monitor_running'] = False
                time.sleep(0.6)  # 等待线程结束
                print("\n✓ 监控已停止")
            
            elif user_input == 'calibrate':
                if not use_real_robot:
                    print("⚠️  需要先连接真实机械臂")
                    continue
                
                print("\n自动计算零点偏移...")
                result = compare_angles()
                if result:
                    ik_angles, sim_angles, real_angles, errors, joint_names = result
                    
                    # 计算建议的偏移量（真实 - 仿真）
                    # errors 只有5个值（对比的关节数），需要扩展为6个（gripper设为0）
                    suggested_offset_5 = errors
                    suggested_offset = np.zeros(6)
                    suggested_offset[:len(errors)] = errors
                    
                    print("\n检测到的角度偏差:")
                    for i, name in enumerate(joint_names):
                        print(f"  {name:20s}: {errors[i]:+7.2f}°")
                    print(f"  {'gripper':20s}: {0.0:+7.2f}° (固定为0)")
                    
                    print(f"\n建议的零点偏移: {suggested_offset}")
                    print("\n是否应用此偏移？")
                    print("  输入 'yes' 应用")
                    print("  输入其他键取消")
                    
                    choice = input("\n> ").strip().lower()
                    if choice == 'yes':
                        # 应用偏移
                        globals()['ZERO_POINT_OFFSET'] = suggested_offset
                        print(f"\n✅ 已应用偏移: {suggested_offset}")
                        print("\n现在发送 IK 解时会自动补偿此偏移")
                    else:
                        print("\n已取消")
            
            elif user_input.startswith('set_offset'):
                parts = user_input.split()
                if len(parts) == 1:
                    print("\n当前零点偏移:")
                    print(f"  {ZERO_POINT_OFFSET}")
                    print("\n用法: set_offset j0 j1 j2 j3 j4 [j5]")
                    print("  j0-j4: 前5个关节的偏移（必需）")
                    print("  j5: gripper 偏移（可选，默认0）")
                    print("\n例如:")
                    print("  set_offset 0 0 5 0 0      # 给 elbow_flex 添加 5° 偏移")
                    print("  set_offset 0 0 5 0 0 0    # 同上，显式指定 gripper 为 0")
                elif len(parts) == 6:
                    # 5个关节偏移，gripper自动设为0
                    try:
                        new_offset = np.zeros(6)
                        for i in range(5):
                            new_offset[i] = float(parts[i + 1])
                        globals()['ZERO_POINT_OFFSET'] = new_offset
                        print(f"✓ 零点偏移已更新为: {new_offset}")
                        print("  (gripper 自动设为 0)")
                    except ValueError:
                        print("错误: 请输入有效的数字")
                elif len(parts) == 7:
                    # 6个关节偏移，包括gripper
                    try:
                        new_offset = np.array([float(parts[i]) for i in range(1, 7)])
                        globals()['ZERO_POINT_OFFSET'] = new_offset
                        print(f"✓ 零点偏移已更新为: {new_offset}")
                    except ValueError:
                        print("错误: 请输入有效的数字")
                else:
                    print("错误: 需要提供 5 个或 6 个关节的偏移值")
            
            elif user_input == 'status':
                print(f"\n状态信息:")
                print(f"  LeRobot 模块: {'✓ 已加载' if LEROBOT_AVAILABLE else '✗ 未加载'}")
                print(f"  真实机械臂: {'✓ 已连接' if use_real_robot else '✗ 未连接'}")
                print(f"  仿真机械臂: ✓ 运行中")
                print(f"  角度监控: {'✓ 运行中' if monitor_running else '✗ 未运行'}")
                print(f"  零点偏移: {ZERO_POINT_OFFSET}")
            
            elif user_input == 'raw' or user_input == 'steps':
                # 显示原始步进值
                if not use_real_robot:
                    print("⚠️  需要先连接真实机械臂")
                    continue
                
                show_raw_steps()
            
            elif user_input.startswith('circle'):
                # 画圆命令
                parts = user_input.split()
                
                if len(parts) == 1:
                    # 使用当前位置作为圆心，默认参数画圆
                    linkState = p.getLinkState(robotId, endEffectorIndex)
                    center = list(linkState[0])
                    print(f"\n使用当前位置作为圆心: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
                    print("使用默认参数: 半径=5cm, 50个点")
                    draw_circle(center)
                
                elif len(parts) == 4:
                    # 指定圆心: circle x y z
                    try:
                        center = [float(parts[1]), float(parts[2]), float(parts[3])]
                        print("使用默认参数: 半径=5cm, 50个点")
                        draw_circle(center)
                    except ValueError:
                        print("❌ 坐标格式错误")
                
                elif len(parts) == 5:
                    # 指定圆心和半径: circle x y z radius
                    try:
                        center = [float(parts[1]), float(parts[2]), float(parts[3])]
                        radius = float(parts[4])
                        draw_circle(center, radius=radius)
                    except ValueError:
                        print("❌ 参数格式错误")
                
                elif len(parts) == 6:
                    # 指定圆心、半径和点数: circle x y z radius steps
                    try:
                        center = [float(parts[1]), float(parts[2]), float(parts[3])]
                        radius = float(parts[4])
                        steps = int(parts[5])
                        draw_circle(center, radius=radius, steps=steps)
                    except ValueError:
                        print("❌ 参数格式错误")
                
                else:
                    print("\n用法:")
                    print("  circle                       - 在当前位置画圆（默认半径5cm）")
                    print("  circle x y z                 - 在指定位置画圆（默认半径5cm）")
                    print("  circle x y z radius          - 在指定位置画圆（指定半径，单位米）")
                    print("  circle x y z radius steps    - 在指定位置画圆（指定半径和点数）")
                    print("\n例如:")
                    print("  circle                       - 当前位置，半径5cm")
                    print("  circle 0.30 0.10 0.25        - 指定位置，半径5cm")
                    print("  circle 0.30 0.10 0.25 0.08   - 指定位置，半径8cm")
                    print("  circle 0.30 0.10 0.25 0.10 100  - 半径10cm，100个点")
            
            elif user_input.startswith('joint'):
                # 关节角度控制命令
                parts = user_input.split()
                
                if len(parts) == 1:
                    # 显示当前关节角度
                    print("\n当前关节角度:")
                    print("\n索引 | 关节名称              | 角度(度)")
                    print("-" * 50)
                    for i in range(min(6, numJoints)):
                        info = p.getJointInfo(robotId, i)
                        jointName = info[1].decode('utf-8')
                        jointType = info[2]
                        if jointType == p.JOINT_REVOLUTE:
                            state = p.getJointState(robotId, i)
                            angle_deg = math.degrees(state[0])
                            print(f"  {i}  | {jointName:20s} | {angle_deg:7.2f}°")
                    
                    print("\n用法:")
                    print("  joint                    - 显示当前关节角度")
                    print("  joint a0 a1 a2 a3 a4     - 设置所有5个关节角度（度数）")
                    print("  joint N angle            - 设置关节N的角度（度数）")
                    print("\n例如:")
                    print("  joint 0 45.0             - 设置关节0为45度")
                    print("  joint 0 0 0 0 0          - 所有关节回零")
                    print("  joint 10 -30 45 -15 0    - 设置所有关节角度")
                
                elif len(parts) == 3:
                    # 设置单个关节角度: joint N angle
                    try:
                        joint_idx = int(parts[1])
                        angle_deg = float(parts[2])
                        
                        if joint_idx < 0 or joint_idx >= numJoints:
                            print(f"❌ 关节索引必须在 0-{numJoints-1} 之间")
                        else:
                            set_joint_angles([angle_deg], [joint_idx])
                    except ValueError:
                        print("❌ 格式错误: joint <关节索引> <角度(度)>")
                
                elif len(parts) == 6:
                    # 设置所有5个关节角度: joint a0 a1 a2 a3 a4
                    try:
                        angles = [float(parts[i]) for i in range(1, 6)]
                        set_joint_angles(angles)
                    except ValueError:
                        print("❌ 请输入有效的角度数值")
                
                else:
                    print("❌ 参数错误，使用 'joint' 查看用法")
            
            elif user_input == 'help':
                print("\n命令列表:")
                print("\n【基础控制】")
                print("  x y z        - 移动到坐标 (例: 0.30 0.10 0.25)")
                print("  current      - 显示当前位置")
                print("  home         - 回到初始位置")
                print("\n【关节控制】")
                print("  joint                    - 显示当前关节角度")
                print("  joint N angle            - 设置关节N的角度（度数）")
                print("  joint a0 a1 a2 a3 a4     - 设置所有5个关节角度（度数）")
                print("\n【轨迹控制】⭐ 新增")
                print("  circle                       - 在当前位置画圆（半径5cm）")
                print("  circle x y z                 - 在指定位置画圆")
                print("  circle x y z radius          - 指定半径（米）")
                print("  circle x y z radius steps    - 指定半径和点数")
                print("\n【位置管理】")
                print("  save         - 保存当前位置")
                print("  load         - 加载最后保存的位置")
                print("  goto N       - 移动到第N个保存的位置")
                print("  list         - 列出所有保存的位置")
                print("\n【机械臂连接】")
                print("  connect      - 连接真实机械臂")
                print("  disconnect   - 断开真实机械臂")
                print("  status       - 显示连接状态")
                print("\n【角度对比】")
                print("  compare      - 对比仿真和真实角度（一次）")
                print("  monitor      - 实时监控角度对比（持续刷新）")
                print("  stop_monitor - 停止实时监控")
                print("  calibrate    - 自动计算并应用零点偏移")
                print("  set_offset   - 手动设置零点偏移")
                print("\n【调试工具】")
                print("  raw / steps  - 实时显示原始步进值（按Enter保存）")
                print("\n【其他】")
                print("  help         - 显示此帮助")
                print("  quit         - 退出程序")
            
            else:
                # 尝试解析为坐标
                parts = user_input.split()
                if len(parts) == 3:
                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                        z = float(parts[2])
                        targetPos = [x, y, z]
                        move_to_position(targetPos)
                    except ValueError:
                        print("错误: 请输入有效的数字 (例: 0.30 0.10 0.25)")
                else:
                    print("错误: 未知命令。输入 'help' 查看帮助")
        
        except KeyboardInterrupt:
            print("\n使用 'quit' 退出程序")
            continue

except Exception as e:
    print(f"发生错误: {e}")

finally:
    # 停止监控线程
    globals()['monitor_running'] = False
    time.sleep(0.1)
    
    globals()['simulationRunning'] = False
    time.sleep(0.1)
    
    # 断开真实机械臂
    if use_real_robot and real_robot:
        try:
            print("\n正在安全关闭真实机械臂...")
            
            # 重要：禁用所有舵机的力矩，让机械臂可以手动移动
            print("  禁用舵机力矩...")
            for name in real_robot.follower_arms:
                try:
                    real_robot.follower_arms[name].write("Torque_Enable", 0)
                    print(f"  ✓ 已禁用 {name} follower 臂力矩")
                except Exception as e:
                    print(f"  ⚠️  禁用 {name} 力矩失败: {e}")
            
            # 等待命令执行
            time.sleep(0.2)
            
            # 断开连接
            print("  断开连接...")
            real_robot.disconnect()
            print("✅ 真实机械臂已安全关闭（力矩已禁用，可以手动移动）")
        except Exception as e:
            print(f"❌ 关闭真实机械臂时出错: {e}")
            print("⚠️  如果机械臂仍然锁住，请重启机械臂电源")
    
    print("\n关闭仿真连接...")
    p.disconnect()
    print("再见！👋")

