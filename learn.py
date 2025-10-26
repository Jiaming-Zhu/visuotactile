
import pybullet as p
import time
import pybullet_data
import math

# 连接到PyBullet
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)

# 加载地面
planeId = p.loadURDF("plane.urdf")

# 加载机械臂（固定底座）
startPos = [0, 0, 0]  # 机械臂放在地面上
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
robotId = p.loadURDF("so101_new_calib.urdf", startPos, startOrientation, 
                     useFixedBase=True)  # 固定底座，防止倾倒

# ==================== 获取机械臂信息 ====================
numJoints = p.getNumJoints(robotId)
print(f"机械臂关节数量: {numJoints}")
print("\n关节信息:")
for i in range(numJoints):
    info = p.getJointInfo(robotId, i)
    print(f"关节 {i}: {info[1].decode('utf-8')} (类型: {info[2]})")

# ==================== 控制方式演示 ====================

# 方式1: 位置控制（最常用）
print("\n=== 演示：位置控制 ===")
# 设置目标关节角度（弧度）
targetPositions = [0.5, 0.3, -0.5, 0.8, 0.2, 0.0]  # 根据您的机械臂调整

# 控制所有可动关节
for i in range(min(6, numJoints)):  # 假设前6个是可动关节
    p.setJointMotorControl2(
        bodyIndex=robotId,
        jointIndex=i,
        controlMode=p.POSITION_CONTROL,
        targetPosition=targetPositions[i],
        force=500  # 最大力矩
    )

# 运行仿真，让机械臂移动到目标位置
for i in range(1000):
    p.stepSimulation()
    time.sleep(1./240.)

# ==================== 读取关节状态 ====================
print("\n=== 当前关节状态 ===")
for i in range(numJoints):
    state = p.getJointState(robotId, i)
    print(f"关节 {i}: 位置={state[0]:.3f}, 速度={state[1]:.3f}, 力={state[3]:.3f}")

# ==================== 方式2: 速度控制 ====================
print("\n=== 演示：速度控制 ===")
for i in range(min(3, numJoints)):
    p.setJointMotorControl2(
        bodyIndex=robotId,
        jointIndex=i,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity=0.5,  # 目标速度（弧度/秒）
        force=500
    )

for i in range(500):
    p.stepSimulation()
    time.sleep(1./240.)

# ==================== 方式3: 力矩控制 ====================
print("\n=== 演示：力矩控制（需要先禁用默认电机）===")
# 先禁用所有关节的电机
for i in range(numJoints):
    p.setJointMotorControl2(
        bodyIndex=robotId,
        jointIndex=i,
        controlMode=p.VELOCITY_CONTROL,
        force=0  # 关闭电机
    )

# 然后施加力矩
for i in range(500):
    p.setJointMotorControl2(
        bodyIndex=robotId,
        jointIndex=0,
        controlMode=p.TORQUE_CONTROL,
        force=10  # 施加的力矩
    )
    p.stepSimulation()
    time.sleep(1./240.)

# ==================== 方式4: 正弦波运动（循环控制）====================
print("\n=== 演示：正弦波运动 ===")
for t in range(2000):
    # 第一个关节做正弦运动
    angle = math.sin(t * 0.01) * math.pi / 4  # ±45度
    p.setJointMotorControl2(
        bodyIndex=robotId,
        jointIndex=0,
        controlMode=p.POSITION_CONTROL,
        targetPosition=angle,
        force=500
    )
    
    p.stepSimulation()
    time.sleep(1./240.)

print("\n仿真结束")
p.disconnect()
