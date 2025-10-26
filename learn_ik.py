
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
startPos = [0, 0, 0]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
robotId = p.loadURDF("so101_new_calib.urdf", startPos, startOrientation, 
                     useFixedBase=True)

# ==================== 获取机械臂信息 ====================
numJoints = p.getNumJoints(robotId)
print(f"机械臂关节数量: {numJoints}")
print("\n所有Link信息:")

# 找到末端执行器
gripperFrameIndex = None
for i in range(numJoints):
    info = p.getJointInfo(robotId, i)
    jointName = info[1].decode('utf-8')
    linkName = info[12].decode('utf-8')
    jointType = info[2]
    
    # 获取link位置
    linkState = p.getLinkState(robotId, i)
    linkPos = linkState[0]
    
    print(f"索引 {i}: {linkName}")
    print(f"  关节: {jointName}, 位置: ({linkPos[0]:.3f}, {linkPos[1]:.3f}, {linkPos[2]:.3f})")
    
    # 查找gripper_frame（夹爪坐标系）
    if 'gripper_frame' in linkName.lower():
        gripperFrameIndex = i
        print(f"  ⭐ 找到夹爪坐标系！这是IK的理想目标点")

# 确定末端执行器索引
if gripperFrameIndex is not None:
    endEffectorIndex = gripperFrameIndex
    print(f"\n✅ 使用末端执行器: 索引 {endEffectorIndex} (gripper_frame_link)")
else:
    endEffectorIndex = numJoints - 1
    print(f"\n⚠️  未找到gripper_frame，使用最后一个link: 索引 {endEffectorIndex}")

print(f"\n{'='*60}")
print(f"IK将控制的位置: {p.getJointInfo(robotId, endEffectorIndex)[12].decode('utf-8')}")
print(f"{'='*60}\n")

# 获取当前末端位置
linkState = p.getLinkState(robotId, endEffectorIndex)
currentPos = linkState[0]  # 世界坐标系中的位置
currentOrn = linkState[1]  # 世界坐标系中的姿态
print(f"\n当前末端位置: {currentPos}")
print(f"当前末端姿态: {currentOrn}")

# ==================== 方法1: IK控制末端位置（最简单）====================
print("\n=== 方法1: 只控制位置 ===")

# 目标位置（x, y, z）
targetPos1 = [0.2, 0.1, 0.3]

# 计算逆运动学
jointPoses = p.calculateInverseKinematics(
    robotId,
    endEffectorIndex,
    targetPos1
)

print(f"计算出的关节角度: {jointPoses[:6]}")  # 显示前6个关节

# 控制机械臂移动到目标位置
for i in range(len(jointPoses)):
    p.setJointMotorControl2(
        bodyIndex=robotId,
        jointIndex=i,
        controlMode=p.POSITION_CONTROL,
        targetPosition=jointPoses[i],
        force=500
    )

# 运行仿真
for i in range(1000):
    p.stepSimulation()
    time.sleep(1./240.)

# 验证末端位置
linkState = p.getLinkState(robotId, endEffectorIndex)
actualPos = linkState[0]
print(f"目标位置: {targetPos1}")
print(f"实际位置: {actualPos}")

# ==================== 方法2: IK控制位置+姿态 ====================
print("\n=== 方法2: 控制位置和姿态 ===")

# 目标位置和姿态
targetPos2 = [0.25, 0, 0.25]
targetOrn2 = p.getQuaternionFromEuler([0, -math.pi/2, 0])  # 末端朝下

# 计算IK（包含姿态）
jointPoses = p.calculateInverseKinematics(
    robotId,
    endEffectorIndex,
    targetPos2,
    targetOrientation=targetOrn2
)

# 控制机械臂
for i in range(len(jointPoses)):
    p.setJointMotorControl2(
        bodyIndex=robotId,
        jointIndex=i,
        controlMode=p.POSITION_CONTROL,
        targetPosition=jointPoses[i],
        force=500
    )

for i in range(1000):
    p.stepSimulation()
    time.sleep(1./240.)

# ==================== 方法3: IK沿轨迹移动（画圆）====================
print("\n=== 方法3: 沿圆形轨迹移动 ===")

# 圆形轨迹参数（修正为合理的大小）
center = [0.2, 0, 0.2]  # 圆心位置
radius = 0.05  # 半径5厘米
steps = 100  # 减少步数，让机械臂有更多时间到达每个点

# 用于记录上一个位置
prevTargetPos = None  # 目标位置
prevActualPos = None  # 实际末端位置

print(f"圆形轨迹: 圆心={center}, 半径={radius}米")
print("红色线 = 目标轨迹")
print("绿色线 = 实际末端轨迹")
print("关键：给机械臂足够时间到达每个点\n")

for step in range(steps):
    # 计算圆形轨迹上的点
    angle = 2 * math.pi * step / steps
    targetPos = [
        center[0] + radius * math.cos(angle),
        center[1] + radius * math.sin(angle),
        center[2]
    ]
    
    # 可选：固定末端姿态
    targetOrn = p.getQuaternionFromEuler([0, -math.pi/2, 0])
    
    # 计算IK
    jointPoses = p.calculateInverseKinematics(
        robotId,
        endEffectorIndex,
        targetPos,
        targetOrientation=targetOrn
    )
    
    # 控制机械臂
    for i in range(len(jointPoses)):
        p.setJointMotorControl2(
            bodyIndex=robotId,
            jointIndex=i,
            controlMode=p.POSITION_CONTROL,
            targetPosition=jointPoses[i],
            force=500,
            maxVelocity=5  # 提高速度限制
        )
    
    # 增加仿真步数，让机械臂有足够时间到达目标
    for _ in range(50):  # 从5步增加到50步！
        p.stepSimulation()
        time.sleep(1./240.)
    
    # 获取实际末端位置
    linkState = p.getLinkState(robotId, endEffectorIndex)
    actualPos = linkState[0]
    
    # 每步都画轨迹
    # 画目标轨迹（红色，细线）
    if prevTargetPos is not None:
        p.addUserDebugLine(prevTargetPos, targetPos, [1, 0, 0], 1, lifeTime=0)
    prevTargetPos = targetPos
    
    # 画实际轨迹（绿色，粗线）
    if prevActualPos is not None:
        p.addUserDebugLine(prevActualPos, actualPos, [0, 1, 0], 4, lifeTime=0)
    prevActualPos = actualPos
    
    # 每10步显示误差
    if step % 10 == 0:
        error = math.sqrt(sum((actualPos[i] - targetPos[i])**2 for i in range(3)))
        print(f"步骤 {step}/{steps}: 误差={error*1000:.2f}毫米")

print("\n圆形轨迹完成")



print("\nIK演示结束！")
print("\n按Ctrl+C关闭窗口")

# 保持窗口打开
try:
    while True:
        p.stepSimulation()
        time.sleep(1./240.)
except KeyboardInterrupt:
    print("\n关闭仿真")
    p.disconnect()

