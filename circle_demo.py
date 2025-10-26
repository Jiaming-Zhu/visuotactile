
import pybullet as p
import time
import pybullet_data
import math

# 连接到PyBullet
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)

# 加载地面和机械臂
planeId = p.loadURDF("plane.urdf")
robotId = p.loadURDF("so101_new_calib.urdf", [0, 0, 0], 
                     p.getQuaternionFromEuler([0, 0, 0]), 
                     useFixedBase=True)

# 找到末端执行器
numJoints = p.getNumJoints(robotId)
gripperFrameIndex = None
for i in range(numJoints):
    info = p.getJointInfo(robotId, i)
    linkName = info[12].decode('utf-8')
    if 'gripper_frame' in linkName.lower():
        gripperFrameIndex = i
        break

if gripperFrameIndex is None:
    gripperFrameIndex = numJoints - 1

endEffectorIndex = gripperFrameIndex
print(f"使用末端执行器索引: {endEffectorIndex}")

# ==================== 获取关节限制 ====================
print("\n获取关节信息和限制...")
jointLowerLimits = []
jointUpperLimits = []
jointRanges = []
restPoses = []

for i in range(numJoints):
    info = p.getJointInfo(robotId, i)
    jointType = info[2]
    
    if jointType == p.JOINT_REVOLUTE:
        lowerLimit = info[8]  # 下限
        upperLimit = info[9]  # 上限
        jointRange = upperLimit - lowerLimit
        
        # 如果关节限制不合理，使用默认值
        if jointRange < 0.01:  # 几乎固定的关节
            lowerLimit = -3.14
            upperLimit = 3.14
            jointRange = 6.28
        
        jointLowerLimits.append(lowerLimit)
        jointUpperLimits.append(upperLimit)
        jointRanges.append(jointRange)
        restPoses.append(0)  # 默认姿态为0
        
        print(f"  关节{i}: 范围 [{math.degrees(lowerLimit):.1f}° 到 {math.degrees(upperLimit):.1f}°]")
    else:
        # 固定关节或其他类型
        jointLowerLimits.append(-3.14)
        jointUpperLimits.append(3.14)
        jointRanges.append(6.28)
        restPoses.append(0)

# ==================== 测试工作空间 ====================
print("\n测试机械臂的可达工作空间...")
print("测试不同的位置组合 (x, y, z)\n")

# 测试不同的位置组合（根据workspace扫描结果调整）
testPositions = [
    # (x, y, z) - 使用find_workspace.py发现的最佳区域
    ([0.25, 0.10, 0.25], "最佳位置1"),
    ([0.30, 0.10, 0.25], "最佳位置2"),
    ([0.25, 0.15, 0.25], "最佳位置3"),
    ([0.30, 0.05, 0.25], "最佳位置4"),
    ([0.20, 0.10, 0.25], "次优位置1"),
    ([0.20, 0.15, 0.25], "次优位置2"),
]

bestPos = None
bestError = float('inf')

for testPos, description in testPositions:
    # 不强制姿态！让IK自由选择（关键改动）
    jointPoses = p.calculateInverseKinematics(
        robotId, 
        endEffectorIndex, 
        testPos
        # 不指定targetOrientation
    )
    
    # 设置关节位置并仿真
    for i in range(len(jointPoses)):
        p.setJointMotorControl2(robotId, i, p.POSITION_CONTROL, 
                               targetPosition=jointPoses[i], force=500)
    
    # 仿真一段时间
    for _ in range(200):
        p.stepSimulation()
        time.sleep(1./240.)
    
    # 检查实际位置
    linkState = p.getLinkState(robotId, endEffectorIndex)
    actualPos = linkState[0]
    error = math.sqrt(sum((actualPos[j] - testPos[j])**2 for j in range(3)))
    
    print(f"  {description:15} {testPos}: 误差={error*1000:.1f}毫米", end="")
    if error < 0.015:  # 误差小于1.5厘米（放宽一点）
        print(" ✓ 可达")
        if error < bestError:
            bestError = error
            bestPos = testPos
    else:
        print(" ✗ 不可达")

if bestPos is None:
    bestPos = [0.2, 0, 0.4]  # 默认值：中距离中位
    print(f"\n⚠️  警告：所有测试点都不理想，使用默认位置 {bestPos}")
    print(f"    建议：调整机械臂初始姿态或测试更多位置")
else:
    print(f"\n✓ 最佳圆心位置: {bestPos}, 误差={bestError*1000:.1f}毫米")

# ==================== 验证圆形轨迹的可达性 ====================
print("\n验证圆形轨迹的所有点是否可达...")

# 尝试不同的圆心和半径组合
testCircles = [
    (bestPos, 0.02, "最佳位置+2cm半径"),
    (bestPos, 0.03, "最佳位置+3cm半径"),
    (bestPos, 0.04, "最佳位置+4cm半径"),
    (bestPos, 0.05, "最佳位置+5cm半径"),
]

selectedCenter = None
selectedRadius = None
maxReachablePoints = 0

for testCenter, testRadius, description in testCircles:
    print(f"\n测试圆: {description}")
    print(f"  圆心: {testCenter}, 半径: {testRadius}米")
    
    # 测试圆周上的8个点
    testAngles = [i * math.pi / 4 for i in range(8)]  # 每45度一个点
    reachableCount = 0
    totalError = 0
    
    for angle in testAngles:
        testPos = [
            testCenter[0] + testRadius * math.cos(angle),
            testCenter[1] + testRadius * math.sin(angle),
            testCenter[2]
        ]
        
        # 计算IK（不强制姿态）
        jointPoses = p.calculateInverseKinematics(
            robotId, endEffectorIndex, testPos
        )
        
        # 设置并仿真
        for i in range(len(jointPoses)):
            p.setJointMotorControl2(robotId, i, p.POSITION_CONTROL,
                                   targetPosition=jointPoses[i], force=500)
        
        for _ in range(200):
            p.stepSimulation()
            time.sleep(1./240.)
        
        # 检查实际位置
        linkState = p.getLinkState(robotId, endEffectorIndex)
        actualPos = linkState[0]
        error = math.sqrt(sum((actualPos[j] - testPos[j])**2 for j in range(3)))
        
        if error < 0.01:  # 误差小于1cm才算可达
            reachableCount += 1
        totalError += error
    
    avgError = totalError / len(testAngles)
    print(f"  可达点数: {reachableCount}/8, 平均误差: {avgError*1000:.1f}毫米")
    
    if reachableCount > maxReachablePoints:
        maxReachablePoints = reachableCount
        selectedCenter = testCenter
        selectedRadius = testRadius

if maxReachablePoints < 6:  # 至少要有75%的点可达
    print(f"\n⚠️  警告：没有找到合适的圆，最多只有{maxReachablePoints}/8点可达")
    print("建议：")
    print("  1. 调整机械臂的初始姿态")
    print("  2. 减小圆的半径")
    print("  3. 调整圆心的位置（特别是高度）")
    print("\n继续使用最佳圆，但效果可能不理想...")

# 如果没有选择到圆（所有测试都失败），使用默认值
if selectedCenter is None:
    selectedCenter = bestPos
    selectedRadius = 0.02  # 非常小的半径
    print(f"\n使用最小圆作为后备：圆心={selectedCenter}, 半径={selectedRadius}米")

# ==================== 画圆演示（等待到达版本）====================
print(f"\n开始画圆演示")
print(f"选择的圆: 圆心={selectedCenter}, 半径={selectedRadius}米")
print(f"可达性: {maxReachablePoints}/8 点\n")

center = selectedCenter
radius = selectedRadius
steps = 40  # 减少点数

# ==================== 初始化：让关节0转到45度 ====================
print("初始化：让关节0（base旋转）转到45度，确保它会参与运动...")
p.setJointMotorControl2(robotId, 0, p.POSITION_CONTROL, 
                       targetPosition=math.radians(45), force=500)

# 等待关节0到达
for _ in range(300):
    p.stepSimulation()
    time.sleep(1./240.)

joint0State = p.getJointState(robotId, 0)
print(f"关节0当前角度: {math.degrees(joint0State[0]):.1f}度\n")

# 不再设置固定的末端姿态
prevActualPos = None

for step in range(steps):
    # 计算目标点
    angle = 2 * math.pi * step / steps
    targetPos = [
        center[0] + radius * math.cos(angle),
        center[1] + radius * math.sin(angle),
        center[2]
    ]
    
    # 画目标点（红色小球）
    sphereId = p.createVisualShape(p.GEOM_SPHERE, radius=0.003, rgbaColor=[1, 0, 0, 1])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=sphereId, basePosition=targetPos)
    
    # 计算IK（不强制姿态，让机械臂自由选择）
    jointPoses = p.calculateInverseKinematics(
        robotId,
        endEffectorIndex,
        targetPos
        # 不指定姿态约束
    )
    
    print(f"\n{'='*80}")
    print(f"步骤 {step+1}/{steps}")
    print(f"目标位置: [{targetPos[0]:.3f}, {targetPos[1]:.3f}, {targetPos[2]:.3f}]")
    print(f"\nIK计算的目标关节角度（度）:")
    for i in range(min(6, len(jointPoses))):  # 只显示前6个关节
        info = p.getJointInfo(robotId, i)
        jointName = info[1].decode('utf-8')
        targetAngle = math.degrees(jointPoses[i])
        print(f"  关节{i} ({jointName:20s}): {targetAngle:7.2f}°")
    
    # 设置目标关节位置
    for i in range(len(jointPoses)):
        p.setJointMotorControl2(
            bodyIndex=robotId,
            jointIndex=i,
            controlMode=p.POSITION_CONTROL,
            targetPosition=jointPoses[i],
            force=500,
            maxVelocity=5
        )
    
    # 等待机械臂到达目标（关键！）
    maxWaitSteps = 500  # 最多等500步
    threshold = 0.02  # 位置误差阈值：2厘米（放宽阈值）
    
    for waitStep in range(maxWaitSteps):
        p.stepSimulation()
        time.sleep(1./240.)
        
        # 检查是否到达
        linkState = p.getLinkState(robotId, endEffectorIndex)
        actualPos = linkState[0]
        
        error = math.sqrt(sum((actualPos[i] - targetPos[i])**2 for i in range(3)))
        
        # 每50步显示一次当前状态
        if waitStep % 50 == 0:
            print(f"\n  [等待 {waitStep}/500] 当前误差={error*1000:.1f}毫米")
            print(f"  实际末端位置: [{actualPos[0]:.3f}, {actualPos[1]:.3f}, {actualPos[2]:.3f}]")
            print(f"  当前关节角度（度）:")
            for i in range(min(6, numJoints)):
                jointState = p.getJointState(robotId, i)
                currentAngle = jointState[0]  # 当前角度（弧度）
                info = p.getJointInfo(robotId, i)
                jointName = info[1].decode('utf-8')
                print(f"    关节{i} ({jointName:20s}): {math.degrees(currentAngle):7.2f}°")
        
        # 如果足够接近，就移动到下一个点
        if error < threshold:
            # 画实际轨迹（绿色粗线）
            if prevActualPos is not None:
                p.addUserDebugLine(prevActualPos, actualPos, [0, 1, 0], 5, lifeTime=0)
            prevActualPos = actualPos
            
            print(f"\n✓ 到达目标！最终误差={error*1000:.1f}毫米")
            print(f"  最终末端位置: [{actualPos[0]:.3f}, {actualPos[1]:.3f}, {actualPos[2]:.3f}]")
            break
    
    else:
        # 超时了，还没到达
        linkState = p.getLinkState(robotId, endEffectorIndex)
        actualPos = linkState[0]
        error = math.sqrt(sum((actualPos[i] - targetPos[i])**2 for i in range(3)))
        
        print(f"\n⚠️  超时！无法到达目标")
        print(f"  目标位置: [{targetPos[0]:.3f}, {targetPos[1]:.3f}, {targetPos[2]:.3f}]")
        print(f"  实际位置: [{actualPos[0]:.3f}, {actualPos[1]:.3f}, {actualPos[2]:.3f}]")
        print(f"  误差: {error*1000:.1f}毫米")
        print(f"  最终关节角度（度）:")
        for i in range(min(6, numJoints)):
            jointState = p.getJointState(robotId, i)
            currentAngle = jointState[0]
            info = p.getJointInfo(robotId, i)
            jointName = info[1].decode('utf-8')
            print(f"    关节{i} ({jointName:20s}): {math.degrees(currentAngle):7.2f}°")
        
        # 还是画出实际轨迹（黄色表示未完全到达）
        if prevActualPos is not None:
            p.addUserDebugLine(prevActualPos, actualPos, [1, 1, 0], 3, lifeTime=0)
        prevActualPos = actualPos

print("\n✓ 圆形轨迹完成！")
print("  绿色粗线 = 实际轨迹")
print("  红色小球 = 目标点")

# 保持窗口打开
print("\n按Ctrl+C关闭...")
try:
    while True:
        p.stepSimulation()
        time.sleep(1./240.)
except KeyboardInterrupt:
    print("关闭")
    p.disconnect()

