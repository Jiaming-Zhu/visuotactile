
import pybullet as p
import time
import pybullet_data
import math

# 初始化
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)

planeId = p.loadURDF("plane.urdf")
robotId = p.loadURDF("so101_new_calib.urdf", [0, 0, 0], 
                     p.getQuaternionFromEuler([0, 0, 0]), 
                     useFixedBase=True)

# 找末端执行器
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
print("误差原因诊断工具")
print("="*80)

# 测试位置（使用find_workspace.py找到的最佳位置）
testPos = [0.30, 0.10, 0.25]  # find_workspace找到的最佳位置，误差0.5mm
print(f"\n测试位置: {testPos} (find_workspace发现的最佳位置)")

# ==================== 测试1: IK求解质量 ====================
print("\n【测试1】IK求解质量")
print("-"*80)

jointPoses = p.calculateInverseKinematics(robotId, endEffectorIndex, testPos)

# 正向运动学验证
for i in range(len(jointPoses)):
    p.resetJointState(robotId, i, jointPoses[i])  # 直接设置，不通过控制器

linkState = p.getLinkState(robotId, endEffectorIndex)
actualPos = linkState[0]
ikError = math.sqrt(sum((actualPos[i] - testPos[i])**2 for i in range(3)))

print(f"IK计算的关节角度直接设置后:")
print(f"  目标位置: [{testPos[0]:.6f}, {testPos[1]:.6f}, {testPos[2]:.6f}]")
print(f"  实际位置: [{actualPos[0]:.6f}, {actualPos[1]:.6f}, {actualPos[2]:.6f}]")
print(f"  误差: {ikError*1000:.3f}毫米")

if ikError < 0.001:
    print(f"  ✅ IK求解非常精确（<1mm），问题不在IK")
elif ikError < 0.005:
    print(f"  ⚠️  IK求解有小误差（1-5mm），可接受")
else:
    print(f"  ❌ IK求解误差较大（>5mm），说明该位置IK难以求解")
    print(f"     原因可能是：位置在工作空间边界，或姿态约束冲突")

# ==================== 测试2: 控制器收敛性 ====================
print("\n【测试2】控制器收敛性")
print("-"*80)

# 重新计算IK
jointPoses = p.calculateInverseKinematics(robotId, endEffectorIndex, testPos)

# 使用控制器，测试不同等待时间
waitTimes = [50, 100, 200, 500, 1000]
errors = []

for waitTime in waitTimes:
    # 重置到初始位置
    for i in range(numJoints):
        p.resetJointState(robotId, i, 0)
    
    # 设置控制目标
    for i in range(len(jointPoses)):
        p.setJointMotorControl2(robotId, i, p.POSITION_CONTROL,
                               targetPosition=jointPoses[i], force=500)
    
    # 等待一段时间
    for _ in range(waitTime):
        p.stepSimulation()
    
    # 检查位置
    linkState = p.getLinkState(robotId, endEffectorIndex)
    actualPos = linkState[0]
    error = math.sqrt(sum((actualPos[j] - testPos[j])**2 for j in range(3)))
    errors.append(error)
    
    print(f"  等待{waitTime:4d}步: 误差={error*1000:6.2f}mm", end="")
    if len(errors) > 1 and abs(errors[-1] - errors[-2]) < 0.0001:
        print(" (已收敛)")
    else:
        print()

if errors[-1] < 0.001:
    print(f"  ✅ 控制器可以精确收敛")
elif errors[-2] > errors[-1]:
    print(f"  ⚠️  控制器仍在收敛，需要更多时间")
else:
    print(f"  ❌ 控制器已收敛，但有稳态误差 {errors[-1]*1000:.2f}mm")

# ==================== 测试3: 关节状态检查 ====================
print("\n【测试3】关节限制检查")
print("-"*80)

hasLimitIssue = False
for i in range(min(numJoints, len(jointPoses))):  # 修复IndexError
    info = p.getJointInfo(robotId, i)
    jointType = info[2]
    
    if jointType == p.JOINT_REVOLUTE:
        jointName = info[1].decode('utf-8')
        lowerLimit = info[8]
        upperLimit = info[9]
        targetAngle = jointPoses[i]
        
        # 检查是否超出限制
        if targetAngle < lowerLimit or targetAngle > upperLimit:
            print(f"  ❌ 关节{i} ({jointName}): 目标角度超出限制")
            print(f"     限制: [{math.degrees(lowerLimit):.1f}°, {math.degrees(upperLimit):.1f}°]")
            print(f"     目标: {math.degrees(targetAngle):.1f}°")
            hasLimitIssue = True
        elif targetAngle < lowerLimit + 0.1 or targetAngle > upperLimit - 0.1:
            print(f"  ⚠️  关节{i} ({jointName}): 接近限制边缘")
            print(f"     限制: [{math.degrees(lowerLimit):.1f}°, {math.degrees(upperLimit):.1f}°]")
            print(f"     目标: {math.degrees(targetAngle):.1f}°")

if not hasLimitIssue:
    print(f"  ✅ 所有关节角度都在安全范围内")

# ==================== 测试4: 不同半径的圆 ====================
print("\n【测试4】测试不同半径的可达性")
print("-"*80)

center = testPos
testRadii = [0.02, 0.03, 0.04, 0.05, 0.08, 0.10]

for radius in testRadii:
    reachableCount = 0
    totalError = 0
    maxError = 0
    
    # 测试8个点
    for angle in [0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi, 5*math.pi/4, 3*math.pi/2, 7*math.pi/4]:
        circlePos = [
            center[0] + radius * math.cos(angle),
            center[1] + radius * math.sin(angle),
            center[2]
        ]
        
        # IK测试（快速，不仿真）
        jointPoses = p.calculateInverseKinematics(robotId, endEffectorIndex, circlePos)
        
        for i in range(len(jointPoses)):
            p.resetJointState(robotId, i, jointPoses[i])
        
        linkState = p.getLinkState(robotId, endEffectorIndex)
        actualPos = linkState[0]
        error = math.sqrt(sum((actualPos[j] - circlePos[j])**2 for j in range(3)))
        
        if error < 0.015:
            reachableCount += 1
        totalError += error
        maxError = max(maxError, error)
    
    avgError = totalError / 8
    
    print(f"  半径{radius*100:4.1f}cm: 可达{reachableCount}/8点, ", end="")
    print(f"平均误差={avgError*1000:5.1f}mm, 最大误差={maxError*1000:5.1f}mm", end="")
    
    if reachableCount == 8 and avgError < 0.01:
        print(" ✅ 完美")
    elif reachableCount >= 6:
        print(" ⚠️  大部分可达")
    else:
        print(" ❌ 超出工作空间")

# ==================== 总结 ====================
print("\n" + "="*80)
print("诊断总结")
print("="*80)

print(f"\n针对位置 {testPos}:")
if ikError < 0.005 and errors[-1] < 0.015:
    print("  ✅ 该位置可以精确到达，误差主要来自:")
    print("     - 控制器需要足够的时间收敛")
    print("     - 可能有小的稳态误差（<15mm）")
    print("\n  建议: 增加等待时间（maxWaitSteps），放宽阈值到20-30mm")
elif ikError > 0.01:
    print("  ❌ 该位置IK求解困难，机械臂难以精确到达")
    print("     - IK算法无法找到精确解")
    print("     - 位置可能在工作空间边界")
    print("\n  建议: 选择IK误差<5mm的位置作为圆心")
else:
    print("  ⚠️  该位置可达，但精度中等")
    print("\n  建议: 适当放宽误差阈值")

print("\n按Ctrl+C关闭...")
try:
    while True:
        p.stepSimulation()
        time.sleep(1./240.)
except KeyboardInterrupt:
    print("关闭")
    p.disconnect()

