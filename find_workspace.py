
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
print("\n正在扫描机械臂工作空间...")
print("这可能需要几十秒，请耐心等待...\n")

# 扫描工作空间
reachablePositions = []

# 在3D空间中采样点
x_range = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
y_range = [-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15]
z_range = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

totalTests = len(x_range) * len(y_range) * len(z_range)
testCount = 0
errorStats = []  # 收集误差统计

for x in x_range:
    for y in y_range:
        for z in z_range:
            testCount += 1
            if testCount % 20 == 0:
                print(f"进度: {testCount}/{totalTests} ({100*testCount/totalTests:.1f}%)")
            
            testPos = [x, y, z]
            
            # 尝试不同的姿态（关键！）
            # 不固定姿态，让IK自由选择
            try:
                jointPoses = p.calculateInverseKinematics(
                    robotId, 
                    endEffectorIndex, 
                    testPos
                    # 不指定targetOrientation，让IK自由
                )
            except:
                continue
            
            # 设置关节
            for i in range(len(jointPoses)):
                p.setJointMotorControl2(robotId, i, p.POSITION_CONTROL,
                                       targetPosition=jointPoses[i], force=500)
            
            # 增加仿真时间
            for _ in range(200):  # 从100增加到200
                p.stepSimulation()
            
            # 检查位置
            linkState = p.getLinkState(robotId, endEffectorIndex)
            actualPos = linkState[0]
            error = math.sqrt(sum((actualPos[j] - testPos[j])**2 for j in range(3)))
            
            errorStats.append(error)
            
            # 放宽阈值
            if error < 0.015:  # 从8mm放宽到15mm
                reachablePositions.append((testPos, error))
                if len(reachablePositions) <= 3:  # 前几个详细输出
                    print(f"  找到可达位置: {testPos}, 误差={error*1000:.1f}mm")

print(f"\n扫描完成！找到 {len(reachablePositions)} 个可达位置\n")

# 显示误差统计
if errorStats:
    errorStats.sort()
    minError = errorStats[0]
    maxError = errorStats[-1]
    medianError = errorStats[len(errorStats)//2]
    avgError = sum(errorStats) / len(errorStats)
    
    print(f"误差统计（{len(errorStats)}个测试点）：")
    print(f"  最小误差: {minError*1000:.1f}毫米")
    print(f"  中位误差: {medianError*1000:.1f}毫米")
    print(f"  平均误差: {avgError*1000:.1f}毫米")
    print(f"  最大误差: {maxError*1000:.1f}毫米")
    print(f"  误差<15mm的点: {len(reachablePositions)}个 ({100*len(reachablePositions)/len(errorStats):.1f}%)\n")

if len(reachablePositions) == 0:
    print("❌ 没有找到任何可达位置！")
    print("\n可能的原因和建议：")
    print("  1. 阈值太严格 → 已放宽到15mm，如果最小误差>15mm，说明测试范围不对")
    print("  2. 测试范围不在工作空间内 → 查看上面的误差统计，调整测试范围")
    print("  3. 末端执行器索引不正确 → 当前使用索引", endEffectorIndex)
    print(f"  4. 如果最小误差很小({minError*1000:.1f}mm)，说明有些点接近可达，可以放宽阈值")
else:
    # 按误差排序
    reachablePositions.sort(key=lambda x: x[1])
    
    print(f"✅ 最佳的10个可达位置：\n")
    for i, (pos, err) in enumerate(reachablePositions[:10]):
        print(f"{i+1:2d}. [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] - 误差={err*1000:.1f}毫米")
    
    # 寻找可以画圆的位置
    print(f"\n寻找适合画圆的位置...")
    bestCircleCenter = None
    bestCircleRadius = 0
    
    for pos, err in reachablePositions[:20]:  # 检查前20个最好的位置
        # 尝试以这个点为圆心，测试不同半径
        for r in [0.02, 0.03, 0.04, 0.05]:
            # 测试圆周上4个点
            testAngles = [0, math.pi/2, math.pi, 3*math.pi/2]
            allReachable = True
            
            for angle in testAngles:
                circlePos = [
                    pos[0] + r * math.cos(angle),
                    pos[1] + r * math.sin(angle),
                    pos[2]
                ]
                
                # 快速检查（不实际移动）
                testOrn = p.getQuaternionFromEuler([0, -math.pi/2, 0])
                jointPoses = p.calculateInverseKinematics(
                    robotId, endEffectorIndex, circlePos, targetOrientation=testOrn
                )
                
                # 设置并检查
                for i in range(len(jointPoses)):
                    p.setJointMotorControl2(robotId, i, p.POSITION_CONTROL,
                                           targetPosition=jointPoses[i], force=500)
                for _ in range(100):
                    p.stepSimulation()
                
                linkState = p.getLinkState(robotId, endEffectorIndex)
                actualPos = linkState[0]
                error = math.sqrt(sum((actualPos[j] - circlePos[j])**2 for j in range(3)))
                
                if error > 0.01:  # 误差超过1cm
                    allReachable = False
                    break
            
            if allReachable and r > bestCircleRadius:
                bestCircleRadius = r
                bestCircleCenter = pos
    
    if bestCircleCenter:
        print(f"\n🎯 找到最佳画圆位置！")
        print(f"   圆心: [{bestCircleCenter[0]:.3f}, {bestCircleCenter[1]:.3f}, {bestCircleCenter[2]:.3f}]")
        print(f"   半径: {bestCircleRadius}米 ({bestCircleRadius*100:.1f}厘米)")
        print(f"\n复制以下参数到 circle_demo.py:")
        print(f"   center = [{bestCircleCenter[0]}, {bestCircleCenter[1]}, {bestCircleCenter[2]}]")
        print(f"   radius = {bestCircleRadius}")
        
        # 在模拟器中标记这个位置
        sphereId = p.createVisualShape(p.GEOM_SPHERE, radius=0.01, rgbaColor=[0, 1, 0, 0.8])
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=sphereId, basePosition=bestCircleCenter)
        
        # 画出圆
        prevPos = None
        for i in range(32):
            angle = 2 * math.pi * i / 32
            circlePos = [
                bestCircleCenter[0] + bestCircleRadius * math.cos(angle),
                bestCircleCenter[1] + bestCircleRadius * math.sin(angle),
                bestCircleCenter[2]
            ]
            
            if prevPos:
                p.addUserDebugLine(prevPos, circlePos, [0, 1, 0], 2, lifeTime=0)
            prevPos = circlePos
        
        print("\n绿色球体 = 圆心")
        print("绿色圆圈 = 建议的画圆轨迹")
    else:
        print(f"\n⚠️  没有找到足够大的可画圆位置")
        print(f"   但可以使用上面列出的单个点进行测试")

print("\n按Ctrl+C关闭...")
try:
    while True:
        p.stepSimulation()
        time.sleep(1./240.)
except KeyboardInterrupt:
    print("关闭")
    p.disconnect()

