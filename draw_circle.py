
import pybullet as p
import time
import pybullet_data
import math

# ==================== 初始化 ====================
physicsClient = p.connect(p.GUI)
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

print(f"机械臂末端执行器: 索引 {endEffectorIndex}")

# ==================== 画圆参数（可调整）====================
center = [0.30, 0.10, 0.25]  # 圆心位置 [x, y, z] - 使用find_workspace找到的最佳位置
radius = 0.1                # 半径（米）- 先用小圆测试
steps = 50                    # 圆周上的点数
threshold = 0.020            # 误差阈值（米）- 20mm

print(f"\n画圆参数:")
print(f"  圆心: {center}")
print(f"  半径: {radius}米 ({radius*100:.1f}厘米)")
print(f"  点数: {steps}")
print(f"  误差阈值: {threshold}米 ({threshold*1000:.0f}毫米)")

# ==================== 画圆 ====================
print(f"\n开始画圆...\n")

prevActualPos = None
successCount = 0
failCount = 0

for step in range(steps):
    # 计算圆周上的目标点
    angle = 2 * math.pi * step / steps
    targetPos = [
        center[0] + radius * math.cos(angle),
        center[1] + radius * math.sin(angle),
        center[2] + radius * math.cos(angle)
    ]
    
    # 在目标点放置标记（红色小球）
    if step % 5 == 0:  # 每5个点放一个标记
        sphereId = p.createVisualShape(p.GEOM_SPHERE, radius=0.003, rgbaColor=[1, 0, 0, 0.5])
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=sphereId, basePosition=targetPos)
    
    # 改进的IK：增加迭代次数，提高精度
    jointPoses = p.calculateInverseKinematics(
        robotId, 
        endEffectorIndex, 
        targetPos,
        maxNumIterations=1000,      # 从默认20增加到1000
        residualThreshold=0.0001    # 降低收敛阈值
    )
    
    # 控制机械臂移动
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
    maxWaitSteps = 300
    reached = False
    
    for waitStep in range(maxWaitSteps):
        p.stepSimulation()
        time.sleep(1./240.)
        
        # 检查是否到达
        linkState = p.getLinkState(robotId, endEffectorIndex)
        actualPos = linkState[0]
        error = math.sqrt(sum((actualPos[i] - targetPos[i])**2 for i in range(3)))
        
        if error < threshold:  # 使用可配置的阈值
            reached = True
            
            # 画实际轨迹（绿色线）
            if prevActualPos is not None:
                p.addUserDebugLine(prevActualPos, actualPos, [0, 1, 0], 5, lifeTime=0)
            prevActualPos = actualPos
            
            successCount += 1
            if step % 10 == 0:
                print(f"进度: {step+1}/{steps} ({100*(step+1)/steps:.1f}%) - 到达 (误差{error*1000:.1f}mm)")
            break
    
    if not reached:
        linkState = p.getLinkState(robotId, endEffectorIndex)
        actualPos = linkState[0]
        error = math.sqrt(sum((actualPos[i] - targetPos[i])**2 for i in range(3)))
        
        # 即使未完全到达，也画轨迹（黄色线）
        if prevActualPos is not None:
            p.addUserDebugLine(prevActualPos, actualPos, [1, 1, 0], 3, lifeTime=0)
        prevActualPos = actualPos
        
        failCount += 1
        print(f"进度: {step+1}/{steps} - 超时 (误差{error*1000:.1f}mm)")

# ==================== 完成 ====================
print(f"\n{'='*60}")
print(f"画圆完成！")
print(f"  成功: {successCount}/{steps} ({100*successCount/steps:.1f}%)")
print(f"  失败: {failCount}/{steps} ({100*failCount/steps:.1f}%)")
print(f"\n图例:")
print(f"  🟢 绿色粗线 = 成功到达的轨迹")
print(f"  🟡 黄色细线 = 未完全到达的轨迹")
print(f"  🔴 红色小球 = 目标点标记")
print(f"{'='*60}")

# 保持窗口打开
print("\n按Ctrl+C关闭...")
try:
    while True:
        p.stepSimulation()
        time.sleep(1./240.)
except KeyboardInterrupt:
    print("关闭")
    p.disconnect()

