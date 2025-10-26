
import pybullet as p
import pybullet_data

# 连接到PyBullet
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)

# 加载地面和机械臂
planeId = p.loadURDF("plane.urdf")
robotId = p.loadURDF("so101_new_calib.urdf", [0, 0, 0], 
                     p.getQuaternionFromEuler([0, 0, 0]), 
                     useFixedBase=True)

# 显示所有link的信息
numJoints = p.getNumJoints(robotId)
print(f"机械臂总共有 {numJoints} 个关节\n")
print("="*80)

for i in range(numJoints):
    info = p.getJointInfo(robotId, i)
    jointIndex = i
    jointName = info[1].decode('utf-8')
    jointType = info[2]
    linkName = info[12].decode('utf-8')
    
    # 获取link的世界坐标位置
    linkState = p.getLinkState(robotId, i)
    linkPos = linkState[0]
    
    # 关节类型
    jointTypeStr = {
        p.JOINT_REVOLUTE: "旋转关节",
        p.JOINT_PRISMATIC: "滑动关节",
        p.JOINT_FIXED: "固定关节",
        p.JOINT_SPHERICAL: "球形关节",
        p.JOINT_PLANAR: "平面关节"
    }.get(jointType, f"未知类型({jointType})")
    
    print(f"索引 {i}:")
    print(f"  关节名: {jointName}")
    print(f"  Link名: {linkName}")
    print(f"  类型: {jointTypeStr}")
    print(f"  世界坐标位置: ({linkPos[0]:.3f}, {linkPos[1]:.3f}, {linkPos[2]:.3f})")
    
    # 标记gripper相关的link
    if 'gripper' in linkName.lower() or 'jaw' in linkName.lower() or 'end' in linkName.lower():
        print(f"  ⭐ 这是夹爪相关的link！")
    
    print()

print("="*80)
print("\n建议的末端执行器索引：")

# 查找包含gripper_frame或tcp的link（通常是末端中心点）
for i in range(numJoints):
    info = p.getJointInfo(robotId, i)
    linkName = info[12].decode('utf-8')
    if 'gripper_frame' in linkName.lower() or 'tcp' in linkName.lower() or 'tool' in linkName.lower():
        print(f"  推荐使用索引 {i}: {linkName} （夹爪坐标系/工具中心点）")
        
# 如果没找到特殊的，推荐最后一个
print(f"  或者使用索引 {numJoints-1}: {p.getJointInfo(robotId, numJoints-1)[12].decode('utf-8')} （最后一个link）")

print("\n提示：")
print("  - gripper_frame_link 通常是夹爪的中心参考点")
print("  - moving_jaw_link 是活动的夹爪部分")
print("  - 一般使用 gripper_frame_link 作为IK目标")

# ==================== 可视化所有重要的link ====================
print("\n正在绘制坐标系...")

# 找到gripper_frame_link的索引
gripperFrameIndex = None
for i in range(numJoints):
    info = p.getJointInfo(robotId, i)
    linkName = info[12].decode('utf-8')
    if 'gripper_frame' in linkName.lower():
        gripperFrameIndex = i
        break

if gripperFrameIndex is None:
    gripperFrameIndex = numJoints - 1

# 绘制坐标系的函数
def drawCoordinateSystem(robotId, linkIndex, axisLength=0.05):
    """在指定link上绘制坐标系（红=X, 绿=Y, 蓝=Z）"""
    linkState = p.getLinkState(robotId, linkIndex)
    pos = linkState[0]
    orn = linkState[1]
    
    # 将四元数转换为旋转矩阵
    rotMatrix = p.getMatrixFromQuaternion(orn)
    
    # 计算三个坐标轴的终点
    xAxis = [pos[0] + rotMatrix[0] * axisLength,
             pos[1] + rotMatrix[3] * axisLength,
             pos[2] + rotMatrix[6] * axisLength]
    
    yAxis = [pos[0] + rotMatrix[1] * axisLength,
             pos[1] + rotMatrix[4] * axisLength,
             pos[2] + rotMatrix[7] * axisLength]
    
    zAxis = [pos[0] + rotMatrix[2] * axisLength,
             pos[1] + rotMatrix[5] * axisLength,
             pos[2] + rotMatrix[8] * axisLength]
    
    # 绘制三个坐标轴
    xLine = p.addUserDebugLine(pos, xAxis, [1, 0, 0], 3, 0)  # 红色 X轴
    yLine = p.addUserDebugLine(pos, yAxis, [0, 1, 0], 3, 0)  # 绿色 Y轴
    zLine = p.addUserDebugLine(pos, zAxis, [0, 0, 1], 3, 0)  # 蓝色 Z轴
    
    return [xLine, yLine, zLine]

# 绘制所有重要link的坐标系
print("\n在模拟器中绘制坐标系：")
print("  红色 = X轴")
print("  绿色 = Y轴")
print("  蓝色 = Z轴")
print()

debugItems = []

# 绘制基座坐标系（较大）
baseLines = drawCoordinateSystem(robotId, 0, axisLength=0.1)
debugItems.extend(baseLines)
print("✓ 基座坐标系（较大的坐标轴）")

# 绘制每个主要关节的坐标系
importantLinks = []
for i in range(numJoints):
    info = p.getJointInfo(robotId, i)
    linkName = info[12].decode('utf-8')
    jointType = info[2]
    
    # 只绘制旋转关节或夹爪相关的link
    if jointType == p.JOINT_REVOLUTE or 'gripper' in linkName.lower() or 'wrist' in linkName.lower():
        lines = drawCoordinateSystem(robotId, i, axisLength=0.04)
        debugItems.extend(lines)
        importantLinks.append((i, linkName))

for idx, name in importantLinks:
    print(f"✓ Link {idx}: {name}")

# 特别标记末端执行器（用球体）
linkState = p.getLinkState(robotId, gripperFrameIndex)
endPos = linkState[0]
sphereId = p.createVisualShape(p.GEOM_SPHERE, radius=0.015, rgbaColor=[1, 0, 0, 0.8])
markerId = p.createMultiBody(baseMass=0, baseVisualShapeIndex=sphereId, 
                             basePosition=endPos)
print(f"\n⭐ 末端执行器位置标记（红球）在 Link {gripperFrameIndex}")

print("\n" + "="*80)
print("模拟器窗口中现在显示了：")
print("  1. 所有主要关节的坐标系（RGB三轴）")
print("  2. 末端执行器位置（红色球体）")
print("  3. 基座有较大的坐标系作为参考")
print("="*80)

input("\n按回车键关闭...")
p.disconnect()

