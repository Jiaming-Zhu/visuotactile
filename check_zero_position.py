#!/usr/bin/env python3
"""
检查 URDF 零点姿态
"""

import pybullet as p
import pybullet_data
import numpy as np
import math

# 连接
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 加载 URDF
robotId = p.loadURDF('so101_new_calib.urdf', [0, 0, 0], useFixedBase=True)

# 设置所有关节为 0
numJoints = p.getNumJoints(robotId)
print('='*80)
print('URDF 零点姿态检查（所有关节角度=0°）')
print('='*80)
print('\n关节零点设置:')
for i in range(numJoints):
    info = p.getJointInfo(robotId, i)
    jointName = info[1].decode('utf-8')
    jointType = info[2]
    if jointType == p.JOINT_REVOLUTE:
        p.resetJointState(robotId, i, 0)
        print(f'  关节 {i} ({jointName:20s}): 0.0° (0.000 rad)')

# 等待物理更新
for _ in range(100):
    p.stepSimulation()

# 找到末端执行器
endEffectorIndex = numJoints - 1
for i in range(numJoints):
    info = p.getJointInfo(robotId, i)
    linkName = info[12].decode('utf-8')
    if 'gripper_frame' in linkName.lower():
        endEffectorIndex = i
        break

linkState = p.getLinkState(robotId, endEffectorIndex)
pos = linkState[0]
orn = linkState[1]
euler = p.getEulerFromQuaternion(orn)

print(f'\n末端执行器索引: {endEffectorIndex}')
print(f'末端位置: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]')
print(f'末端姿态(欧拉角): [{math.degrees(euler[0]):.1f}°, {math.degrees(euler[1]):.1f}°, {math.degrees(euler[2]):.1f}°]')

# 读取各关节实际状态
print('\n各关节状态验证:')
for i in range(numJoints):
    info = p.getJointInfo(robotId, i)
    jointName = info[1].decode('utf-8')
    jointType = info[2]
    if jointType == p.JOINT_REVOLUTE:
        state = p.getJointState(robotId, i)
        print(f'  {jointName:20s}: {math.degrees(state[0]):7.2f}°')

print('\n' + '='*80)
print('重要：LeRobot 校准的零位定义')
print('='*80)
print('''
根据 LeRobot 文档和代码（feetech_calibration.py line 454-456）:

校准零位定义：
  "直的水平姿态，夹爪向上且关闭"
  
具体要求：
  - 所有关节处于"四分之一圈"位置
  - 这个姿态在校准后对应所有电机角度为 0°
  - 如果设置所有 Goal_Position 为 0，机械臂会移动到这个姿态

换句话说：
  ✓ LeRobot 校准零位 = 所有舵机角度 = 0°
  ✓ URDF 零位 = 所有关节角度 = 0°
  
两者**必须对应**，否则 IK 解会导致错误的姿态！
''')

print('\n' + '='*80)
print('检查结论')
print('='*80)
print('''
需要验证：
1. 将真实机械臂设置为校准零位（所有舵机 Goal_Position = 0°）
2. 测量真实机械臂的末端位置
3. 对比 URDF 零位的末端位置
4. 如果位置差异大于 5cm，说明零点不对应！

建议测试步骤：
  a) 连接真实机械臂
  b) motors.write("Goal_Position", [0, 0, 0, 0, 0, 0])  # 所有舵机设为 0°
  c) 测量末端位置
  d) 对比上面显示的 URDF 末端位置
  e) 如果不一致，需要：
     - 方案1: 修改 URDF 文件的 joint origin
     - 方案2: 重新校准真实机械臂
     - 方案3: 在代码中添加零点偏移补偿
''')

p.disconnect()

