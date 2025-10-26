# IK 控制真实机械臂 - 完整说明

## 📋 概述

已成功将 PyBullet IK 解写入真实 SO101 机械臂的功能集成到 `interactive_control.py` 中。

## ✅ 已完成的工作

### 1. **核心功能实现**
- ✅ 实现 `write_to_real_robot()` 函数，将 IK 解写入真实机械臂
- ✅ 参照 `manipulator.py` 的 `teleop_step` 方法（line 503-530）
- ✅ 完整的单位转换流程：弧度 → 度数 → 步进值
- ✅ 安全检查：限制单步移动幅度（可选）

### 2. **新增命令**
- `connect` - 连接真实机械臂
- `disconnect` - 断开真实机械臂
- `status` - 显示连接状态

### 3. **文档完善**
- ✅ `单位转换与校准说明.md` - 详细的理论说明
- ✅ `单位转换流程图.txt` - ASCII 流程图
- ✅ `IK_控制真实机械臂使用说明.md` - 使用指南
- ✅ 代码中添加详细注释

## 🔑 关键问题解答

### Q1: IK 解的输出格式是什么？
**A:** PyBullet 的 `calculateInverseKinematics()` 输出**弧度**（rad）

```python
ik_solution = p.calculateInverseKinematics(robotId, endEffectorIndex, targetPos)
# 输出示例: [0.785, -1.571, 2.356, ...]  # 弧度
```

### Q2: 机械臂能接受什么格式的输入？
**A:** 取决于是否有校准：

#### 有校准模式（当前使用）✓
```python
# 输入：度数（°）
motors.write("Goal_Position", [45.0, -90.0, 135.0])  

# LeRobot 自动转换：度数 → 步进值
# 舵机接收：步进值（如 [2500, 1596, 3500]）
```

#### 无校准模式
```python
# 输入：步进值（0-4095）
motors.write("Goal_Position", [2500, 1596, 3500])

# 无转换，直接发送
# 舵机接收：步进值（[2500, 1596, 3500]）
```

### Q3: 有校准和无校准有什么区别？
**A:** 关键区别在于 **LeRobot 是否自动转换单位**

| 特性 | 有校准模式 | 无校准模式 |
|------|-----------|-----------|
| 输入单位 | 度数（°） | 步进值 |
| 输入范围 | -180° ~ 180° | 0 ~ 4095 |
| 自动转换 | ✓ 是 | ✗ 否 |
| 统一坐标系 | ✓ 是 | ✗ 否 |
| 方向纠正 | ✓ 支持 | ✗ 不支持 |
| 零位统一 | ✓ 所有舵机零位为 0° | ✗ 各自独立 |

**转换位置**: `feetech.py` line 1181-1182
```python
if data_name in CALIBRATION_REQUIRED and self.calibration is not None:
    values = self.revert_calibration(values, motor_names)
    # 度数 → 步进值
```

## 📊 完整的单位转换流程

```
PyBullet IK 输出
    ↓ (弧度: [0.785, -1.571, 2.356])
    
np.rad2deg()  ← interactive_control.py line 154
    ↓ (度数: [45.0, -90.0, 135.0])
    
torch.from_numpy().float()
    ↓ (tensor: [45.0, -90.0, 135.0])
    
ensure_safe_goal_position() (可选)
    ↓ (安全检查后的度数)
    
goal_pos.numpy().astype(np.float32)
    ↓ (numpy float32: [45.0, -90.0, 135.0])
    
motors.write("Goal_Position", ...)  ← interactive_control.py line 190
    ↓ (传递度数到 LeRobot)
    
revert_calibration()  ← feetech.py line 1181-1182 (自动)
    ↓ (步进值: [2500, 1596, 3500])
    
串口发送到舵机
    ↓
    
舵机执行移动
```

## 🚀 使用方法

### 1. 启动程序
```bash
cd /home/martina/Y3_Project/learn_PyBullet
python interactive_control.py
```

### 2. 连接真实机械臂
```
> connect
```

输出示例：
```
正在连接真实机械臂...
Connecting main follower arm.
Connecting main leader arm.
正在运行校准...
✅ 真实机械臂已连接
```

### 3. 发送目标位置
```
> 0.30 0.10 0.25
```

输出示例：
```
目标位置: [0.300, 0.100, 0.250]
计算IK（迭代优化中...） ✓ (迭代3次)
IK求解误差: 0.15毫米
IK计算的关节角度:
  关节0 (waist               ):   15.23°
  关节1 (shoulder            ):  -45.67°
  关节2 (elbow               ):   78.91°
  关节3 (wrist_angle         ):  -12.34°
  关节4 (wrist_rotate        ):    5.67°
  关节5 (gripper             ):    0.00°

将IK解写入真实机械臂...
IK解（弧度）: [ 0.266 -0.797  1.377 -0.215  0.099  0.   ]
IK解（度数）: [ 15.23 -45.67  78.91 -12.34   5.67   0.  ]
✓ 写入 main follower 臂（度数 → 自动转换为步进值）
  耗时: 12.34ms
✅ 成功写入真实机械臂

提示：LeRobot 已自动将度数转换为舵机步进值（0-4095）
```

### 4. 查看状态
```
> status
```

输出：
```
状态信息:
  LeRobot 模块: ✓ 已加载
  真实机械臂: ✓ 已连接
  仿真机械臂: ✓ 运行中
```

### 5. 断开连接
```
> disconnect
```

## 🔧 代码实现细节

### 核心函数：`write_to_real_robot()`

位置：`interactive_control.py` line 113-204

```python
def write_to_real_robot(joint_angles_rad):
    """
    单位转换流程：
    1. PyBullet IK 输出：弧度（rad）
    2. 转换为度数：np.rad2deg()
    3. 输入到 LeRobot：度数（float32）
    4. LeRobot 自动转换：度数 → 步进值（revert_calibration）
    5. 发送到舵机：步进值
    """
    
    # 步骤 1：弧度 → 度数
    joint_angles_deg = np.rad2deg(joint_angles_rad[:6])
    
    # 步骤 2-5：写入舵机
    for name in real_robot.follower_arms:
        goal_pos = torch.from_numpy(joint_angles_deg).float()
        
        # 可选：安全检查
        if real_robot.config.max_relative_target is not None:
            present_pos = real_robot.follower_arms[name].read("Present_Position")
            present_pos = torch.from_numpy(present_pos)
            goal_pos = ensure_safe_goal_position(
                goal_pos, present_pos, real_robot.config.max_relative_target
            )
        
        # 写入（LeRobot 自动转换度数为步进值）
        goal_pos_np = goal_pos.numpy().astype(np.float32)
        real_robot.follower_arms[name].write("Goal_Position", goal_pos_np)
```

### LeRobot 自动转换

位置：`feetech.py` line 1181-1182

```python
def write(self, data_name, values, motor_names=None):
    # ... 省略 ...
    
    # 关键：自动转换度数为步进值
    if data_name in CALIBRATION_REQUIRED and self.calibration is not None:
        values = self.revert_calibration(values, motor_names)
        # 输入：度数 → 输出：步进值
    
    # 发送到舵机
```

### 转换实现

位置：`feetech.py` line 837-890

```python
def revert_calibration(self, values, motor_names):
    """度数 → 步进值"""
    for i, name in enumerate(motor_names):
        # 1. 度数 → 分辨率范围
        values[i] = values[i] / 180.0 * (resolution // 2)
        
        # 2. 减去零位偏移
        values[i] -= homing_offset
        
        # 3. 应用旋转方向
        if drive_mode:
            values[i] *= -1
    
    return values.astype(np.int32)
```

## ⚠️ 注意事项

### 1. 校准文件必须存在
校准文件路径：`lerobot/calibration/so101_follower_main.json`

如果不存在，`robot.connect()` 时会进入交互式校准流程。

### 2. 单位必须正确
```python
# ✓ 正确：输入度数
motors.write("Goal_Position", [45.0, -90.0])

# ✗ 错误：输入弧度
motors.write("Goal_Position", [0.785, -1.571])  # 会被当作度数！

# ✗ 错误：输入步进值（有校准模式下）
motors.write("Goal_Position", [2500, 1596])  # 会被当作度数！
```

### 3. 安全距离
建议先在仿真中测试目标位置是否可达，再写入真实机械臂。

### 4. 急停准备
操作真实机械臂时，随时准备按急停按钮。

## 📁 文件清单

```
learn_PyBullet/
├── interactive_control.py          # 主程序（已更新）
├── README_IK控制真实机械臂.md       # 本文档
├── IK_控制真实机械臂使用说明.md     # 使用指南
├── 单位转换与校准说明.md           # 理论详解
└── 单位转换流程图.txt              # ASCII 流程图
```

## 🔍 调试与验证

### 检查校准状态
```python
if real_robot.follower_arms["main"].calibration is not None:
    print("✓ 有校准模式")
    print(f"校准文件: {real_robot.calibration_dir}")
else:
    print("✗ 无校准模式")
```

### 测试小角度移动
```python
# 移动 5°
motors.write("Goal_Position", [5.0, 0, 0, 0, 0, 0])
time.sleep(1)

# 读取实际位置
pos = motors.read("Present_Position")
print(f"目标: 5.0°, 实际: {pos[0]:.1f}°")
```

### 对比仿真和真实
```python
# IK 解
ik_rad = [0.785, -1.571, 2.356, 0, 0, 0]
ik_deg = np.rad2deg(ik_rad)

# 写入
motors.write("Goal_Position", ik_deg)
time.sleep(2)

# 读取
actual = motors.read("Present_Position")

# 对比
error = np.abs(ik_deg - actual)
print(f"误差: {error}")
```

## 📚 参考资料

### 相关源码
- `manipulator.py` line 503-530 - 遥操作写入逻辑
- `feetech.py` line 1113-1210 - 写入函数实现
- `feetech.py` line 837-890 - 校准转换实现
- `feetech.py` line 648-787 - 校准应用（读取）

### 文档
- [LeRobot 官方文档](https://github.com/huggingface/lerobot)
- [FeeeTech 舵机手册](https://www.feetechrc.com/)
- [PyBullet 文档](https://pybullet.org/)

## ❓ 常见问题

### Q: 为什么要转换为度数而不是直接用弧度？
A: LeRobot 的有校准模式接受度数作为标准单位，这样可以统一不同型号舵机的接口。

### Q: 能不能跳过校准直接写入步进值？
A: 可以，但需要：
1. 不调用 `robot.connect()`，手动创建 `MotorsBus`
2. 确保 `self.calibration = None`
3. 输入步进值而不是度数
4. 需要自己处理零位偏移和旋转方向

不推荐，因为很容易出错。

### Q: 如何验证转换是否正确？
A: 测试已知角度：
```python
# 测试 0°（零位）
motors.write("Goal_Position", [0, 0, 0, 0, 0, 0])
time.sleep(2)
pos = motors.read("Present_Position")
print(pos)  # 应该接近 [0, 0, 0, 0, 0, 0]

# 测试 45°
motors.write("Goal_Position", [45, 0, 0, 0, 0, 0])
time.sleep(2)
pos = motors.read("Present_Position")
print(pos[0])  # 应该接近 45.0
```

## 🎉 总结

✅ **已完成**：
- PyBullet IK 解可以正确写入真实机械臂
- 单位转换流程完全清晰：弧度 → 度数 → 步进值
- 有校准模式和无校准模式的区别已明确
- 代码参照 `manipulator.py` 的标准实现
- 文档完整，包含理论、流程图和示例

✅ **使用正确**：
- 输入到 LeRobot：度数（°）
- LeRobot 自动转换：度数 → 步进值
- 舵机接收：步进值

现在可以安全地使用 `interactive_control.py` 来控制真实机械臂了！🚀

---

**版本**: v1.0  
**日期**: 2025-10-25  
**作者**: AI Assistant

