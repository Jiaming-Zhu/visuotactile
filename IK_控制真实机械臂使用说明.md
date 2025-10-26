# IK 控制真实机械臂使用说明

## 功能概述

`interactive_control.py` 现在支持将 PyBullet 计算的 IK 解直接写入真实的 SO101 机械臂。该功能参照了 `manipulator.py` 中的遥操作控制逻辑。

## 核心功能

### 1. 双模式运行
- **仿真模式**：在 PyBullet 中模拟机械臂运动
- **真实模式**：同时控制仿真和真实机械臂

### 2. 关键函数

#### `write_to_real_robot(joint_angles_rad)`
将关节角度写入真实机械臂，参照 `manipulator.py` 的 `teleop_step` 方法。

**实现要点**：
```python
# 1. 弧度转度数（LeRobot 使用度数单位）
joint_angles_deg = np.rad2deg(joint_angles_rad[:6])

# 2. 转换为 PyTorch tensor
goal_pos = torch.from_numpy(joint_angles_deg).float()

# 3. 可选的安全检查（限制单步移动幅度）
if real_robot.config.max_relative_target is not None:
    present_pos = real_robot.follower_arms[name].read("Present_Position")
    present_pos = torch.from_numpy(present_pos)
    goal_pos = ensure_safe_goal_position(goal_pos, present_pos, 
                                        real_robot.config.max_relative_target)

# 4. 转换为 numpy float32
goal_pos_np = goal_pos.numpy().astype(np.float32)

# 5. 写入舵机
real_robot.follower_arms[name].write("Goal_Position", goal_pos_np)
```

## 使用方法

### 第一步：启动程序
```bash
cd /home/martina/Y3_Project/learn_PyBullet
python interactive_control.py
```

### 第二步：连接真实机械臂
```
> connect
```

程序会自动：
- 加载 SO101RobotConfig 配置
- 连接舵机总线
- 运行校准流程
- 启用扭矩

### 第三步：发送目标位置
```
> 0.30 0.10 0.25
```

程序会：
1. 在 PyBullet 中计算 IK 解
2. 显示计算的关节角度（度数）
3. **自动将 IK 解写入真实机械臂**
4. 同时在仿真中显示运动

### 第四步：查看状态
```
> status
```

输出示例：
```
状态信息:
  LeRobot 模块: ✓ 已加载
  真实机械臂: ✓ 已连接
  仿真机械臂: ✓ 运行中
```

### 第五步：断开连接
```
> disconnect
```

## 新增命令

| 命令 | 功能 |
|------|------|
| `connect` | 连接真实机械臂 |
| `disconnect` | 断开真实机械臂 |
| `status` | 显示连接状态 |

## 安全特性

### 1. 单位转换
- PyBullet IK 输出：弧度（rad）
- LeRobot 输入：度数（°），范围 -180 到 180

### 2. 移动幅度限制
- 如果配置了 `max_relative_target`，会自动限制单步移动幅度
- 使用 `ensure_safe_goal_position` 函数防止突然的大幅移动

### 3. 错误处理
- 连接失败时不会影响仿真
- 写入失败时会显示错误信息但程序继续运行
- 程序退出时自动断开机械臂连接

## 代码对照

### 参考源码：`manipulator.py` line 503-530

```python
# 步骤 2：将 leader 位置发送给 follower 臂
follower_goal_pos = {}
for name in self.follower_arms:
    before_fwrite_t = time.perf_counter()
    goal_pos = leader_pos[name]
    
    # 安全检查：限制单步移动幅度
    if self.config.max_relative_target is not None:
        present_pos = self.follower_arms[name].read("Present_Position")
        present_pos = torch.from_numpy(present_pos)
        goal_pos = ensure_safe_goal_position(goal_pos, present_pos, 
                                            self.config.max_relative_target)
    
    follower_goal_pos[name] = goal_pos
    
    # 将目标位置写入 follower 舵机
    goal_pos = goal_pos.numpy().astype(np.float32)
    self.follower_arms[name].write("Goal_Position", goal_pos)
```

### 实现代码：`interactive_control.py` line 113-167

完全参照上述逻辑实现，主要区别：
- 数据源：IK 计算结果（弧度）→ 需要转换为度数
- 只控制 follower 臂（前 6 个关节，不包括夹爪）

## 工作流程

```
用户输入坐标
    ↓
PyBullet 计算 IK（弧度）
    ↓
转换为度数
    ↓
创建 PyTorch tensor
    ↓
可选：安全检查（限制移动幅度）
    ↓
转换为 numpy float32
    ↓
写入真实机械臂（Goal_Position）
    ↓
同时控制仿真运动
```

## 调试信息

程序会显示详细的执行信息：

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
关节角度（度）: [ 15.23 -45.67  78.91 -12.34   5.67   0.  ]
✓ 写入 main follower 臂，耗时: 12.34ms
✅ 成功写入真实机械臂
```

## 注意事项

1. **首次使用前**：确保已完成机械臂校准，校准文件应位于 `lerobot/calibration/` 目录
2. **安全距离**：建议先在仿真中测试目标位置是否可达
3. **急停准备**：操作真实机械臂时，随时准备按急停按钮
4. **工作空间**：确保目标位置在机械臂工作空间内

## 故障排除

### 问题 1：无法连接真实机械臂
```
❌ LeRobot 模块未加载，无法连接真实机械臂
```
**解决**：检查 lerobot 路径是否正确，确保 `sys.path` 包含 lerobot 目录

### 问题 2：写入失败
```
❌ 写入真实机械臂失败: ...
```
**解决**：
- 检查舵机总线连接
- 确认舵机电源开启
- 检查串口权限（可能需要 `sudo`）

### 问题 3：运动不流畅
**原因**：可能是单步移动幅度过大
**解决**：在 SO101RobotConfig 中设置 `max_relative_target` 参数

## 扩展功能建议

- [ ] 添加实时位置反馈（读取真实机械臂位置并显示）
- [ ] 支持轨迹规划（一次性发送多个目标点）
- [ ] 添加力反馈监控
- [ ] 支持录制和回放动作序列

