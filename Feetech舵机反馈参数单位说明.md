# Feetech 舵机反馈参数单位说明

## 📊 完整参数单位表

| 参数名称 | 英文名称 | 单位 | 数值范围 | 说明 |
|---------|---------|------|---------|------|
| **位置** | `Present_Position` | **度（°）** | -180° ~ +180° | 校准后的角度值 |
| **速度** | `Present_Speed` | **步/秒** | 0 ~ 65535 | 原始步进值（无校准） |
| **负载** | `Present_Load` | **原始值** | 0 ~ 65535 | 实际常见 0-2048+，非精确百分比 ⚠️ |
| **电流** | `Present_Current` | **毫安（mA）** | 0 ~ 65535 | 实际电流值 |
| **电压** | `Present_Voltage` | **0.1伏（dV）** | 50 ~ 250 | 实际值需除以10（如：120 = 12.0V）|
| **温度** | `Present_Temperature` | **摄氏度（℃）** | 0 ~ 100 | 舵机内部温度 |
| **移动状态** | `Moving` | **布尔值** | 0 或 1 | 0=静止，1=移动中 |

---

## 🔍 详细说明

### 1. **Present_Position（当前位置）**

**单位**：度（°）

```python
position = motors.read("Present_Position")
# 返回：numpy.ndarray，如 [45.2, -30.5, 90.0]
# 单位：度（°），范围：-180° ~ +180°
```

**特点**：
- ✅ 已经过校准处理
- ✅ 考虑了 `homing_offset`（零点偏移）
- ✅ 考虑了 `drive_mode`（旋转方向）
- 📌 这是最常用的反馈参数

**原始值转换**：
```
校准后角度 = (原始步进值 + homing_offset) / (4096/2) * 180°
如果 drive_mode=1（反转），则取负值
```

---

### 2. **Present_Speed（当前速度）**

**单位**：步/秒（steps/s）

```python
speed = motors.read("Present_Speed")
# 返回：numpy.ndarray，如 [1024, 512, 0]
# 单位：步/秒
```

**特点**：
- ❌ **未经过校准**，返回原始步进值
- 数据格式：2字节无符号整数（0-65535）
- 高位（bit 15）：方向标志（0=CW顺时针，1=CCW逆时针）
- 低15位（bit 0-14）：速度大小

**转换为度/秒**：
```python
# 假设分辨率为 4096 步/圈
steps_per_second = speed & 0x7FFF  # 去除方向位
degrees_per_second = steps_per_second / 4096 * 360

# 例如：
# speed = 1024 → 1024 / 4096 * 360 = 90°/s
```

**方向判断**：
```python
is_ccw = (speed & 0x8000) != 0  # 检查最高位
direction = "逆时针" if is_ccw else "顺时针"
```

---

### 3. **Present_Load（当前负载）**

**单位**：原始数值（无标准单位）⚠️

```python
load = motors.read("Present_Load")
# 返回：numpy.ndarray，如 [250, 500, 1200, 2000]
# 原始数值，范围：0-65535（理论最大）
# 实际使用：通常 0-2048，重负载可能超过
```

**特点**：
- 数据格式：2字节无符号整数（0-65535）
- **实际观测范围**：0-2048+（取决于负载）
- 包含方向信息（通过位运算提取）
- **不是简单的百分比！**

**数值含义**（经验值）：
```python
# 原始值直接使用（不需要除以10）
if load < 100:
    print("轻负载或空载")
elif load < 500:
    print("正常负载")
elif load < 1000:
    print("中等负载")
elif load < 1500:
    print("较大负载")
elif load < 2000:
    print("重负载")
else:
    print("极重负载/堵转/异常")
```

**转换为大致百分比（参考）**：
```python
# 方法1：基于1024作为100%基准（可能不准确）
load_pct_ref1 = (load / 1024.0) * 100

# 方法2：基于2048作为100%基准
load_pct_ref2 = (load / 2048.0) * 100

# 例如：
# load = 250  → 约 12-24%
# load = 500  → 约 24-49%  
# load = 1000 → 约 49-98%
# load = 2000 → 约 98-196%（超过100%表示过载）
```

**⚠️ 重要提示**：
- `Present_Load` **不是精确的百分比**
- 不同舵机型号可能有不同的映射关系
- 主要用于**相对比较**和**异常检测**
- 数值 >1500-2000 通常表示异常负载

---

### 4. **Present_Current（当前电流）**

**单位**：毫安（mA）

```python
current = motors.read("Present_Current")
# 返回：numpy.ndarray，如 [120, 85, 200]
# 单位：毫安（mA）
```

**特点**：
- ✅ 直接读取的值即为毫安数
- 数据格式：2字节无符号整数
- **典型范围**：0-2000 mA（取决于舵机型号）
- STS3215 最大电流：约 2000-2500 mA

**参考值**：
- 0-50 mA：空载或轻微移动
- 50-200 mA：正常运动
- 200-500 mA：较大负载
- 500-1000 mA：重负载
- \>1000 mA：接近极限（可能触发过流保护）

---

### 5. **Present_Voltage（当前电压）**

**单位**：0.1伏（dV，分伏）

```python
voltage_raw = motors.read("Present_Voltage")
# 返回：numpy.ndarray，如 [120, 119, 121]
# 单位：0.1V（需要除以10）

# 转换为实际电压
voltage_v = voltage_raw / 10.0
# 例如：120 → 12.0V
```

**特点**：
- 数据格式：1字节无符号整数（0-255）
- **存储值 = 实际电压 × 10**
- **典型工作电压**：11.0V ~ 13.0V（12V 标称）

**转换公式**：
```python
actual_voltage = raw_value * 0.1  # 或者 raw_value / 10.0
```

**参考值**：
| 原始值 | 实际电压 | 状态 |
|-------|---------|------|
| 110 | 11.0V | 电压偏低 ⚠️ |
| 120 | 12.0V | 正常 ✅ |
| 125 | 12.5V | 正常 ✅ |
| 130 | 13.0V | 电压偏高 ⚠️ |
| 140 | 14.0V | 过压风险 ❌ |

---

### 6. **Present_Temperature（当前温度）**

**单位**：摄氏度（℃）

```python
temp = motors.read("Present_Temperature")
# 返回：numpy.ndarray，如 [35, 42, 38]
# 单位：摄氏度（℃）
```

**特点**：
- ✅ 直接读取的值即为摄氏度
- 数据格式：1字节无符号整数（0-255）
- **典型范围**：20℃ ~ 80℃
- **最高限制**：通常设为 70-80℃

**参考值**：
- 20-40℃：正常工作温度 ✅
- 40-55℃：轻微发热（长时间运行）
- 55-65℃：较热（需要注意）⚠️
- 65-75℃：过热警告 ⚠️
- \>75℃：危险，可能触发过热保护 ❌

---

### 7. **Moving（移动状态）**

**单位**：布尔值（0 或 1）

```python
moving = motors.read("Moving")
# 返回：numpy.ndarray，如 [0, 1, 0]
# 0 = 静止，1 = 移动中
```

**特点**：
- 数据格式：1字节（但只使用最低位）
- 0：舵机已到达目标位置或静止
- 1：舵机正在移动到目标位置

**应用**：
```python
# 等待舵机到达目标位置
while motors.read("Moving", "joint1")[0] == 1:
    time.sleep(0.01)
print("舵机已到达目标位置")
```

---

## 💻 使用示例代码

### 示例1：读取所有状态参数

```python
from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus

# 配置舵机总线
config = FeetechMotorsBusConfig(
    port="/dev/ttyUSB0",
    motors={"joint1": (1, "sts3215")}
)

# 连接
motors = FeetechMotorsBus(config)
motors.connect()

# 读取各种参数
position = motors.read("Present_Position")      # 度（°）
speed = motors.read("Present_Speed")            # 步/秒
load = motors.read("Present_Load")              # 千分比（需除以10）
current = motors.read("Present_Current")        # 毫安（mA）
voltage = motors.read("Present_Voltage")        # 0.1伏（需除以10）
temp = motors.read("Present_Temperature")       # 摄氏度（℃）
moving = motors.read("Moving")                  # 0或1

print(f"位置: {position[0]:.2f}°")
print(f"速度: {speed[0] & 0x7FFF} 步/秒")
print(f"负载: {load[0]} (原始值)")  # 直接显示原始值
print(f"电流: {current[0]} mA")
print(f"电压: {voltage[0] / 10:.1f} V")
print(f"温度: {temp[0]}℃")
print(f"移动: {'是' if moving[0] else '否'}")

motors.disconnect()
```

### 示例2：监控舵机健康状态

```python
import time
import numpy as np

def monitor_motor_health(motors, motor_name="joint1", duration=10):
    """监控舵机健康状态"""
    start_time = time.time()
    
    while time.time() - start_time < duration:
        # 读取状态
        current = motors.read("Present_Current", motor_name)[0]
        voltage = motors.read("Present_Voltage", motor_name)[0] / 10.0
        temp = motors.read("Present_Temperature", motor_name)[0]
        load = motors.read("Present_Load", motor_name)[0]  # 直接使用原始值
        
        # 健康检查
        warnings = []
        if current > 1000:
            warnings.append(f"⚠️  电流过高: {current} mA")
        if voltage < 11.0 or voltage > 13.0:
            warnings.append(f"⚠️  电压异常: {voltage:.1f} V")
        if temp > 65:
            warnings.append(f"⚠️  温度过高: {temp}℃")
        if load > 1500:
            warnings.append(f"⚠️  负载过高: {load}")
        
        # 显示状态
        status = "⚠️  警告" if warnings else "✅ 正常"
        print(f"{status} | 电流:{current:4d}mA | 电压:{voltage:4.1f}V | 温度:{temp:2d}℃ | 负载:{load:4d}")
        
        for warning in warnings:
            print(f"  {warning}")
        
        time.sleep(0.5)

# 使用
monitor_motor_health(motors, "joint1", duration=30)
```

### 示例3：速度和负载方向分析

```python
def analyze_motor_motion(motors, motor_name="joint1"):
    """分析舵机运动方向和负载方向"""
    speed_raw = motors.read("Present_Speed", motor_name)[0]
    load_raw = motors.read("Present_Load", motor_name)[0]
    
    # 速度分析
    speed_magnitude = speed_raw & 0x7FFF  # 低15位
    speed_ccw = (speed_raw & 0x8000) != 0  # 最高位
    speed_direction = "逆时针(CCW)" if speed_ccw else "顺时针(CW)"
    speed_deg_per_sec = speed_magnitude / 4096 * 360
    
    # 负载分析
    load_value = load_raw  # 直接使用原始值
    
    print(f"速度: {speed_deg_per_sec:.1f}°/s ({speed_direction})")
    print(f"负载: {load_value} (原始值)")
    
    # 判断运动状态
    if speed_magnitude < 10:
        print("状态: 静止")
    elif load_value < 100:
        print("状态: 轻负载或空载")
    elif load_value < 1000:
        print("状态: 正常负载")
    elif load_value < 1500:
        print("状态: 中等负载")
    else:
        print("状态: 重负载或异常")

# 使用
analyze_motor_motion(motors, "joint1")
```

---

## 📚 参考文档

1. **Feetech STS3215 舵机手册**：
   - https://docs.google.com/spreadsheets/d/1GVs7W1VS1PqdhA1nW-abeyAHhTUxKUdR/edit

2. **LeRobot Feetech 模块源码**：
   - `/lerobot/lerobot/common/robot_devices/motors/feetech.py`

3. **控制表定义**：
   - 见 `SCS_SERIES_CONTROL_TABLE` (Line 122-174)

---

## 🎯 总结

| 参数 | 单位 | 是否校准 | 用途 |
|------|------|---------|------|
| `Present_Position` | **度（°）** | ✅ 是 | 位置反馈（最常用） |
| `Present_Speed` | 步/秒 | ❌ 否 | 速度监控 |
| `Present_Load` | 原始值 | ❌ 否 | 负载监控（非精确，0-2048+） |
| `Present_Current` | **mA** | ✅ 是 | 电流监控（健康检查） |
| `Present_Voltage` | 0.1V | ❌ 否 | 电压监控（需除以10） |
| `Present_Temperature` | **℃** | ✅ 是 | 温度监控（健康检查） |
| `Moving` | 布尔值 | ✅ 是 | 运动状态 |

**关键提示**：
- ✅ `Present_Position`、`Present_Current`、`Present_Temperature` 可直接使用
- ⚠️ `Present_Load` 是原始数值（非标准百分比），实际范围可达 2000+
- ⚠️ `Present_Speed` 需要位操作提取方向和大小
- ⚠️ `Present_Voltage` 需要除以 10 才是实际电压

---

希望这份文档能帮助你理解 Feetech 舵机的反馈参数！🎉

