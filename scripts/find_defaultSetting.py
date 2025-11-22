"""
读取机械臂当前所有设置（电机控制表寄存器）。

说明：
- 依赖 LeRobot 的 ManipulatorRobot 和对应电机总线封装（Feetech/Dynamixel）。
- 会连接机器人，遍历 follower（以及存在时的 leader）每个电机，
  读取控制表中所有可读项（EEPROM+RAM），并以 JSON 打印。
- 默认使用 So101 配置，并将校准目录指向当前文件所在目录（与项目中其它脚本一致）。

使用：
    python learn_PyBullet/find_defaultSetting.py

提示：
- 若未安装本地包，请先在仓库根目录执行：
      cd lerobot && pip install -e .[dev,test]
"""

from __future__ import annotations

DEFAULT_SETTING_EXAMPLE = """
{
  "arms": {
    "follower:main": {
      "port": "/dev/ttyACM1",
      "bus_type": "feetech",
      "motors": {
        "shoulder_pan": {
          "index": 1,
          "model": "sts3215"
        },
        "shoulder_lift": {
          "index": 2,
          "model": "sts3215"
        },
        "elbow_flex": {
          "index": 3,
          "model": "sts3215"
        },
        "wrist_flex": {
          "index": 4,
          "model": "sts3215"
        },
        "wrist_roll": {
          "index": 5,
          "model": "sts3215"
        },
        "gripper": {
          "index": 6,
          "model": "sts3215"
        }
      },
      "calibration": {
        "homing_offset": [
          -2049,
          -2005,
          -2084,
          -2070,
          -2010,
          -1887
        ],
        "drive_mode": [
          0,
          0,
          0,
          0,
          0,
          0
        ],
        "start_pos": [
          2010,
          2993,
          1113,
          2024,
          2047,
          1893
        ],
        "end_pos": [
          3073,
          -2020,
          2034,
          3094,
          -1062,
          3070
        ],
        "calib_mode": [
          "DEGREE",
          "DEGREE",
          "DEGREE",
          "DEGREE",
          "DEGREE",
          "LINEAR"
        ],
        "motor_names": [
          "shoulder_pan",
          "shoulder_lift",
          "elbow_flex",
          "wrist_flex",
          "wrist_roll",
          "gripper"
        ]
      },
      "readings": {
        "shoulder_pan": {
          "Model": 777,
          "ID": 1,
          "Baud_Rate": 0,
          "Return_Delay": 0,
          "Response_Status_Level": 1,
          "Min_Angle_Limit": 736,
          "Max_Angle_Limit": 3475,
          "Max_Temperature_Limit": 70,
          "Max_Voltage_Limit": 80,
          "Min_Voltage_Limit": 40,
          "Max_Torque_Limit": 1000,
          "Phase": 12,
          "Unloading_Condition": 44,
          "LED_Alarm_Condition": 47,
          "P_Coefficient": 16,
          "D_Coefficient": 32,
          "I_Coefficient": 0,
          "Minimum_Startup_Force": 16,
          "CW_Dead_Zone": 1,
          "CCW_Dead_Zone": 1,
          "Protection_Current": 500,
          "Angular_Resolution": 1,
          "Offset": 2023,
          "Mode": 0,
          "Protective_Torque": 20,
          "Protection_Time": 200,
          "Overload_Torque": 80,
          "Speed_closed_loop_P_proportional_coefficient": 10,
          "Over_Current_Protection_Time": 200,
          "Velocity_closed_loop_I_integral_coefficient": 200,
          "Torque_Enable": 1,
          "Acceleration": 254,
          "Goal_Position": -0.087890625,
          "Goal_Time": 0,
          "Goal_Speed": 0,
          "Torque_Limit": 1000,
          "Lock": 0,
          "Present_Position": 0.52734375,
          "Present_Speed": 0,
          "Present_Load": 0,
          "Present_Voltage": 49,
          "Present_Temperature": 25,
          "Status": 0,
          "Moving": 0,
          "Present_Current": 0,
          "Maximum_Acceleration": 254
        },
        "shoulder_lift": {
          "Model": 777,
          "ID": 2,
          "Baud_Rate": 0,
          "Return_Delay": 0,
          "Response_Status_Level": 1,
          "Min_Angle_Limit": 815,
          "Max_Angle_Limit": 3199,
          "Max_Temperature_Limit": 70,
          "Max_Voltage_Limit": 80,
          "Min_Voltage_Limit": 40,
          "Max_Torque_Limit": 1000,
          "Phase": 12,
          "Unloading_Condition": 44,
          "LED_Alarm_Condition": 47,
          "P_Coefficient": 16,
          "D_Coefficient": 32,
          "I_Coefficient": 0,
          "Minimum_Startup_Force": 16,
          "CW_Dead_Zone": 1,
          "CCW_Dead_Zone": 1,
          "Protection_Current": 500,
          "Angular_Resolution": 1,
          "Offset": 4060,
          "Mode": 0,
          "Protective_Torque": 20,
          "Protection_Time": 200,
          "Overload_Torque": 80,
          "Speed_closed_loop_P_proportional_coefficient": 10,
          "Over_Current_Protection_Time": 200,
          "Velocity_closed_loop_I_integral_coefficient": 200,
          "Torque_Enable": 1,
          "Acceleration": 254,
          "Goal_Position": -129.55078125,
          "Goal_Time": 0,
          "Goal_Speed": 0,
          "Torque_Limit": 1000,
          "Lock": 0,
          "Present_Position": -105.029296875,
          "Present_Speed": 0,
          "Present_Load": 0,
          "Present_Voltage": 49,
          "Present_Temperature": 25,
          "Status": 0,
          "Moving": 0,
          "Present_Current": 0,
          "Maximum_Acceleration": 254
        },
        "elbow_flex": {
          "Model": 777,
          "ID": 3,
          "Baud_Rate": 0,
          "Return_Delay": 0,
          "Response_Status_Level": 1,
          "Min_Angle_Limit": 850,
          "Max_Angle_Limit": 3063,
          "Max_Temperature_Limit": 70,
          "Max_Voltage_Limit": 80,
          "Min_Voltage_Limit": 40,
          "Max_Torque_Limit": 1000,
          "Phase": 12,
          "Unloading_Condition": 44,
          "LED_Alarm_Condition": 47,
          "P_Coefficient": 16,
          "D_Coefficient": 32,
          "I_Coefficient": 0,
          "Minimum_Startup_Force": 16,
          "CW_Dead_Zone": 1,
          "CCW_Dead_Zone": 1,
          "Protection_Current": 500,
          "Angular_Resolution": 1,
          "Offset": 1972,
          "Mode": 0,
          "Protective_Torque": 20,
          "Protection_Time": 200,
          "Overload_Torque": 80,
          "Speed_closed_loop_P_proportional_coefficient": 10,
          "Over_Current_Protection_Time": 200,
          "Velocity_closed_loop_I_integral_coefficient": 200,
          "Torque_Enable": 1,
          "Acceleration": 254,
          "Goal_Position": 31.552734375,
          "Goal_Time": 0,
          "Goal_Speed": 0,
          "Torque_Limit": 1000,
          "Lock": 0,
          "Present_Position": 31.81640625,
          "Present_Speed": 0,
          "Present_Load": 0,
          "Present_Voltage": 49,
          "Present_Temperature": 25,
          "Status": 0,
          "Moving": 0,
          "Present_Current": 0,
          "Maximum_Acceleration": 254
        },
        "wrist_flex": {
          "Model": 777,
          "ID": 4,
          "Baud_Rate": 0,
          "Return_Delay": 0,
          "Response_Status_Level": 1,
          "Min_Angle_Limit": 811,
          "Max_Angle_Limit": 3183,
          "Max_Temperature_Limit": 70,
          "Max_Voltage_Limit": 80,
          "Min_Voltage_Limit": 40,
          "Max_Torque_Limit": 1000,
          "Phase": 12,
          "Unloading_Condition": 44,
          "LED_Alarm_Condition": 47,
          "P_Coefficient": 16,
          "D_Coefficient": 32,
          "I_Coefficient": 0,
          "Minimum_Startup_Force": 16,
          "CW_Dead_Zone": 1,
          "CCW_Dead_Zone": 1,
          "Protection_Current": 500,
          "Angular_Resolution": 1,
          "Offset": 2040,
          "Mode": 0,
          "Protective_Torque": 20,
          "Protection_Time": 200,
          "Overload_Torque": 80,
          "Speed_closed_loop_P_proportional_coefficient": 10,
          "Over_Current_Protection_Time": 200,
          "Velocity_closed_loop_I_integral_coefficient": 200,
          "Torque_Enable": 1,
          "Acceleration": 254,
          "Goal_Position": 94.921875,
          "Goal_Time": 0,
          "Goal_Speed": 0,
          "Torque_Limit": 1000,
          "Lock": 0,
          "Present_Position": 94.921875,
          "Present_Speed": 0,
          "Present_Load": 0,
          "Present_Voltage": 49,
          "Present_Temperature": 26,
          "Status": 0,
          "Moving": 0,
          "Present_Current": 0,
          "Maximum_Acceleration": 254
        },
        "wrist_roll": {
          "Model": 777,
          "ID": 5,
          "Baud_Rate": 0,
          "Return_Delay": 0,
          "Response_Status_Level": 1,
          "Min_Angle_Limit": 45,
          "Max_Angle_Limit": 3946,
          "Max_Temperature_Limit": 70,
          "Max_Voltage_Limit": 80,
          "Min_Voltage_Limit": 40,
          "Max_Torque_Limit": 1000,
          "Phase": 12,
          "Unloading_Condition": 44,
          "LED_Alarm_Condition": 47,
          "P_Coefficient": 16,
          "D_Coefficient": 32,
          "I_Coefficient": 0,
          "Minimum_Startup_Force": 16,
          "CW_Dead_Zone": 1,
          "CCW_Dead_Zone": 1,
          "Protection_Current": 500,
          "Angular_Resolution": 1,
          "Offset": 1977,
          "Mode": 0,
          "Protective_Torque": 20,
          "Protection_Time": 200,
          "Overload_Torque": 80,
          "Speed_closed_loop_P_proportional_coefficient": 10,
          "Over_Current_Protection_Time": 200,
          "Velocity_closed_loop_I_integral_coefficient": 200,
          "Torque_Enable": 1,
          "Acceleration": 254,
          "Goal_Position": 0.3515625,
          "Goal_Time": 0,
          "Goal_Speed": 0,
          "Torque_Limit": 1000,
          "Lock": 0,
          "Present_Position": 0.703125,
          "Present_Speed": 0,
          "Present_Load": 0,
          "Present_Voltage": 49,
          "Present_Temperature": 26,
          "Status": 0,
          "Moving": 0,
          "Present_Current": 0,
          "Maximum_Acceleration": 254
        },
        "gripper": {
          "Model": 777,
          "ID": 6,
          "Baud_Rate": 0,
          "Return_Delay": 0,
          "Response_Status_Level": 1,
          "Min_Angle_Limit": 1499,
          "Max_Angle_Limit": 3010,
          "Max_Temperature_Limit": 70,
          "Max_Voltage_Limit": 80,
          "Min_Voltage_Limit": 40,
          "Max_Torque_Limit": 500,
          "Phase": 12,
          "Unloading_Condition": 44,
          "LED_Alarm_Condition": 47,
          "P_Coefficient": 16,
          "D_Coefficient": 32,
          "I_Coefficient": 0,
          "Minimum_Startup_Force": 16,
          "CW_Dead_Zone": 1,
          "CCW_Dead_Zone": 1,
          "Protection_Current": 250,
          "Angular_Resolution": 1,
          "Offset": 1009,
          "Mode": 0,
          "Protective_Torque": 20,
          "Protection_Time": 200,
          "Overload_Torque": 25,
          "Speed_closed_loop_P_proportional_coefficient": 10,
          "Over_Current_Protection_Time": 200,
          "Velocity_closed_loop_I_integral_coefficient": 200,
          "Torque_Enable": 1,
          "Acceleration": 254,
          "Goal_Position": 0.0,
          "Goal_Time": 0,
          "Goal_Speed": 0,
          "Torque_Limit": 500,
          "Lock": 0,
          "Present_Position": 0.5097705721855164,
          "Present_Speed": 0,
          "Present_Load": 0,
          "Present_Voltage": 49,
          "Present_Temperature": 26,
          "Status": 0,
          "Moving": 0,
          "Present_Current": 0,
          "Maximum_Acceleration": 254
        }
      }
    }
  }
}
"""

import json
from pathlib import Path
from typing import Any, Dict, List


def _to_py_scalar(x):
    try:
        import numpy as np

        if isinstance(x, (np.generic,)):
            return x.item()
        if isinstance(x, (list, tuple, np.ndarray)):
            return [ _to_py_scalar(v) for v in list(x) ]
    except Exception:
        pass
    return x


def _detect_bus_type(bus) -> str:
    mod = getattr(bus.__class__, "__module__", "")
    if "feetech" in mod:
        return "feetech"
    if "dynamixel" in mod:
        return "dynamixel"
    return mod or "unknown"


def _enumerate_data_keys(bus) -> List[str]:
    bus_type = _detect_bus_type(bus)
    # 取一个型号以获取控制表定义
    any_model = next(iter(bus.motors.values()))[1]
    if bus_type == "feetech":
        from lerobot.common.robot_devices.motors.feetech import MODEL_CONTROL_TABLE

        return list(MODEL_CONTROL_TABLE[any_model].keys())
    elif bus_type == "dynamixel":
        from lerobot.common.robot_devices.motors.dynamixel import MODEL_CONTROL_TABLE

        return list(MODEL_CONTROL_TABLE[any_model].keys())
    else:
        # 回退：尝试常用键集合
        return [
            "ID",
            "Torque_Enable",
            "Goal_Position",
            "Present_Position",
            "Present_Speed",
            "Present_Load",
            "Present_Current",
            "Present_Voltage",
            "Present_Temperature",
        ]


def _read_all_from_bus(bus) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    info["port"] = getattr(bus, "port", getattr(bus, "port_handler", None))
    info["bus_type"] = _detect_bus_type(bus)

    # 电机元信息
    motor_meta: Dict[str, Dict[str, Any]] = {}
    for name, (idx, model) in bus.motors.items():
        motor_meta[name] = {"index": idx, "model": model}
    info["motors"] = motor_meta

    # 读取控制表所有键
    keys = _enumerate_data_keys(bus)
    names = list(bus.motors.keys())
    # 初始化每个电机的读数字典
    per_motor: Dict[str, Dict[str, Any]] = {n: {} for n in names}

    for key in keys:
        try:
            values = bus.read(key)  # numpy array/sequence，对应所有电机
        except Exception as e:
            # 某些键可能不支持同步/被固件屏蔽，跳过但记录原因
            for n in names:
                per_motor[n][key] = {
                    "error": f"{type(e).__name__}: {e}"
                }
            continue

        # 分发到各电机
        for i, n in enumerate(names):
            try:
                per_motor[n][key] = _to_py_scalar(values[i])
            except Exception:
                per_motor[n][key] = _to_py_scalar(values)

    # 附带校准信息（若存在）
    calib = getattr(bus, "calibration", None)
    if calib is not None:
        info["calibration"] = calib

    info["readings"] = per_motor
    return info


def main() -> None:
    try:
        from lerobot.common.robot_devices.robots.configs import So101RobotConfig
        from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
    except Exception as e:  # pragma: no cover - 运行环境问题直接提示
        raise SystemExit(
            "导入 LeRobot 失败，请先安装本地包：\n"
            "  cd lerobot && pip install -e .[dev,test]\n"
            f"原始错误：{type(e).__name__}: {e}"
        )

    calib_dir = str(Path(__file__).parent)
    # 只连接 follower 手臂，避免在无设备时触发 leader 的校准与读写
    cfg = So101RobotConfig(
        calibration_dir=calib_dir,
        leader_arms={},  # 禁用 leader 端口（例如 /dev/ttyACM0）
        cameras={},      # 本脚本不需要相机
    )
    robot = ManipulatorRobot(cfg)

    # 连接并读取
    robot.connect()
    try:
        result: Dict[str, Any] = {"arms": {}}

        # follower 臂
        for name, bus in robot.follower_arms.items():
            result["arms"][f"follower:{name}"] = _read_all_from_bus(bus)

        # leader 臂（若存在）
        for name, bus in robot.leader_arms.items():
            result["arms"][f"leader:{name}"] = _read_all_from_bus(bus)

        # 打印 JSON（便于复制/保存）
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        # 断开（保持安全，避免残留力矩）
        try:
            # 尝试关闭所有 follower 力矩
            for arm in robot.follower_arms.values():
                try:
                    arm.write("Torque_Enable", 0)
                except Exception:
                    pass
        except Exception:
            pass
        robot.disconnect()


if __name__ == "__main__":
    main()
