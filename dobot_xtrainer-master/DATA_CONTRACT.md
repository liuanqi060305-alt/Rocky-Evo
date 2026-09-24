# X-Trainer 数据契约（Data Contract）

schema_version: **2** ｜ dataset_version: **2.0**

> 本文件由 `scripts/gen_data_contract.py` 自动生成，请勿手工编辑。

生成时间：2026-09-08T23:41:48

---

## 目录结构

```
<任务名>/
├── episodes_to_delete.txt        # 任务级，待删名单
├── dataset_statistics.json       # 任务级，由 scripts/dataset_stats.py 生成
└── collect_data/
    └── <episode_id>/             # 目录名 = YYYYMMDDHHMMSS
        ├── meta.json             # ★V2 新增：episode 级 metadata
        ├── topImg/    {i}.jpg    # i 从 0 起连续
        ├── leftImg/   {i}.jpg
        ├── rightImg/  {i}.jpg
        ├── observation/ {i}.pkl  # state + action + 时间戳
        └── replay.mp4            # 衍生物，非必需
```

## Observation

| 字段 | 类型 | Shape | 单位 | 含义 | 是否必需 |
|---|---|---|---|---|---|
| images.top | uint8 | (480, 640, 3) | — | 顶部相机，磁盘上 **BGR**，flip=True | 必需 |
| images.left | uint8 | (480, 640, 3) | — | 左腕相机，磁盘上 **BGR**，flip=False | 必需 |
| images.right | uint8 | (480, 640, 3) | — | 右腕相机，磁盘上 **BGR**，flip=True | 必需 |
| state | float64 | (14,) | rad / normalized | 机械臂实测关节角。0-5 左臂 rad，6 左夹爪 [0,1]，7-12 右臂 rad，13 右夹爪 [0,1] | 必需 |
| timestamp_ns | int | scalar | ns | `time.time_ns()` 墙钟。image/state/action 共用同一戳 | 必需(V2) |
| monotonic_ns | int | scalar | ns | `time.monotonic_ns()`，算间隔用，不受 NTP 调时影响 | 必需(V2) |
| schema_version | int | scalar | — | 固定 2。V1 数据无此键 | 必需(V2) |

**通道序**：磁盘上的 jpg 是 **BGR**。链路为 sensor `bgr8` → `read()` 转 RGB → `run_control.py` 再转回 BGR → `cv2.imwrite`。用 `cv2.imread` 读出即 BGR，喂给期望 RGB 的模型前必须 `cv2.cvtColor(..., COLOR_BGR2RGB)`。

## Action

| 字段 | 类型 | Shape | 单位 | 含义 | 是否必需 |
|---|---|---|---|---|---|
| control | float64 | (14,) | rad / normalized | **绝对**关节角目标，维度布局与 state 完全相同 | 必需 |

### 维度布局（state 与 action 相同）

| Index | Name | 单位 | 说明 |
|---|---|---|---|
| 0 | 左臂 J1 | rad | 由控制器「度」经 np.deg2rad 转换 |
| 1 | 左臂 J2 | rad | 由控制器「度」经 np.deg2rad 转换 |
| 2 | 左臂 J3 | rad | 由控制器「度」经 np.deg2rad 转换 |
| 3 | 左臂 J4 | rad | 由控制器「度」经 np.deg2rad 转换 |
| 4 | 左臂 J5 | rad | 由控制器「度」经 np.deg2rad 转换 |
| 5 | 左臂 J6 | rad | 由控制器「度」经 np.deg2rad 转换 |
| 6 | 左夹爪 | normalized [0,1] | 0=全闭，1=全开 |
| 7 | 右臂 J1 | rad | 由控制器「度」经 np.deg2rad 转换 |
| 8 | 右臂 J2 | rad | 由控制器「度」经 np.deg2rad 转换 |
| 9 | 右臂 J3 | rad | 由控制器「度」经 np.deg2rad 转换 |
| 10 | 右臂 J4 | rad | 由控制器「度」经 np.deg2rad 转换 |
| 11 | 右臂 J5 | rad | 由控制器「度」经 np.deg2rad 转换 |
| 12 | 右臂 J6 | rad | 由控制器「度」经 np.deg2rad 转换 |
| 13 | 右夹爪 | normalized [0,1] | 0=全闭，1=全开 |

### 时序契约

```
state[t]  ──执行 action[t]──▶  state[t+1]
```

`action[t]` 与 `state[t]` **同帧配对**，`action[t]` 是从 `state[t]` 出发的绝对目标。转 LeRobot 等格式时**无需移位**。

实测依据：`|control[t] − state[t]|` = 0.948°，`|control[t] − state[t+1]|` = 0.677°，后者更小（比值 1.40×），确认 action 是前向目标而非当前位置。

## Metadata（`meta.json`）

| 字段 | 类型 | 含义 | 是否必需 |
|---|---|---|---|
| schema_version | int | 2 | 必需 |
| dataset_version | str | 数据集版本号，当前 "2.0" | 必需 |
| episode_id | str | = episode 目录名，录制起始时刻 `YYYYMMDDHHMMSS` | 必需 |
| task_name | str | 任务名，来自 run_control.py 的 project_name | 必需 |
| episode_start_time | int | episode 起始 `time.time_ns()` | 必需 |
| episode_start_time_iso | str | 上者的人可读形式 | 必需 |
| episode_start_monotonic_ns | int | episode 起始 `time.monotonic_ns()` | 必需 |
| timestamp | str | = episode_start_time_iso，人可读别名 | 必需 |
| num_frames | int|null | 帧数。录制结束时回填 | 必需 |
| duration_s | float|null | 实际时长（由时间戳算出）。录制结束时回填 | 必需 |
| fps_measured | float|null | **实测帧率**，取代硬编码 25。录制结束时回填 | 必需 |
| left_position | list[7]|null | 左臂 TCP `[X,Y,Z,rx,ry,rz,_]`，位置 mm / 姿态 deg。录制开始时采样一次 | 可空 |
| left_angle | list[7]|null | 左臂 7 维关节角，rad（第 7 维为归一化夹爪） | 可空 |
| right_position | list[7]|null | 右臂 TCP，同 left_position | 可空 |
| right_angle | list[7]|null | 右臂 7 维关节角 | 可空 |
| velocity_available | bool | **恒 false**。无真实关节速度，V2 已不导出该字段 | 必需 |
| ee_pose_available | bool | **恒 false**。`ee_pos_quat` 恒零，V2 已不导出 | 必需 |
| gripper_feedback | bool | **恒 false**。从手无夹爪反馈，state[6]/[13] 实为上一帧指令 | 必需 |
| data_type | str | `teleop` / `replay` / `policy_rollout` | 必需 |
| success | bool|null | 任务是否成功。**null = 未标注**，需人工填 | 可空 |
| valid | bool|null | 数据是否可用于训练。**null = 未标注** | 可空 |
| note | str | 自由文本备注 | 可空 |
| image_layout | dict | 相机名、目录、分辨率、磁盘通道序、flip 设置 | 必需 |
| state_layout | dict | 14 维的索引划分与单位 | 必需 |
| action_layout | dict | 动作语义声明（absolute joint target） | 必需 |

## V2 已移除的字段

以下字段在 V1 中存在但**不含任何有效信息**，V2 不再落盘。其不可用性改由 metadata 的 `*_available` 标志显式声明。

| 字段 | 移除理由 |
|---|---|
| joint_velocities | `dobot.py:200` 与 joint_positions 是同一个变量；V1 实测 494/494 帧完全相等。字段名是 velocity 而内容是 position，会直接误导训练 |
| ee_pos_quat | `dobot.py:196` 硬编码 `np.zeros(7)`；V1 实测 494/494 帧全零。需要 TCP 请用 `env.get_XYZrxryrz_state()` |
| gripper_position | `dobot.py:108` 硬编码 `[1.0]`；V1 实测恒为 `[1,1]`。真实夹爪值在 state 的第 6/13 维 |

## 版本兼容

| | V1（旧） | V2（新） |
|---|---|---|
| 判定方式 | 无 `schema_version` 键 | `schema_version == 2` |
| 时间戳 | 无 | `timestamp_ns` + `monotonic_ns` |
| `meta.json` | 无 | 有 |
| 无效字段 | 有 3 个 | 已移除 |

**读取方式统一走 `scripts/episode_io.py`**，不要直接 `pickle.load`。该模块把 V1/V2 归一化成同一视图：V1 缺失的时间戳返回 `None`，V1 的 3 个无效字段被剥离。

```python
from scripts import episode_io
ep = episode_io.load_episode(episode_dir)
ep['state']         # (T, 14) float64
ep['action']        # (T, 14) float64
ep['timestamp_ns']  # (T,) int64，V1 数据为 None
ep['source_schema'] # 1 或 2
episode_io.effective_fps(episode_dir)   # 真实帧率，V1 回落到 fallback
```

## 实测样本情况

| episode_id | schema | 帧数 | 时间戳 | meta.json | 实测fps |
|---|---|---|---|---|---|
| 20260906011917 | V1 | 494 | 无 | 无 | 未知 |
| 20260908120000 | V2 | 5 | 有 | 有 | 18.02 |
