# X-Trainer 数据格式审计报告

审计日期：2026-09-08
审计范围：`~/dobot_xtrainer/datasets/` 下全部已采集数据 + `experiments/run_control.py` 采集链路全部源码
审计性质：**只读**。本次审计未修改任何代码、未修改任何数据。

**样本量声明**：磁盘上目前只有 **1 条 episode**（`20260906011917`，494 帧）。
第二、六、八部分要求的「随机抽取若干 Episode」无法执行 —— 总体只有一条，已全量普查。
所有统计结论的样本量 n=1 episode / 494 帧，**不足以代表未来批量采集的分布**。

---

## 1. 数据目录结构

```
datasets/
└── plug_and_unplug_task/              ← 任务目录，名字来自 run_control.py:33 project_name
    ├── episodes_to_delete.txt         ← 0 字节，空
    └── collect_data/
        └── 20260906011917/            ← episode 目录，名字 = 录制起始时刻 YYYYMMDDHHMMSS
            ├── topImg/       0.jpg … 493.jpg   (494 个)
            ├── leftImg/      0.jpg … 493.jpg   (494 个)
            ├── rightImg/     0.jpg … 493.jpg   (494 个)
            ├── observation/  0.pkl … 493.pkl   (494 个)
            └── replay.mp4                       (9.9 MB)
```

| 路径 | 类别 | 内容 | 是否必须 | 产生位置 |
|---|---|---|---|---|
| `topImg/*.jpg` | 图像 | 顶部相机 | **必须** | `run_control.py:564` |
| `leftImg/*.jpg` | 图像 | 左腕相机 | **必须** | `run_control.py:565` |
| `rightImg/*.jpg` | 图像 | 右腕相机 | **必须** | `run_control.py:566` |
| `observation/*.pkl` | State + Action | 5 键 dict，含 state 与 action | **必须** | `run_control.py:570` → `format_obs.py:save_frame` |
| `replay.mp4` | 其它（衍生） | 三路横向拼接回放视频 | 非必须 | `run_control.py:554` → `spawn_episode_video`，纯人工检视用 |
| `episodes_to_delete.txt` | Metadata（任务级） | 待删 episode 名单 | 非必须 | `scripts/check_episodes.py:281` 写入 |

**目录名是唯一的 episode 级元数据载体** —— episode 内部没有任何 json/yaml/txt 元数据文件（已 `find` 确认）。

---

## 2. 图像数据

共 **3 路相机**。设备 SN 在 `scripts/dobot_config/dobot_settings.ini` 的 `[CAMERA]` 段。

| 项 | Camera 1 (top) | Camera 2 (left) | Camera 3 (right) |
|---|---|---|---|
| 目录 | `topImg/` | `leftImg/` | `rightImg/` |
| 文件格式 | JPEG（`cv2.imwrite`，默认质量 95） | 同 | 同 |
| 分辨率 | 480×640×3 | 480×640×3 | 480×640×3 |
| dtype | uint8 | uint8 | uint8 |
| **磁盘上的通道序** | **BGR** | **BGR** | **BGR** |
| `flip`（旋转 180°） | True | False | True |
| 文件命名 | `{idx}.jpg`，idx 从 0 起 | 同 | 同 |
| 是否带时间戳 | **否**（时间戳只在 episode 目录名上） | 同 | 同 |
| 是否连续编号 | **是**，0..493 无缺口 | 是，0..493 | 是，0..493 |
| 文件大小 | 72–77 KB | 52–66 KB | 51–83 KB |

### 通道序的完整链路（易错点，务必按此理解）

1. `realsense_camera.py` 用 `rs.format.bgr8` 从传感器取帧
2. `read()` 内 `color_image[:, :, ::-1]` → **返回 RGB**
3. `run_control.py:146` 再 `[:, :, ::-1]` → **变回 BGR**
4. `cv2.imwrite` 吃 BGR，**因此磁盘上的 jpg 是 BGR 存储、颜色正确**

结论：用 `cv2.imread` 读出来即 BGR，喂给期望 RGB 的模型前必须 `cv2.cvtColor(..., COLOR_BGR2RGB)`。
用 PIL 读则直接得到正确的 RGB。

### 保存频率

| 指标 | 值 |
|---|---|
| 代码节流目标 | `run_control.py:591-592` 目标 0.04 s，即 **25 Hz** |
| **实测帧间隔（mtime）** | 中位 0.0520 s，均值 0.0545 s |
| **实测实际频率** | **18.3 Hz** |
| episode 总时长 | 26.9 s（494 帧） |

**实际采集频率是 18.3 Hz，不是 25 Hz。** 主循环体耗时超过 40 ms，`if total_time < 0.04` 的补睡不生效。

两点说明：
- 频率由 mtime 推算。mtime 是写盘完成时刻，非曝光时刻，存在系统缓冲误差。但 494 帧 / 26.9 s = 18.4 Hz 与逐帧中位数自洽，量级可信。**精确的曝光间隔需在采集端打时间戳才能确定 —— 当前数据无此信息，标记为「待人工确认」**。
- `replay.mp4` 以 fps=25.0 硬编码写出（`run_control.py:182`），比实际快约 1.37 倍，**回放视频的速度不代表真实速度**。

### 图像质量普查（494 帧 × 3 路 = 1482 张，全量）

| 检查项 | top | left | right |
|---|---|---|---|
| 图片损坏（`imread` 返回 None/空） | 0 | 0 | 0 |
| 纯黑帧（mean < 1） | 0 | 0 | 0 |
| **重复图片**（MD5 全等） | **0** | **0** | **0** |
| 掉帧（mtime 间隔 > 0.1 s） | 0 | 0 | 0 |
| 图片缺失（编号缺口） | 0 | 0 | 0 |
| 空文件 | 0 | 0 | 0 |
| 亮度 mean（min–max） | 60.6–63.2 | 45.8–81.0 | 45.5–85.8 |
| 亮度 mean（整段平均） | 62.2 | 61.9 | 66.8 |

无黑屏、无重复、无掉帧、无缺失、无损坏。

**亮度偏低**：三路均值 62–67（满量程 255），约为中性灰的一半，与现场光线偏暗的观察一致。这不是数据错误，但会影响视觉模型效果。

**top 相机动态范围异常窄**：整段亮度 mean 只在 60.6–63.2 波动（跨度 2.6），而 left/right 跨度 35.2/40.3。可能是顶部视野内容变化本就小，也可能是该相机自动曝光锁定。**成因待人工确认**（看一眼 `replay.mp4` 顶部画面即可判断）。

---

## 3. Robot State

### 存储位置

`observation/{idx}.pkl` —— 单个 pickle dict，**5 个键**，每个文件恰好 849 字节（494/494 一致）。

```python
{
  'joint_positions':  ndarray (14,) float64,
  'joint_velocities': ndarray (14,) float64,
  'ee_pos_quat':      ndarray (14,) float64,
  'gripper_position': ndarray (2,)  float64,
  'control':          ndarray (14,) float64,   # ← 这是 Action，见第 4 部分
}
```

代码路径：`run_control.py:570` → `format_obs.py:save_frame()` → `obs["control"] = action` 后整体 `pickle.dump`。
`obs` 由 `env.get_obs()` 产生（`env.py:65-85`），底层 `dobot.py:193-203`。

### `joint_positions` (14,) —— 主 state

左臂 0–6、右臂 7–13。归属依据：`agent.py:50-51` 中 `np.concatenate([agent_left.act(), agent_right.act()])`。

| Index | Name | Description | Unit | 实测范围 |
|---|---|---|---|---|
| 0 | left_j1 | 左臂关节 1 | rad | −1.609 … −1.361 |
| 1 | left_j2 | 左臂关节 2 | rad | −0.388 … 0.092 |
| 2 | left_j3 | 左臂关节 3 | rad | −2.090 … −1.457 |
| 3 | left_j4 | 左臂关节 4 | rad | −0.042 … 2.391 |
| 4 | left_j5 | 左臂关节 5 | rad | 1.127 … 1.454 |
| 5 | left_j6 | 左臂关节 6 | rad | 1.430 … 1.819 |
| 6 | left_gripper | 左夹爪开合 | **归一化 [0,1]**，非 rad | 0.000 … 0.976 |
| 7 | right_j1 | 右臂关节 1 | rad | 1.376 … 1.591 |
| 8 | right_j2 | 右臂关节 2 | rad | −0.255 … 0.302 |
| 9 | right_j3 | 右臂关节 3 | rad | 1.498 … 2.070 |
| 10 | right_j4 | 右臂关节 4 | rad | −1.872 … 0.115 |
| 11 | right_j5 | 右臂关节 5 | rad | −1.553 … −1.334 |
| 12 | right_j6 | 右臂关节 6 | rad | −1.857 … −1.377 |
| 13 | right_gripper | 右夹爪开合 | **归一化 [0,1]**，非 rad | 0.000 … 0.968 |

单位依据：
- 关节 = 弧度。`dobot.py:105-106` 从控制器取「度」后 `np.deg2rad()` 转换。
- 夹爪 = 无量纲 [0,1]。`dynamixel.py:107-114` 显式 `(pos - open) / (close - open)` 后 `min(max(0, g_pos), 1)` 截断。
- 左右臂符号相反（如 dim2 为负、dim9 为正）与 `run_inference.py:178-179` 的安全边界一致（左臂 J3 ∈ (−2.6, 0)，右臂 J3 ∈ (0, 2.6)），互相印证。

### 其余三个键：**均为占位符，不含有效信息**

| 键 | Shape | 实测 | 源码证据 | 判定 |
|---|---|---|---|---|
| `joint_velocities` | (14,) | 非夹爪 12 维 **494/494 帧完全等于 `joint_positions`**；夹爪维恒为 1.0 | `dobot.py:200` `"joint_velocities": joints` —— 与 `joint_positions` 是**同一个变量** | **不是速度。是位置的副本。不可用** |
| `ee_pos_quat` | (14,) | **494/494 帧全零** | `dobot.py:196` `pos_quat = np.zeros(7)` 硬编码，双臂拼成 14 | **恒零占位。无 TCP 信息** |
| `gripper_position` | (2,) | **恒为 `[1.0, 1.0]`**，唯一值只有 1 种 | `dobot.py:197` `gripper_pos = np.array([joints[-1]])`，而 `dynamixel.py:108` 里从手侧 `gripper_pos = [1.0]` 硬编码 | **恒 1 占位。真实夹爪开合只在 dim 6/13** |

**从手机械臂没有夹爪位置反馈。** `dobot.py:108` 写死 `gripper_pos = [1.0]`，所以 `joint_positions[6]`/`[13]` 从控制器读回来永远是 1.0。
这两维之所以在 pkl 里有真实值，是 `run_control.py:573-574` 用上一次的 action 覆写的结果 —— 见第 5 部分。

---

## 4. Robot Action

### 存储位置

`observation/{idx}.pkl` 里的 **`control`** 键，shape (14,) float64。

写入链路：`format_obs.py:15` `obs["control"] = action`，其中 `action` 来自 `run_control.py:527` 的 `agent.act({})` —— 即**主手（遥操作手柄）的当前关节读数**，经 `servo_action_check()` 限幅后的值。

### 判定结论

| 问题 | 结论 | 依据 |
|---|---|---|
| 当前位置 or 下一时刻目标？ | **下一时刻目标** | 实测：`\|control[t] − state[t]\|` = 0.016553 rad (0.948°)；`\|control[t] − state[t+1]\|` = 0.011818 rad (0.677°)。**更接近 state[t+1]，比值 1.40×** |
| Joint or TCP？ | **Joint** | `env.step(action)` → `command_joint_state(joints)`；且 `ee_pos_quat` 恒零，全链路无 TCP 量 |
| Delta or Absolute？ | **Absolute（绝对关节角）** | 量纲与 `joint_positions` 一致（见上表 control 范围列，与 state 范围逐维吻合）；`command_joint_state` 直接下发绝对角 |

**Action = 绝对关节角目标（absolute joint target），单位与 state 完全相同。**

补充：`control[t]` 与 `state[t+1]` 仍有 0.677° 残差，因为从手是伺服跟随主手，存在跟踪误差，不会精确到位。这个残差本身就是「目标 vs 实际」的证据。

### 逐维含义

| Index | Name | Meaning | Unit |
|---|---|---|---|
| 0 | left_j1_target | 左臂关节 1 目标角 | rad |
| 1 | left_j2_target | 左臂关节 2 目标角 | rad |
| 2 | left_j3_target | 左臂关节 3 目标角 | rad |
| 3 | left_j4_target | 左臂关节 4 目标角 | rad |
| 4 | left_j5_target | 左臂关节 5 目标角 | rad |
| 5 | left_j6_target | 左臂关节 6 目标角 | rad |
| 6 | left_gripper_target | 左夹爪目标开合 | 归一化 [0,1] |
| 7 | right_j1_target | 右臂关节 1 目标角 | rad |
| 8 | right_j2_target | 右臂关节 2 目标角 | rad |
| 9 | right_j3_target | 右臂关节 3 目标角 | rad |
| 10 | right_j4_target | 右臂关节 4 目标角 | rad |
| 11 | right_j5_target | 右臂关节 5 目标角 | rad |
| 12 | right_j6_target | 右臂关节 6 目标角 | rad |
| 13 | right_gripper_target | 右夹爪目标开合 | 归一化 [0,1] |

---

## 5. State 与 Action 时序关系

### 结论：`Action[t]` 配对 `State[t]`，配对是**正确**的模仿学习约定

`run_control.py` 主循环内的执行顺序（录制分支）：

```
第 527 行   action = agent.act({})            ← 读主手，得到本帧 action[t]
   ...
第 564-566  cv2.imwrite(... img_list[n])      ← 存图像[t]
第 570 行   save_frame(obs_dir, idx, obs, action)
                                              ← 存 (obs, action[t])
                                                注意：obs 是上一轮第 572 行的产物
第 572 行   obs = env.step(action, flag_in)   ← 下发 action[t]，读回新 obs
第 573-574  obs["joint_positions"][6]  = action[6]
            obs["joint_positions"][13] = action[13]   ← 用 action 覆写夹爪维
```

因此第 570 行落盘的那一对是：

```
state[t]  = env.step() 在下发 action[t-1] 之后读回的机械臂实测角
action[t] = 本帧主手读数，即从 state[t] 出发要去的目标
```

这正是标准约定：**state[t] 是执行 action[t] 之前的状态，action[t] 是从该状态发出的指令。** 转换到 LeRobot 时 `state ← joint_positions`、`actions ← control` 直接对应，无需移位。

### 实测铁证：夹爪维的精确一步滞后

因为 `state` 的夹爪维是被 `action` 覆写的，可以用它反查时序：

| 维度 | `\|state[t] − control[t−1]\|` | `\|state[t] − control[t]\|` |
|---|---|---|
| dim 6（左夹爪） | **0.00e+00** | 4.02e−03 |
| dim 13（右夹爪） | **0.00e+00** | 3.92e−03 |

`state[t]` 的夹爪维**精确等于** `control[t−1]`，误差为 0。这既证明了覆写发生在 `save_frame` 之后（所以存的是上一轮的值），也证明了 `state[t]` / `action[t]` 的错位关系正是「前状态 / 当前指令」。

### 一个必须知道的副作用

`state[6]` 和 `state[13]` **不是传感器测量值，是上一帧的指令值**（因为从手无夹爪反馈，见第 3 部分）。

对模仿学习无害 —— 夹爪指令通常能到位，用上一帧指令近似当前开合是常见做法，`run_inference.py:219-220` 推理时也做同样的事，训练/推理一致。**但它不是真实观测，不能当作夹爪状态的 ground truth 用于误差分析。**

### 图像的时序

图像在第 564-566 行落盘，取自相机线程持续更新的 `img_list`。**图像没有独立时间戳**，其采样时刻与 `state[t]`（上一轮末读取）存在一个未量化的偏移。

`img_list` 用整帧引用交换保证单帧自洽（`run_control.py:120-128` 注释，实测撕裂率 0%），但**图像与 state 的精确时间对齐程度无法从现有数据确认 —— 标记「待人工确认」**。以 18.3 Hz 计，偏移量级不超过一帧（约 55 ms）。

---

## 6. 同步关系检查

episode `20260906011917` 全量核对：

```
topImg      : 494
leftImg     : 494
rightImg    : 494
observation : 494   （State 与 Action 同在一个 pkl 内，天然等长）
```

| 检查项 | 结果 |
|---|---|
| 四个目录文件数 | 全部 494，**一致** |
| 四个目录的帧号**集合**是否逐一相同 | **完全相同**（不只是数量相同） |
| 帧号范围 | 0 … 493，无缺口 |
| State 与 Action 长度 | 均 494。二者存在同一 pkl 内，结构上不可能不等长 |

**无长度差异，无需解释成因。**

值得注意：State 与 Action 打包在同一文件里，这个设计**天然排除了两者长度不一致的可能**。相比 ALOHA 等把 action 单独存一个数组的方案，这一点更稳健。

首帧编号为 0 的成因（`run_control.py:555-563`）：`idx += 1` 后紧跟 `if mk_dir(left_dir): idx = 0`，新建目录时返回 True 从而把 idx 重置为 0。所以每条 episode 都从 `0.jpg` 开始。

---

## 7. Metadata

### 目前**已经**保存的

| 项 | 载体 | 形式 | 备注 |
|---|---|---|---|
| episode 起始时刻 | **episode 目录名** | `YYYYMMDDHHMMSS`，如 `20260906011917` | 由 `run_control.py:99` 按 B 键时生成 |
| 任务名 | **任务目录名** | `plug_and_unplug_task` | 来自 `run_control.py:33` |
| 帧序号 | **文件名** | `0.jpg` / `0.pkl` | 隐式表达时序 |
| 待删名单 | `episodes_to_delete.txt` | 任务级文本，当前 0 字节 | 由 `check_episodes.py` 写 |

**仅此四项，全部靠文件名/目录名隐式承载。没有任何结构化元数据文件。**

### 目前**缺少**的

| 缺失项 | 影响 |
|---|---|
| **每帧时间戳** | 无法确定真实采集频率（只能用 mtime 近似），无法确定图像与 state 的精确对齐 |
| **实际采集帧率** | 代码目标 25 Hz，实测 18.3 Hz。下游若按 25 Hz 解释会导致时间轴错误 |
| 相机 SN ↔ 位置映射 | 只存在 ini 里，不随数据走。换机/重插后无法回溯当时哪台相机在哪个位置 |
| 相机内参 / flip 设置 | `flip` 是 top=True/left=False/right=True，不随数据保存 |
| 成功/失败标记 | 无法区分成功演示与失败演示，训练时无法筛选 |
| 数据有效性标记 | 只有一个全局 `episodes_to_delete.txt`，无 per-episode 标记 |
| 任务语言描述（prompt） | VLA 训练必需。当前只能从目录名推导 |
| 软件版本 / git commit | 无法复现采集时的代码状态 |
| 操作者 | 多人采集时无法区分风格差异 |
| 夹爪标定值（open/close 角） | `gripper_config` 只在 ini。夹爪归一化依赖它，换标定后旧数据量纲改变且无从察觉 |

### 建议未来增加（**仅建议，本次未修改任何程序**）

按需求方给出的字段清单，逐项给出可行性判断：

| 建议字段 | 可行性 | 说明 |
|---|---|---|
| `left_position` | ✅ 可加 | 左臂 TCP 位置。**注意 `ee_pos_quat` 当前恒零不可用**，需调 `env.get_XYZrxryrz_state()`（该接口已存在，`run_inference.py:187` 在用） |
| `left_angle` | ✅ 已有等价物 | 即 `joint_positions[0:6]`，已保存 |
| `right_position` | ✅ 可加 | 同 `left_position`，取右臂分量 |
| `right_angle` | ✅ 已有等价物 | 即 `joint_positions[7:13]` |
| `data_type` | ✅ 可加 | 建议枚举：`teleop` / `replay` / `policy_rollout` |
| `success` | ⚠ 需人工判定 | 程序无法自动判断插拔是否成功，需采集后人工标注或加外部传感器。（**后续决定**：改为「留存即成功」—— 失败的当场 `reject-human` 删掉，留存即视为成功，免去逐条标注。见 `docs/XTRAINER_DATA_FLOW.md` 第 7 节） |
| `valid` | ✅ 可加 | 建议替代现在的全局 `episodes_to_delete.txt`，改为 per-episode 标记 |
| `note` | ✅ 可加 | 自由文本，记录异常情况 |

**额外强烈建议**（优先级高于上面几项，因为影响训练正确性）：

1. **每帧时间戳** —— 存进 pkl，如 `obs["timestamp"] = time.time()`。这是当前最大的信息缺口，直接影响时间轴正确性。
2. **episode 级 `meta.json`** —— 一次写入，含 fps 实测值、相机 SN 映射、flip 设置、`gripper_config`、git commit、prompt 文本。
3. **prompt 文本** —— VLA 训练必需，不应只靠目录名反推。

---

## 8. 异常数据检查

全量普查（1 episode / 494 帧 / 1482 张图 / 494 个 pkl）：

| 检查项 | 结果 | 说明 |
|---|---|---|
| **NaN** | **0** | `joint_positions` 与 `control` 全 14 维逐帧检查 |
| **Inf** | **0** | 同上 |
| 图片损坏 | **0** | 1482 张全部 `imread` 成功且非空 |
| 空文件 | **0** | `find -size 0` 无命中 |
| pkl 大小异常 | **0** | 494 个文件**全部恰好 849 字节** |
| Action 长度异常 | **0** | 全部 (14,) |
| State 长度异常 | **0** | 全部 (14,) |
| 图片数量异常 | **0** | 三路各 494，与 pkl 数量一致 |
| Episode 提前结束 | **无法判定** | 见下 |
| Episode 结束过晚 | **无法判定** | 见下 |

### 「提前/过晚结束」无法判定的原因

Episode 的起止完全由**人按 B 键**控制（`run_control.py:95-102`），程序不记录任务是否完成。判断 26.9 s / 494 帧是否合理，需要知道这次插拔任务的预期时长 —— 数据里没有这个信息。

**标记「待人工确认」**：请看 `replay.mp4` 判断首尾是否有多余的空转或截断。这也正是第 7 部分建议加 `success` 字段的原因。

### 发现的一处数值异常（已查明，无害）

`joint_positions` 的夹爪维出现 12 个**非规格化浮点数**（denormal double）：

```
dim 6  帧 174–179
dim 13 帧 275–280
例：帧174 = 1.263793e-302,  帧179 = 1.284571e-322
```

**排查过程**：初看形似未初始化内存。追查完整序列后确认是**等比衰减**：

```
帧168: 1.263793e-278
帧169: 1.263793e-282     每帧 ×1e-4
...                       严格等比
帧179: 1.284571e-322
帧180: 0.000000e+00      ← 落到 0
```

夹爪指令平滑趋零的过程中穿过了 float64 的非规格化区间。**不是数据损坏。**

**对下游的影响：无。** 实测转成 float32（LeRobot 的存储类型）后这 12 个值全部变为精确 0 —— float32 最小非规格化数约 1.4e−45，远大于 1e−302。而它们的物理意图本来就是「夹爪全闭 = 0」，归零后语义正确。

（说明：此现象大概率源于主手侧的一阶滤波/衰减实现。**具体产生位置未追到源码，标记「待人工确认」** —— 但既然实测对下游零影响，不阻塞转换工作。）

---

## 9. Data Contract

```
Episode
├── images
│   ├── top          uint8 (494, 480, 640, 3)  BGR on disk   必须
│   ├── left         uint8 (494, 480, 640, 3)  BGR on disk   必须
│   └── right        uint8 (494, 480, 640, 3)  BGR on disk   必须
│
├── state            float64 (494, 14)          必须   ← pkl['joint_positions']
│
├── action           float64 (494, 14)          必须   ← pkl['control']
│
├── unused           （存在但无信息，不要使用）
│   ├── joint_velocities  float64 (494, 14)   = state 的副本
│   ├── ee_pos_quat       float64 (494, 14)   恒 0
│   └── gripper_position  float64 (494, 2)    恒 1
│
└── metadata
    ├── episode_id   str   "20260906011917"   必须（目录名）
    ├── task_name    str   "plug_and_unplug_task"  必须（目录名）
    ├── frame_index  int   0..493             必须（文件名）
    └── timestamp    ——    缺失
```

### 字段规格

| 字段 | 类型 | Shape | 含义 | 是否必须 |
|---|---|---|---|---|
| `images.top` | uint8 | (T, 480, 640, 3) | 顶部相机，**磁盘上 BGR**，flip=True | 必须 |
| `images.left` | uint8 | (T, 480, 640, 3) | 左腕相机，**磁盘上 BGR**，flip=False | 必须 |
| `images.right` | uint8 | (T, 480, 640, 3) | 右腕相机，**磁盘上 BGR**，flip=True | 必须 |
| `state` | float64 | (T, 14) | 机械臂实测关节角。0–5 左臂 rad，6 左夹爪 [0,1]，7–12 右臂 rad，13 右夹爪 [0,1]。**夹爪两维实为上一帧指令，非测量值** | 必须 |
| `action` | float64 | (T, 14) | **绝对**关节角目标，维度定义同 state | 必须 |
| `joint_velocities` | float64 | (T, 14) | **非速度**，是 state 的副本 | 不可用 |
| `ee_pos_quat` | float64 | (T, 14) | 恒零占位，无 TCP 信息 | 不可用 |
| `gripper_position` | float64 | (T, 2) | 恒 `[1,1]` 占位 | 不可用 |
| `episode_id` | str | — | 录制起始时刻 `YYYYMMDDHHMMSS` | 必须（隐式） |
| `task_name` | str | — | 任务目录名 | 必须（隐式） |
| `frame_index` | int | — | 0 起连续 | 必须（隐式） |
| `fps` | float | — | **实测 18.3 Hz**，非代码标称的 25 | 缺失，需外部提供 |
| `timestamp` | float | — | 每帧采集时刻 | **缺失** |
| `prompt` | str | — | 任务语言描述 | **缺失**（VLA 必需） |
| `success` / `valid` | bool | — | 演示质量标记 | **缺失** |

### 时序契约

```
state[t]  ──execute action[t]──▶  state[t+1]
```

`action[t]` 与 `state[t]` 同帧配对，`action[t]` 是从 `state[t]` 出发的绝对目标。转 LeRobot 时无需移位。

---

## 10. 已发现的问题

按严重程度排序。

### P0 — 影响训练正确性，必须处理

| # | 问题 | 证据 | 建议（不修改代码） |
|---|---|---|---|
| 1 | **实际帧率 18.3 Hz，代码标称 25 Hz** | mtime 实测中位 0.0520 s / 均值 0.0545 s；494 帧 / 26.9 s | 转换脚本的 `fps` 参数应填 **18** 或实测值，不要沿用 25。否则 LeRobot 数据集时间轴被压缩 1.37×，影响任何依赖 `timestamp` 的逻辑。**当前 `convert_xtrainer_data_to_lerobot.py` 的 `DEFAULT_FPS = 25` 需要复核** |
| 2 | **数据量严重不足** | 磁盘仅 1 条 episode / 494 帧 / 26.9 s | 手册建议单任务 50–100 条。当前无法训练，也无法做任何有统计意义的分布检查 |
| 3 | **缺每帧时间戳** | pkl 5 键中无时间信息 | 无法验证图像与 state 的对齐，无法事后核算真实帧率。建议采集端补 `obs["timestamp"]` |

### P1 — 会导致误用，需在文档中明确

| # | 问题 | 证据 |
|---|---|---|
| 4 | **`joint_velocities` 是 `joint_positions` 的副本，不是速度** | `dobot.py:200` 同一变量；实测 494/494 帧非夹爪维完全相等。若误当速度用会引入完全错误的特征 |
| 5 | **`ee_pos_quat` 恒零** | `dobot.py:196` `np.zeros(7)` 硬编码；实测 494/494 全零。想用 TCP 必须改走 `env.get_XYZrxryrz_state()` |
| 6 | **`gripper_position` 恒 `[1,1]`** | `dobot.py:108` / `dynamixel.py:108` 硬编码。真实夹爪值只在 `joint_positions[6]`/`[13]` |
| 7 | **`state[6]`/`state[13]` 不是测量值，是上一帧指令** | 从手无夹爪反馈；实测 `\|state[t] − control[t−1]\| = 0` |
| 8 | **磁盘图像是 BGR 不是 RGB** | 链路：sensor bgr8 → `read()` 转 RGB → `run_control.py:146` 转回 BGR → `imwrite`。用 `cv2.imread` 读出即 BGR，喂 RGB 模型必须转换 |

### P2 — 可用性/可维护性

| # | 问题 | 说明 |
|---|---|---|
| 9 | **缺 prompt 文本** | VLA 训练必需，当前只能从目录名反推 |
| 10 | **缺 success / valid 标记** | 无法筛掉失败演示 |
| 11 | **相机 SN 映射、flip、gripper_config 不随数据保存** | 换机或重新标定后旧数据无法解释，且不会报错 |
| 12 | **`replay.mp4` 以 25 fps 写出** | `run_control.py:182` 硬编码，比实际快 1.37×。人工检视时对速度的判断会失真 |
| 13 | **图像亮度整体偏低（mean 62–67 / 255）** | 与现场光线偏暗一致。非数据错误，但影响视觉模型效果，建议采集前补光 |
| 14 | **top 相机动态范围异常窄**（跨度 2.6 vs 左右 35/40） | 成因待人工确认：视野内容变化小，或自动曝光锁定 |

### 已排除的疑似问题

| 疑点 | 结论 |
|---|---|
| 夹爪维 12 个非规格化浮点数（1e−302 … 1e−322） | **无害**。是 ×1e−4 等比衰减穿过非规格化区间；转 float32 后全部归零，物理语义（夹爪全闭）正确 |
| 重复帧 / 掉帧 / 黑屏 / 损坏 / 空文件 / NaN / Inf | **全部 0 命中**。数据完整性良好 |
| State 与 Action 长度不一致 | **不可能发生**。二者存于同一 pkl，结构上等长 |

---

## 11. 待人工确认清单

以下无法从代码和现有数据得出结论，不做推测：

1. **真实曝光时刻与帧率** —— 现有 18.3 Hz 由 mtime 推算（写盘时刻 ≠ 曝光时刻）。精确值需采集端打时间戳。
2. **图像与 state 的精确时间对齐量** —— 图像取自相机线程异步更新的 `img_list`，与 state 的读取时刻存在未量化偏移（量级 < 1 帧 ≈ 55 ms）。
3. **Episode 起止是否恰当** —— 起止由人按 B 键决定，程序无完成度判据。需看 `replay.mp4` 人工判断首尾有无空转或截断。
4. **top 相机动态范围窄的成因** —— 视野内容变化小，还是自动曝光被锁定。
5. **夹爪指令等比衰减的产生位置** —— 未在源码中定位到该滤波/衰减实现。因对下游零影响，不阻塞转换工作。
6. **本审计结论的普适性** —— 全部基于 **1 条 episode**。批量采集后需重跑一次审计，确认结论仍成立。

---

## 12. 对 OpenPI Converter 的直接结论

供下一阶段直接引用：

| 项 | 值 |
|---|---|
| `state` 来源 | `pkl['joint_positions']`，(14,) float64 |
| `actions` 来源 | `pkl['control']`，(14,) float64，**绝对关节角** |
| 时序处理 | `state[t]` / `action[t]` 同帧配对，**无需移位** |
| 图像来源 | `topImg` / `leftImg` / `rightImg`，磁盘 **BGR**，需 `cv2.cvtColor(..., COLOR_BGR2RGB)` |
| 图像尺寸 | 480×640×3 uint8 |
| **fps** | **实测 18.3，非 25 —— 需复核转换脚本的 `DEFAULT_FPS`** |
| 帧号 | 0 起连续，四个目录集合完全一致，可直接用交集或任一目录枚举 |
| 不要读取的键 | `joint_velocities`、`ee_pos_quat`、`gripper_position`（全为占位符） |
| delta / absolute | action 是绝对值。若模型需要 delta，需自行相减（关节维转 delta、夹爪维保持绝对是常见做法） |
| prompt | 数据内无此字段，需从任务目录名生成或外部指定 |

---

*本报告仅记录审计结果。未修改任何代码，未修改任何数据。*

