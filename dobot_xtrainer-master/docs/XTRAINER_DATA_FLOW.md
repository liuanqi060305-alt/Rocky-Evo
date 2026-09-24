# X-Trainer 项目数据流

本文件定义整个项目的数据目录职责与数据流转。**核心原则：同一批真实 Raw Dataset 只有一个事实来源。**

最后更新：2026-09-10

---

## 1. 目录架构

```
/home/iml/RockyEVO/                     ← 项目总目录
│
├── dobot_xtrainer/                     ← 数采工程（正式）
│   ├── datasets/                       ← ★ 唯一 Raw Dataset 实体
│   └── dobot_xtrainer-master/
│       ├── scripts/data_paths.py       ← 统一路径配置层
│       ├── config/data_paths.example.yaml
│       ├── tools/check_data_paths.py
│       └── tools/gen_dataset_manifest.py
│
├── openpi/                             ← VLA 工程（只读 Raw）
│   ├── examples/xtrainer/convert_xtrainer_data_to_lerobot.py
│   ├── src/openpi/policies/xtrainer_policy.py
│   └── src/openpi/training/config.py
│
└── data/                               ← 项目统一数据根
    ├── raw -> ../dobot_xtrainer/datasets   ← 软链接，非副本
    ├── processed/
    ├── lerobot/
    ├── rejected/
    └── test_configs/
```

`/home/iml/dobot_xtrainer/` 是**旧副本**，保留作备份，**不再从该副本启动正式采集**。

---

## 2. 四类数据职责

| 目录 | 职责 | 谁写 | 谁读 |
|---|---|---|---|
| `data/raw` | 机器人真实采集数据 | **仅数采工程** | 所有下游（只读） |
| `data/processed` | 质检索引、过滤清单、统计结果、中间产物 | QC 工具 | Converter |
| `data/lerobot` | Converter 产出的训练格式 | Converter | 训练 |
| `data/rejected` | 坏数据的**索引与说明**，非实体 | QC 工具 | 人工 / Converter |
| `data/test_configs` | 真机泛化测试配置 | 人工 | 推理评测 |

**约束：**

- Raw 采集完成后应视为不可变。不在 raw 里做格式转换，不覆盖，不删。
- 转换产物必须与 Raw 分离。
- `test_configs` 不放训练用 demonstration。
- Checkpoint 不进 Raw。

---

## 3. 路径配置

不写死开发机路径。优先级（高→低）：

1. CLI 参数 —— `run_control.py --save-data-path` / Converter `--raw-dir`
2. 环境变量 —— `XTRAINER_RAW_DATA_ROOT`、`XTRAINER_PROCESSED_DATA_ROOT`、`XTRAINER_LEROBOT_DATA_ROOT`、`XTRAINER_TEST_CONFIG_ROOT`
3. 配置文件 —— `config/data_paths.yaml`（模板见 `.example.yaml`，本机无需创建）
4. 内置默认 —— `<项目总目录>/data/{raw,...}`

因为默认值已正确指向 `data/raw`（软链接到唯一实体），**本机无需任何额外配置**。

软链接只是便利层，不是唯一实现 —— 换机时设环境变量即可，无需软链接。

```bash
# 采集侧
python -c "from scripts.data_paths import paths; print(paths.describe())"
# 体检
python tools/check_data_paths.py
```

---

## 4. 数据流

### 【数据采集】

```
X-Trainer 双臂 + 三路 D405
        ↓
run_control.py（遥操作主循环，25Hz 目标 / 实测约 18Hz）
        ↓
Raw Episode（topImg / leftImg / rightImg / observation / meta.json）
        ↓
唯一 Raw Dataset：data/raw/<dataset_version>/collect_data/<episode_id>/
```

**Raw Episode 状态**：来自真实机器人采集。包含 camera、state、action、timestamp、metadata。

### 【数据准备】

```
Raw Episode（刚采完：success=null, valid=null）
        ↓
操作员当场判断
        ├── 失败 → reject-human → 写 rejected_log.jsonl → 立刻删除原始 Episode
        └── 成功 → 什么都不做，留在盘上（留存即成功）
        ↓
build-train-manifest（自动审查 + 出清单，一步完成）
        ↓
自动质量审查（引擎 scripts/episode_audit.py，审所有 valid=null 的）
        ↓  顺带把 success=null 落成 success=true + success_source=implicit_retained
        ┌───────────────┴───────────────┐
   PASS: valid=true              FAIL: valid=false + quality_issues
        ↓                               ↓
   进 train_manifest.json        quality_rejected_manifest.json
        ↓                        （不自动删除，等人工确认）
train_manifest.json  ←── Converter 只转这里面的 Episode
        ↓
Converter（convert_xtrainer_data_to_lerobot.py）—— 只读 Raw
        ↓
LeRobot Dataset（$HF_LEROBOT_HOME 或 data/lerobot）
```

**留存即成功（implicit success）**：人工失败的 Episode 在采集后立刻删除，所以磁盘上留存的按定义就是成功示范 —— 不要求逐条人工标 `success`。收录标准是 `success != false AND valid != false`。

`DATA_AUDIT.md` 是**一次性的格式审计报告**（字段语义、已知问题），`episode_audit.py` 是**每条都跑的自动审查**，两者不同。

**Converter 必须遵守**（依据 DATA_AUDIT.md 第 3 节实测）：

- **不能**把 `joint_velocities` 当速度 —— 它在 V1 里是 `joint_positions` 的副本（`dobot.py:200` 同一变量，实测 494/494 帧相等），V2 已不再导出
- **不能**把 `ee_pos_quat` 当真实位姿 —— 硬编码 `np.zeros(7)`，实测全零，V2 已不再导出
- **不能**把 `gripper_position` 当传感器反馈 —— 硬编码 `[1.0]`，恒为 `[1,1]`，V2 已不再导出
- `state[6]` / `state[13]` 是上一帧夹爪**指令**而非测量值（从手无夹爪反馈）
- 磁盘 jpg 是 **BGR**，喂 RGB 模型前须 `cv2.cvtColor`
- fps 用实测值，不要假设 25

### 【训练】

```
LeRobot Dataset
        ↓
XTrainer DataConfig（LeRobotXTrainerDataConfig）
        ↓  top→base_0_rgb / left→left_wrist_0_rgb / right→right_wrist_0_rgb
        ↓  state = 14 维关节角(rad) + 归一化夹爪；action = 绝对关节角目标
        ↓  prompt 来自数据集 task 字段（prompt_from_task=True）
Normalization Statistics（compute_norm_stats.py）
        ↓  ★ 必须用 X-Trainer 自己的数据
        ↓  ★ 禁止复用 Realman / ALOHA / DROID 的统计量
π0.5 Fine-tuning（pi05_xtrainer_lora / pi05_xtrainer_full）
        ↓
Checkpoint
```

### 【离线推理检查】（上真机之前必做）

```
Checkpoint + 历史 Observation
        ↓
检查：action shape / 左右臂顺序 / 动作数量级 / 夹爪 / 连续性
        ↓
通过后才上真机
```

### 【部署】

```
Top / Left / Right Camera + Robot State
        ↓
X-Trainer Observation Adapter（run_inference_openpi.py）
        ↓  observation/{state,top_image,left_image,right_image} + prompt
π0.5 Policy Server（serve_policy.py，uv 环境）
        ↓  WebSocket
Action Chunk
        ↓
Denormalization（服务端 Unnormalize + AbsoluteActions）
        ↓
Safety Check（关节增量 / J3-J4 限位 / TCP 工作空间 / 相机线程存活）
        ↓
X-Trainer Robot
```

---

## 5. 数据集版本命名

采用：`xtrainer_pin_v0_debug` / `xtrainer_pin_v1` / `xtrainer_pin_v1_full` / `xtrainer_pin_v2`

**禁止**：`final`、`final2`、`new`、`latest`、`最终版`、`最新版`。

每个正式版本应能追溯到 Raw / Converted / LeRobot / Norm Stats / Training Experiment / Checkpoint —— 由 `dataset_manifest.json` 承载。

> 当前既有目录名是 `plug_and_unplug_task`（早于本规范）。**不重命名**，因为它已被 `repo_id`、norm stats 路径、`episodes_to_delete.txt` 引用。下一批正式数据启用新命名。

---

## 6. Dataset Manifest

每个 dataset version 一份 `data/raw/<version>/dataset_manifest.json`：

```bash
python tools/gen_dataset_manifest.py --task <version> --notes "说明"
```

含 `dataset_version` / `created_at` / `raw_root` / `episode_count` / `valid_episode_count` / `invalid_episode_count` / `task` / `notes`，以及待训练阶段回填的 `lerobot_repo_id` / `norm_stats_path` / `training_experiment` / `checkpoint_path`（未知则为 `null`，不编造）。

---

## 7. 人工示范阶段的数据策略（团队正式决定）

### 两类失败必须分开对待

| 失败来源 | 策略 |
|---|---|
| **人工遥操作失误**（操作员失误、任务失败、错误轨迹、中途误操作） | 记录原因后**删除原始大文件**，不长期占硬盘 |
| **VLA / RLT 自主推理失败**、human intervention | **后续重要数据资产**，不适用上面的删除策略，另行设计 |

**不要把两者混为一谈。** 当前团队已明确：人工遥操作失误轨迹不是当前 Offline RL 所需的失败数据，因此本阶段**不建立** `human_failure_dataset/`。

### 三种最终状态

| 状态 | 判定 | 处理 |
|---|---|---|
| **A. 人工失败** | 操作员当场判失败 | `reject-human`：删除原始 Episode，仅留轻量 rejected log |
| **B. 成功但坏数据** | `valid=false` | 暂不进训练；**不自动删除**，人工确认后可显式删 |
| **C. 成功且合格** | `success != false` + `valid=true` | 进入 `train_manifest.json`，用于 π0.5 |

另有 `mark-failed`：事后才发现某条不该用，但暂时不想删 —— 写 `success=false`，保留数据、排除训练。

### success 与 valid 的区别

- `success` —— 人工示范**任务**是否成功。**默认 true（留存即成功）**，只有显式写 `false` 才排除。
- `valid` —— **数据本身**是否满足训练质量。由自动审查判定。
- `valid` 三态：`true` / `false` / `null`（未审查）。**`null` ≠ `false`**：前者是还没查，后者是已判定为坏数据。

刚采完是 `success=null, valid=null`。采集程序不在这里猜 `success`（录制刚结束、人还没判断，写 true 就是编造）；审查时才把它落成显式 `true` + `success_source="implicit_retained"`，让 `meta.json` 自解释，下游不必先知道这条策略。

### 为什么技术审查失败的数据不自动删

审查规则本身可能误判。实例：既有 episode 曾因「关节增量 0.1749 rad 超过 0.17 阈值」被判 FAIL，追查后确认那是**快速人工操作的正常峰值**（p99=7.75°，峰值处是一段连续快动），而 0.17 rad 原本是 `run_inference.py` 的**推理期**安全阈值，不适用于遥操作数据。阈值已改为提示 0.17 / 硬错 0.35（物理上界约 2 倍）。

所以第一版只标 `valid=false` 等人工确认，`delete-invalid` 是显式命令且要求 `valid` 已为 `false`。

### 命令

常规流程只需两条：

```bash
# 做失败了 -> 当场记录并删除（交互确认，须输入 DELETE）
python scripts/manage_episodes.py reject-human <episode_id> --reason operator_failure

# 采完 -> 自动审查所有未审查条目 + 生成训练清单
python scripts/manage_episodes.py build-train-manifest
```

其余：

```bash
# 列出所有 Episode 的 success / valid 状态
python scripts/manage_episodes.py list

# 事后排除但保留数据 / 撤销上一条
python scripts/manage_episodes.py mark-failed <episode_id> --reason task_failed
python scripts/manage_episodes.py mark-success <episode_id>

# 只审查不出清单
python scripts/manage_episodes.py audit <episode_id>
python scripts/manage_episodes.py audit-pending      # 所有留存且 valid=null

# 人工确认确为坏数据后显式删除
python scripts/manage_episodes.py delete-invalid <episode_id>
```

批量分布概览仍用 `python scripts/check_episodes.py`，它与 audit 共用同一判定引擎 `scripts/episode_audit.py`，不会给出矛盾结论。

### 产出文件

| 文件 | 位置 | 内容 |
|---|---|---|
| `train_manifest.json` | `<raw>/<task>/` | 唯一训练清单，Converter 优先读取 |
| `rejected_log.jsonl` / `.csv` | `<rejected>/<task>/` | 删除记录（轻量，不含图片/视频/pkl） |
| `quality_rejected_manifest.json` | `<rejected>/<task>/` | `success=true` 但 `valid=false` 的清单 |

`scripts/delete_bad_episodes.sh` 是 `check_episodes.py` 每次运行都会覆盖重写的历史遗留产物（裸 `rm -rf`，无路径校验，`cd` 目标是上次运行时的目录）。**不要用它删数据**，一律走 `manage_episodes.py`。

---

## 8. Git 规则

**不进版本库**：raw / lerobot / converted 数据、`*.ckpt` `*.pth` `*.safetensors`、`*.mp4`、`*.hdf5`、`assets/`、`config/data_paths.yaml`、`__pycache__`。

**必须进版本库**：`dataset_manifest.json`、`rejected_manifest.json`、`meta.json`、`config/data_paths.example.yaml`、`docs/**`、`DATA_AUDIT.md`、`DATA_CONTRACT.md`。

体检：`python tools/check_data_paths.py`（会报出被跟踪的 >10MB 文件）。

---

## 9. 相关文档

| 文件 | 内容 |
|---|---|
| `docs/数据采集完整流程.md` | **端到端操作流程与命令**（从开机到 train_manifest） |
| `DATA_AUDIT.md` | 第一版数据格式审计报告（实测结论与已知问题） |
| `DATA_CONTRACT.md` | 字段级数据契约（自动生成） |
| `doc/遥操作与数据采集操作手册.md` | 硬件操作细节（灯语、按钮、对齐姿态） |
| `~/RockyEVO/openpi/examples/xtrainer/README.md` | VLA 侧环境与训练命令 |
