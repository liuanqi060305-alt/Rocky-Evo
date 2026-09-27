# 深圳大学智算平台：X-Trainer VLA 第二版训练流程

本文用于把 Dobot X-Trainer 新采集的数据整理成一个可复现的 V2 数据集，并在深圳大学智算平台使用单卡 H100 完成 pi0.5 全量微调、checkpoint 验证和真机评测。

## 1. 本轮训练方案

- 训练数据默认采用“第一版 54 条 + 审查通过的新训练集数据”，避免只训练新数据造成原始位置能力遗忘。
- 按完整位置条件划分 `train` 和 `test`，不要把同一位置随机拆分到两边。
- 语言指令继续统一使用 `plug and unplug`。
- 新建 LeRobot repo id：`xtrainer/plug_and_unplug_task_v2`。
- 新建 OpenPI 配置：`pi05_xtrainer_full_v2`，超参数与学校 H100 第一版保持一致，只更换数据集 repo id。
- 新建实验名，不覆盖 `plug_v1_54eps_full_h100`。
- 从 `pi05_base` 开始一轮新的 30000-step 全量微调；旧 V1 的 `29999` 不作为 `--resume` 目标。

选择重新从基座训练，是为了让 V1 和 V2 可以公平比较。`--resume` 只用于 V2 训练意外中断后，继续同一个 V2 实验目录。

## 2. 当前数据状态（2026-09-27 22:12 检查）

- Raw 目录：`/home/iml/RockyEVO/data/raw/plug_and_unplug_task`
- 实体目录：`/home/iml/RockyEVO/dobot_xtrainer/datasets/plug_and_unplug_task`
- 检查时共有 188 个 episode，约 14GB。
- 第一版 54 条已经是 `success=true, valid=true`，并标注为“左右均水平 0 度”。
- 四组 ±15 度位置数据共 81 条，已经标为 `split=train`。
- 另有 53 条没有位置标签，其中包含 2026-09-27 新采的 50 条和此前 3 条未分类数据。
- 当前 `train_manifest.json` 仍只包含第一版 54 条；现在直接转换只会得到旧数据集。
- 随后采集进程已正常退出，最终稳定在 188 条；最后一条为 288 帧。
- 已对 134 条 `valid=null` 数据完成只读审查：`PASS=82`、`FAIL=52`，没有写回任何状态。
- 52 条失败只涉及两类阈值：39 条少于 300 帧，27 条存在大于 0.35rad（20°）的单帧动作跳变，其中 14 条同时命中两类；没有发现图像损坏、模态帧号错位或 NaN/Inf。

如果重新开始采集，这些数字会继续变化；本轮训练应以 188 条冻结范围为起点。

## 3. 阶段 A：冻结数据范围

### A1. 正常结束采集

先停止当前 episode 的录制，再正常退出采集程序。不要在文件仍写入时运行审查、打包或转换。

```bash
pgrep -af run_control.py
```

确认没有 `run_control.py` 后，再继续下面步骤。最后一个 episode 必须同时具有完整的 `topImg`、`leftImg`、`rightImg`、`observation` 和 `meta.json`。

### A2. 确定最新一批的标签和用途

对每一组位置明确以下信息：

- 人类可读名称，例如“左右均水平 0 度”；
- `position_id`；
- 左右侧角度或位移；
- `split=train` 或 `split=test`。

如果所有位置都标为 `train`，模型可以使用全部数据，但之后不能把这些位置称为“未见位置泛化测试”。推荐至少保留一个完整位置条件作为 `test`，真机测试时只在该条件上评价泛化能力。

未标注 episode 在转换器中会为了兼容历史数据被当成 `train`，因此正式转换前必须先处理当前未分类数据，不能依赖这个回退行为。

### A3. 数据质量审查

```bash
conda activate x_trainer
cd /home/iml/RockyEVO/dobot_xtrainer/dobot_xtrainer-master
export XTRAINER_TASK_NAME=plug_and_unplug_task

# 先只读审查，不写状态
python scripts/manage_episodes.py --task plug_and_unplug_task audit-pending --no-write

# 人工查看报告并决定阈值命中数据是否保留后，再正式审查并生成训练白名单
python scripts/manage_episodes.py --task plug_and_unplug_task build-train-manifest
```

审查失败只会写 `valid=false`，不会自动删除。本轮只读审查中的 52 条阈值失败不能直接等同于坏示范：短轨迹可能是高效完成任务，动作跳变也可能是正常快速操作。需要结合回放、跳变所在关节、任务是否完成，以及各位置组数量平衡逐条复核。也不要为了凑数量未经查看就批量改成 `valid=true`。

按照当前严格阈值，最终训练集会是旧 54 条加 82 条新 PASS，共 136 条；人工复核保留部分阈值命中数据后，数量会增加。四个位置组必须分别统计，避免某一组因为短轨迹较多而被系统性少采样。

生成后检查：

```bash
python scripts/manage_episodes.py --task plug_and_unplug_task list
python scripts/check_episodes.py
python tools/gen_dataset_manifest.py \
  --task plug_and_unplug_task \
  --notes "V2：第一版54条加位置泛化新数据；训练前冻结快照"
```

必须记录最终四个数字：Raw 总数、训练集数量、测试集数量、排除数量。

## 4. 阶段 B：创建不可变 Raw 快照

不要把仍会继续采集的目录直接交给平台转换。生成一个带日期和 episode 数量的快照，排除训练不需要的 `replay.mp4`，同时保留 `meta.json`、`train_manifest.json` 和 `dataset_manifest.json`。

```bash
DATE=$(date +%Y%m%d)
EPISODES=188  # 示例；替换成停止采集并审查后的实际 Raw 总数
SNAP="plug_v2_${EPISODES}eps_${DATE}"
OUT="/home/iml/RockyEVO/data/processed/${SNAP}_raw.tar.zst"

cd /home/iml/RockyEVO/dobot_xtrainer/datasets
tar --zstd -cf "$OUT" \
  --exclude='plug_and_unplug_task/collect_data/*/replay.mp4' \
  --exclude='plug_and_unplug_task/position_views' \
  plug_and_unplug_task

cd "$(dirname "$OUT")"
sha256sum "$(basename "$OUT")" | tee "$(basename "$OUT").sha256"
```

Raw 中约 1.24GB 是 `replay.mp4`，转换器不会读取它们。用单个归档上传也能避免传输约 25 万个小文件。

## 5. 阶段 C：上传到学校持久存储

先使用平台当前提供的 Bita 登录令牌登录。令牌不要写进文档或 Git。

```bash
bita login \
  -u a1173895530 \
  -p '<当前Bita登录令牌>' \
  -t szdx \
  -e https://console.aicloud.szu.edu.cn
```

上传归档和校验文件：

```bash
bita upload \
  -e https://console.aicloud.szu.edu.cn \
  -b home \
  -c 1964608207239503873 \
  -o "tm904895221620000/a1173895530/datasets/raw_snapshots/${SNAP}_raw.tar.zst" \
  -l "$OUT"

bita upload \
  -e https://console.aicloud.szu.edu.cn \
  -b home \
  -c 1964608207239503873 \
  -o "tm904895221620000/a1173895530/datasets/raw_snapshots/${SNAP}_raw.tar.zst.sha256" \
  -l "${OUT}.sha256"
```

## 6. 阶段 D：创建 H100 训练实例

在智算平台创建调试任务：

- GPU：单卡 H100 80GB；
- 镜像：已经验证过的 `loongforge_pi0.5`；
- 时长：建议先给 6 小时，完成后及时关闭；
- 挂载以下三个持久目录，源路径和挂载路径保持一致：
  - `/share/home/tm904895221620000/a1173895530/datasets`
  - `/share/home/tm904895221620000/a1173895530/openpi训练`
  - `/share/home/tm904895221620000/a1173895530/weights`

创建实例属于会消耗算力额度的操作，由项目成员在平台界面确认。实例启动后复制当前 SSH 命令中的 `root@...` 主机名；旧实例 ID 不能复用。

## 7. 阶段 E：平台内解包和转换 LeRobot

SSH 登录后：

```bash
ACCOUNT=/share/home/tm904895221620000/a1173895530
BASE="$ACCOUNT/openpi训练"
ARCHIVE="$ACCOUNT/datasets/raw_snapshots/${SNAP}_raw.tar.zst"
SNAP_ROOT="$ACCOUNT/datasets/raw_snapshots/$SNAP"

cd "$(dirname "$ARCHIVE")"
sha256sum -c "${ARCHIVE}.sha256"

mkdir -p "$SNAP_ROOT"
tar --zstd -xf "$ARCHIVE" -C "$SNAP_ROOT"

RAW="$SNAP_ROOT/plug_and_unplug_task"
test -f "$RAW/train_manifest.json"
test -d "$RAW/collect_data"
```

转换时只取 `split=train`，测试位置不会进入模型：

```bash
cd "$BASE/openpi"

HF_LEROBOT_HOME="$BASE/hf_lerobot" \
HF_DATASETS_CACHE="$BASE/hf_datasets_cache" \
HF_HUB_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
UV_OFFLINE=1 \
.venv/bin/python examples/xtrainer/convert_xtrainer_data_to_lerobot.py \
  --raw-dir "$RAW" \
  --repo-id xtrainer/plug_and_unplug_task_v2 \
  --generalization-split train \
  --prompt 'plug and unplug' \
  2>&1 | tee "$BASE/logs/${SNAP}_convert.log"
```

转换结束后检查：

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import json
p = Path('/share/home/tm904895221620000/a1173895530/openpi训练/hf_lerobot/xtrainer/plug_and_unplug_task_v2/meta')
info = json.loads((p / 'info.json').read_text())
tasks = [json.loads(x) for x in (p / 'tasks.jsonl').read_text().splitlines()]
print(info['total_episodes'], info['total_frames'], info['fps'])
print(tasks)
assert info['total_episodes'] > 0
assert tasks == [{'task_index': 0, 'task': 'plug and unplug'}]
PY
```

## 8. 阶段 F：V2 配置与归一化统计

正式运行前，`config.py` 中必须存在 `pi05_xtrainer_full_v2`，其模型、优化器和超参数与 `pi05_xtrainer_full` 相同，唯一的数据差异是：

```text
repo_id = xtrainer/plug_and_unplug_task_v2
```

然后重新计算 norm stats。数据变了就必须重算，不能复用 V1 的统计量。

```bash
BASE=/share/home/tm904895221620000/a1173895530/openpi训练
cd "$BASE/openpi"

TRAIN_CONFIG=pi05_xtrainer_full_v2 \
./scripts/compute_norm_stats_xtrainer_szu.sh \
  2>&1 | tee "$BASE/logs/plug_v2_norm_stats.log"
```

检查文件存在且不是 V1 路径：

```bash
test -f "$BASE/openpi/assets/pi05_xtrainer_full_v2/xtrainer/plug_and_unplug_task_v2/norm_stats.json"
```

## 9. 阶段 G：启动 H100 全量微调

超参数保持学校文档版本：batch 32、30000 steps、seed 42、num workers 2、Cosine LR（warmup 1000、peak 2.5e-5、末值 2.5e-6）、AdamW、梯度裁剪 1.0、EMA 0.99、每 1000 步保存、每 5000 步永久保留。

第一次启动使用一个从未存在过的实验名，不加 `--overwrite`：

```bash
BASE=/share/home/tm904895221620000/a1173895530/openpi训练
cd "$BASE/openpi"

TRAIN_EPISODES=185  # 示例；替换成转换后 info.json 中的 total_episodes
DATE=20260927       # 使用本轮冻结数据的日期
EXP="plug_v2_${TRAIN_EPISODES}eps_full_h100_${DATE}"
mkdir -p "$BASE/logs" "$BASE/run"

nohup setsid env \
  TRAIN_CONFIG=pi05_xtrainer_full_v2 \
  EXP_NAME="$EXP" \
  BATCH_SIZE=32 \
  WANDB_MODE=offline \
  ./scripts/train_xtrainer_szu.sh \
  > "$BASE/logs/$EXP.log" 2>&1 < /dev/null &

echo $! > "$BASE/run/$EXP.pid"
echo "$EXP"
```

`--overwrite` 会删除同名 checkpoint，只有明确要从头重跑且确认旧目录不再需要时才能使用。

## 10. 阶段 H：监控、异常恢复和完成判定

```bash
BASE=/share/home/tm904895221620000/a1173895530/openpi训练
EXP="plug_v2_${TRAIN_EPISODES}eps_full_h100_${DATE}"

tail -F "$BASE/logs/$EXP.log"
ps -fp "$(cat "$BASE/run/$EXP.pid")"
nvidia-smi
grep -E 'Step|loss=|Saving|Traceback|out of memory|Killed' "$BASE/logs/$EXP.log" | tail -50
```

loss 应整体下降，但单个训练 loss 不能证明真机成功率。出现短期波动正常；出现 NaN、持续上升、OOM、进程消失或 traceback 才需要停止排查。

实例意外结束后，在同一个实验名下续训：

```bash
TRAIN_CONFIG=pi05_xtrainer_full_v2 \
EXP_NAME="$EXP" \
BATCH_SIZE=32 \
WANDB_MODE=offline \
./scripts/train_xtrainer_szu.sh --resume
```

不要对 V1 的 `plug_v1_54eps_full_h100` 使用这条 V2 续训命令。

完成标准：

1. 日志走完 30000 steps，末尾没有 traceback；
2. 训练进程正常退出；
3. checkpoint 目录中存在最终步目录，按当前实现通常为 `29999`；
4. `params`、`train_state`、`assets/xtrainer/plug_and_unplug_task_v2/norm_stats.json` 均存在；
5. 保留转换日志、训练日志、Raw SHA256 和最终 manifest。

## 11. 阶段 I：checkpoint 服务和真机评测

不必下载约 42GB 的完整 checkpoint；优先在平台启动 policy server，再通过 SSH 隧道连接真机电脑。

服务端的配置和 checkpoint 必须同时使用 V2：

```bash
CHECKPOINT_DIR="$BASE/checkpoints/pi05_xtrainer_full_v2/$EXP/29999" \
PORT=8000 \
DEFAULT_PROMPT='plug and unplug' \
./scripts/serve_xtrainer_szu.sh
```

`serve_xtrainer_szu.sh` 当前默认写死 V1 配置，正式运行 V2 前需要先改成可通过环境变量指定 `POLICY_CONFIG`，否则即使目录指向 V2，也可能加载错误的 norm stats。

真机侧顺序：

1. 建立 SSH 隧道；
2. `python experiments/check_openpi_server.py`；
3. 第一次使用 `--dry-run` 检查观测和动作数值；
4. 正式运行，按位置条件分别记录成功率；
5. 用相同相机、光照、初始姿态、次数和失败定义比较 V1 与 V2；
6. 至少比较 5000、10000、15000、20000、25000、29999 中若干 checkpoint，最终一步不一定真机表现最好。

## 12. 分工

项目成员必须完成：

- 正常结束当前采集程序；
- 告知最新一批数据对应的物体位置、角度，以及哪些位置用于训练或留出测试；
- 在智算平台确认创建 H100 实例，因为这会消耗算力额度；
- 提供当前实例 SSH 主机名；
- 完成真机现场安全检查、急停监护和成功/失败判定。

Codex 可以完成：

- 扫描并审查全部 episode，逐条汇总异常，按确认后的规则补齐标签；
- 生成并校验 `train_manifest.json`、`dataset_manifest.json` 和版本化标签清单；
- 创建去除 replay 的 Raw 快照、SHA256，并执行 Bita 上传；
- 增加 `pi05_xtrainer_full_v2` 配置，并让训练、norm stats、服务脚本支持 V2；
- 登录已创建的实例，校验挂载、解包、转换 LeRobot、核对 episode/frame/prompt；
- 计算 norm stats、启动训练、持续检查 loss、GPU、checkpoint 和异常；
- 在中断后安全续训，训练完成后验证 checkpoint 完整性；
- 配置 V2 policy server、SSH 隧道和客户端命令；
- 把代码、清单和操作文档用中文提交说明纳入 Rocky-Evo 版本管理。
