# OpenPI 训练适配补丁

## V3：全量 188 条

`pi05_xtrainer_full_v3_all188.patch` 用于已经具备 X-Trainer V1/V2 适配的 OpenPI 代码树。它新增：

- 配置名：`pi05_xtrainer_full_v3_all188`；
- 数据集：`xtrainer/plug_and_unplug_task_v3_all188`；
- 单卡 H100、全参数微调、batch size 32；
- 50,000 个训练 step，最终 checkpoint 为 `49999`；
- 每 1,000 step 保存，5,000 的整数倍永久保留。

50,000 steps 不是由 Episode 数量自动生成的。它使 188 条、63,495 帧数据获得约 `50000 × 32 / 63495 = 25.2` 次等效遍历，与 V2 的约 24.9 次接近。

在 OpenPI 根目录执行：

```bash
git apply --check /path/to/pi05_xtrainer_full_v3_all188.patch
git apply /path/to/pi05_xtrainer_full_v3_all188.patch
.venv/bin/python -c "from openpi.training.config import get_config; print(get_config('pi05_xtrainer_full_v3_all188'))"
```

如果配置名已经存在，说明补丁已应用，不要重复执行。

## V2：107 条

`pi05_xtrainer_full_v2_107eps.patch` 是 V2 的历史正式补丁，用于已经具备 X-Trainer V1 适配的 OpenPI 代码树。它只包含 107 条联合数据训练需要的增量变更：

- 新增 `pi05_xtrainer_full_v2_107eps`；
- 数据集指向 `xtrainer/plug_and_unplug_task_v2_107eps`；
- 保持 V1 H100 全量微调超参不变；
- 让训练脚本为 V2 full config 默认选择 batch size 32；
- 让服务脚本通过 `POLICY_CONFIG` 选择模型配置。

在 OpenPI 根目录执行：

```bash
git apply --check /path/to/pi05_xtrainer_full_v2_107eps.patch
git apply /path/to/pi05_xtrainer_full_v2_107eps.patch
bash -n scripts/train_xtrainer_szu.sh scripts/serve_xtrainer_szu.sh
.venv/bin/python -c "from openpi.training.config import get_config; print(get_config('pi05_xtrainer_full_v2_107eps'))"
```

如果配置名已经存在，说明补丁已应用，不要重复执行。

`pi05_xtrainer_full_v2_104eps.patch` 只保留为历史记录；104 条方案已被 107 条方案取代，平台上不要再应用它。
