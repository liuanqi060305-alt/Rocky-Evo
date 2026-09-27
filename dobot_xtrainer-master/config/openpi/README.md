# OpenPI V2 适配补丁

`pi05_xtrainer_full_v2_107eps.patch` 是当前正式补丁，用于已经具备 X-Trainer V1 适配的 OpenPI 代码树。它只包含本轮 107 条联合数据训练需要的增量变更：

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
