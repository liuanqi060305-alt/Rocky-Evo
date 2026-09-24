# Rocky Evo

本仓库保存 Rocky Evo 双臂 VLA 项目的 X-Trainer 客户端、数据采集、相机管理、
真机推理、rollout 记录和回放工具。

## VLA baseline

首个基线版本使用标签 `vla-baseline-v1`。该版本保留：

- 三路 RealSense 相机采集及故障恢复；
- 双臂数据采集、数据审查与回放工具；
- OpenPI 客户端、连续 rollout、成功/失败标记和成功率统计；
- 异步策略请求和动作块执行；
- 当前实机安全边界及复位流程。

客户端入口位于 `dobot_xtrainer-master/experiments/run_inference_openpi.py`，操作与
部署命令见 `dobot_xtrainer-master/scripts/README.md`。

OpenPI 服务端继续使用独立仓库管理。本基线对应的上游 OpenPI 提交为
`215abfb217dbac7d5f1273282331b9b1866c0479`，本仓库不保存模型权重、checkpoint、
数据集或 rollout 视频。

## 本机配置

复制并填写设备配置：

```bash
cp dobot_xtrainer-master/scripts/dobot_config/dobot_settings.example.ini \
   dobot_xtrainer-master/scripts/dobot_config/dobot_settings.ini
```

真实的 `dobot_settings.ini`、数据、权重、日志和回滚备份均被 Git 忽略。GUI 所需的
sudo 密码可写入本机配置，也可通过 `XTRAINER_SUDO_PASSWORD` 环境变量提供。
