#!/bin/bash
# ──────────────────────────────────────────────────
# Dobot Xtrainer GUI 一键启动脚本
# ──────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "  Dobot Xtrainer 控制面板"
echo "========================================"

# 激活 conda。本机是 Miniforge（~/miniforge3），不是 miniconda3 —— 原先只找
# miniconda3，找不到就静默跳过，于是用系统 python 跑、缺依赖报错。
# 两个路径都试，谁在用谁。
for CONDA_SH in "$HOME/miniforge3/etc/profile.d/conda.sh" \
                "$HOME/miniconda3/etc/profile.d/conda.sh"; do
    if [ -f "$CONDA_SH" ]; then
        source "$CONDA_SH"
        conda activate x_trainer
        break
    fi
done

if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "✗ 没找到 conda，或 x_trainer 环境不存在" >&2
    exit 1
fi
echo "环境: $CONDA_DEFAULT_ENV ($(python -V 2>&1))"

# 权限检查取代原先的「明文口令 + sudo chmod 777」（理由见 run_gui.py 注释）
if ! id -nG | tr ' ' '\n' | grep -qx dialout; then
    echo "✗ 当前用户不在 dialout 组，串口会 Permission denied" >&2
    echo "  修复：sudo usermod -aG dialout \$USER  然后重新登录" >&2
    exit 1
fi

echo "启动 GUI..."
echo ""

python run_gui.py
