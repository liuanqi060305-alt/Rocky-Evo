#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "用法: $0 <智算实例主机名>" >&2
    echo "示例: $0 a21260898526949376894161" >&2
}

if [[ $# -ne 1 ]]; then
    usage
    exit 2
fi

INSTANCE_HOST="$1"
if [[ ! "$INSTANCE_HOST" =~ ^a[0-9]+$ ]]; then
    echo "实例主机名格式不正确: $INSTANCE_HOST" >&2
    echo "请复制智算平台 SSH 命令中 root@ 后面的内容。" >&2
    exit 2
fi

KEY_PATH="${SZU_SSH_KEY:-/home/iml/.ssh/id_rsa}"
PROXY_ADDR="${SZU_PROXY_ADDR:-member.aicloud.szu.edu.cn:30027}"
LOCAL_PORT="${LOCAL_PORT:-8000}"
REMOTE_PORT="${REMOTE_PORT:-8000}"
STATE_DIR="${XDG_RUNTIME_DIR:-/tmp}/xtrainer-szu-inference"
PID_FILE="$STATE_DIR/tunnel.pid"
TUNNEL_LOG="$STATE_DIR/tunnel.log"

if [[ ! -r "$KEY_PATH" ]]; then
    echo "SSH 私钥不存在或不可读: $KEY_PATH" >&2
    exit 1
fi

mkdir -p "$STATE_DIR"

SSH_OPTIONS=(
    -i "$KEY_PATH"
    -o IdentitiesOnly=yes
    -o BatchMode=yes
    -o ConnectTimeout=15
    -o ServerAliveInterval=30
    -o ServerAliveCountMax=3
    -o StrictHostKeyChecking=accept-new
    -o "ProxyCommand=nc -X 5 -x $PROXY_ADDR %h %p"
)

echo "[1/3] 连接智算实例并启动推理服务: $INSTANCE_HOST"
ssh "${SSH_OPTIONS[@]}" "root@$INSTANCE_HOST" 'bash -s' <<'REMOTE'
set -euo pipefail
BASE="/share/home/tm904895221620000/a1173895530/openpi训练"
LOG="$BASE/logs/serve_plug_v1_29999.log"
PID="$BASE/run/serve_plug_v1_29999.pid"
mkdir -p "$BASE/logs" "$BASE/run"

if pgrep -af '[s]cripts/serve_policy.py.*--port=8000' >/dev/null; then
    echo "远端推理服务已经运行。"
else
    cd "$BASE/openpi"
    if [[ -x ./scripts/serve_xtrainer_szu.sh ]]; then
        nohup setsid ./scripts/serve_xtrainer_szu.sh > "$LOG" 2>&1 < /dev/null &
    else
        # Older persistent copies of the repository do not contain the helper
        # script, so start the same service directly.
        nohup setsid env \
            HF_LEROBOT_HOME="$BASE/hf_lerobot" \
            HF_DATASETS_CACHE="$BASE/hf_datasets_cache" \
            OPENPI_DATA_HOME="/share/home/tm904895221620000/a1173895530/weights" \
            HF_HUB_OFFLINE=1 \
            HF_DATASETS_OFFLINE=1 \
            UV_OFFLINE=1 \
            XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
            PYTHONUNBUFFERED=1 \
            .venv/bin/python scripts/serve_policy.py \
                --port=8000 \
                --default-prompt="plug and unplug" \
                policy:checkpoint \
                --policy.config=pi05_xtrainer_full \
                --policy.dir="$BASE/checkpoints/pi05_xtrainer_full/plug_v1_54eps_full_h100/29999" \
            > "$LOG" 2>&1 < /dev/null &
    fi
    echo $! > "$PID"
    echo "远端推理服务正在加载模型，日志: $LOG"
fi
REMOTE

if curl -fsS --max-time 2 "http://127.0.0.1:$LOCAL_PORT/healthz" >/dev/null 2>&1; then
    echo "[2/3] 本地端口 $LOCAL_PORT 已连接到健康的推理服务。"
else
    if [[ -f "$PID_FILE" ]]; then
        OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
        if [[ "$OLD_PID" =~ ^[0-9]+$ ]] && kill -0 "$OLD_PID" 2>/dev/null; then
            kill "$OLD_PID" 2>/dev/null || true
            wait "$OLD_PID" 2>/dev/null || true
        fi
        rm -f "$PID_FILE"
    fi

    if ss -ltn "sport = :$LOCAL_PORT" | tail -n +2 | grep -q .; then
        echo "本地端口 $LOCAL_PORT 已被其他程序占用，请先关闭占用程序。" >&2
        exit 1
    fi

    echo "[2/3] 建立本地 SSH 隧道 127.0.0.1:$LOCAL_PORT"
    # Start in a separate session so closing the launcher terminal does not
    # tear down the port-forward process.
    nohup setsid ssh "${SSH_OPTIONS[@]}" \
        -o ExitOnForwardFailure=yes \
        -N -L "$LOCAL_PORT:127.0.0.1:$REMOTE_PORT" \
        "root@$INSTANCE_HOST" > "$TUNNEL_LOG" 2>&1 < /dev/null &
    TUNNEL_PID=$!
    echo "$TUNNEL_PID" > "$PID_FILE"

    for _ in $(seq 1 120); do
        if curl -fsS --max-time 2 "http://127.0.0.1:$LOCAL_PORT/healthz" >/dev/null 2>&1; then
            break
        fi
        if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
            echo "SSH 隧道提前退出：" >&2
            cat "$TUNNEL_LOG" >&2
            exit 1
        fi
        sleep 2
    done
fi

if ! curl -fsS --max-time 5 "http://127.0.0.1:$LOCAL_PORT/healthz" >/dev/null; then
    echo "等待 4 分钟后服务仍未就绪。" >&2
    echo "查看远端日志：" >&2
    echo "ssh ... root@$INSTANCE_HOST 'tail -n 100 /share/home/tm904895221620000/a1173895530/openpi训练/logs/serve_plug_v1_29999.log'" >&2
    exit 1
fi

echo "[3/3] 推理服务已就绪: http://127.0.0.1:$LOCAL_PORT"
echo "SSH 隧道 PID: $(cat "$PID_FILE" 2>/dev/null || echo 'existing')"
echo "现在可以运行 experiments/run_inference_openpi.py。"
