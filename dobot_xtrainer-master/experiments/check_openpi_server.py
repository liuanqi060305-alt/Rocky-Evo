"""只检查 openpi WebSocket 服务，不初始化相机或机械臂。"""

import argparse
import json
import signal
import sys

from openpi_client import websocket_client_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--timeout", type=int, default=15)
    args = parser.parse_args()

    def timeout_handler(_signum, _frame):
        raise TimeoutError(
            f"在 {args.timeout}s 内未连上 ws://{args.host}:{args.port}；"
            "检查智算服务日志和 SSH 隧道"
        )

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(args.timeout)
    try:
        client = websocket_client_policy.WebsocketClientPolicy(
            host=args.host,
            port=args.port,
        )
    except TimeoutError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        signal.alarm(0)
    metadata = client.get_server_metadata()
    print(f"OK: connected to ws://{args.host}:{args.port}")
    print(json.dumps(metadata, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
