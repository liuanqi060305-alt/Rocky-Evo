"""
把录制的轨迹原样在真机上重放（按采集频率）。

从 observation/*.pkl 读取每帧的 'control'(14维 action)，用与采集相同的
ZMQClientRobot + RobotEnv 通道 env.step(action, [1,1]) 依次下发。

前置：先在另一终端启动 robot server：
    python experiments/launch_nodes.py

用法：
    python scripts/replay_robot.py [轨迹目录] [--hz 25] [--robot_port 6002] [--yes]

安全：
    - 回放会让真机运动，执行前请确认工作区无人无障碍。
    - 首帧会从当前姿态"缓慢插值"移动到轨迹起点，避免突跳。
    - 不加 --yes 时会要求手动确认后才开始。
"""
import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import time
import glob
import pickle
import argparse
import numpy as np
from dobot_control.env import RobotEnv
from dobot_control.robots.robot_node import ZMQClientRobot

from scripts.data_paths import paths


def _default_dir():
    """默认取任务下最新一条 episode。

    原实现写死 episode `20260718210124`，该目录已不存在，默认参数早已失效。
    """
    task = os.environ.get("XTRAINER_TASK_NAME", "plug_and_unplug_task")
    cdir = paths.collect_dir(task)
    if not cdir.is_dir():
        return str(cdir)
    eps = sorted((d for d in cdir.iterdir() if d.is_dir()), key=lambda p: p.name)
    return str(eps[-1]) if eps else str(cdir)


DEFAULT_DIR = _default_dir()


def load_actions(traj_dir):
    """按帧号顺序读取所有 pkl 里的 control(action)，返回 (N,14) 数组。"""
    obs_dir = os.path.join(traj_dir, "observation")
    files = glob.glob(os.path.join(obs_dir, "*.pkl"))
    if not files:
        raise SystemExit(f"[ERROR] {obs_dir} 下没有 pkl")
    files.sort(key=lambda p: int(os.path.basename(p)[:-4]))  # 按数字帧号排序
    actions = []
    for p in files:
        with open(p, "rb") as f:
            d = pickle.load(f)
        actions.append(np.asarray(d["control"], dtype=float))
    return np.array(actions)


def smooth_move(env, target, step=0.01, period=0.04):
    """从当前关节姿态缓慢插值移动到 target(14维)，避免首帧突跳。"""
    curr = env.get_obs()["joint_positions"]
    max_delta = float(np.abs(curr - target).max())
    steps = max(int(max_delta / step), 1)
    print(f"  首帧对齐：max_delta={max_delta:.3f} rad, 分 {steps} 步缓慢移动...")
    for jnt in np.linspace(curr, target, steps):
        env.step(jnt, np.array([1, 1]))
        time.sleep(period)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj_dir", nargs="?", default=DEFAULT_DIR, help="轨迹目录")
    ap.add_argument("--hz", type=float, default=25.0, help="回放频率(采集约25Hz)")
    ap.add_argument("--robot_port", type=int, default=6002)
    ap.add_argument("--hostname", default="127.0.0.1")
    ap.add_argument("--yes", action="store_true", help="跳过确认直接开始")
    args = ap.parse_args()

    traj_dir = args.traj_dir.rstrip("/")
    actions = load_actions(traj_dir)
    period = 1.0 / args.hz
    print(f"[replay_robot] {traj_dir}")
    print(f"  帧数={len(actions)}  回放频率={args.hz}Hz  周期={period*1000:.0f}ms")

    print("\n" + "=" * 60)
    print("⚠️  即将在【真实机械臂】上重放轨迹，机械臂会运动！")
    print("    请确认：工作区无人无障碍、急停在手边、robot server 已启动。")
    print("=" * 60)
    if not args.yes:
        if input("确认开始请输入 yes: ").strip().lower() != "yes":
            print("已取消。")
            return

    robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    env = RobotEnv(robot_client)
    try:
        smooth_move(env, actions[0])  # 缓慢对齐到起点
        print("  开始回放...")
        for i, action in enumerate(actions):
            tic = time.time()
            env.step(action, np.array([1, 1]))
            if i % 100 == 0:
                print(f"  帧 {i}/{len(actions)}")
            dt = time.time() - tic
            if dt < period:
                time.sleep(period - dt)
        print("[replay_robot] 回放完成。")
    except KeyboardInterrupt:
        print("\n手动中断 (Ctrl+C)，停止回放。")
    finally:
        robot_client.close()


if __name__ == "__main__":
    main()
