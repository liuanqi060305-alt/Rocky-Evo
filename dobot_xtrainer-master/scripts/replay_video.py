"""
从采集的原始数据（topImg/leftImg/rightImg）生成三路横向拼接的回放视频。
不驱动机械臂，纯离线可视化。

用法：
    python scripts/replay_video.py [轨迹目录] [--fps 25] [--out 输出.mp4] [--show]
    默认轨迹目录 = DEFAULT_DIR；默认输出 = 轨迹目录/replay.mp4
"""
import os
import argparse
import cv2
import numpy as np

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.data_paths import paths  # noqa: E402


def _default_dir():
    """默认取任务下最新一条 episode。

    原实现写死了 episode `20260719142708`，该目录早已不存在，默认参数是失效的。
    改成动态取最新，并且路径经统一配置层解析。
    """
    task = os.environ.get("XTRAINER_TASK_NAME", "plug_and_unplug_task")
    cdir = paths.collect_dir(task)
    if not cdir.is_dir():
        return str(cdir)
    eps = sorted((d for d in cdir.iterdir() if d.is_dir()), key=lambda p: p.name)
    return str(eps[-1]) if eps else str(cdir)


DEFAULT_DIR = _default_dir()
CAMS = ["topImg", "leftImg", "rightImg"]  # 拼接顺序：上 / 左 / 右
W, H = 640, 480


def frame_indices(traj_dir):
    d = os.path.join(traj_dir, "topImg")
    if not os.path.isdir(d):
        raise SystemExit(f"[ERROR] 找不到 topImg 目录: {d}")
    idxs = [int(f[:-4]) for f in os.listdir(d) if f.endswith(".jpg")]
    return sorted(idxs)


def load_stitched(traj_dir, i):
    """读第 i 帧三路图并横向拼接；缺帧用黑底红字占位。"""
    imgs = []
    for cam in CAMS:
        p = os.path.join(traj_dir, cam, f"{i}.jpg")
        im = cv2.imread(p)
        if im is None:
            im = np.zeros((H, W, 3), dtype=np.uint8)
            cv2.putText(im, f"MISSING {cam}/{i}", (20, H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            im = cv2.resize(im, (W, H))
        cv2.putText(im, cam, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        imgs.append(im)
    canvas = np.concatenate(imgs, axis=1)
    cv2.putText(canvas, f"frame {i}", (10, H - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj_dir", nargs="?", default=DEFAULT_DIR, help="轨迹目录")
    ap.add_argument("--fps", type=float, default=None,
                    help="视频帧率。默认从数据的真实时间戳推算(V2)；"
                         "旧数据(V1)无时间戳时回落到 25")
    ap.add_argument("--out", default=None, help="输出 mp4 路径")
    ap.add_argument("--show", action="store_true", help="边写边预览窗口")
    args = ap.parse_args()

    traj_dir = args.traj_dir.rstrip("/")
    out_path = args.out or os.path.join(traj_dir, "replay.mp4")
    idxs = frame_indices(traj_dir)
    if not idxs:
        raise SystemExit(f"[ERROR] {traj_dir} 下没有帧")

    # V2：帧率优先取真实时间戳算出的实测值。显式 --fps 仍然最高优先，
    # 便于故意慢放/快放检视。
    if args.fps is not None:
        fps, src = args.fps, "命令行指定"
    else:
        try:
            from scripts.episode_io import effective_fps
        except ImportError:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from scripts.episode_io import effective_fps
        fps = effective_fps(traj_dir, fallback=25.0)
        src = "实测时间戳" if abs(fps - 25.0) > 1e-9 else "回落默认(无时间戳)"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W * len(CAMS), H))
    if not writer.isOpened():
        raise SystemExit(f"[ERROR] 无法创建视频: {out_path}")

    print(f"[replay_video] {traj_dir}")
    print(f"  帧数={len(idxs)} (帧号 {idxs[0]}~{idxs[-1]})  fps={fps:.3f} ({src})  -> {out_path}")
    for n, i in enumerate(idxs):
        canvas = load_stitched(traj_dir, i)
        writer.write(canvas)
        if args.show:
            cv2.imshow("replay", canvas)
            if cv2.waitKey(max(1, int(1000 / fps))) == 27:  # ESC 退出
                break
        if n % 100 == 0:
            print(f"  写入 {n}/{len(idxs)}")
    writer.release()
    if args.show:
        cv2.destroyAllWindows()
    print(f"[replay_video] 完成，视频已保存: {out_path}")


if __name__ == "__main__":
    main()
