"""
批量对多条轨迹生成三路拼接回放视频。

用法：
    # 默认：处理 collect_data 下最新的 101 条轨迹
    python scripts/batch_replay_video.py

    # 指定数量
    python scripts/batch_replay_video.py --latest 50

    # 指定轨迹目录列表（空格分隔）
    python scripts/batch_replay_video.py --dirs /path/traj1 /path/traj2

    # 自定义输出根目录（默认每条轨迹的 replay.mp4 放在轨迹目录下）
    python scripts/batch_replay_video.py --out_root /path/to/output_videos

    # 自定义帧率、显示预览
    python scripts/batch_replay_video.py --fps 25 --show
"""
import os
import sys
import argparse
import cv2
import numpy as np
from glob import glob

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# ---- 配置 ----
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.data_paths import paths  # noqa: E402

# 路径经统一配置层解析，不再写死开发机绝对路径
DEFAULT_COLLECT_DIR = str(paths.collect_dir(
    os.environ.get("XTRAINER_TASK_NAME", "plug_and_unplug_task")))
CAMS = ["topImg", "leftImg", "rightImg"]
W, H = 640, 480


# ---- 核心函数（复用 replay_video.py 逻辑） ----
def frame_indices(traj_dir):
    d = os.path.join(traj_dir, "topImg")
    if not os.path.isdir(d):
        return []
    idxs = [int(f[:-4]) for f in os.listdir(d) if f.endswith(".jpg")]
    return sorted(idxs)


def load_stitched(traj_dir, i):
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


def generate_one(traj_dir, out_path, fps=25.0, show=False):
    """为单条轨迹生成回放视频，返回 (成功/失败, 帧数)。"""
    idxs = frame_indices(traj_dir)
    if not idxs:
        return False, 0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W * len(CAMS), H))
    if not writer.isOpened():
        return False, 0

    for n, i in enumerate(idxs):
        canvas = load_stitched(traj_dir, i)
        writer.write(canvas)
        if show:
            cv2.imshow("batch_replay", canvas)
            if cv2.waitKey(1) == 27:  # ESC
                show = False
    writer.release()
    return True, len(idxs)


def get_latest_dirs(collect_dir, n):
    """获取 collect_dir 下按名称排序的最新 n 个目录。"""
    all_dirs = sorted(
        [os.path.join(collect_dir, d) for d in os.listdir(collect_dir)
         if os.path.isdir(os.path.join(collect_dir, d))]
    )
    return all_dirs[-n:] if n > 0 else all_dirs


# ---- 主入口 ----
def main():
    ap = argparse.ArgumentParser(description="批量生成回放视频")
    ap.add_argument("--latest", type=int, default=101,
                    help="处理 collect_data 下最新的 N 条 (默认 101)")
    ap.add_argument("--dirs", nargs="+", default=None,
                    help="手动指定轨迹目录列表")
    ap.add_argument("--collect_dir", default=DEFAULT_COLLECT_DIR,
                    help="collect_data 根目录")
    ap.add_argument("--out_root", default=None,
                    help="输出根目录 (默认每条视频放在对应轨迹目录下)")
    ap.add_argument("--fps", type=float, default=25.0, help="视频帧率 (默认 25)")
    ap.add_argument("--show", action="store_true", help="边生成边预览 (ESC 跳过预览)")
    args = ap.parse_args()

    # ---- 确定轨迹目录列表 ----
    if args.dirs:
        traj_dirs = [d.rstrip("/") for d in args.dirs]
    else:
        traj_dirs = get_latest_dirs(args.collect_dir, args.latest)

    if not traj_dirs:
        raise SystemExit("[ERROR] 没有找到轨迹目录")

    print(f"[batch_replay_video] 共 {len(traj_dirs)} 条轨迹待处理")
    print(f"  collect_dir = {args.collect_dir}")
    print(f"  fps = {args.fps}")
    if args.out_root:
        os.makedirs(args.out_root, exist_ok=True)
        print(f"  out_root = {args.out_root}")

    # ---- 逐条生成 ----
    ok_cnt = 0
    fail_list = []
    for idx, traj_dir in enumerate(traj_dirs):
        name = os.path.basename(traj_dir)
        if args.out_root:
            out_path = os.path.join(args.out_root, f"{name}_replay.mp4")
        else:
            out_path = os.path.join(traj_dir, "replay.mp4")

        print(f"\n[{idx+1}/{len(traj_dirs)}] {name}", flush=True)
        try:
            success, n_frames = generate_one(traj_dir, out_path, args.fps, args.show)
            if success:
                ok_cnt += 1
                print(f"  ✓ 完成  帧数={n_frames}  -> {out_path}")
            else:
                fail_list.append(name)
                print(f"  ✗ 失败（无帧或无法写入）")
        except Exception as e:
            fail_list.append(name)
            print(f"  ✗ 异常: {e}")

    # ---- 汇总 ----
    print("\n" + "=" * 60)
    print(f"[batch_replay_video] 处理完毕: 成功 {ok_cnt}/{len(traj_dirs)}")
    if fail_list:
        print(f"  失败 ({len(fail_list)}):")
        for f in fail_list:
            print(f"    - {f}")
    else:
        print("  全部成功 ✓")

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
