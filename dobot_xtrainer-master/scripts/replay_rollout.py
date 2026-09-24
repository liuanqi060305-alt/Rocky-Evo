#!/usr/bin/env python3
"""离线回放 OpenPI 真机推理 rollout，不连接或驱动机械臂。

默认回放 plug_and_unplug_task 下最新一条有数据的 rollout：

    python scripts/replay_rollout.py

指定 rollout、改变速度或导出视频：

    python scripts/replay_rollout.py /path/to/rollout --speed 0.5
    python scripts/replay_rollout.py /path/to/rollout --out replay.mp4 --no-show

窗口按键：空格=播放/暂停，a/d=前/后一帧，j/l=前/后十帧，q/ESC=退出。
"""

import argparse
import json
import os
import pathlib
import pickle
import sys
from typing import Dict, List, Optional, Tuple

# OpenCV wheel 的 Qt 插件不带字体，避免预览时反复打印字体警告。
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2
import numpy as np


CAMERAS = ("topImg", "leftImg", "rightImg")
CAMERA_LABELS = ("TOP", "LEFT", "RIGHT")
IMAGE_SIZE = (640, 480)
PANEL_HEIGHT = 220
FALLBACK_FPS = 25.0

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
ROCKY_ROOT = REPO_ROOT.parents[1]
DEFAULT_ROOT = ROCKY_ROOT / "data" / "rollouts" / "plug_and_unplug_task"


def numeric_indices(directory: pathlib.Path, suffix: str) -> List[int]:
    indices = []
    if not directory.is_dir():
        return indices
    for path in directory.glob("*" + suffix):
        try:
            indices.append(int(path.stem))
        except ValueError:
            continue
    return sorted(indices)


def find_latest_rollout(root: pathlib.Path) -> pathlib.Path:
    candidates = []
    for meta_path in root.glob("session_*/rollout_*/meta.json"):
        rollout = meta_path.parent
        if numeric_indices(rollout / "observation", ".pkl"):
            candidates.append(rollout)
    if not candidates:
        raise FileNotFoundError(f"{root} 下没有包含 observation/*.pkl 的 rollout")
    return max(candidates, key=lambda p: (p / "meta.json").stat().st_mtime_ns)


def resolve_rollout(path_arg: Optional[str], root_arg: str) -> pathlib.Path:
    if path_arg:
        rollout = pathlib.Path(path_arg).expanduser().resolve()
    else:
        rollout = find_latest_rollout(pathlib.Path(root_arg).expanduser().resolve())
    if not rollout.is_dir():
        raise FileNotFoundError(f"rollout 目录不存在: {rollout}")
    if not (rollout / "observation").is_dir():
        raise FileNotFoundError(f"缺少 observation 目录: {rollout}")
    return rollout


def load_meta(rollout: pathlib.Path) -> Dict:
    path = rollout / "meta.json"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_records(rollout: pathlib.Path) -> Tuple[List[int], List[Dict]]:
    indices = numeric_indices(rollout / "observation", ".pkl")
    if not indices:
        raise FileNotFoundError(f"{rollout}/observation 下没有 pkl 数据")
    records = []
    for index in indices:
        path = rollout / "observation" / f"{index}.pkl"
        with path.open("rb") as f:
            record = pickle.load(f)
        state = np.asarray(record.get("joint_positions"), dtype=np.float64)
        action = np.asarray(record.get("control"), dtype=np.float64)
        if state.shape != (14,) or action.shape != (14,):
            raise ValueError(
                f"{path} 的 state/action 维度异常: {state.shape}/{action.shape}"
            )
        records.append(record)
    return indices, records


def measured_fps(records: List[Dict]) -> float:
    if len(records) < 2:
        return FALLBACK_FPS
    for key in ("monotonic_ns", "timestamp_ns"):
        values = [record.get(key) for record in records]
        if all(value is not None for value in values):
            span = (int(values[-1]) - int(values[0])) / 1e9
            if span > 0:
                return (len(values) - 1) / span
    return FALLBACK_FPS


def elapsed_seconds(records: List[Dict]) -> np.ndarray:
    for key in ("monotonic_ns", "timestamp_ns"):
        values = [record.get(key) for record in records]
        if all(value is not None for value in values):
            base = int(values[0])
            return np.asarray([(int(value) - base) / 1e9 for value in values])
    return np.arange(len(records), dtype=np.float64) / FALLBACK_FPS


def load_camera_image(rollout: pathlib.Path, camera: str, index: int) -> np.ndarray:
    width, height = IMAGE_SIZE
    path = rollout / camera / f"{index}.jpg"
    image = cv2.imread(str(path))
    if image is None:
        image = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(
            image,
            f"MISSING {camera}/{index}.jpg",
            (25, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    elif image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image


def format_vector(values: np.ndarray) -> str:
    return "[" + "  ".join(f"{value:+.3f}" for value in values) + "]"


def result_color(result: str) -> Tuple[int, int, int]:
    return {
        "success": (30, 210, 30),
        "failure": (30, 30, 230),
        "aborted": (0, 180, 255),
        "error": (180, 40, 180),
    }.get(result.lower(), (220, 220, 220))


def render_frame(
    rollout: pathlib.Path,
    meta: Dict,
    indices: List[int],
    records: List[Dict],
    elapsed: np.ndarray,
    position: int,
    paused: bool = False,
) -> np.ndarray:
    index = indices[position]
    record = records[position]
    images = []
    for camera, label in zip(CAMERAS, CAMERA_LABELS):
        image = load_camera_image(rollout, camera, index)
        cv2.rectangle(image, (0, 0), (130, 38), (0, 0, 0), -1)
        cv2.putText(
            image, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (60, 230, 60), 2, cv2.LINE_AA,
        )
        images.append(image)

    camera_canvas = np.concatenate(images, axis=1)
    width = camera_canvas.shape[1]
    panel = np.full((PANEL_HEIGHT, width, 3), 24, dtype=np.uint8)

    state = np.asarray(record["joint_positions"], dtype=np.float64)
    action = np.asarray(record["control"], dtype=np.float64)
    arm_delta = np.concatenate((action[:6] - state[:6], action[7:13] - state[7:13]))
    result = str(meta.get("result") or "unfinished")
    duration = float(elapsed[-1]) if len(elapsed) else 0.0
    latency = record.get("infer_latency_ms")
    latency_text = "n/a" if latency is None else f"{float(latency):.0f} ms"
    mode = "PAUSED" if paused else "PLAYING"

    cv2.putText(
        panel,
        f"{mode} | result={result.upper()} | condition={meta.get('condition') or '(none)'} "
        f"| operator={meta.get('operator') or '(none)'}",
        (15, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
        result_color(result), 2, cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"frame {position + 1}/{len(indices)} (file {index}) | "
        f"time {elapsed[position]:.2f}/{duration:.2f}s | infer={latency_text} "
        f"| chunk={record.get('chunk_index', 'n/a')} "
        f"| plans={record.get('ensemble_sources', 1)} "
        f"| max action-state delta={np.max(np.abs(arm_delta)):.4f} rad",
        (15, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
        (230, 230, 230), 1, cv2.LINE_AA,
    )
    cv2.putText(panel, "STATE  L " + format_vector(state[:7]), (15, 94),
                cv2.FONT_HERSHEY_SIMPLEX, 0.53, (200, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, "ACTION L " + format_vector(action[:7]), (15, 124),
                cv2.FONT_HERSHEY_SIMPLEX, 0.53, (120, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, "STATE  R " + format_vector(state[7:]), (960, 94),
                cv2.FONT_HERSHEY_SIMPLEX, 0.53, (200, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, "ACTION R " + format_vector(action[7:]), (960, 124),
                cv2.FONT_HERSHEY_SIMPLEX, 0.53, (120, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(
        panel,
        "SPACE pause/play | a/d -/+1 frame | j/l -/+10 frames | q/ESC quit",
        (15, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
        (190, 190, 190), 1, cv2.LINE_AA,
    )

    bar_x0, bar_x1, bar_y = 15, width - 15, PANEL_HEIGHT - 22
    cv2.line(panel, (bar_x0, bar_y), (bar_x1, bar_y), (80, 80, 80), 8)
    fraction = position / max(1, len(indices) - 1)
    cv2.line(panel, (bar_x0, bar_y),
             (bar_x0 + int((bar_x1 - bar_x0) * fraction), bar_y),
             result_color(result), 8)
    return np.concatenate((camera_canvas, panel), axis=0)


def export_video(
    output: pathlib.Path,
    rollout: pathlib.Path,
    meta: Dict,
    indices: List[int],
    records: List[Dict],
    elapsed: np.ndarray,
    fps: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    size = (IMAGE_SIZE[0] * len(CAMERAS), IMAGE_SIZE[1] + PANEL_HEIGHT)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    if not writer.isOpened():
        raise OSError(f"无法创建视频: {output}")
    try:
        for position in range(len(indices)):
            writer.write(render_frame(
                rollout, meta, indices, records, elapsed, position, paused=False
            ))
            if position % 100 == 0:
                print(f"[export] {position}/{len(indices)}", flush=True)
    finally:
        writer.release()
    print(f"[export] saved: {output}", flush=True)


def interactive_replay(
    rollout: pathlib.Path,
    meta: Dict,
    indices: List[int],
    records: List[Dict],
    elapsed: np.ndarray,
    fps: float,
    speed: float,
) -> None:
    position = 0
    paused = False
    frame_delay_ms = max(1, int(round(1000.0 / (fps * speed))))
    while True:
        canvas = render_frame(
            rollout, meta, indices, records, elapsed, position, paused=paused
        )
        cv2.imshow("OpenPI rollout replay", canvas)
        key = cv2.waitKeyEx(0 if paused else frame_delay_ms)
        if key in (ord("q"), ord("Q"), 27):
            break
        if key == ord(" "):
            paused = not paused
            continue
        if key in (ord("a"), ord("A"), 2424832, 81):
            position = max(0, position - 1)
            paused = True
            continue
        if key in (ord("d"), ord("D"), 2555904, 83):
            position = min(len(indices) - 1, position + 1)
            paused = True
            continue
        if key in (ord("j"), ord("J")):
            position = max(0, position - 10)
            paused = True
            continue
        if key in (ord("l"), ord("L")):
            position = min(len(indices) - 1, position + 10)
            paused = True
            continue
        if not paused:
            position += 1
            if position >= len(indices):
                position = len(indices) - 1
                paused = True
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="离线回放 OpenPI 真机 rollout（不会驱动机械臂）"
    )
    parser.add_argument("rollout_dir", nargs="?", help="rollout 目录；默认取最新一条")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="默认搜索根目录")
    parser.add_argument("--fps", type=float, default=None,
                        help="覆盖数据实测帧率；导出和播放均使用该值")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="窗口播放速度倍率，例如 0.5 或 2")
    parser.add_argument("--out", help="导出 MP4 路径")
    parser.add_argument("--no-show", action="store_true", help="不打开回放窗口")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.speed <= 0:
        raise ValueError("--speed 必须大于 0")
    if args.fps is not None and args.fps <= 0:
        raise ValueError("--fps 必须大于 0")
    if args.no_show and not args.out:
        raise ValueError("--no-show 必须同时指定 --out，否则没有任何输出")

    rollout = resolve_rollout(args.rollout_dir, args.root)
    meta = load_meta(rollout)
    indices, records = load_records(rollout)
    elapsed = elapsed_seconds(records)
    data_fps = measured_fps(records)
    fps = args.fps if args.fps is not None else data_fps

    print(f"[rollout] {rollout}")
    print(
        f"[summary] result={meta.get('result')} frames={len(indices)} "
        f"duration={elapsed[-1]:.3f}s measured_fps={data_fps:.3f} playback_fps={fps:.3f}"
    )
    print("[safety] offline replay only; no robot connection or command")

    if args.out:
        export_video(
            pathlib.Path(args.out).expanduser().resolve(),
            rollout, meta, indices, records, elapsed, fps,
        )
    if not args.no_show:
        interactive_replay(rollout, meta, indices, records, elapsed, fps, args.speed)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
