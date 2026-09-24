#!/usr/bin/env python3
"""Read one RealSense color stream without starting the robot/data pipeline."""

import argparse
import sys
import time

import pyrealsense2 as rs


DEFAULT_SERIAL = "352122271318"


def _info(device, key):
    try:
        if device.supports(key):
            return device.get_info(key)
    except Exception:
        pass
    return "unknown"


def list_devices():
    devices = rs.context().query_devices()
    print("[CAM] Detected RealSense devices:")
    if len(devices) == 0:
        print("  (none)")
        return []

    serials = []
    for device in devices:
        serial = _info(device, rs.camera_info.serial_number)
        serials.append(serial)
        print(
            "  name={}; serial={}; USB={}; port={}".format(
                _info(device, rs.camera_info.name),
                serial,
                _info(device, rs.camera_info.usb_type_descriptor),
                _info(device, rs.camera_info.physical_port),
            )
        )
    return serials


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test one RealSense camera's color stream, without robot control."
    )
    parser.add_argument("--serial", default=DEFAULT_SERIAL,
                        help="camera serial number (default: top camera)")
    parser.add_argument("--fps", type=int, default=30,
                        help="color stream FPS for diagnosis (default: 30)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--timeout-ms", type=int, default=5000,
                        help="wait timeout for each frame (default: 5000)")
    parser.add_argument("--frames", type=int, default=30,
                        help="number of successful frames to read (default: 30)")
    parser.add_argument("--max-errors", type=int, default=3,
                        help="stop after this many consecutive read errors")
    parser.add_argument("--preview", action="store_true",
                        help="show frames in an OpenCV window")
    args = parser.parse_args()
    if args.fps <= 0 or args.width <= 0 or args.height <= 0:
        parser.error("fps, width and height must be positive")
    if args.timeout_ms <= 0 or args.frames <= 0 or args.max_errors <= 0:
        parser.error("timeout-ms, frames and max-errors must be positive")
    return args


def main():
    args = parse_args()
    serials = list_devices()
    if args.serial not in serials:
        print("[ERROR] Camera {} is not currently enumerated.".format(args.serial),
              file=sys.stderr)
        return 2

    config = rs.config()
    config.enable_device(args.serial)
    config.enable_stream(
        rs.stream.color,
        args.width,
        args.height,
        rs.format.bgr8,
        args.fps,
    )

    pipeline = rs.pipeline()
    preview = None
    if args.preview:
        import cv2
        import numpy as np
        preview = (cv2, np)

    started = time.monotonic()
    successful_frames = 0
    consecutive_errors = 0
    print(
        "[CAM] Starting color-only stream: serial={}, {}x{} @ {} FPS; "
        "timeout={} ms".format(
            args.serial, args.width, args.height, args.fps, args.timeout_ms
        ),
        flush=True,
    )

    try:
        profile = pipeline.start(config)
        device = profile.get_device()
        print(
            "[CAM] Pipeline started: name={}, USB={}, port={}".format(
                _info(device, rs.camera_info.name),
                _info(device, rs.camera_info.usb_type_descriptor),
                _info(device, rs.camera_info.physical_port),
            ),
            flush=True,
        )

        while successful_frames < args.frames and consecutive_errors < args.max_errors:
            try:
                frames = pipeline.wait_for_frames(args.timeout_ms)
                color = frames.get_color_frame()
                if not color:
                    raise RuntimeError("frameset contains no color frame")

                successful_frames += 1
                consecutive_errors = 0
                if successful_frames == 1 or successful_frames % 10 == 0:
                    print(
                        "[CAM] Got frame {}/{} after {:.2f}s".format(
                            successful_frames, args.frames, time.monotonic() - started
                        ),
                        flush=True,
                    )

                if preview is not None:
                    cv2, np = preview
                    image = np.asanyarray(color.get_data())
                    cv2.imshow("RealSense diagnostic", image)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
            except Exception as error:
                consecutive_errors += 1
                print(
                    "[CAM] Read error {}/{}: {}".format(
                        consecutive_errors, args.max_errors, error
                    ),
                    flush=True,
                )

    except Exception as error:
        print("[ERROR] Could not start pipeline: {}".format(error), file=sys.stderr)
        return 3
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        if preview is not None:
            preview[0].destroyAllWindows()

    elapsed = time.monotonic() - started
    print(
        "[CAM] Finished: {}/{} frames in {:.2f}s; consecutive_errors={}".format(
            successful_frames, args.frames, elapsed, consecutive_errors
        ),
        flush=True,
    )
    return 0 if successful_frames == args.frames else 1


if __name__ == "__main__":
    sys.exit(main())
