import sys
import os
import argparse
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from scripts.manipulate_utils import load_ini_data_camera
from dobot_control.cameras.realsense_camera import RealSenseCamera, get_device_ids
import numpy as np
import cv2


parser = argparse.ArgumentParser(description="Preview the three inference cameras")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--exposure-us", type=float, default=11000)
parser.add_argument("--gain", type=float, default=16)
args = parser.parse_args()
if args.fps <= 0:
    parser.error("--fps must be positive")
if args.exposure_us < 0:
    parser.error("--exposure-us cannot be negative")
if args.exposure_us > 0 and args.gain < 0:
    parser.error("--gain cannot be negative")
manual_exposure = args.exposure_us if args.exposure_us > 0 else None
manual_gain = args.gain if manual_exposure is not None else None


# camera init
device_ids = get_device_ids()
print(f"Found {len(device_ids)} devices: ", device_ids)

camera_dict = load_ini_data_camera()
rs_list = []
show_canvas = np.zeros((480, 640*3, 3), dtype=np.uint8)

try:
    rs_list.append(RealSenseCamera(
        flip=True, device_id=camera_dict["top"], fps=args.fps,
        allow_hardware_reset=False,
        exposure_us=manual_exposure, gain=manual_gain,
    ))
    rs_list.append(RealSenseCamera(
        flip=False, device_id=camera_dict["left"], fps=args.fps,
        allow_hardware_reset=False,
        exposure_us=manual_exposure, gain=manual_gain,
    ))
    rs_list.append(RealSenseCamera(
        flip=True, device_id=camera_dict["right"], fps=args.fps,
        allow_hardware_reset=False,
        exposure_us=manual_exposure, gain=manual_gain,
    ))
    while 1:
        # 读取三台相机画面并拼接
        for i in range(len(rs_list)):
            _img, _ = rs_list[i].read()
            _img = _img[:, :, ::-1]  # RGB → BGR
            gray = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
            mean = float(gray.mean())
            saturated = float(np.mean(gray >= 245)) * 100
            panel = np.asarray(_img, dtype="uint8").copy()
            cv2.rectangle(panel, (0, 0), (640, 36), (0, 0, 0), -1)
            cv2.putText(
                panel,
                f"mean={mean:.1f} saturated={saturated:.1f}%",
                (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if saturated <= 8.0 else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            show_canvas[:, int(640*i):int(640*(i+1))] = panel
        cv2.imshow("0", show_canvas)
        key = cv2.waitKey(1)
        # 按 Esc 键退出
        if key == 27:
            break
        # 点窗口 ❌ 按钮退出
        if cv2.getWindowProperty("0", cv2.WND_PROP_VISIBLE) < 1:
            break
finally:
    # 初始化中途或读取异常时也释放已经打开的相机。
    cv2.destroyAllWindows()
    for cam in rs_list:
        cam.close()
    print("相机已关闭，程序退出。")
