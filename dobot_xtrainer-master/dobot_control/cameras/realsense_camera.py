import time
from typing import List, Optional, Tuple
import numpy as np
from dobot_control.cameras.camera import CameraDriver
import cv2
import pyrealsense2 as rs

# 启动/重建后的首帧需要额外时间，不能用稳定取流时的帧间隔估算。
STARTUP_TIMEOUT_MS = 5000
# 正常取流时尽快检测断流；这个超时不用于启动后的首帧。
FRAME_TIMEOUT_MS = 300
# 单次 300ms 超时可能只是 USB 调度抖动。先在现有 pipeline 上再等几次，避免
# 把一次瞬时丢帧放大成 pipeline 重建，继而触发设备硬复位。
FRAME_WAITS_BEFORE_RESTART = 3
# 读取重试轮数。共享 USB3 Hub 上的边缘链路重建一次往往不够。
READ_RETRIES = 5
# 普通 pipeline 重建无法修复 UVC -71/-32 时，设备仍会被系统枚举到，但不会
# 再产生图像。只有多轮 pipeline 重建都失败后才复位故障相机；过早、频繁硬复位
# 可能使设备直接从 USB 总线消失。
HARD_RESET_AFTER_FAILURES = 4
HARD_RESET_TIMEOUT_S = 12.0


def get_device_ids() -> List[str]:
    """只枚举当前在线设备；查询操作不得隐式复位整组相机。"""
    ctx = rs.context()
    devices = ctx.query_devices()
    device_ids = []
    for dev in devices:
        device_ids.append(dev.get_info(rs.camera_info.serial_number))
    return device_ids


def list_connected_serials(retries: int = 3, delay: float = 1.0) -> List[str]:
    """已连接相机的 serial 列表。

    固件卡死时(常见于 USB hub 热插拔之后)query_devices() 会返回正确的数量,
    但构造 device 对象就抛 'hwmon command 0x10 ... failed' —— 此时 serial 读不出来,
    enable_device() 自然匹配不上,报出来的却是含糊的 'No device connected'。
    这里把这两种情况区分开,让调用方能给出可操作的提示。
    """
    last_err = None
    for attempt in range(retries):
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            serials = []
            unreadable = 0
            for i in range(len(devices)):
                try:
                    serials.append(devices[i].get_info(rs.camera_info.serial_number))
                except Exception as e:
                    unreadable += 1
                    last_err = e
            if serials or not unreadable:
                return serials
            # 全部读不出来:固件没响应,重试无用但代价很低
        except Exception as e:
            last_err = e
        if attempt < retries - 1:
            time.sleep(delay)

    if last_err is not None:
        print(f"[CAM] 无法读取相机 serial: {last_err}", flush=True)
    return []


class RealSenseCamera(CameraDriver):
    def __repr__(self) -> str:
        return f"RealSenseCamera(device_id={self._device_id})"

    def __init__(
        self,
        device_id: Optional[str] = None,
        flip: bool = False,
        fps: int = 90,
        allow_hardware_reset: bool = True,
        exposure_us: Optional[float] = None,
        gain: Optional[float] = None,
    ):
        import pyrealsense2 as rs
        print("init", device_id)
        self._device_id = device_id
        self._flip = flip
        self._allow_hardware_reset = allow_hardware_reset
        self._exposure_us = exposure_us
        self._gain = gain
        # 存盘/控制频率由调用方主循环决定，相机 fps 只决定最新画面的最大帧龄。
        # 采集可按需要使用 90fps；三路共用 Hub 的真机推理默认使用 30fps，以减少
        # 带宽和供电压力。当前物理链路不稳定时，不能用一次短时跑通推断长期可靠。
        self._fps = fps
        self._pipeline = None
        self._waiting_for_first_frame = True
        try:
            self._start_pipeline()
            for _ in range(50):
                self.read()
        except BaseException:
            self.close()
            raise

    def close(self):
        """释放当前 pipeline；初始化失败和重试耗尽时也必须清理。"""
        pipeline, self._pipeline = self._pipeline, None
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass

    def _start_pipeline(self):
        """建立（或重建）pipeline。"""
        config = rs.config()
        if self._device_id is not None:
            config.enable_device(self._device_id)

        # 深度流已禁用：三路相机共享同一条 USB3 总线，深度+彩色同时开启时
        # 带宽争用会导致部分相机 wait_for_frames 超时（一路独占、其余饿死）。
        # 工程内所有调用点均为 image, _ = read()，深度数据未被使用。
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, self._fps)
        self._pipeline = rs.pipeline()
        self._waiting_for_first_frame = True
        try:
            profile = self._pipeline.start(config)
        except RuntimeError as e:
            raise RuntimeError(self._diagnose_start_failure(e)) from e
        self._configure_exposure(profile)
        print(f"[CAM {self._device_id}] color 640x480 @ {self._fps}fps; "
              f"first-frame timeout={STARTUP_TIMEOUT_MS}ms", flush=True)

    def _configure_exposure(self, profile):
        """Apply optional manual exposure after every pipeline (re)start.

        D405 auto exposure is constrained by the frame interval. In this setup it
        settles near 11 ms at 90 fps but reaches 33 ms at 30 fps, making inference
        frames roughly twice as bright as the training images. Supplying an exposure
        keeps the sensor response stable while retaining the lower USB bandwidth of
        the 30 fps stream.
        """
        if self._exposure_us is None and self._gain is None:
            return
        if profile is None:
            raise RuntimeError("pipeline did not return a profile for exposure setup")

        # D405 exposes RGB, depth, and infrared profiles through one "Stereo
        # Module" and therefore raises from first_color_sensor(). Select the
        # sensor by its advertised stream profiles instead.
        sensor = None
        for candidate in profile.get_device().query_sensors():
            try:
                if any(
                    stream_profile.stream_type() == rs.stream.color
                    for stream_profile in candidate.get_stream_profiles()
                ):
                    sensor = candidate
                    break
            except Exception:
                continue
        if sensor is None:
            raise RuntimeError(f"camera {self._device_id} has no color stream sensor")
        if not sensor.supports(rs.option.enable_auto_exposure):
            raise RuntimeError(f"camera {self._device_id} cannot disable auto exposure")
        sensor.set_option(rs.option.enable_auto_exposure, 0.0)

        configured = []
        for option, value, name in (
            (rs.option.exposure, self._exposure_us, "exposure_us"),
            (rs.option.gain, self._gain, "gain"),
        ):
            if value is None:
                continue
            if not sensor.supports(option):
                raise RuntimeError(f"camera {self._device_id} does not support {name}")
            limits = sensor.get_option_range(option)
            value = float(value)
            if not limits.min <= value <= limits.max:
                raise ValueError(
                    f"camera {self._device_id} {name}={value:g} is outside "
                    f"[{limits.min:g}, {limits.max:g}]"
                )
            sensor.set_option(option, value)
            configured.append(f"{name}={sensor.get_option(option):g}")

        print(
            f"[CAM {self._device_id}] manual exposure: {', '.join(configured)}",
            flush=True,
        )

    def _diagnose_start_failure(self, err: Exception) -> str:
        """把 'No device connected' 翻译成能照着做的提示。"""
        serials = list_connected_serials()
        if serials:
            if self._device_id in serials:
                return (f"相机 {self._device_id} 在线但 pipeline 启动失败: {err}. "
                        f"通常是被其它进程占用(检查是否有残留的 python 进程),"
                        f"或 USB 带宽不足。")
            return (f"相机 {self._device_id} 未连接。当前在线的是: {serials}. "
                    f"请核对 dobot_settings.ini 的 [CAMERA] 配置。")

        # 一个 serial 都读不出来:区分「没插」和「固件卡死」
        try:
            n = len(rs.context().query_devices())
        except Exception:
            n = 0
        if n:
            return (f"检测到 {n} 台 RealSense,但固件无响应(hwmon 查询失败),"
                    f"读不出 serial,因此 {self._device_id} 匹配不上。"
                    f"这不是配置问题 —— 请重新插拔相机 USB hub(拔掉等 5 秒再插回),"
                    f"或用 root 复位 USB 端口。原始错误: {err}")
        return (f"没有检测到任何 RealSense 相机。请检查 USB 连接和供电。"
                f"原始错误: {err}")

    def _restart_pipeline(self):
        """USB 抖动后重建 pipeline。出错时 librealsense 会内部 stop 掉 pipeline，
        之后再调 wait_for_frames 只会报 'cannot be called before start()'，必须重建。"""
        self.close()
        time.sleep(0.5)
        self._start_pipeline()

    def _hardware_reset_pipeline(self):
        """硬复位当前设备，并等待它重新枚举后重建 pipeline。"""
        self.close()
        if self._device_id is None:
            raise RuntimeError("cannot hardware-reset a camera without device_id")

        device = None
        for candidate in rs.context().query_devices():
            try:
                if candidate.get_info(rs.camera_info.serial_number) == self._device_id:
                    device = candidate
                    break
            except Exception:
                continue
        if device is None:
            raise RuntimeError(f"camera {self._device_id} is not available for hardware reset")

        print(f"[CAM {self._device_id}] pipeline rebuild did not recover frames; "
              "performing device hardware reset", flush=True)
        device.hardware_reset()
        deadline = time.monotonic() + HARD_RESET_TIMEOUT_S
        while time.monotonic() < deadline:
            time.sleep(0.5)
            if self._device_id in list_connected_serials(retries=1, delay=0):
                self._start_pipeline()
                return
        raise RuntimeError(
            f"camera {self._device_id} did not re-enumerate within "
            f"{HARD_RESET_TIMEOUT_S:.0f}s after hardware reset"
        )

    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None,  # farthest: float = 0.12
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Read a frame from the camera.

        Args:
            img_size: The size of the image to return. If None, the original size is returned.
            farthest: The farthest distance to map to 255.

        Returns:
            np.ndarray: The color image, shape=(H, W, 3)
            np.ndarray: The depth image, shape=(H, W, 1)
        """

        # 链路是 marginal 的（dmesg 常见 uvcvideo -71/-32），单次抖动不应让调用方线程死掉。
        # 稳定取流阶段先在同一 pipeline 上短等数次；确认持续无帧后才进入重建。
        last_err = None
        color_frame = None
        for attempt in range(READ_RETRIES):
            if self._pipeline is None:
                try:
                    self._start_pipeline()
                except Exception as e:
                    last_err = e

            # 启动后的 5 秒等待本身已经足够长，只等一次；稳定阶段允许连续三次
            # 300ms 超时，以吸收偶发的 USB 抖动。
            wait_count = (1 if self._waiting_for_first_frame
                          else FRAME_WAITS_BEFORE_RESTART)
            for _ in range(wait_count):
                if self._pipeline is None:
                    break
                try:
                    timeout_ms = (STARTUP_TIMEOUT_MS if self._waiting_for_first_frame
                                  else FRAME_TIMEOUT_MS)
                    frames = self._pipeline.wait_for_frames(timeout_ms)
                    color_frame = frames.get_color_frame()
                    if color_frame:
                        self._waiting_for_first_frame = False
                        break
                    last_err = RuntimeError("no color frame in frameset")
                except Exception as e:
                    last_err = e
            if color_frame:
                break
            print(f"[CAM {self._device_id}] read failed "
                  f"({attempt + 1}/{READ_RETRIES}): {last_err}", flush=True)
            if attempt == READ_RETRIES - 1:
                break
            try:
                if (self._allow_hardware_reset
                        and attempt + 1 == HARD_RESET_AFTER_FAILURES):
                    self._hardware_reset_pipeline()
                else:
                    self._restart_pipeline()
            except Exception as e:
                self.close()
                last_err = e
                print(f"[CAM {self._device_id}] pipeline restart failed: {e}", flush=True)
        if not color_frame:
            self.close()
            raise RuntimeError(
                f"camera {self._device_id} unrecoverable after {READ_RETRIES} "
                f"attempts: {last_err}")
        color_image = np.asanyarray(color_frame.get_data())
        # 深度流已禁用，返回零占位以保持 (image, depth) 的返回签名不变
        depth_image = np.zeros(color_image.shape[:2], dtype=np.uint16)
        if img_size is None:
            image = color_image[:, :, ::-1]
            depth = depth_image
        else:
            image = cv2.resize(color_image, img_size)[:, :, ::-1]
            depth = cv2.resize(depth_image, img_size)

        # rotate 180 degree's because everything is upside down in order to center the camera
        if self._flip:
            image = cv2.rotate(image, cv2.ROTATE_180)
            depth = cv2.rotate(depth, cv2.ROTATE_180)[:, :, None]
        else:
            depth = depth[:, :, None]

        return image, depth

def _debug_read(camera, save_datastream=False):

    cv2.namedWindow("image")
    # cv2.namedWindow("depth")
    counter = 0
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    while True:
        # time.sleep(0.1)
        tic = time.time()
        image, depth = camera.read()

        _, image_ = cv2.imencode('.jpg', image, encode_param)

        key = cv2.waitKey(1)
        cv2.imshow("image", image[:, :, ::-1])
        # cv2.imshow("depth", depth)
        toc = time.time()
        print(image_.shape, toc - tic)
        counter += 1


if __name__ == "__main__":
    device_ids = get_device_ids()
    print(f"Found {len(device_ids)} devices")
    print(device_ids)
    rs = RealSenseCamera(flip=True, device_id=device_ids[0])
    im, depth = rs.read()
    _debug_read(rs, save_datastream=True)
