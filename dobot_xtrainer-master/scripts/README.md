# scripts 脚本说明

本目录存放遥操作数据采集流程中用到的**辅助脚本**与**公共工具模块**。
带编号（`1_` ~ `6_`）的脚本大致按使用先后顺序排列。

> 运行约定：所有脚本均从**项目根目录**执行，例如
> `python scripts/1_find_port.py`
> （各脚本已在开头把项目根目录加入 `sys.path`，可直接 import `dobot_control` / `scripts`。）

---

## 一、可执行脚本

### 1_find_port.py —— 识别并写入串口号
自动扫描串口，逐个尝试波特率，识别出**左右主手**和**左右夹爪**分别对应哪个串口，
并把结果写回 `scripts/dobot_config/dobot_settings.ini`。
- 前置：接好主手、夹爪的 USB。
- 需要至少检测到 4 个串口，否则报错。
```bash
python scripts/1_find_port.py
```

### 2_get_offset.py —— 主手关节 offset 标定
读取主手当前各关节角，与 ini 里配置的 `start_joints` 对比，
用于计算/校准 `joint_offsets`（把主手摆到标准起始姿态时运行）。
```bash
python scripts/2_get_offset.py
```

### 3_just_buttonA.py —— 主手力矩开关工具
只连主手（不连从手/相机/server），用按键控制主手力矩，
方便**开机前手动把主手摆到与从手初始姿态一致**。
- 短按 Button A：主手 锁定 ⇄ 解锁（解锁=力矩关，可自由拖动）。
- 长按 Button A：切换 servo 标志（此工具里主要用短按）。
- Ctrl+C 退出（按键线程已设 daemon，一次即可）。
```bash
python scripts/3_just_buttonA.py
```
⚠️ 解锁前先握稳主手，力矩归零瞬间会因重力下坠。

### 5_camera_read.py —— 相机预览/测试
读取 3 台 RealSense 相机并显示画面，用于确认相机连接、编号（top/left/right）是否正确。
该预览与推理端一致，使用 `30fps + 11000us 固定曝光 + gain 16`。每个画面顶部会显示
灰度均值和高光饱和比例；`saturated` 超过 8% 时文字变红，不能直接开始真机推理。
```bash
python scripts/5_camera_read.py
```
若现场灯光改变，可用 `--exposure-us=9000` 等值试调；确定后把同一数值传给推理客户端
的 `--camera-exposure-us`。目标是三路灰度均值大致 70～95、`saturated` 接近 0～1%。

### 6_dataset_count.py —— 统计轨迹最大帧数
遍历某个数据集下 `collect_data/` 里所有轨迹，统计其中**最长一条的帧数**（`max_step`）。
常用于后续转 hdf5 时确定对齐长度。
```bash
python scripts/6_dataset_count.py --dataset_name plug_and_unplug_task
```

### script_collect2train.py —— 原始数据转训练数据(hdf5)
把 `collect_data/{时间戳}/`（pkl + 三路 jpg）转换为训练用的 **hdf5**（`train_data/`），
并可生成对照视频（`output_videos/`）。是采集 → 训练之间的数据打包步骤。
```bash
python scripts/script_collect2train.py
```
> 注：`4_collect2train_data.py` 为早期/未完成版本，实际转换以 `script_collect2train.py` 为准。

---

## 二、回放脚本（本项目新增）

### replay_rollout.py —— OpenPI 推理数据回放（不驱动机械臂）
回放 `data/rollouts/` 保存的真机推理数据，同时显示三路相机、成功/失败结果、
状态、策略动作、推理延迟和进度。
```bash
# 自动选择最新一条 rollout，在窗口中交互回放
python scripts/replay_rollout.py

# 指定 rollout，以 0.5 倍速度播放
python scripts/replay_rollout.py <rollout目录> --speed 0.5

# 离线导出带数据面板的 MP4
python scripts/replay_rollout.py <rollout目录> --out replay.mp4 --no-show
```
- 窗口按键：空格播放/暂停，`a/d` 前后 1 帧，`j/l` 前后 10 帧，`q`/ESC 退出。
- 默认根据每帧时间戳恢复实测速度；`--fps` 可覆盖播放和导出帧率。
- 只读取文件，不连接相机、推理服务或机械臂。

### replay_video.py —— 离线视频回放（不驱动机械臂）
把一条轨迹的 top/left/right 三路图像横向拼成一个 `1920×480` 视频。
```bash
# 默认轨迹目录，输出到 该目录/replay.mp4
python scripts/replay_video.py

# 指定目录 + 边生成边预览(ESC 退出)
python scripts/replay_video.py <轨迹目录> --show

# 自定义输出与帧率
python scripts/replay_video.py <轨迹目录> --out /tmp/a.mp4 --fps 25
```
- `--fps` 默认 25，对应采集频率约 25Hz。
- 缺帧用黑底红字占位，不会中断。

### batch_replay_video.py —— 批量生成回放视频
对多条轨迹批量调用 `replay_video.py` 的拼接逻辑，一次生成所有视频。
```bash
# 处理 collect_data 下最新的 101 条（默认）
python scripts/batch_replay_video.py

# 指定数量 / 指定目录列表
python scripts/batch_replay_video.py --latest 50
python scripts/batch_replay_video.py --dirs /path/traj1 /path/traj2

# 统一输出到某个目录（默认每条视频放在对应轨迹目录下）
python scripts/batch_replay_video.py --out_root /path/to/output_videos

# 边生成边预览 (ESC 跳过预览)
python scripts/batch_replay_video.py --show
```
- `--fps` 默认 25。
- 失败的单条会跳过并继续，最终打印汇总。

### replay_robot.py —— 真机原样重放（会驱动机械臂）
读每帧 pkl 的 `control`(14维 action)，用与采集相同的通道
`env.step(action,[1,1])` 按频率下发到真机。
```bash
# 终端 A：先起 robot server
python experiments/launch_nodes.py

# 终端 B：重放（默认需输入 yes 确认）
python scripts/replay_robot.py
python scripts/replay_robot.py <轨迹目录> --hz 25 --yes
```
- 🔴 **会让机械臂运动**：执行前确认工作区无人、急停在手边、server 已启动。
- 首帧自动从当前姿态**缓慢插值**到轨迹起点，避免突跳。
- Ctrl+C 可随时中断，退出时关闭 ZMQ 连接。
- `--hz` 回放频率（默认 25），`--yes` 跳过确认，`--robot_port` 默认 6002。

---

## 三、公共工具模块（被其它脚本 import，不单独运行）

| 模块 | 作用 |
|------|------|
| `manipulate_utils.py` | 读 ini 配置、灯控 `set_light`、位姿初始化 `robot_pose_init`、动作安全检查 `servo_action_check` / `pose_check` / `dynamic_approach` 等 |
| `function_util.py` | 通用工具：串口扫描 `scan_port`、建目录 `mk_dir`、日志 `log_write`、精确延时 `wait_period`、存视频 `save_videos` 等 |
| `format_obs.py` | `save_frame`：把单帧 obs+action 打包成 pkl 落盘（采集时调用） |
| `__init__.py` | 包标记（空） |

---

## 四、配置文件

`scripts/dobot_config/dobot_settings.ini` 集中存放：
- `[CAMERA]`：三台相机序列号（top/left/right）
- `[HAND_LEFT] / [HAND_RIGHT]`：主手串口、波特率、joint_ids、offset、sign、夹爪配置、起始姿态
- `[GRIPPER_LEFT] / [GRIPPER_RIGHT]`：夹爪 id、串口、开合位置
- `[COMPUTER]`：sudo 口令（供 `scan_port` 提权设置串口权限用）

`1_find_port.py`、`2_get_offset.py` 会写回此文件；其余脚本从中读取配置。

---

## 五、典型使用顺序

1. `1_find_port.py`　识别串口 →
2. `2_get_offset.py`　标定主手 offset →
3. `5_camera_read.py`　确认相机 →
4. `3_just_buttonA.py`　（可选）开机前摆主手姿态 →
5. `experiments/launch_nodes.py` + `experiments/run_control.py`　正式遥操作采集 →
6. `replay_video.py` / `replay_robot.py`　回看或真机重放 →
7. `6_dataset_count.py` + `script_collect2train.py`　统计并转换为训练数据

> 更详细的图文版操作手册见 `doc/遥操作与数据采集操作手册.md`。

---

## 六、完整数据采集流程

> 涉及程序：`experiments/launch_nodes.py`（robot server）+ `experiments/run_control.py`（采集主程序）。
> 二者是**两个独立进程**，需各开一个终端。

### 重要前提：程序不会「停下来等你输入」
`run_control.py` 初始化完、打印出 `Ok, let's start` 后会进入**高速主循环持续运行**。
在你**按下主手按钮**之前，从手不跟随、也不录制——体感上像「停着等你」，
它等待的是**按键事件**（Button A / Button B），不是 `input()` 阻塞。

### 按键速查（主手按钮）
| 操作 | 按钮 | 效果 |
|------|------|------|
| 短按（<0.5s） | Button A（黄色） | 主手 锁定 ⇄ 解锁（力矩开/关，用于手动摆姿） |
| 长按（>1s） | Button A（黄色） | 从手 跟随 servo 开 / 关 |
| 按一下 | Button B | 录制 开 / 关（**必须已开 servo 才生效**） |

左右手 Button A 独立控制各自一侧，可单臂操作。

### 灯语
| 灯色 | 含义 |
|------|------|
| 灭 | 初始 / 未开始跟随 |
| 黄 | servo 已开，遥操作中，**未录制** |
| 绿 | **正在录制** |
| 红 | 出错（多为主从手偏差过大 / 动作跳变过大） |

### 分阶段步骤

**阶段 0：开机前物理准备**
- 确认 2 台 Nova 从手、2 只主手、3 台 RealSense 相机均通电、连线正常。
- 确认从手网络连通（左 `192.168.5.1` / 右 `192.168.5.2`）、已开启 TCP/IP 模式。
- 确认从手处于安全、可自由运动的位置（下一步会自动移动）。

**阶段 1：启动 robot server（终端 A）**
```bash
python experiments/launch_nodes.py
```
- 会使能两台从手、初始化夹爪（夹爪开合一次），随后阻塞等待 client。
- ⚠️ 本步**不会**让从手移动到初始姿态，仅上电使能。
- 此终端**保持开启，不要关闭**。

**阶段 2：启动采集主程序（终端 B）**
```bash
python experiments/run_control.py
```
自动依次执行：相机初始化 → 主手串口连接 → 连从手 → 固件检查 →
`robot_pose_init`（**从手自动缓慢归位到初始姿态**）→ 打印 `Ok, let's start` → 进主循环。
- 🔴 归位阶段从手会**自行运动**，人先离开工作范围。

**阶段 3：对齐主手与从手（人工，关键）**
1. 双手分别握稳左右主手（务必先握稳，解锁瞬间会因重力下坠）。
2. 短按黄色按钮（Button A）解锁主手 → 力矩关闭，可自由拖动。
3. 拖动主手，使其姿态与对应从手（初始姿态）尽量一致。
4. 对齐后继续握住即可，无需重新锁定（下一步长按会接管力矩）。

**阶段 4：开启从手跟随（servo）**
- 长按黄色按钮（Button A，>1s）开启该侧从手跟随。
- 触发瞬间执行 `dynamic_approach`：从手缓慢移到主手姿态并做偏差检查。
  - 偏差过大 → **红灯**报错：重新摆姿对齐后再长按。
  - 成功 → **黄灯亮**，进入遥操作待录制状态。

**阶段 5：遥操作 + 录制**
- 移动主手，从手实时跟随。
- 按 **Button B** 开始录制 → **绿灯**，开始落盘；每次开始录制生成一个新时间戳文件夹（= 一条轨迹）。
- 完成一条后再按 **Button B** 停止 → 回到**黄灯**。
- 重复「B 开 → 操作 → B 停」采集多条。

**阶段 6：结束**
- 长按 Button A 关闭该侧 servo（从手停止跟随）。
- Ctrl+C 退出采集程序（已做 daemon + finally 清理，一次即可干净退出）。
- 关闭 server 终端；从手按常规流程收尾。

### 一句话记忆
> 起 server → 起采集(从手自动归位) → 握主手 → 短按解锁 → 摆姿对齐 →
> 长按开跟随(黄) → Button B 录制(绿) → 再按停 → 长按关跟随 → Ctrl+C

### 数据落盘位置
```
<save_data_path>/<project_name>/collect_data/{时间戳}/
    ├── topImg/       顶部相机 jpg（0.jpg, 1.jpg, ...）
    ├── leftImg/      左相机 jpg
    ├── rightImg/     右相机 jpg
    └── observation/  每帧 obs+action 的 pkl
```
每个 `{时间戳}` 文件夹 = 一条轨迹，对应一次「Button B 开 → 关」。

### 常见坑与安全提醒
- **必须先长按开 servo（黄灯），Button B 才录得上**；未开 servo 时按 B 不保存数据。
- **解锁主手前务必先握稳**，否则力矩归零瞬间下坠。
- **红灯 = 对齐失败**，松开重新摆姿再长按。
- server 未起 / 网络不通时，采集程序会卡在 "Waiting to connect the robot..."（当前 ZMQ client 无接收超时，会一直等待）。
- 采集目录若属主为 root，会在录制建文件夹时报 `PermissionError`；用
  `sudo chown -R <用户>:<用户> <datasets目录>` 修复属主即可。

## 七、本地 OpenPI 真机推理

本机 RTX 3090 使用下载到本地的全量 checkpoint `29999`。模型服务默认使用
`8001`，避免与智算平台 SSH 隧道常用的 `8000` 冲突。

**终端 A：启动本地模型服务**
```bash
cd /home/iml/RockyEVO/openpi
./scripts/serve_xtrainer_local.sh
```
首次推理会执行 JAX 编译，等待约 20～30 秒；之后单次推理约 200 ms。本机实测
占用约 22.1 GB 显存，因此运行时不要同时训练其他模型。

**终端 B：机械臂服务（已运行时不要重复启动）**
```bash
cd /home/iml/RockyEVO/dobot_xtrainer/dobot_xtrainer-master
conda activate x_trainer
python experiments/launch_nodes.py
```

`launch_nodes.py` 会先以只读方式检查左右控制器，再使能机械臂和连接夹爪。若显示
`mode=9(ERROR)` 或 `mode=11(COLLISION)`，先检查急停、碰撞、机械干涉和控制柜报警。
只读复查命令为：

```bash
python scripts/check_dobot_alarm.py
```

确认机械臂周围安全、急停已释放且报警原因已消除后，可在 Dobot 控制软件中清错，
也可以显式清除指定控制器报警。例如只清左臂：

```bash
python scripts/check_dobot_alarm.py --clear left
```

该命令只执行 `ClearError()` 并重新读取两臂状态，不使能机械臂、不下发运动。模式不再
是 `ERROR/COLLISION` 后，再重新运行 `launch_nodes.py`。若清错后立即恢复为 mode 9，
不要反复清除，应根据控制柜或 Dobot 软件显示的具体报警处理硬件原因。

**终端 C：本地真机推理客户端**
```bash
cd /home/iml/RockyEVO/dobot_xtrainer/dobot_xtrainer-master
conda activate x_trainer

python experiments/run_inference_openpi.py \
  --remote-host=127.0.0.1 \
  --remote-port=8001 \
  --condition=original_position \
  --operator=iml \
  --show-img
```

按键与灯光：

| 按键 | 动作 | 录制与灯光 |
|------|------|------------|
| `r` | 开始一次推理 | 创建 rollout，全程记录，绿灯 |
| `s` | 标记成功 | 停止本条记录并写入成功，橙灯；客户端继续运行 |
| `f` | 标记失败 | 停止本条记录并写入失败，橙灯；客户端继续运行 |
| `h` | 中止并复位/待机复位 | 推理中将本条记为 `ABORTED` 后复位；客户端继续运行 |
| `q` | 结束程序 | 推理中按下会将本条记为 `ABORTED` |
| `e` | 紧急停止 | 停止、亮红灯并结束程序 |

客户端默认在每次按 `r` 后先自动经过安全位，再移动到当前 checkpoint 所用 54 条
训练示范的首帧控制目标中位数。这样可避免旧的整数角度复位姿态使左右腕关节偏离
训练起点约 13.8°/8.7°。若需要完全手动通过 `h` 控制复位，启动时增加
`--no-reset-before-rollout`。

`s`、`f`、`h` 都只结束当前 rollout，随后可继续按 `r` 开始下一次；只有 `q`、
`e` 或 `Ctrl+C` 会结束客户端进程。动作默认每执行 5 步重新请求一次视觉推理，
用于在插接阶段及时修正横向误差。推理相机默认使用 `640x480@30fps`，采集程序
仍保持 `90fps`；30fps 已高于策略请求频率，并将三相机共享 USB Hub 的流量降到
约三分之一。由于 D405 自动曝光会随帧周期从训练时约 11ms 增长到 30fps 下的
33ms，推理端默认固定为 `11000us, gain=16`，使亮度接近 90fps 训练图像。开始
rollout 前若任一路高光饱和像素超过 8%，客户端会拒绝启动并打印三路曝光统计。
`--camera-exposure-us=0` 可恢复自动曝光，但不适用于当前训练环境。

客户端默认开启异步推理：后台约每 5 步请求一次新动作块，前台控制循环保持
连续执行已有动作，不再停住等待约 200ms 的模型响应。新结果到达后会按实际等待
期间已经执行的步数跳过过期动作，并用 3 步连接新旧关节目标。客户端还会把对同一
控制时刻有效的最近 3 个动作块做时间集成，默认权重约为
`0.57 / 0.29 / 0.14`（新到旧）；这可抑制随机重规划在每个块边界交替反向，同时
保留最新画面的主要权重。运行中请求超过 2 秒或 50 步动作缓存耗尽时，本条会安全
失败并回到待机。Dobot 往返通信使实测控制频率约为 15～16Hz，与训练数据的 15fps
一致；`40ms` 只是循环的最短等待周期，不代表机械臂实际达到 25Hz。

`--open-loop-horizon` 在异步模式下表示后台重规划间隔。当前模型建议保持默认 5；
数值太小会频繁随机重采样，数值太大则降低视觉纠偏频率。排查问题时可增加
`--no-async-inference` 回退到原同步控制，此时会重新出现周期性等待和停顿。
`--temporal-ensemble-chunks=1` 可关闭时间集成，只用于 A/B 排查，不建议用于当前
checkpoint。

如果终端出现 `ZMQError: Operation cannot be accomplished in current state`，它是
REQ socket 在一次请求超时后仍等待旧回复所产生的二次错误，并不是关节限位。客户端
现在会在超时后重建 socket，服务端也会把机械臂异常作为明确的
`RobotServerError` 返回。若看到 `left/right robot error` 或 `[ROBOT ALARM]`，应先
查看 Dobot 控制器报警、清除报警，再由操作员重启 `launch_nodes.py`；程序不会自动
重发可能已经执行过的运动命令。

推理中任一路相机超过 0.75 秒没有新画面时，客户端会把本条记为
`failure/camera_stale`、停止下发模型动作并留在待机状态。相机读取线程会继续尝试
重新连接；画面恢复后可再次按 `r`。若终端已经出现 `Evaluation stopped.`，说明
客户端确实已经退出，此后输入的 `h`、`r` 会被 Shell 当作系统命令，必须先由操作员
重新运行客户端。
