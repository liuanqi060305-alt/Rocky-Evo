"""统一数据路径配置层。

存在的理由：本项目有两个代码工程（数采 + VLA），但同一批真实 Raw Dataset
必须只有一个事实来源。任何工具都不应写死 `~/dobot_xtrainer/datasets` 这类
开发机绝对路径 —— 换机、换副本、换目录就全断。

优先级（高 -> 低）：
    1. 函数显式传参 / CLI 参数
    2. 环境变量  XTRAINER_RAW_DATA_ROOT 等
    3. 配置文件  config/data_paths.yaml（若存在）
    4. 内置默认：相对本文件位置推导出的项目 data/ 目录

四类数据职责（详见 docs/XTRAINER_DATA_FLOW.md）：
    raw       机器人真实采集数据。数采工程写，其余只读，不在此做格式转换
    processed 质检索引、过滤清单、统计结果、中间产物。不覆盖 raw
    lerobot   Converter 从 raw 生成的训练格式。训练只读这里
    rejected  坏数据的索引/说明/软链接，不移动 raw 实体
    test_configs 真机泛化测试配置，不放训练用 demonstration

用法：
    from scripts.data_paths import paths
    paths.raw            # Raw 数据根
    paths.task_dir("plug_and_unplug_task")
    paths.collect_dir("plug_and_unplug_task")
"""
import os
import pathlib
from typing import Optional

# 本文件位于 <repo>/scripts/data_paths.py
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
# <repo> 的上一级是数采工程根（含 datasets/），再上一级是项目总目录（含 data/）
_PROJECT_DIR = _REPO_ROOT.parent.parent

ENV_RAW = "XTRAINER_RAW_DATA_ROOT"
ENV_PROCESSED = "XTRAINER_PROCESSED_DATA_ROOT"
ENV_LEROBOT = "XTRAINER_LEROBOT_DATA_ROOT"
ENV_TEST_CONFIG = "XTRAINER_TEST_CONFIG_ROOT"
ENV_REJECTED = "XTRAINER_REJECTED_DATA_ROOT"
ENV_CONFIG_FILE = "XTRAINER_DATA_PATHS_CONFIG"

CONFIG_CANDIDATES = (
    _REPO_ROOT / "config" / "data_paths.yaml",
    _PROJECT_DIR / "config" / "data_paths.yaml",
)


def _load_config_file() -> dict:
    """读 config/data_paths.yaml。缺 PyYAML 或文件不存在时静默返回空 dict。

    刻意不把 yaml 作为硬依赖：数采环境是 Python 3.8 + 固定 requirements，
    不应为路径配置引入新依赖。没有 yaml 时环境变量与默认值仍然可用。
    """
    path = os.environ.get(ENV_CONFIG_FILE)
    candidates = [pathlib.Path(path)] if path else list(CONFIG_CANDIDATES)
    for p in candidates:
        if not p.is_file():
            continue
        try:
            import yaml  # noqa: PLC0415
        except ImportError:
            print(f"[data_paths] 发现 {p} 但未安装 PyYAML，忽略该配置文件")
            return {}
        try:
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[data_paths] 读取 {p} 失败，忽略: {e}")
            return {}
    return {}


class DataPaths:
    """四类数据根目录的解析结果。构造时即固定，不随后续环境变量变化。"""

    def __init__(self, project_dir: Optional[pathlib.Path] = None):
        self.project_dir = pathlib.Path(project_dir) if project_dir else _PROJECT_DIR
        cfg = _load_config_file()
        default_data = self.project_dir / "data"
        # 必须在任何 _resolve 之前初始化：_resolve 会往里写来源记录
        self._sources = {}

        self.raw = self._resolve(ENV_RAW, cfg.get("raw_data_root"), default_data / "raw")
        self.processed = self._resolve(ENV_PROCESSED, cfg.get("processed_data_root"),
                                       default_data / "processed")
        self.lerobot = self._resolve(ENV_LEROBOT, cfg.get("lerobot_data_root"),
                                     default_data / "lerobot")
        self.test_configs = self._resolve(ENV_TEST_CONFIG, cfg.get("test_config_root"),
                                          default_data / "test_configs")
        self.rejected = self._resolve(ENV_REJECTED, cfg.get("rejected_data_root"),
                                      default_data / "rejected")

    def _resolve(self, env_key, cfg_val, default) -> pathlib.Path:
        if env_key and os.environ.get(env_key):
            val, src = os.environ[env_key], f"env:{env_key}"
        elif cfg_val:
            val, src = cfg_val, "config"
        else:
            val, src = default, "default"
        p = pathlib.Path(str(val)).expanduser()
        self._sources[str(p)] = src
        return p

    def source_of(self, p) -> str:
        """返回某路径的来源（env / config / default），用于诊断。"""
        return getattr(self, "_sources", {}).get(str(p), "unknown")

    # ── Raw 下的常用子路径 ──
    def task_dir(self, task_name: str) -> pathlib.Path:
        """<raw>/<task_name>/ —— 一个数据集版本"""
        return self.raw / task_name

    def collect_dir(self, task_name: str) -> pathlib.Path:
        """<raw>/<task_name>/collect_data/ —— episode 实际所在"""
        return self.task_dir(task_name) / "collect_data"

    def episode_dir(self, task_name: str, episode_id: str) -> pathlib.Path:
        return self.collect_dir(task_name) / str(episode_id)

    def reject_list(self, task_name: str) -> pathlib.Path:
        """沿用既有文件名，保持与 check_episodes.py / Converter 的兼容"""
        return self.task_dir(task_name) / "episodes_to_delete.txt"

    def manifest(self, task_name: str) -> pathlib.Path:
        return self.task_dir(task_name) / "dataset_manifest.json"

    def statistics(self, task_name: str) -> pathlib.Path:
        return self.task_dir(task_name) / "dataset_statistics.json"

    def describe(self) -> str:
        lines = ["数据路径配置："]
        for name in ("raw", "processed", "lerobot", "rejected", "test_configs"):
            p = getattr(self, name)
            real = p.resolve() if p.exists() else None
            flag = "✓" if p.exists() else "✗ 不存在"
            extra = ""
            if p.is_symlink():
                extra = f"  (软链接 -> {os.readlink(p)})"
            elif real and real != p:
                extra = f"  (实体 {real})"
            lines.append(f"  {name:<13} {p}  [{self.source_of(p)}] {flag}{extra}")
        return "\n".join(lines)


# 模块级单例，供各脚本直接 import
paths = DataPaths()


if __name__ == "__main__":
    print(paths.describe())
