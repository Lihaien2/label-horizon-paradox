"""Shared base configuration: date ranges, paths, device, horizons, lookback.

这些是固定的项目常量（不是可调超参数）。可调的训练超参数放在各个入口脚本
（``scripts/run_search.py``、``scripts/run_blo.py``）顶部。
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (_HERE, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch

# ---------------------------------------------------------------------------
# 日期范围（按时间顺序切分，无未来信息泄漏）
# ---------------------------------------------------------------------------
TRAIN_START, TRAIN_END = "2020-01-02", "2023-12-31"
VALID_START, VALID_END = "2024-01-02", "2024-12-31"
TEST_START, TEST_END = "2025-01-02", "2026-06-30"

# 数据构建日期范围。数据从 2019-06 开始（TRAIN_START 前半年的滚动窗口预热）。
# 标签分位数阈值只在 TRAIN_START..TRAIN_END 上计算（对 valid/test 无泄漏）。
DATA_START = "2019-06-01"
DATA_END = "2026-06-30"

# ---------------------------------------------------------------------------
# 设备、回看窗口、候选 horizon 与推理目标
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 【改动】回看窗口：模型输入为过去 LOOKBACK 个交易日的特征序列（1 天 -> 10 天）。
LOOKBACK = 10

# 【改动】候选标签 horizon：原 1..10 改为 1..5（标准化累积标签）。
HORIZONS = list(range(1, 6))   # h = 1..5

# 【改动】推理目标固定为 h=5（原为 10）。
TARGET_H = 5

# ---------------------------------------------------------------------------
# 股票池：只做沪深300（原为全市场 / CSI300+500+1000）
# ---------------------------------------------------------------------------
# 沪深300指数代码（Tushare / Wind 通用：000300.SH）。
UNIVERSE_INDEX_CODE = "000300.SH"
# 用于数据文件名后缀的简称（如 member_mask_csi300.parquet）。
UNIVERSE_NAME = "csi300"

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
RESULTS_DIR = _ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
DATA_DIR = _ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "a_share_data"
PROCESSED_DIR = DATA_DIR / "a_share_data_processed"

# 损失曲线 / 图表输出目录（【改动】新增，用于报告分析）。
LOSSES_DIR = RESULTS_DIR / "losses"
FIGURES_DIR = RESULTS_DIR / "figures"
