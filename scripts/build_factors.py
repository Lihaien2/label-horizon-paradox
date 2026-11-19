"""计算沪深300的 Alpha158 因子（DATA_START~DATA_END），保存到 data/。

（【改动】原为全市场 Alpha158；现改为只算沪深300成分股，数据源 Tushare。）

从 Tushare 后复权 OHLCV 出发，用 factors_calculate.alpha158_db.compute_alpha158 计算
158 个因子。按约一年一批（含 rolling-60 预热）计算、拼接、去重，再截断到目标区间。

输出（data/a_share_data/）：
  - factors.parquet：完整因子（长表：date × wind_code × 158 因子）

默认不覆盖已存在文件（除非 --force）。

用法：
  python scripts/build_factors.py              # 计算（文件已存在则跳过）
  python scripts/build_factors.py --force      # 强制重算
"""
from __future__ import annotations
import sys, warnings, time, argparse
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SRC = _ROOT / "src"
for _p in (_HERE, _ROOT, _SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd

from src.config import (
    DATA_START, DATA_END, RAW_DATA_DIR, UNIVERSE_INDEX_CODE,
)
from factors_calculate.alpha158_db import compute_alpha158, ALPHA158_NAMES
from data_fetch.index_members_db import get_index_member_mask

# 分批：约一年一批，向前推半年做 rolling-60 预热（与上一批有重叠用于去重）。
# 每批起于目标年前 6 个月，用于滚动窗口预热。保存时截断到 DATA_START..DATA_END。
BATCHES = [
    ("2019-01-01", "2020-06-30"),  # 2019-06 预热 + 2020 上半年
    ("2019-06-01", "2020-12-31"),  # 2020
    ("2020-06-01", "2021-12-31"),  # 2021
    ("2021-06-01", "2022-12-31"),  # 2022
    ("2022-06-01", "2023-12-31"),  # 2023
    ("2023-06-01", "2024-12-31"),  # 2024
    ("2024-06-01", "2025-12-31"),  # 2025
    ("2025-06-01", "2026-06-30"),  # 2026
]


def build(start: str, end: str, out_dir: Path, force: bool, batches=None):
    """分批计算因子 + 拼接 + 截断 + 保存。"""
    path = out_dir / "factors.parquet"
    if path.exists() and not force:
        print(f"[skip] factors.parquet 已存在 ({path.stat().st_size/1e6:.1f}M)", flush=True)
        return

    print(f"\n=== 计算沪深300 Alpha158 因子 {start}~{end}（每批含 6 个月预热） ===", flush=True)
    print(f"  ALPHA158 因子数：{len(ALPHA158_NAMES)}", flush=True)

    # 沪深300历史成分（全区间并集，无幸存者偏差）。
    member_mask = get_index_member_mask(UNIVERSE_INDEX_CODE, start, end, force_refresh=force)
    codes = sorted(member_mask.columns.tolist())
    print(f"  沪深300历史成分股票数：{len(codes)}", flush=True)

    t0 = time.time()
    bs = batches if batches is not None else BATCHES
    parts = []
    for i, (s, e) in enumerate(bs):
        print(f"[{i+1}/{len(bs)}] 计算因子 {s}~{e}...", flush=True)
        t1 = time.time()
        long_df = compute_alpha158(s, e, index_code=None, codes=codes, as_frame="long")
        print(f"    完成：{long_df.shape}，耗时 {time.time()-t1:.0f}s", flush=True)
        parts.append(long_df)
    print(f"拼接 {len(parts)} 批...", flush=True)
    long_df = pd.concat(parts)
    long_df = long_df[~long_df.index.duplicated(keep="first")]  # 去重（预热重叠）
    print(f"  拼接+去重后：{long_df.shape}，耗时 {time.time()-t0:.0f}s", flush=True)

    # 截断到保存区间
    dates = long_df.index.get_level_values("date")
    long_df = long_df[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    print(f"  截断到 {start}~{end}：{long_df.shape}", flush=True)
    per_day = long_df.groupby(level="date").size()
    print(f"  每日股票数 min/中位/max = {per_day.min()}/{int(per_day.median())}/{per_day.max()}", flush=True)

    long_df.to_parquet(path)
    print(f"  [saved] factors.parquet shape={long_df.shape} ({path.stat().st_size/1e6:.1f}M)", flush=True)
    print(f"\n=== 完成，总耗时 {time.time()-t0:.0f}s ===", flush=True)


def main():
    ap = argparse.ArgumentParser(description="计算沪深300 Alpha158 因子（Tushare）")
    ap.add_argument("--force", action="store_true", help="强制重算（覆盖已存在文件）")
    ap.add_argument("--start", default=DATA_START, help=f"开始日期（默认 {DATA_START}）")
    ap.add_argument("--end", default=DATA_END, help=f"结束日期（默认 {DATA_END}）")
    args = ap.parse_args()

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"force={args.force}（已存在文件 {'覆盖' if args.force else '跳过'}）", flush=True)
    build(args.start, args.end, RAW_DATA_DIR, args.force)


if __name__ == "__main__":
    main()
