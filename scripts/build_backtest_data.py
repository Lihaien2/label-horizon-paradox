"""下载回测数据到 data/a_share_data/（DATA_START~DATA_END）——Tushare + 仅沪深300。

（【改动】原为 DolphinDB + 全市场 + 沪深300/500/1000；现改为 Tushare，只下载沪深300。）

职责：从 Tushare 拉取沪深300历史成分股（PIT）与其日线行情，加工成 parquet 存到本地。
训练/回测运行时只读 data/ 下的 parquet，不再连接数据库。

输出（data/a_share_data/）：
  - adjclose.parquet       ：后复权收盘价（NAV 标记价 + 收益计算）
  - vwap_adj.parquet       ：后复权 VWAP = 成交额/成交量 近似 × 复权因子（成交价 + 标签）
  - limit_up.parquet       ：涨停掩码（不能买入）
  - limit_down.parquet     ：跌停掩码（不能卖出）
  - tradable_mask.parquet  ：可交易掩码（排除停牌）
  - member_mask_csi300.parquet ：沪深300逐日成分掩码（PIT）
  - index_close.parquet    ：沪深300日收盘（回测基准）

默认不覆盖已存在文件（除非 --force）。

用法：
  python scripts/build_backtest_data.py              # 下载缺失文件
  python scripts/build_backtest_data.py --force      # 强制重新下载
  python scripts/build_backtest_data.py --start 2025-01-02 --end 2025-01-10
"""
from __future__ import annotations
import sys, warnings, time, argparse
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (_HERE, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd

from data_fetch import marketdata_db as md
from data_fetch.index_members_db import get_index_member_mask
from src.config import (
    DATA_START, DATA_END, RAW_DATA_DIR, UNIVERSE_INDEX_CODE, UNIVERSE_NAME,
)


def _save(name: str, df, out_dir: Path, force: bool) -> bool:
    """保存 parquet；文件已存在且未 --force 时跳过。返回是否真的写入。"""
    path = out_dir / f"{name}.parquet"
    if path.exists() and not force:
        print(f"  [skip] {name}.parquet 已存在 ({path.stat().st_size/1e6:.1f}M)", flush=True)
        return False
    df.to_parquet(path)
    print(f"  [saved] {name}.parquet shape={df.shape} ({path.stat().st_size/1e6:.1f}M)", flush=True)
    return True


def build(start: str, end: str, out_dir: Path, force: bool):
    """下载沪深300成分 + 行情并落盘。"""
    print(f"\n[1] 获取沪深300（{UNIVERSE_INDEX_CODE}）成分 {start}~{end}...", flush=True)
    t0 = time.time()
    member_mask = get_index_member_mask(UNIVERSE_INDEX_CODE, start, end, force_refresh=force)
    codes = sorted(member_mask.columns.tolist())  # 历史上进出的全部成分（无幸存者偏差）
    print(f"  历史成分股票数：{len(codes)}", flush=True)

    print(f"\n[2] 下载日线行情 {start}~{end}（{len(codes)} 只）...", flush=True)
    ohlcv = md.get_adj_ohlcv(codes, start, end, force_refresh=force)
    _save("adjclose", ohlcv["close"], out_dir, force)
    _save("vwap_adj", ohlcv["vwap"], out_dir, force)

    print(f"\n[3] 涨跌停 / 可交易掩码...", flush=True)
    limit_up, limit_down = md.get_limit_status(codes, start, end, force_refresh=force)
    _save("limit_up", limit_up, out_dir, force)
    _save("limit_down", limit_down, out_dir, force)
    # ---------------- 修改开始 ----------------
    # 原来的代码（注释掉）：
    # tradable = md.get_tradable_mask(codes, start, end, force_refresh=force)

    # 新代码：直接从日线收盘价推导可交易状态,避开 Tushare 限流
    # 如果收盘价是 NaN，说明当天停牌（或者还没上市/已退市）
    adjclose_df = ohlcv["close"]
    tradable = adjclose_df.notna()
    # ---------------- 修改结束 ----------------
    _save("tradable_mask", tradable, out_dir, force)

    print(f"\n[4] 指数成分掩码 / 指数收盘...", flush=True)
    _save(f"member_mask_{UNIVERSE_NAME}", member_mask, out_dir, force)
    index_close = md.get_index_close([UNIVERSE_INDEX_CODE], start, end, force_refresh=force)
    _save("index_close", index_close, out_dir, force)

    print(f"\n=== 完成，总耗时 {time.time()-t0:.0f}s ===", flush=True)


def main():
    ap = argparse.ArgumentParser(description="下载沪深300回测数据到 data/（Tushare）")
    ap.add_argument("--force", action="store_true", help="强制重新下载（覆盖已存在文件）")
    ap.add_argument("--start", default=DATA_START, help=f"开始日期（默认 {DATA_START}）")
    ap.add_argument("--end", default=DATA_END, help=f"结束日期（默认 {DATA_END}）")
    args = ap.parse_args()

    out_dir = RAW_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== 下载沪深300回测数据 {args.start}~{args.end} -> {out_dir} ===", flush=True)
    print(f"  force={args.force}（已存在文件 {'覆盖' if args.force else '跳过'}）", flush=True)
    build(args.start, args.end, out_dir, args.force)

    # 校验
    print(f"\n=== 校验 ===", flush=True)
    for f in sorted(out_dir.iterdir()):
        if f.suffix == ".parquet":
            print(f"  {f.name} ({f.stat().st_size/1e6:.1f}M)", flush=True)


if __name__ == "__main__":
    main()
