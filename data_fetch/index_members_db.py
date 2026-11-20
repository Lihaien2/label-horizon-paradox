"""指数成分股（PIT，Point-in-Time）——Tushare 实现。

数据源：Tushare ``index_weight`` 接口（月度成分快照，2000积分起）。
根据 S_CON_INDATE/S_CON_OUTDATE 类似的"快照区间"重建逐日成分标记，避免幸存者偏差。

本项目只做沪深300（000300.SH）。Tushare 的 index_weight 对沪深300可能同时接受
"000300.SH"（Wind 代码）与 "399300.SZ"（深交所代码）；本模块会先尝试传入的
index_code，若为空再尝试另一个代码。

用法：
    from data_fetch.index_members_db import get_index_member_mask, get_index_members
    mask = get_index_member_mask("000300.SH", "2020-01-02", "2026-06-30")
    codes = get_index_members("000300.SH", "2024-06-28")
"""

from __future__ import annotations

import time

import pandas as pd

from data_fetch.cache import cache_key, read_cache, write_cache
from data_fetch.calendar_db import trading_dates
from data_fetch.connection import pro

__all__ = ["get_index_members", "get_index_member_mask", "INDEX_CSI300"]

INDEX_CSI300 = "000300.SH"

# 沪深300在 Tushare 里的两个等价代码（用于回退尝试）。
_CSI300_ALT = "399300.SZ"


def _fetch_snapshots(index_code: str, start: str, end: str) -> pd.DataFrame:
    """按月循环拉取指数月度成分快照（避免 index_weight 单次大范围被截断）。"""
    api = pro()
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    month_starts = pd.date_range(start_ts.replace(day=1), end_ts.replace(day=1), freq="MS")
    parts = []
    for ms in month_starts:
        me = ms + pd.offsets.MonthEnd(0)
        lo = max(ms, start_ts).strftime("%Y%m%d")
        hi = min(me, end_ts).strftime("%Y%m%d")
        time.sleep(0.12)  # 限速（index_members_db 顶部已 import time）
        try:
            d = api.index_weight(index_code=index_code, start_date=lo, end_date=hi)
        except Exception as ex:
            print(f"  [warn] index_weight {index_code} {lo}-{hi} 失败: {ex}")
            continue
        if d is None or d.empty:
            continue
        d = d[d["con_code"].notna()].copy()
        d["con_code"] = d["con_code"].astype(str)
        d["trade_date"] = d["trade_date"].astype(str).str.replace("-", "", regex=False)
        parts.append(d[["con_code", "trade_date", "weight"]])
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(subset=["con_code", "trade_date"])
    return out


def _snapshots_with_fallback(index_code: str, start: str, end: str) -> tuple[pd.DataFrame, str]:
    """尝试获取快照；若目标代码为空则回退到备选代码。返回 (df, 实际使用的代码)。"""
    df = _fetch_snapshots(index_code, start, end)
    if not df.empty:
        return df, index_code
    # 沪深300两个代码互备；避免单个代码在 Tushare 侧取不到数据。
    alt = _CSI300_ALT if index_code.upper() == INDEX_CSI300 else INDEX_CSI300
    df2 = _fetch_snapshots(alt, start, end)
    if not df2.empty:
        print(f"  [index_members] {index_code} 无快照，改用 {alt}")
        return df2, alt
    raise RuntimeError(
        f"index_weight 在 {start}~{end} 无任何快照（尝试了 {index_code} / {alt}）。"
        "请确认 Token、积分（>=2000）与指数代码。"
    )


def get_index_member_mask(
    index_code: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """每日成分标记（宽表：日期 × 股票，True=当日属于该指数）。

    基于 index_weight 月度快照：每个快照状态延续到下一个快照前一天（阶梯式近似，
    非交易所公告级精确生效日）。返回 (T x N) bool 表，index=交易日（由交易日历对齐）。

    Args:
        index_code: 指数代码，如 "000300.SH"。
        start/end: 日期范围（闭区间）。
        force_refresh: 是否强制刷新缓存。

    Returns:
        (T x N) bool 宽表，index=交易日，columns=wind_code（含历史进出的全部成分）。
    """
    key = cache_key("idxmember", index_code, str(start), str(end))
    if not force_refresh:
        cached = read_cache("idxmember", key)
        if cached is not None:
            return cached

    s = pd.Timestamp(start).strftime("%Y%m%d")
    e = pd.Timestamp(end).strftime("%Y%m%d")
    snap, used_code = _snapshots_with_fallback(index_code, s, e)

    # 快照按 trade_date 升序；每只股票在某快照区间内视为成分股。
    snap = snap.sort_values("trade_date")
    snapshot_dates = sorted(snap["trade_date"].unique())
    codes = sorted(snap["con_code"].unique())

    # 交易日轴（与回测时间轴一致）。
    dates = pd.DatetimeIndex(trading_dates(start, end))
    mask = pd.DataFrame(False, index=dates, columns=codes)

    # 构造每个快照的生效区间 [snap_date, next_snap_date - 1天]，最后一个延续到 end。
    for i, snap_date in enumerate(snapshot_dates):
        interval_start = max(s, snap_date)
        if i + 1 < len(snapshot_dates):
            next_dt = pd.to_datetime(snapshot_dates[i + 1], format="%Y%m%d")
            interval_end = (next_dt - pd.Timedelta(days=1)).strftime("%Y%m%d")
        else:
            interval_end = e
        if interval_start > interval_end:
            continue
        # 该快照当日的成分列表。
        members = snap[snap["trade_date"] == snap_date]["con_code"].unique()
        seg = (dates >= pd.to_datetime(interval_start, format="%Y%m%d")) & \
              (dates <= pd.to_datetime(interval_end, format="%Y%m%d"))
        for code in members:
            if code in mask.columns:
                mask.loc[seg, code] = True

    write_cache("idxmember", key, mask)
    print(f"  [index_members] {used_code} 每日成分掩码：{mask.shape}，"
          f"日均成分 {int(mask.sum(axis=1).mean())} 只")
    return mask


def get_index_members(
    index_code: str,
    date: str | pd.Timestamp,
    force_refresh: bool = False,
) -> list[str]:
    """某交易日的指数成分股列表（PIT），升序返回 wind_code 列表。"""
    date = pd.Timestamp(date)
    mask = get_index_member_mask(index_code, date, date, force_refresh=force_refresh)
    if date not in mask.index:
        return []
    return sorted(mask.columns[mask.loc[date]].tolist())
