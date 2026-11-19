"""交易日历（Tushare 实现）。

数据源：Tushare ``trade_cal`` 接口（SSE 交易所，A股交易日）。
回测时间轴的唯一事实来源，带本地 parquet 缓存。

（【改动】原为 DolphinDB AShareCalendar，现改用 Tushare trade_cal。）
"""

from __future__ import annotations

import pandas as pd

from data_fetch.cache import cache_key, read_cache, write_cache
from data_fetch.connection import pro

__all__ = ["trading_dates", "align_to_trading", "is_trading_day", "next_trading_day", "prev_trading_day"]

# A股日历用上交所（SSE）；上交所与深交所交易日一致。
_EXCH = "SSE"
_CACHE_NAME = "calendar"


def _fetch_all() -> pd.DatetimeIndex:
    """从 Tushare 拉取全部 SSE 交易日，返回 DatetimeIndex。"""
    api = pro()
    df = api.trade_cal(exchange=_EXCH, start_date="19900101", end_date="20301231", is_open="1")
    if df is None or df.empty:
        raise RuntimeError("Tushare trade_cal 返回空；请检查 Token 与积分。")
    # cal_date 为 'YYYYMMDD' 字符串 -> Timestamp
    dates = pd.to_datetime(df["cal_date"], format="%Y%m%d")
    return pd.DatetimeIndex(sorted(dates))


def _get_all_dates() -> pd.DatetimeIndex:
    """获取全部交易日（带缓存）。"""
    key = cache_key(_CACHE_NAME, _EXCH, "all")
    cached = read_cache(_CACHE_NAME, key)
    if cached is not None:
        return pd.DatetimeIndex(cached["date"])
    dates = _fetch_all()
    write_cache(_CACHE_NAME, key, pd.DataFrame({"date": dates}))
    return dates


def trading_dates(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex:
    """返回 [start, end]（闭区间）内的交易日，升序 DatetimeIndex。"""
    all_dates = _get_all_dates()
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    mask = (all_dates >= start) & (all_dates <= end)
    return all_dates[mask]


def align_to_trading(date: str | pd.Timestamp, direction: str = "forward") -> pd.Timestamp:
    """把日期对齐到最近的交易日（forward/backward/nearest）。"""
    date = pd.Timestamp(date)
    all_dates = _get_all_dates()
    if direction == "forward":
        after = all_dates[all_dates >= date]
        return after[0] if len(after) else all_dates[-1]
    if direction == "backward":
        before = all_dates[all_dates <= date]
        return before[-1] if len(before) else all_dates[0]
    if direction == "nearest":
        idx = all_dates.get_indexer([date], method="nearest")[0]
        return all_dates[idx]
    raise ValueError(f"direction must be forward/backward/nearest, got {direction}")


def is_trading_day(date: str | pd.Timestamp) -> bool:
    """判断某天是否为交易日。"""
    return pd.Timestamp(date) in _get_all_dates()


def next_trading_day(date: str | pd.Timestamp) -> pd.Timestamp:
    """下一个交易日（不含当天）。"""
    date = pd.Timestamp(date)
    all_dates = _get_all_dates()
    after = all_dates[all_dates > date]
    return after[0] if len(after) else all_dates[-1]


def prev_trading_day(date: str | pd.Timestamp) -> pd.Timestamp:
    """上一个交易日（不含当天）。"""
    date = pd.Timestamp(date)
    all_dates = _get_all_dates()
    before = all_dates[all_dates < date]
    return before[-1] if len(before) else all_dates[0]
