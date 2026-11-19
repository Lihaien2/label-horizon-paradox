"""行情数据层（Tushare 实现，【改动】替代原 DolphinDB）。

数据源（Tushare Pro API，2000积分及以上）：
  - daily        ：日线 OHLCV（open/high/low/close/vol/amount/pre_close/pct_chg）
  - adj_factor   ：复权因子（2000积分起）
  - suspend_d    ：每日停复牌（2000积分起）
  - index_daily  ：指数日线（用于基准收益）

所有行情统一返回 **宽表**（index=date，columns=wind_code/ts_code），与向量化回测引擎
的矩阵形状一致；并带本地 parquet 缓存。

关键约定（与论文/demo 对齐）：
  - 收益用"后复权"价格计算（避免除权缺口）：adj_price = raw_price * adj_factor。
  - VWAP：日线无直接 VWAP，用成交额/成交量近似：vwap(元/股) = amount*10 / vol。
    （tushare 中 amount 单位千元、vol 单位手，故 amount*1000/(vol*100)=amount*10/vol）
    后复权 VWAP = vwap_raw * adj_factor。
  - 停牌：tushare daily 不含停牌日（无行），并用 suspend_d 显式剔除停牌日。
  - 涨跌停：limit_list_d 需 5000 积分，2000 积分档不可用，故由收盘价 vs 前收盘价
    按板块涨跌停幅度近似（主板10%、创业板300xxx自2020-08-24起20%）。见
    ``_limit_ratio``。若你有 5000+ 积分，可改用 limit_list_d 获得精确涨跌停。
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from data_fetch.cache import cache_key, read_cache, write_cache
from data_fetch.connection import pro

__all__ = [
    "get_adj_ohlcv",
    "get_adjclose",
    "get_vwap_adj",
    "get_limit_status",
    "get_tradable_mask",
    "get_index_close",
    "get_universe_ohlcv_long",
]

# 每次接口调用的间隔（秒），避免触发 Tushare 频控；可用环境变量 TUSHARE_SLEEP 覆盖。
REQUEST_INTERVAL = float(os.environ.get("TUSHARE_SLEEP", "0.12"))


def _rate_limit() -> None:
    """调用 Tushare 接口前的简单限速（2000积分档主要受 suspend_d 200次/分钟限制）。"""
    time.sleep(REQUEST_INTERVAL)


def _ts_range(start: str | pd.Timestamp, end: str | pd.Timestamp) -> tuple[str, str]:
    """把起止日期转成 Tushare 需要的 'YYYYMMDD' 字符串。"""
    s = pd.Timestamp(start).strftime("%Y%m%d")
    e = pd.Timestamp(end).strftime("%Y%m%d")
    return s, e


# ───────────────────────── 原始数据抓取（长表，带缓存） ─────────────────────────


def get_universe_ohlcv_long(
    codes: list[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """逐只股票抓取日线 + 复权因子，返回合并后的长表。

    返回列：ts_code, trade_date, open, high, low, close, pre_close, pct_chg,
            vol, amount, adj_factor, vwap_raw, vwap_adj。

    说明：tushare 的 daily 不含停牌日（停牌日无行），故"有无行"即近似可交易。
    """
    key = cache_key("universe_ohlcv", "|".join(sorted(codes)), str(start), str(end))
    if not force_refresh:
        cached = read_cache("universe_ohlcv", key)
        if cached is not None:
            return cached

    api = pro()
    s, e = _ts_range(start, end)
    parts = []
    n = len(codes)
    for i, code in enumerate(codes, 1):
        _rate_limit()
        try:
            d = api.daily(ts_code=code, start_date=s, end_date=e)
        except Exception as ex:  # 单只失败不中断，记录后继续
            print(f"  [warn] daily {code} 失败: {ex}")
            continue
        if d is None or d.empty:
            continue

        _rate_limit()
        try:
            af = api.adj_factor(ts_code=code, start_date=s, end_date=e)
        except Exception as ex:
            af = None
            print(f"  [warn] adj_factor {code} 失败: {ex}")

        d = d.copy()
        d["trade_date"] = d["trade_date"].astype(str)
        if af is not None and not af.empty:
            af = af.copy()
            af["trade_date"] = af["trade_date"].astype(str)
            d = d.merge(af[["trade_date", "adj_factor"]], on="trade_date", how="left")
        else:
            d["adj_factor"] = np.nan

        # 后复权 OHLC（原始价格 × 复权因子）。
        for col in ("open", "high", "low", "close"):
            d[f"{col}_adj"] = d[col] * d["adj_factor"]
        # VWAP 近似：amount(千元)*10 / vol(手) = 元/股；再后复权。
        d["vwap_raw"] = np.where(d["vol"] > 0, d["amount"] * 10.0 / d["vol"], np.nan)
        d["vwap_adj"] = d["vwap_raw"] * d["adj_factor"]

        parts.append(d)
        if i % 50 == 0 or i == n:
            print(f"  [ohlcv] 已抓取 {i}/{n} 只", flush=True)

    if not parts:
        raise RuntimeError("未抓到任何股票日线；请检查 Token、积分与股票代码。")
    long = pd.concat(parts, ignore_index=True)
    long["ts_code"] = long["ts_code"].astype(str)
    write_cache("universe_ohlcv", key, long)
    return long


def get_suspend_long(
    codes: list[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """抓取每日停牌记录（长表：ts_code, trade_date, suspend_type）。

    suspend_type == 'S' 表示当日停牌。用于构建 tradable_mask。
    """
    key = cache_key("suspend", "|".join(sorted(codes)), str(start), str(end))
    if not force_refresh:
        cached = read_cache("suspend", key)
        if cached is not None:
            return cached

    api = pro()
    s, e = _ts_range(start, end)
    parts = []
    for code in codes:
        _rate_limit()
        try:
            d = api.suspend_d(ts_code=code, start_date=s, end_date=e)
        except Exception as ex:
            print(f"  [warn] suspend_d {code} 失败（继续，缺省视为可交易）: {ex}")
            continue
        if d is None or d.empty:
            continue
        d = d.copy()
        d["trade_date"] = d["trade_date"].astype(str)
        d["ts_code"] = d["ts_code"].astype(str)
        parts.append(d[["ts_code", "trade_date", "suspend_type"]])
    if not parts:
        # 无停牌记录时返回空表（全部视为可交易）。
        return pd.DataFrame(columns=["ts_code", "trade_date", "suspend_type"])
    out = pd.concat(parts, ignore_index=True)
    write_cache("suspend", key, out)
    return out


# ───────────────────────── 宽表构建 ─────────────────────────


def _long_to_wide(long: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """长表 -> 宽表（index=trade_date[Timestamp], columns=ts_code）。

    注意：把 'YYYYMMDD' 字符串索引转成 DatetimeIndex，便于下游用 Timestamp 切片对齐。
    """
    wide = long.pivot(index="trade_date", columns="ts_code", values=value_col).sort_index()
    wide.index = pd.to_datetime(wide.index, format="%Y%m%d")
    return wide


def get_adj_ohlcv(
    codes: list[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """后复权 OHLCV 宽表字典。

    返回 {open, high, low, close, vwap, volume} 六个 (T x N) 宽表（均后复权，
    vwap 为后复权 VWAP，volume 为原始成交量（手））。用于 Alpha158 因子计算。
    """
    long = get_universe_ohlcv_long(codes, start, end, force_refresh=force_refresh)
    return {
        "open": _long_to_wide(long, "open_adj"),
        "high": _long_to_wide(long, "high_adj"),
        "low": _long_to_wide(long, "low_adj"),
        "close": _long_to_wide(long, "close_adj"),
        "vwap": _long_to_wide(long, "vwap_adj"),
        "volume": _long_to_wide(long, "vol"),
    }


def get_adjclose(
    codes: list[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """后复权收盘价宽表（用于 NAV 标记与收益计算）。"""
    return get_adj_ohlcv(codes, start, end, force_refresh=force_refresh)["close"]


def get_vwap_adj(
    codes: list[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """后复权 VWAP 宽表（用于成交价与标签计算）。"""
    return get_adj_ohlcv(codes, start, end, force_refresh=force_refresh)["vwap"]


def _limit_ratio(code: str, date: pd.Timestamp) -> float:
    """返回某股票在某日的涨跌停幅度。

    主板 10%；创业板（300/301 开头）自 2020-08-24 起 20%（此前 10%）；
    科创板（688/689）20%（沪深300通常不含）。ST 股 5% 未单独处理（沪深300无 ST）。
    """
    if code.startswith(("688", "689")):
        return 0.20
    if code.startswith(("300", "301")):
        return 0.20 if date >= pd.Timestamp("2020-08-24") else 0.10
    return 0.10


def get_limit_status(
    codes: list[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """涨跌停状态矩阵（limit_up, limit_down），由收盘价 vs 前收盘价近似。

    limit_up   = close >= round(pre_close * (1 + limit), 2)（涨停价）
    limit_down = close <= round(pre_close * (1 - limit), 2)（跌停价）

    （limit_list_d 需 5000 积分，2000 档不可用；此处用价格近似，详见模块 docstring。）
    """
    key_up = cache_key("limit_up", str(start), str(end))
    key_dn = cache_key("limit_dn", str(start), str(end))
    if not force_refresh:
        cu, cd = read_cache("limit_up", key_up), read_cache("limit_dn", key_dn)
        if cu is not None and cd is not None:
            return cu, cd

    long = get_universe_ohlcv_long(codes, start, end, force_refresh=force_refresh)
    close = _long_to_wide(long, "close")
    pre_close = _long_to_wide(long, "pre_close")

    # 逐日逐股计算涨跌停幅度（按板块）。为效率，先把代码映射为每日幅度矩阵。
    dates = close.index
    col_ratio = pd.DataFrame(1.0, index=dates, columns=close.columns)
    for code in close.columns:
        col_ratio[code] = _limit_ratio(code, pd.Timestamp(dates[-1]))  # 近似用末日期判断板块
    # 创业板 2020-08-24 之前为 10%，之后为 20%：对 300/301 代码按日期分段修正。
    for code in close.columns:
        if code.startswith(("300", "301")):
            before = dates < pd.Timestamp("2020-08-24")
            col_ratio.loc[before, code] = 0.10
            col_ratio.loc[~before, code] = 0.20

    # 涨跌停价（四舍五入到分，与交易所规则一致）。
    limit_up_price = (pre_close * (1 + col_ratio)).round(2)
    limit_down_price = (pre_close * (1 - col_ratio)).round(2)

    limit_up = (close >= limit_up_price).fillna(False)
    limit_down = (close <= limit_down_price).fillna(False)
    write_cache("limit_up", key_up, limit_up)
    write_cache("limit_dn", key_dn, limit_down)
    return limit_up, limit_down


def get_tradable_mask(
    codes: list[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """可交易掩码宽表（True=可交易，即非停牌）。

    停牌判定：tushare daily 停牌日无行（close 为 NaN）；再叠加 suspend_d 的
    suspend_type=='S' 显式剔除。二者取交集。
    """
    key = cache_key("tradable", str(start), str(end))
    if not force_refresh:
        cached = read_cache("tradable", key)
        if cached is not None:
            return cached

    long = get_universe_ohlcv_long(codes, start, end, force_refresh=force_refresh)
    close = _long_to_wide(long, "close")
    tradable = close.notna()  # 有日线行 => 当日有交易

    # 显式剔除 suspend_d 记录的停牌日（如接口可用）。
    susp = get_suspend_long(codes, start, end, force_refresh=force_refresh)
    if not susp.empty:
        susp_s = susp[susp["suspend_type"] == "S"]
        if not susp_s.empty:
            susp_wide = susp_s.pivot_table(
                index="trade_date", columns="ts_code", values="suspend_type",
                aggfunc="count", fill_value=0,
            ).sort_index()
            susp_wide.index = pd.to_datetime(susp_wide.index, format="%Y%m%d")  # 对齐 DatetimeIndex
            susp_flag = susp_wide.reindex(index=close.index, columns=close.columns).fillna(0) > 0
            tradable = tradable & (~susp_flag)

    tradable = tradable.fillna(False)
    write_cache("tradable", key, tradable)
    return tradable


def get_index_close(
    codes: list[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """指数日收盘价宽表（index=date, columns=指数代码）。

    数据源：Tushare index_daily（未复权，用作基准收益比较）。
    """
    key = cache_key("idxclose", "|".join(codes), str(start), str(end))
    if not force_refresh:
        cached = read_cache("idxclose", key)
        if cached is not None:
            return cached

    api = pro()
    s, e = _ts_range(start, end)
    parts = []
    for code in codes:
        _rate_limit()
        try:
            d = api.index_daily(ts_code=code, start_date=s, end_date=e)
        except Exception as ex:
            print(f"  [warn] index_daily {code} 失败: {ex}")
            continue
        if d is None or d.empty:
            continue
        d = d.copy()
        d["trade_date"] = d["trade_date"].astype(str)
        d["ts_code"] = d["ts_code"].astype(str)
        parts.append(d[["ts_code", "trade_date", "close"]])
    if not parts:
        raise RuntimeError("index_daily 未返回任何指数收盘价；请检查 Token/积分/代码。")
    long = pd.concat(parts, ignore_index=True)
    wide = _long_to_wide(long, "close")
    write_cache("idxclose", key, wide)
    return wide
