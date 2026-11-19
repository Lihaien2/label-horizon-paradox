<h1 align="center">Data Pipeline</h1>

This document describes the data layer: market data (Tushare), Alpha158 factors,
label construction, and factor preprocessing（当前为 **沪深300 + 回看10天 + ffill** 版本）。

## Overview

All market data is read from parquet files under `data/a_share_data/` at
runtime — no database connection is needed for training/backtesting. The
download scripts (`scripts/build_*.py`) pull from **Tushare** once and cache
locally.

```
Tushare ──build_backtest_data.py──> data/a_share_data/ (raw parquet, CSI300)
                                        │
              build_factors.py ─────────┤  (compute Alpha158)
                                        │
              cache_factors_processed.py ──> data/a_share_data_processed/ (preprocessed factors)
                                        │
              run_search / run_blo ──────┴──> training
```

## Data Source: Tushare（2000积分档）

Token 从环境变量 `TUSHARE_TOKEN` 读取。用到的接口与积分要求：

| 接口 | 用途 | 积分 |
|---|---|---|
| `daily` | 日线 OHLCV | 120 起 |
| `adj_factor` | 复权因子 | 2000 起 |
| `index_weight` | 指数月度成分（PIT） | 2000 起 |
| `suspend_d` | 每日停复牌 | 2000 起 |
| `trade_cal` | 交易日历 | 基础 |
| `index_daily` | 指数日线（基准） | 基础 |

> `limit_list_d`（精确涨跌停）需 **5000 积分**，2000 档不可用；本项目用"收盘价 vs
> 前收盘价"按板块涨跌停幅度近似（主板10%、创业板20%、科创板20%）。

## Data Layout

`data/a_share_data/`（原始，来自 Tushare，仅沪深300历史成分）：

| File | Shape | Description |
|---|---|---|
| `adjclose.parquet` | T × N (wide) | 后复权收盘价（= raw close × adj_factor；NAV 标记 + 收益计算） |
| `vwap_adj.parquet` | T × N | 后复权 VWAP = amount×10/vol × adj_factor（成交价 + 标签） |
| `limit_up.parquet` | T × N | 涨停掩码（近似，见上） |
| `limit_down.parquet` | T × N | 跌停掩码（近似） |
| `tradable_mask.parquet` | T × N | 可交易掩码（非停牌：suspend_d + 日线行存在） |
| `member_mask_csi300.parquet` | T × N | 沪深300逐日成分掩码（PIT，月度快照阶梯近似） |
| `index_close.parquet` | T × 1 | 沪深300日收盘（基准） |
| `factors.parquet` | long (date, code) × 158 | Alpha158 factors |

`data/a_share_data_processed/`:

| File | Description |
|---|---|
| `factors.parquet` | Preprocessed factors (69 standardized + 89 kept), NaN thresholds from train period |

## Date Ranges

Defined in `src/config.py`:

| Constant | Value | Purpose |
|---|---|---|
| `DATA_START` | 2019-06-01 | Data build start (half-year before train, for rolling-60 warmup) |
| `DATA_END` | 2026-06-30 | Data build end |
| `TRAIN_START` / `TRAIN_END` | 2020-01-02 / 2023-12-31 | Training |
| `VALID_START` / `VALID_END` | 2024-01-02 / 2024-12-31 | Validation |
| `TEST_START` / `TEST_END` | 2025-01-02 / 2026-06-30 | Test (final eval + backtest) |

Label quantile thresholds and factor de-extreme thresholds are computed on
`TRAIN_START..TRAIN_END` only.

## Alpha158 Factors

`factors_calculate/alpha158.py` defines the 158 factor names (consistent with
Qlib). Computation (`factors_calculate/alpha158_db.py`) pulls **post-adjusted
OHLCV from Tushare** (`marketdata_db.get_adj_ohlcv`) and computes factors in
yearly batches with rolling-60 warmup, concatenated and de-duplicated.

## Factor Preprocessing

`src/data_utils.py:load_factors_tanhw` applies **intrinsic-class preprocessing**:

- **69 heavy-tailed factors** (price ratios, MA, ROC, etc.): cross-sectional
  `tanh(median / (IQR·2))`; outliers beyond train-set `q001/q999` set to NaN.
- **89 bounded factors** (candlestick ratios, statistics): kept as-is.

The preprocessed cache is built once by `scripts/cache_factors_processed.py`.

## Labels

Post-adjusted VWAP return (`src/data_utils.py:make_label`):

```
r_h[t] = vwap_adj[t+h+1] / vwap_adj[t+1] - 1
```

The `+1` offset aligns entry at next-day VWAP. Labels are:

1. **Quantile-trimmed**: outside `[q0.5, q99.5]` (per-horizon, train-set) → NaN.
2. **Cross-sectional z-score + clip ±3**: per-day, per-horizon.

Candidate horizons are now **h = 1..5**; the inference target is **Δ = 5**.

## Lookback Window & Suspension Handling（【改动】）

The model input is a sequence of the past **`LOOKBACK = 10`** trading days of
factors (`[N, 10, 158]`). Building it (`src/data_utils.py`):

1. `prepare_factor_series` aligns each stock's factor series to the full trading
   calendar and **forward-fills (ffill)** suspension gaps along time (causal).
2. A stock must have a complete 10-day window (no leading NaN, e.g. newly listed)
   to be scored; otherwise it is dropped on that day.

## Data Availability

This repo does **not** redistribute market data. The `data/` directory is gitignored.
Set `TUSHARE_TOKEN` and run `scripts/build_backtest_data.py` to fetch CSI300 data.

```bash
export TUSHARE_TOKEN=你的token
python scripts/build_backtest_data.py            # market data + CSI300 constituents
python scripts/build_factors.py                  # Alpha158 factors
python scripts/cache_factors_processed.py        # preprocess factors
```

`build_backtest_data.py` does not overwrite existing files by default; use `--force`.

## Build Scripts (Tushare)

- `data_fetch/connection.py`: Tushare token + `pro()`.
- `data_fetch/marketdata_db.py`: daily OHLCV, adj factor, VWAP (amount/vol), limit status, tradable, index close.
- `data_fetch/index_members_db.py`: CSI300 daily constituent masks (PIT, `index_weight` monthly snapshots).
- `data_fetch/calendar_db.py`: trading calendar (`trade_cal`).
