<h1 align="center">Architecture</h1>

This document describes the code structure and module responsibilities
（当前为 **Tushare + 沪深300 + LSTM + 回看10天 + h=1..5** 版本）。

## Module Map

```
src/                core library
├── config.py       fixed constants (dates, paths, device, horizons=1..5, lookback=10, universe)
├── model.py        AlphaLSTM (+ AlphaMLP 保留) + differentiable IC/RankIC losses
├── data_utils.py   factor sequence build (ffill), label construction, universe, PIT prediction, metrics
├── blo_trainer.py  HorizonTeacher + BLOTrainer (warmup + bi-level loop + writeback)
└── plotting.py     report figures + summary tables (matplotlib, Agg)

scripts/            entry points (tunable hyperparameters live here)
├── build_backtest_data.py   download CSI300 market data + constituents (Tushare)
├── build_factors.py         compute Alpha158 factors (Tushare OHLCV)
├── cache_factors_processed.py  preprocess factors (69 standardized + 89 kept) → cache
├── run_search.py            brute-force horizon search ("no BLO") → IC ranking + loss curves
├── run_blo.py               bi-level optimization → learned λ + loss curves
├── run_backtest.py          ladder backtest on saved alphas (top-30 within CSI300)
└── run_plot.py              generate figures + summary tables

engine/             backtest engine
├── optim_backtest.py   period-by-period backtest (stateful account)
├── ladder_backtest.py  staggered sub-portfolio (5 sub-accounts, summed NAV)
├── optimizer.py        OptimizerProtocol + TopN / ThreeIndexTopN / Normalizer
└── base/
    ├── account.py      stateful holdings + NAV
    ├── constraints.py  limit-up/down, weight cap, residual
    ├── costs.py        stamp duty + commission
    ├── timeline.py     DailyClose timepoint
    └── execution.py    VWAP / close execution

factors_calculate/  Alpha158
├── alpha158.py         158 factor names (Qlib-consistent)
└── alpha158_db.py      compute factors from Tushare post-adjusted OHLCV

data_fetch/         Tushare access
├── connection.py       Tushare token + pro_api
├── marketdata_db.py    EOD OHLCV, adj factor, VWAP, limits, tradable, index close
├── index_members_db.py CSI300 constituent masks (PIT, via index_weight)
├── calendar_db.py      trading calendar (via trade_cal)
└── cache.py            parquet cache

performance/        metrics
└── metrics.py          return/risk (sharpe, ann, mdd, calmar) + factor (IC, ICIR, RankIC)

data/               parquet market data (not in git)
├── a_share_data/          raw (prices, masks, factors) — CSI300 only
└── a_share_data_processed/ preprocessed factors

results/            outputs (not in git)
├── alphas/         per-horizon test alphas (search output)
├── blo/            BLO results (λ, rc) + loss curve data
├── backtest/       backtest NAV + summaries
├── models/         saved model states
├── losses/         per-epoch / per-iter loss history (CSV)
└── figures/        PNG figures + summary.md

docs/               documentation
```

## Design Principles

### Decoupled stages

The stages communicate only through saved parquet/CSV files:

- `run_search.py` → `results/alphas/h{h}/alpha_*.parquet` + `results/losses/*.csv`
- `run_blo.py` → `results/blo/blo_results.csv` + `results/blo/blo_loss_agg.csv`
- `run_backtest.py` reads `results/alphas/` → `results/backtest/`
- `run_plot.py` reads all of the above → `results/figures/`

No stage imports another's code.

### Config separation

- **`src/config.py`**: fixed project constants (dates, paths, device, horizons,
  lookback, universe). Not tunable.
- **`scripts/run_*.py`**: tunable hyperparameters (lr, epochs, seeds, batch
  sizes, LSTM hidden size) at the top of each entry script.

### Stateless optimizers, stateful account

Optimizers (`engine/optimizer.py`) are pure functions — all state (holdings,
NAV) lives in `Account` (`engine/base/account.py`).

### Differentiable IC

`src/model.py` implements daily cross-sectional IC as a differentiable PyTorch
operation. BLO's inner loop uses it directly as a loss, with second-order
autodiff through `torch.func.functional_call` + `autograd.grad(create_graph=True)`.

### Point-in-time correctness

- Factor preprocessing thresholds: train-period only.
- Label quantile thresholds: train-period only.
- Index constituent masks: daily PIT (via Tushare `index_weight` monthly snapshots).
- Tradable universe: daily PIT (suspend_d + daily row existence).
- Lookback sequence features: **forward-filled (ffill)** along time; stocks still
  lacking a full 10-day window (newly listed) are dropped.

No look-ahead anywhere in the pipeline.

## Data Flow

```
Tushare ──build_backtest_data.py──> data/a_share_data/ (raw parquet, CSI300)
   │
   └──build_factors.py──> data/a_share_data/factors.parquet
                                    │
                                    cache_factors_processed.py
                                    │
                                    ▼
                 data/a_share_data_processed/factors.parquet (69 standardized + 89 kept)
                                    │
                 ┌──────────────────┼──────────────────┐
                 ▼                  ▼                  ▼
           run_search.py       run_blo.py          run_backtest.py
                 │                  │                  │
                 ▼                  ▼                  ▼
           results/alphas/    results/blo/       results/backtest/
           results/losses/    results/blo/       (NAV + metrics)
                 │                  │
                 └────────┬─────────┘
                          ▼
                    run_plot.py → results/figures/
```
