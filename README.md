<h1 align="center">The Label Horizon Paradox — Open-Source Demo</h1>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="requirements.txt"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://openreview.net/forum?id=G43CIfmmxh"><img src="https://img.shields.io/badge/ICML-2026-red.svg" alt="ICML 2026"></a>
</p>

An open-source demo of the bi-level optimization (BLO) method from **"The
Label Horizon Paradox: Rethinking Supervision Targets in Financial Forecasting"**
([ICML 2026](https://openreview.net/forum?id=G43CIfmmxh)).

The paper uncovers a counter-intuitive phenomenon: **the optimal training label
horizon often deviates from the inference target**. Training on the exact target
horizon is not necessarily best; an intermediate horizon can provide a better
signal-to-noise trade-off. This repo provides a self-contained demo that
operationalizes that insight with a BLO framework that **auto-learns the
label-horizon weights** within a single training run.

> This is a standalone demo, not the paper's experimental codebase. It builds the
> core method (BLO + brute-force baseline + ladder backtest) on public daily-frequency
> data and the open-source Alpha158 factor set. Some details differ from the paper's
> description (see [docs/blo.md](docs/blo.md#differences-from-the-paper)).

> **本仓库当前版本已按用户需求改造**：数据源换成 **Tushare**、股票池仅 **沪深300**、
> 回看窗口 **1 天 → 10 天**（LSTM 序列输入）、候选标签 **h=1..10 → 1..5**、推理目标
> **固定 h=5**、停牌缺失用 **ffill**、模型用 **LSTM**、子账户 **10→5**（持有期 10天→5天）、
> 回测 **TopN=30**，并新增报告用图片/表格输出。详见 [docs/改动说明.md](docs/改动说明.md)。

---

## Task

We forecast the cross-sectional **5-day VWAP return** of CSI-300 constituents — the
return from the next-day VWAP (`t+1`) to the VWAP 5 days later (`t+6`), i.e. the
target horizon `Δ = 5`. This matches a 5-day rebalancing cadence.

Instead of training on this target directly, we treat the label horizon as a
learnable choice: the model can be supervised by the `h`-day VWAP return for any
`h = 1..5`. The question is which `h` gives the best out-of-sample performance
on the fixed target `Δ = 5`. BLO learns the horizon weights `λ` automatically;
the brute-force search trains one model per `h` to establish the IC ranking for
comparison.

The model input is a **lookback sequence of 10 trading days** (`L = 10`) of
Alpha158 factors (`[B, 10, 158]`), encoded by an **LSTM**.

---

## TL;DR

- **Universe**: CSI 300 constituents (point-in-time); **features**: Alpha158
  (158 factors from Microsoft's [Qlib](https://github.com/microsoft/qlib)).
- **Labels**: adjusted VWAP returns for `h = 1..5` (5 candidate horizons).
- **Inference target**: `Δ = 5` (5-day VWAP return, aligned with a 5-day
  rebalancing cadence).
- **Model**: LSTM (`[B, 10, 158] → hidden(64) → 1`).
- **Method**: bi-level optimization learns a softmax weight vector `λ ∈ R^5`
  over the 5 candidate horizons; the student is trained on the λ-weighted
  composite label and evaluated on the fixed target `Δ = 5`. Compared against
  the brute-force search (one model per horizon, i.e. "no BLO").
- **Data**: A-share daily data via **Tushare** (2000+ points), CSI 300 only.

---

## Pipeline

```
Alpha158 factors ────┐
                     ├──> build_tensors ──> BLO trainer ──> learned λ
VWAP labels (h=1..5) ┘                       (warmup + bi-level)
                                                     │
                     brute-force search ──> IC ranking ──> rc(λ, IC)
                                                     │
                                                     ▼
                     saved alphas ──> LadderBacktest ──> Sharpe / Ann / MDD
```

Three entry scripts, each decoupled:

| Script | Purpose | Output |
|---|---|---|
| `scripts/run_search.py` | Brute-force ("no BLO"): train one LSTM per `h`, produce the IC ranking | `results/alphas/h{h}/alpha_*.parquet` |
| `scripts/run_blo.py` | BLO: auto-learn horizon weights `λ` | `results/blo/blo_results.csv` |
| `scripts/run_backtest.py` | Ladder backtest on saved alphas (top-30 within CSI 300) | `results/backtest/*.csv` |
| `scripts/run_plot.py` | Generate report figures + tables | `results/figures/*.png`, `summary.md` |

---

## Installation

```bash
git clone https://github.com/EFundAI/label-horizon-paradox.git
cd label-horizon-paradox
pip install -r requirements.txt   # torch, pandas, numpy, pyarrow, tushare, matplotlib
```

**Data**: this repo does **not** redistribute market data. The build scripts
(`scripts/build_backtest_data.py`, `scripts/build_factors.py`) pull from
**Tushare**; set your token first:

```bash
export TUSHARE_TOKEN=你的token   # Linux/macOS
# 或 Windows: set TUSHARE_TOKEN=你的token
```

The build scripts produce the parquet files under `data/a_share_data/` (see
[docs/data.md](docs/data.md#data-layout)); training/backtest only read parquet.

---

## Quick Start

```bash
# 0. Token
export TUSHARE_TOKEN=你的token

# 1. Build market data + Alpha158 factors (one-time; CSI 300 only)
python scripts/build_backtest_data.py
python scripts/build_factors.py
python scripts/cache_factors_processed.py

# 2. Brute-force horizon search ("no BLO"; baseline IC ranking)
python scripts/run_search.py

# 3. Bi-level optimization (auto-learn λ)
python scripts/run_blo.py

# 4. Backtest saved alphas (top-30 within CSI 300)
python scripts/run_backtest.py --mode agg --topn 30

# 5. Report figures/tables (incl. the 4-curve loss figure)
python scripts/run_plot.py
```

---

## Key Results

> The original demo's result tables (all-A-shares, AlphaMLP, h=1..10) are **no
> longer applicable** after the change to CSI 300 + LSTM + lookback-10 + h=1..5.
> After running the pipeline, the numbers appear in:
>
> - **IC ranking**: `results/losses/search_ic_ranking.csv` (and `results/figures/ic_ranking.png`)
> - **BLO λ / rc**: `results/blo/blo_results.csv` (and `results/figures/lambda_distribution.png`)
> - **Backtest**: `results/backtest/backtest_agg_topn_summary.csv` (and `results/figures/backtest_nav.png`)
> - **Loss curves (BLO vs no-BLO, train+test)**: `results/figures/loss_curve_blo_vs_noblo.png`
> - **Everything summarized**: `results/figures/summary.md`

The key result to look for is the **Label Horizon Paradox**: in the brute-force
search, IC on the fixed target `Δ = 5` should peak at some `h ≠ 5` (i.e. the
target horizon is *not* the best training horizon), and BLO's learned `λ` should
recover that ranking (positive Spearman `rc`).

---

## Documentation

- [docs/改动说明.md](docs/改动说明.md) — 本次改动的需求、方案、坑与运行指引（中文）。
- [docs/data.md](docs/data.md) — Data pipeline: Tushare, Alpha158 factors, VWAP labels, preprocessing.
- [docs/search.md](docs/search.md) — Brute-force horizon search and the IC ranking baseline.
- [docs/blo.md](docs/blo.md) — Bi-level optimization: method, writeback, and differences from the paper.
- [docs/backtest.md](docs/backtest.md) — Ladder backtest engine, optimizers, costs, benchmarks.
- [docs/architecture.md](docs/architecture.md) — Code structure and module responsibilities.

---

## Repository Layout

```
label-horizon-paradox/
├── src/                  # core: config, model (AlphaLSTM), data_utils, blo_trainer, plotting
├── scripts/              # entry points: build_*.py, run_search.py, run_blo.py, run_backtest.py, run_plot.py
├── engine/               # backtest engine (ladder, optimizer, account, costs, execution)
│   └── base/             #   account, constraints, costs, timeline, execution
├── factors_calculate/    # Alpha158 factor names + computation (Tushare OHLCV)
├── data_fetch/           # Tushare market data + index constituents
├── performance/          # return/risk + IC/RankIC metrics
├── data/                 # parquet market data (CSI 300)
│   ├── a_share_data/          # raw (prices, masks, factors)
│   └── a_share_data_processed/# preprocessed factors (69 standardized + 89 kept)
├── results/              # alphas/, blo/, backtest/, models/, losses/, figures/
└── docs/                 # documentation
```

---

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{
song2026the,
  title={The Label Horizon Paradox: Rethinking Supervision Targets in Financial Forecasting},
  author={Chen-Hui Song and Shuoling Liu and Liyuan Chen},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026},
  url={https://openreview.net/forum?id=G43CIfmmxh}
}
```

---

## Disclaimer

This project is for **academic research purposes only** and does **not
constitute investment advice**. The backtest is a simplified signal-validation
framework, not a production simulator; real trading involves risks, costs, and
constraints not modeled here (see [docs/backtest.md](docs/backtest.md#simplifications--limitations)
for details). The authors are not liable for any financial decisions made
based on this code or its results.

---

## License

This project is released under the [MIT License](LICENSE).
