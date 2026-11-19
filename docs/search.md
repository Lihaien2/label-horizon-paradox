<h1 align="center">Brute-Force Horizon Search（无 BLO）</h1>

This document describes the brute-force baseline: train one LSTM per horizon
`h ∈ {1..5}`, evaluate on the target `h=5`, and produce the IC ranking that BLO
is compared against（当前为 **沪深300 + LSTM + 回看10天** 版本）。

## Purpose

The Label Horizon Paradox asks: is the inference target `Δ = 5` the best training
horizon? The brute-force search answers this empirically by training a separate
LSTM for each candidate horizon `h ∈ {1..5}` and measuring out-of-sample IC on
the fixed target `h=5`.

The resulting IC ranking serves two purposes:

1. **Evidence of the paradox**: if IC peaks at `h ≠ 5`, the target is not the
   best training horizon.
2. **Evaluation target for BLO**: BLO's learned `λ` is compared to the IC
   ranking via Spearman rank correlation `rc` (see [blo.md](blo.md#evaluation)).

## Method

`scripts/run_search.py`:

For each horizon `h`:

1. Build the `h`-day VWAP return label (trim + z-score + clip ±3).
2. Train `N_SEED=15` independent **LSTM** models (SGD + momentum, 4096-sample
   batches), supervised by the `h`-day label:
   - 15 epochs, early stopping with `skip=1`, `patience=3`; best-epoch selected
     by validation IC on the `h=5` valid label.
3. For each seed, predict test-period alpha (point-in-time, sequence input) and save.
4. Aggregate: average the 15 per-seed alphas → `alpha_agg.parquet`.

The input is a **10-day lookback sequence** (`[N, 10, 158]`) of Alpha158 factors,
with suspension gaps **forward-filled (ffill)**; stocks lacking a full window
(newly listed) are dropped on that day.

## Configuration

```python
N_SEED = 15
SKIP = 1
PATIENCE = 3
EPOCHS = 15
BATCH_SIZE = 4096
MOMENTUM = 0.95
LR = 0.05
WEIGHT_DECAY = 1e-4
# LSTM 结构
LSTM_HIDDEN = 64
LSTM_LAYERS = 1
LSTM_DROPOUT = 0.2
```

`HORIZONS = [1,2,3,4,5]` and `TARGET_H = 5` come from `src.config`.

## Output

`results/alphas/h{h}/`:

- `alpha_seed{si}.parquet` — per-seed test alpha (wide: date × wind_code).
- `alpha_agg.parquet` — averaged alpha (the one used for backtest).

`results/models/h{h}/model_seed{si}.pt` — saved model states.

`results/losses/`:

- `search_h{h}_seed{si}_loss.csv` — per-seed per-epoch train/valid/test loss & IC.
- `search_loss_agg.csv` — aggregated loss (for `run_plot.py`).
- `search_ic_ranking.csv` — per-h IC/ICIR/RankIC/RankICIR (for IC figure + BLO R_AGG).

## Results

> The original demo's IC table (all-A-shares, AlphaMLP, h=1..10) no longer applies.
> Re-run to obtain the CSI300 + LSTM + lookback-10 numbers; they are written to
> `results/losses/search_ic_ranking.csv` and visualized in `results/figures/ic_ranking.png`.

The paradox holds if IC on the fixed target `h=5` peaks at some `h ≠ 5`.

## Point-in-Time Prediction

`predict_alpha_pit` (`src/data_utils.py`) generates test alphas using only
information available up to each day. Features are preprocessed with train-period
thresholds; the universe is the point-in-time CSI300 & tradable mask
(`get_universe_mask`). Each stock must have a complete 10-day window (ffill'd);
otherwise it is dropped (alpha = NaN) — aligned with training (`build_xy_seq`).

## Metrics

`metrics_for_alpha` (`src/data_utils.py`) computes daily IC, RankIC, ICIR,
RankICIR between an alpha and a label, over the test period.
