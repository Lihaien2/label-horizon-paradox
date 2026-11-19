<h1 align="center">Bi-Level Optimization (BLO)</h1>

This document describes the BLO method as implemented in this demo（当前为
**沪深300 + LSTM + 回看10天 + h=1..5 + 目标h=5** 版本）. The implementation follows
the paper's core idea but differs in several details (see [Differences from the paper](#differences-from-the-paper)).

## The Label Horizon Paradox

Conventional financial forecasting trains the model on the same horizon as the
inference target `Δ`. The paper shows this is not always optimal: as the label
horizon extends, **signal realization** competes with **noise accumulation**.
The optimal supervision horizon `δ*` sits where marginal signal gain equals
marginal noise penalty — which can be *shorter* than `Δ`.

This demo instantiates the idea in a daily-frequency setting:

- **Candidate horizons**: `h = 1..5` (5-day VWAP returns).
- **Inference target**: `Δ = 5` (5-day return, aligned with a 5-day rebalancing cadence).
- **Input**: 10-day lookback sequence of Alpha158 factors (`[N, 10, 158]`), ffill.
- **Model**: LSTM.
- **Brute-force evidence**: training one model per `h` and evaluating on `h=5`
  shows whether IC peaks at `h ≠ 5` (the paradox).

## Method

BLO learns a softmax weight vector `λ ∈ R^5` over the 5 candidate horizons. The
student model is trained on the `λ`-weighted composite label; `λ` is optimized so
that the resulting student generalizes best on the fixed target `Δ = 5`.

### Teacher

`HorizonTeacher` (`src/blo_trainer.py`) holds a logits vector; `forward()` returns
`softmax(logits)` — the 5 horizon weights summing to 1.

### Warmup

A few epochs of standard supervision with a **balanced diff-z label**:

1. Per-day returns `r_1..r_5` (trim-tailed).
2. Differences `d_1=r_1, d_2=r_2-r_1, ..., d_5=r_5-r_4` — each marginal return
   gets equal weight (a naive mean label would overweight `r_1`).
3. Cross-sectional z-score + clip ±3, then equal-weighted mean across horizons.
4. SGD + 4096-batch + momentum 0.95, fixed epochs.

### Bi-Level Loop

Each iteration (one sample per update):

1. **Sample**: random center `t`; inner set from days before `t`, outer set from
   days after `t` (each `batch_days=10`, `gap_days=10`). Inner entirely before
   outer — time-forward, no leakage.
2. **Inner loop** (`M=1` differentiable SGD step): from current student params `θ`,
   compute weighted IC loss `L_inner = 1 - Σ_δ λ_δ · IC_δ` on the inner set, one
   gradient step to `θ*` (`functional_call` + `autograd.grad(create_graph=True)`).
3. **Outer loop**: evaluate `θ*` on the outer set with `1 - IC` against target
   `Δ = 5`, plus entropy regularizer `−γH(λ)`; backprop to update `λ` (Adam).
4. **Writeback**: write `θ*` back into the student so inner updates accumulate.

`λ` is averaged over a post-burnin window (`LAM_START:LAM_END`) per seed, then
across seeds for the ensemble.

### Evaluation

The learned `λ` is compared to the brute-force search IC ranking via Spearman
rank correlation `rc = spearmanr(R_AGG, λ)`. `rc ≈ 1` means BLO recovered the
ranking without running the full sweep.

> **R_AGG is now auto-computed** (`run_blo.py:compute_r_agg`) from the search's
> `alpha_agg.parquet` test IC, instead of being hard-coded — so it stays correct
> after any data/param change. If the search has not been run, a placeholder is
> used with a warning.

## Configuration

`scripts/run_blo.py` exposes all tunable hyperparameters at the top:

```python
N_ITERS = 180
LAM_START = 160
LAM_END = 180
WARMUP_EPOCHS = 2
WARMUP_BATCHSIZE = 4096
WARMUP_SEEDS = [0,1,2,3,4,5,6,7,8,9]
N_SEED = 20
BATCH_DAYS = 10
GAP_DAYS = 10
LR_INNER = 0.05
LR_OUTER = 0.1
INNER_MOMENTUM = 0.9
INNER_STEPS = 1
ENTROPY_WEIGHT = 1e-4
LSTM_HIDDEN = 64
LSTM_LAYERS = 1
LSTM_DROPOUT = 0.2
```

## Results

> The original demo's λ/rc tables (all-A, AlphaMLP, h=1..10) no longer apply.
> Re-run to obtain the CSI300 + LSTM numbers; they are written to
> `results/blo/blo_results.csv` and visualized in `results/figures/lambda_distribution.png`.
> Loss curves are written to `results/blo/blo_loss_agg.csv` (for the 4-curve loss figure).

## Differences from the Paper

| Aspect | Paper | This demo |
|---|---|---|
| **Setting** | Intraday / interday, LSTM on CSI500 | Daily-frequency, LSTM on **CSI300** |
| **Features** | Minute-level features | Public Alpha158 (158 daily factors) |
| **Lookback** | Minute-level patches | **10 trading days** |
| **Warmup label** | Mean of standardized candidate labels | Diff-z (marginal-return) label, equal-weighted |
| **Sampling** | Intra-batch split (B_in / B_out) | Random-dispersed: inner before outer with a gap |
| **Ensemble** | A single trajectory per update | Multiple trajectories sampled and averaged |
| **Horizons** | Minute-level `δ ∈ {1..Δ}` | Daily `h ∈ {1..5}`, target `Δ=5` |
