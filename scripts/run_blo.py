"""BLO（双层优化）自动学习标签 horizon 权重 λ——LSTM + 5 个候选 horizon。

（【改动】模型 AlphaMLP -> AlphaLSTM，回看窗口 10 天序列，h 1..10 -> 1..5，目标 h=5；
新增损失曲线记录与 R_AGG 自动重算。）

方法（沿用原 demo，仅把数据/模型/标签维度换掉）：
  - teacher：共享 logits 向量 λ∈R^Δ，softmax 归一化为 horizon 权重。
  - warmup：用均衡 diff-z 标签做几个 epoch 的标准监督，先学出表征。
  - 双层迭代（随机分散采样，每次一个样本）：
      * inner：在 inner 集上对 θ 做 M 步可微 SGD（加权 IC 损失 Σ λ_δ·IC_δ）；
      * outer：用目标 Δ 的 IC 损失 + 熵正则更新 λ；
      * write-back：把 θ* 写回 student，使 inner 更新跨迭代累积。
  - 损失曲线：每步记录 train/test 目标标签 MSE 与 test IC（固定子样本，省时）。

R_AGG：由 run_search.py 产出的聚合 alpha 的测试 IC **自动重算**（不再手填），
避免换数据/换参数后排名过期。若搜索 alpha 不存在，则用占位并告警。

输出：
  - results/blo/blo_results.csv
  - results/blo/blo_warmup{ws}.csv
  - results/blo/blo_loss_agg.csv（供 run_plot.py 画 BLO 损失曲线）
  - results/blo/log_blo.txt
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import warnings
import contextlib
import io
import time
import json
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (_HERE, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import pandas as pd
import torch
torch.backends.cudnn.enabled = False   # ← 加这一行

from src.config import (
    DEVICE, HORIZONS, RESULTS_DIR, TARGET_H, LOOKBACK,
    TRAIN_START, TRAIN_END, VALID_START, VALID_END, TEST_START, TEST_END,
)
from src.blo_trainer import BLOTrainer
from src.data_utils import (
    load_factors_tanhw, load_label_data, compute_label_quantile, make_label,
    get_universe_mask, prepare_factor_series, _window_slice, metrics_for_alpha,
)

# ---- BLO 超参数（为"弱信号下不复现悖论"做了一轮针对性调整）----
# 诊断：M=1 的 meta-gradient 偏向"和目标最像的标签"(h5)，writeback 正反馈强化，
# 加上弱信号高噪声，λ 会系统性滑向 h5。下面几项分别针对"短视 / 噪声 / 塌缩"。
N_ITERS = 240        # 180 -> 240：更多步平均 λ 噪声
LAM_START = 220      # λ 平均窗口 [LAM_START:LAM_END]（最后 20 步）
LAM_END = 240
WARMUP_EPOCHS = 3    # 2 -> 3：更好的初始表征
WARMUP_BATCHSIZE = 4096

# WARMUP_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 每个 warmup seed 跑一遍完整 BLO
WARMUP_SEEDS = [0, 1, 2]   # 3 个 warmup 做稳定性验证

N_SEED = 20          # 每个 warmup 下的 BLO 集成 seed 数
BATCH_DAYS = 16      # 10 -> 16：更多交易日，IC/meta-gradient 估计更稳
GAP_DAYS = 20        # 10 -> 20：inner/outer 拉得更开，outer 更"样本外"
LR_INNER = 5e-3      # 1e-2 -> 5e-3：配合 INNER_STEPS=2，总位移不变但分两步
LR_OUTER = 0.1
INNER_MOMENTUM = 0.9
INNER_STEPS = 2      # 1 -> 2：让 inner 真正走两步，outer 更像"训练后泛化"（治短视）
ENTROPY_WEIGHT = 5e-4  # 1e-4 -> 5e-4：更强的熵正则，防止 λ 塌缩到 h5
# LSTM 结构超参（与 run_search 保持一致）。
LSTM_HIDDEN = 64
LSTM_LAYERS = 1
LSTM_DROPOUT = 0.2
# 损失曲线用固定子样本大小（每步评估省时；完整集评估见 run_search 的无 BLO 分支）。
LOSS_EVAL_SAMPLES = 20000

BLO_DIR = RESULTS_DIR / "blo"
BLO_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = BLO_DIR / "blo_results.csv"
LOSS_CSV = BLO_DIR / "blo_loss_agg.csv"
LOG = BLO_DIR / "log_blo.txt"
_lines = []

# R_AGG 占位（若 run_search 的 alpha 存在，main() 里会自动重算并覆盖）。
R_AGG = np.arange(1, len(HORIZONS) + 1, dtype=float)


def p(msg=""):
    print(msg, flush=True)
    _lines.append(str(msg))
    with open(LOG, "w") as f:
        f.write("\n".join(_lines))


def rc(avg):
    """Spearman 相关：集成 λ 与 R_AGG 的秩相关（等价于 Spearman(IC, λ)）。"""
    return float(np.corrcoef(R_AGG, pd.Series(avg).rank().values)[0, 1])


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def compute_r_agg(vwap_adj, lo_target, hi_target):
    """【改动】由 run_search 的聚合 alpha 自动计算 h=1..5 的 IC 排名。

    返回长度 = len(HORIZONS) 的数组，值 = IC 排名（5=最好，1=最差），与 λ"越大越重要"
    的约定一致。若搜索 alpha 不存在返回 None（调用方回退占位并告警）。
    """
    ret_eval = make_label(vwap_adj, TARGET_H, lo_target, hi_target)
    ics = {}
    for h in HORIZONS:
        path = RESULTS_DIR / "alphas" / f"h{h}" / "alpha_agg.parquet"
        if not path.exists():
            return None
        alpha = pd.read_parquet(path)
        ics[h] = metrics_for_alpha(alpha, ret_eval)["IC"]
    ordered = sorted(ics.items(), key=lambda kv: kv[1])  # IC 从小到大
    rank = {h: i + 1 for i, (h, _) in enumerate(ordered)}
    p(f"  搜索 IC: " + " ".join(f"h{h}={ics[h]:.4f}" for h in HORIZONS))
    return np.array([rank[h] for h in HORIZONS], dtype=float)


def build_tensors(long_df, factor_cols, vwap_adj, dates_set, horizons, universe_mask, thr,
                  lookback=LOOKBACK, factor_series=None):
    """构建 BLO 张量 (X, Y_all, Y_warmup, date_ids)——序列版（停牌 ffill）。

    X: (N, lookback, D) 回看序列。
    Y_all: (N, Delta) 各 horizon 标签（逐日截面 z-score + clip ±3）。
    Y_warmup: (N,) diff-z 等权标签（warmup 用）。
    date_ids: (N,) 日期 id（按日聚合 IC 用）。

    factor_series: 可选，prepare_factor_series 的返回值（复用避免重复计算）。
    """
    ret_parts = []
    for h in horizons:
        r = vwap_adj.shift(-(h + 1)) / vwap_adj.shift(-1) - 1.0
        r.index.name = "date"
        lo, hi = thr[h]
        r = r.mask((r < lo) | (r > hi))  # 截尾（inf 也会被 hi 截掉）
        ret_parts.append(r.stack().rename(f"ret_{h}"))
    ret_long = pd.concat(ret_parts, axis=1)
    ret_long.index.names = ["date", "wind_code"]
    ret_long = ret_long.replace([np.inf, -np.inf], np.nan)
    ret_cols = [f"ret_{h}" for h in horizons]

    if factor_series is None:
        all_dates, stock_arrs = prepare_factor_series(long_df, factor_cols)
    else:
        all_dates, stock_arrs = factor_series
    pos = {d: i for i, d in enumerate(all_dates)}

    mask_long = universe_mask.stack().rename("in_universe")
    mask_long.index.names = ["date", "wind_code"]
    mask_dates = set(mask_long.index.get_level_values("date"))
    ret_dates = set(ret_long.index.get_level_values("date"))

    X_parts, Y_parts, Yw_parts, date_keys = [], [], [], []
    for tj in sorted(dates_set):
        if tj not in ret_dates:
            continue
        p = pos.get(tj)
        if p is None or p < lookback - 1:
            continue
        r_day = ret_long.xs(tj, level="date").dropna()  # 各 horizon 标签均有效
        if tj in mask_dates:
            in_idx = mask_long.xs(tj, level="date")  # bool Series（index=wind_code）
            in_idx = in_idx[in_idx].index
            r_day = r_day.loc[r_day.index.intersection(in_idx)]
        if len(r_day) < 20:
            continue
        rows, keep = [], []
        for c in r_day.index:
            s = _window_slice(stock_arrs, c, p, lookback)  # 停牌已 ffill
            if s is None:
                continue
            rows.append(s)
            keep.append(c)
        if len(keep) < 20:
            continue
        X_j = np.stack(rows, axis=0)  # (n, lookback, D)
        R_j = r_day.loc[keep][ret_cols].to_numpy(dtype=np.float32)  # (n, Delta)
        # 逐 horizon 截面 z-score + clip ±3。
        mu = R_j.mean(axis=0)
        sd = R_j.std(axis=0, ddof=1)
        sd = np.where(sd < 1e-12, 1.0, sd)
        Y_j = np.clip((R_j - mu) / sd, -3.0, 3.0).astype(np.float32)
        # warmup 标签：diff-z（d1=r1, d2=r2-r1, ..., d5=r5-r4）等权平均。
        diffs = np.concatenate([R_j[:, :1], R_j[:, 1:] - R_j[:, :-1]], axis=1)
        d_mu = diffs.mean(axis=0, keepdims=True)
        d_sd = diffs.std(axis=0, keepdims=True, ddof=1)
        d_sd = np.where(d_sd < 1e-12, 1.0, d_sd)
        diffs_z = np.clip((diffs - d_mu) / d_sd, -3.0, 3.0).astype(np.float32)
        yw = diffs_z.mean(axis=1)
        X_parts.append(X_j)
        Y_parts.append(Y_j)
        Yw_parts.append(yw)
        date_keys.append(tj)

    if not X_parts:
        raise RuntimeError("build_tensors 未产出任何样本；检查数据与日期范围。")

    X = np.concatenate(X_parts, axis=0)
    Y_all = np.concatenate(Y_parts, axis=0)
    Y_warmup = np.concatenate(Yw_parts, axis=0)
    date_ids_list = []
    for i, tj in enumerate(date_keys):
        date_ids_list.extend([i] * len(X_parts[i]))
    date_ids = np.array(date_ids_list, dtype=np.int64)

    X = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)
    Y_all = torch.as_tensor(Y_all, dtype=torch.float32, device=DEVICE)
    Y_warmup = torch.as_tensor(Y_warmup, dtype=torch.float32, device=DEVICE)
    date_ids = torch.as_tensor(date_ids, dtype=torch.int64, device=DEVICE)
    return X, Y_all, Y_warmup, date_ids


def main():
    global R_AGG
    p("=" * 100)
    p(f"BLO | LSTM 回看{LOOKBACK}天 | 缓存因子(69 tanh + 89 keep) | 标签截尾0.5%+z+clip3 | {N_SEED}seed")
    p(f"  随机分散采样(batch={BATCH_DAYS}, gap={GAP_DAYS}) | warmup{WARMUP_EPOCHS}ep(bs={WARMUP_BATCHSIZE}, diff-z) "
      f"{N_ITERS}iters | λ avg [{LAM_START}:{LAM_END}]")
    p(f"  M={INNER_STEPS} lr_inner={LR_INNER} lr_outer={LR_OUTER} momentum={INNER_MOMENTUM} | "
      f"LSTM hidden={LSTM_HIDDEN} layers={LSTM_LAYERS}")
    p(f"  train {TRAIN_START}~{TRAIN_END} valid {VALID_START}~{VALID_END} test {TEST_START}~{TEST_END}")
    p("=" * 100)

    t0 = time.time()
    long_df, factor_cols = load_factors_tanhw(use_cache=True)
    vwap_adj = load_label_data()
    tr_s, tr_e = pd.Timestamp(TRAIN_START), pd.Timestamp(TRAIN_END)
    vadj_tr = vwap_adj.loc[tr_s:tr_e]
    thr = {h: compute_label_quantile(vadj_tr, h) for h in HORIZONS}
    p(f"  标签分位数阈值(train q0.5/q99.5): " +
      " ".join(f"h{h}=[{thr[h][0]:.3f},{thr[h][1]:.3f}]" for h in HORIZONS))

    # 【改动】股票池：沪深300成分 & 非停牌（每日 PIT）。
    universe_mask = get_universe_mask(TRAIN_START, TEST_END)
    # 【改动】预计算因子序列（ffill），复用给 train/valid/test 三个 build_tensors。
    factor_series = prepare_factor_series(long_df, factor_cols)
    train_dates = {d for d in long_df.index.get_level_values("date")
                   if pd.Timestamp(TRAIN_START) <= d <= pd.Timestamp(TRAIN_END)}
    valid_dates = {d for d in long_df.index.get_level_values("date")
                   if pd.Timestamp(VALID_START) <= d <= pd.Timestamp(VALID_END)}
    test_dates = {d for d in long_df.index.get_level_values("date")
                  if pd.Timestamp(TEST_START) <= d <= pd.Timestamp(TEST_END)}

    Xtr, Ytr, Yw_tr, dtr = build_tensors(long_df, factor_cols, vwap_adj, train_dates, HORIZONS, universe_mask, thr, factor_series=factor_series)
    Xva, Yva, Yw_va, dva = build_tensors(long_df, factor_cols, vwap_adj, valid_dates, HORIZONS, universe_mask, thr, factor_series=factor_series)
    Xte, Yte, Yw_te, dte = build_tensors(long_df, factor_cols, vwap_adj, test_dates, HORIZONS, universe_mask, thr, factor_series=factor_series)
    p(f"  数据加载+张量构建：train={tuple(Xtr.shape)} valid={tuple(Xva.shape)} "
      f"test={tuple(Xte.shape)} ({time.time()-t0:.0f}s)")

    target_idx = HORIZONS.index(TARGET_H)  # 【改动】目标 horizon 在 Y_all 中的列下标（4）

    # 损失曲线用固定子样本（每步评估省时，固定种子保证可复现）。
    rng = np.random.RandomState(0)
    tr_idx = rng.choice(Xtr.shape[0], size=min(LOSS_EVAL_SAMPLES, Xtr.shape[0]), replace=False)
    te_idx = rng.choice(Xte.shape[0], size=min(LOSS_EVAL_SAMPLES, Xte.shape[0]), replace=False)
    train_eval_X = Xtr[tr_idx]
    train_eval_Y = Ytr[tr_idx, target_idx]
    test_X = Xte[te_idx]
    test_Y = Yte[te_idx, target_idx]
    test_date_ids = dte[te_idx]

    # 自动重算 R_AGG（由搜索 IC 排名；无搜索 alpha 则占位并告警）。
    R_AGG = compute_r_agg(vwap_adj, thr[TARGET_H][0], thr[TARGET_H][1])
    if R_AGG is None:
        R_AGG = np.arange(1, len(HORIZONS) + 1, dtype=float)
        p("  [警告] 未找到 run_search 的聚合 alpha，R_AGG 用占位 [1..5]；"
          "请先运行 scripts/run_search.py，否则 rc 无意义。")
    p(f"  R_AGG（IC 排名，5=最好）= {R_AGG.tolist()}")

    t1 = time.time()
    all_rows = []
    blo_loss_rows = []
    for ws in WARMUP_SEEDS:
        # 共享 warmup student（一次 diff-z warmup，复用给所有 BLO seed）。
        p(f"\n[warmup {WARMUP_EPOCHS}ep seed={ws}] 共享 student (diff-z)...")
        torch.manual_seed(ws)
        np.random.seed(ws)
        tw = BLOTrainer(Xtr.shape[2], len(HORIZONS), DEVICE,
                        lr_inner=LR_INNER, lr_outer=LR_OUTER, inner_steps=INNER_STEPS,
                        inner_momentum=INNER_MOMENTUM, entropy_weight=ENTROPY_WEIGHT,
                        batch_days=BATCH_DAYS, gap_days=GAP_DAYS,
                        hidden_size=LSTM_HIDDEN, num_layers=LSTM_LAYERS, dropout=LSTM_DROPOUT)
        tw.warmup(Xtr, Yw_tr, dtr, n_epochs=WARMUP_EPOCHS, batch_size=WARMUP_BATCHSIZE)
        shared_student = {k: v.detach().clone() for k, v in tw.student.state_dict().items()}
        del tw
        torch.cuda.empty_cache()
        p(f"  warmup{ws} done ({time.time()-t1:.0f}s)")

        seed_lams, seed_rcs, seed_argmaxs = [], [], []
        for bs in range(N_SEED):
            torch.manual_seed(bs)
            np.random.seed(bs)
            t = BLOTrainer(Xtr.shape[2], len(HORIZONS), DEVICE,
                           lr_inner=LR_INNER, lr_outer=LR_OUTER, inner_steps=INNER_STEPS,
                           inner_momentum=INNER_MOMENTUM, entropy_weight=ENTROPY_WEIGHT,
                           batch_days=BATCH_DAYS, gap_days=GAP_DAYS,
                           hidden_size=LSTM_HIDDEN, num_layers=LSTM_LAYERS, dropout=LSTM_DROPOUT)
            t.student.load_state_dict(shared_student)
            with quiet():
                history = t.train(Xtr, Ytr, dtr, target_idx, N_ITERS,
                                  val_X=Xva, val_Y_target=Yva[:, target_idx], val_date_ids=dva,
                                  train_eval_X=train_eval_X, train_eval_Y=train_eval_Y,
                                  test_X=test_X, test_Y=test_Y, test_date_ids=test_date_ids)
            # 平均 λ（后 burn-in 窗口）。
            lam = np.array([h["lambda"] for h in history])[LAM_START:LAM_END].mean(axis=0)
            seed_lams.append(lam)
            seed_rcs.append(rc(lam))
            seed_argmaxs.append(int(np.argmax(lam)) + 1)
            # 【改动】记录每步损失（去掉 λ 数组，只留标量，便于写 CSV）。
            for hrec in history:
                blo_loss_rows.append({
                    "warmup_seed": ws, "blo_seed": bs,
                    "iter": hrec["iter"], "outer_loss": hrec["outer_loss"],
                    "val_ic": hrec["val_ic"], "train_loss": hrec["train_loss"],
                    "valid_loss": hrec.get("valid_loss", float("nan")),  # 【改动】新增验证集损失
                    "test_loss": hrec["test_loss"], "test_ic": hrec["test_ic"],
                })
            del t
            if bs % 5 == 0 or bs == N_SEED - 1:
                p(f"    [warmup{ws}] seed{bs}/{N_SEED} rc={seed_rcs[-1]:.3f} argmax=h{seed_argmaxs[-1]} "
                  f"(累计 {time.time()-t1:.0f}s)")

        seed_lams = np.array(seed_lams)
        ens = seed_lams.mean(axis=0)
        p(f"  [warmup{ws}] ens_rc={rc(ens):.3f} rc_mean={np.mean(seed_rcs):.3f} "
          f"argmax=h{int(np.argmax(ens))+1} 正seed={sum(1 for r in seed_rcs if r > 0)}/{N_SEED} "
          f"({time.time()-t1:.0f}s)")

        row = {
            "warmup_seed": ws, "warmup_epochs": WARMUP_EPOCHS,
            "batch_days": BATCH_DAYS, "gap_days": GAP_DAYS, "n_iters": N_ITERS,
            "ens_rc": rc(ens), "rc_mean": float(np.mean(seed_rcs)),
            "rc_std": float(np.std(seed_rcs)), "ens_argmax": int(np.argmax(ens)) + 1,
            "ens_spread": float(ens.max() - ens.min()),
            "n_pos": int(sum(1 for r in seed_rcs if r > 0)),
            "seed_rcs": json.dumps([round(r, 3) for r in seed_rcs]),
            "seed_argmaxs": json.dumps(seed_argmaxs),
            "ens_lambda": json.dumps([round(float(x), 4) for x in ens]),
        }
        all_rows.append(row)
        pd.DataFrame([row]).to_csv(BLO_DIR / f"blo_warmup{ws}.csv", index=False)

    p(f"\n=== Results ({len(WARMUP_SEEDS)} warmup seed(s)) ===")
    p(f"  per-warmup ens_rc: {[round(r['ens_rc'], 3) for r in all_rows]}")
    p(f"  per-warmup argmax: {[r['ens_argmax'] for r in all_rows]}")
    if len(all_rows) > 1:
        grand_lams = np.array([json.loads(r["ens_lambda"]) for r in all_rows]).mean(axis=0)
        p(f"  grand ens_rc (all warmup) = {rc(grand_lams):.3f}")
        p(f"  grand argmax = h{int(np.argmax(grand_lams))+1}")
        p(f"  grand lambda = {[round(float(x), 4) for x in grand_lams]}")
    p(f"  elapsed = {time.time()-t1:.0f}s")

    pd.DataFrame(all_rows).to_csv(RESULTS_CSV, index=False)
    pd.DataFrame(blo_loss_rows).to_csv(LOSS_CSV, index=False)
    p(f"  results saved: {RESULTS_CSV}")
    p(f"  loss saved: {LOSS_CSV}")
    p(f"  per-warmup results in: {BLO_DIR}/blo_warmup*.csv")
    p(f"\n*** All done ***")


if __name__ == "__main__":
    main()
