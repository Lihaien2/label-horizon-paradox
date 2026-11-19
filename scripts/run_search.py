"""暴力搜索（无 BLO）：对每个 horizon h（1..5）训练一个 LSTM。

（【改动】模型 AlphaMLP -> AlphaLSTM，回看窗口 1天 -> 10天，h 范围 1..10 -> 1..5，
推理目标 h=5；新增每 epoch 的损失曲线记录。）

对每个 horizon h，训练 N_SEED 个独立 LSTM，监督信号为该 h 天的 VWAP 收益；
从 epoch `skip` 起按验证集 IC 选最优 epoch（早停），保存逐 seed 与聚合 alpha。
同时记录每个 epoch 的 train/valid/test 损失与 IC，供 run_plot.py 绘制
"有 BLO vs 无 BLO"损失曲线（无 BLO 分支取 h=5 的模型）。

配置：
  - 因子：缓存（69 tanh + 89 keep）。
  - 标签：后复权 VWAP 收益，分位数截尾 + 截面 z-score + clip ±3。
  - 测试评估：h5 标签（分位数截尾）。
  - 验证集：纯 IC 选最优 epoch，patience=3。

输出：
  - results/alphas/h{h}/alpha_seed{si}.parquet + alpha_agg.parquet
  - results/models/h{h}/model_seed{si}.pt
  - results/losses/search_h{h}_seed{si}_loss.csv（每 seed 损失）
  - results/losses/search_loss_agg.csv（跨 seed 聚合损失，供画图）
  - log_search.txt
"""
from __future__ import annotations

import sys
import warnings
import time
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (_HERE, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from src.config import (
    DEVICE, HORIZONS, RESULTS_DIR, TARGET_H, LOOKBACK, LOSSES_DIR,
    TRAIN_START, TRAIN_END, VALID_START, VALID_END, TEST_START, TEST_END,
)
from src.model import AlphaLSTM, daily_ic  # 【改动】AlphaMLP -> AlphaLSTM
from src.data_utils import (
    load_factors_tanhw, load_label_data, make_label, compute_label_quantile,
    build_xy_seq, get_universe_mask, predict_alpha_pit, prepare_factor_series,
    metrics_for_alpha, evaluate_mse_loss,
)

# ---- 可调超参数 ----
N_SEED = 15
SKIP = 1
PATIENCE = 3
WEIGHT_DECAY = 1e-4
EPOCHS = 15
BATCH_SIZE = 4096
MOMENTUM = 0.95
LR = 0.05
# LSTM 结构超参（【改动】新增）。
LSTM_HIDDEN = 64
LSTM_LAYERS = 1
LSTM_DROPOUT = 0.2
# TARGET_H（评估目标 horizon = 5）与 LOOKBACK（回看 10 天）从 src.config 导入。

LOSS_DIR = LOSSES_DIR
LOSS_DIR.mkdir(parents=True, exist_ok=True)


def train_horizon(Xt, yt, dtr, Xv, yv, dv, Xte, yte, dte, in_dim, seed,
                  lr, momentum, weight_decay, epochs, patience, batch_size, skip,
                  hidden, layers, dropout):
    """训练一个 LSTM；从 epoch `skip` 起按验证 IC 早停；记录每 epoch 损失/IC。

    返回 (best_state, best_ep, loss_hist)。
    loss_hist 为 list[dict]，含 epoch/train_loss/valid_loss/valid_ic/test_loss/test_ic。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = AlphaLSTM(in_dim, hidden_size=hidden, num_layers=layers, dropout=dropout).to(DEVICE)
    opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    n = len(Xt)
    best = -np.inf
    best_state = None
    best_ep = None
    bad = 0
    loss_hist = []

    for epoch in range(epochs):
        # ---- 训练一个 epoch，并累计该 epoch 的 MSE（作为 train loss）----
        model.train()
        train_sum, nb = 0.0, 0
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 10:
                continue
            xb, yb = Xt[idx], yt[idx]
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_sum += float(loss.item()) * len(idx)
            nb += len(idx)
        train_loss = train_sum / nb if nb else float("nan")

        # ---- 评估：验证 IC（早停依据）+ train/valid/test 损失与 IC（供画图）----
        model.eval()
        with torch.no_grad():
            v_ic = float(daily_ic(model(Xv), yv, dv).item())
            valid_loss = evaluate_mse_loss(model, Xv, yv)
            test_loss = evaluate_mse_loss(model, Xte, yte)
            test_ic = float(daily_ic(model(Xte), yte, dte).item())
        loss_hist.append({
            "epoch": epoch + 1, "train_loss": train_loss, "valid_loss": valid_loss,
            "valid_ic": v_ic, "test_loss": test_loss, "test_ic": test_ic,
        })

        if epoch < skip:
            continue
        if v_ic > best:
            best = v_ic
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_ep = epoch + 1
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    return best_state, best_ep, loss_hist


def main():
    print(f"=== h1..5 保存 alpha | 验证集纯 IC 选最优 | 缓存因子 | LSTM 回看{LOOKBACK}天 ===")
    print(f"  测试用 h{TARGET_H} 截尾标签 | lr={LR} mom={MOMENTUM} "
          f"epochs={EPOCHS} patience={PATIENCE} | LSTM hidden={LSTM_HIDDEN}")
    vwap_adj = load_label_data()
    tr_s, tr_e = pd.Timestamp(TRAIN_START), pd.Timestamp(TRAIN_END)
    vadj_tr = vwap_adj.loc[tr_s:tr_e]

    # 目标 horizon（h=5）的分位数阈值与评估标签。
    lo_tgt, hi_tgt = compute_label_quantile(vadj_tr, TARGET_H)
    ret_eval = make_label(vwap_adj, TARGET_H, lo_tgt, hi_tgt)

    # 【改动】股票池：沪深300成分 & 非停牌（每日 PIT），训练/验证/测试共用同一口径。
    universe_mask = get_universe_mask(TRAIN_START, TEST_END)

    fl, factor_cols = load_factors_tanhw(use_cache=True)
    # 【改动】预计算按股票对齐 + ffill 的因子序列（多处复用，避免重复计算）。
    factor_series = prepare_factor_series(fl, factor_cols)
    train_dates = {d for d in fl.index.get_level_values("date")
                   if pd.Timestamp(TRAIN_START) <= d <= pd.Timestamp(TRAIN_END)}
    valid_dates = {d for d in fl.index.get_level_values("date")
                   if pd.Timestamp(VALID_START) <= d <= pd.Timestamp(VALID_END)}
    test_dates = {d for d in fl.index.get_level_values("date")
                  if pd.Timestamp(TEST_START) <= d <= pd.Timestamp(TEST_END)}

    # 验证集/测试集张量（目标 h5 标签，用于早停与损失/IC 记录）。
    ret_target = make_label(vwap_adj, TARGET_H, lo_tgt, hi_tgt)
    Xva_np, yva_np, dva_np = build_xy_seq(fl, ret_target, valid_dates, factor_cols, universe_mask, clip=3.0, factor_series=factor_series)
    Xte_np, yte_np, dte_np = build_xy_seq(fl, ret_target, test_dates, factor_cols, universe_mask, clip=3.0, factor_series=factor_series)
    Xv = torch.as_tensor(Xva_np, dtype=torch.float32, device=DEVICE)
    yv = torch.as_tensor(yva_np, dtype=torch.float32, device=DEVICE)
    dv = torch.as_tensor(dva_np, dtype=torch.int64, device=DEVICE)
    Xte = torch.as_tensor(Xte_np, dtype=torch.float32, device=DEVICE)
    yte = torch.as_tensor(yte_np, dtype=torch.float32, device=DEVICE)
    dte = torch.as_tensor(dte_np, dtype=torch.int64, device=DEVICE)

    t_all = time.time()
    agg_loss_rows = []
    ic_rows = []  # 【改动】记录每个 h 的聚合 IC（供画 IC 排名图 + BLO 自动 R_AGG）
    for h in HORIZONS:
        lo, hi = compute_label_quantile(vadj_tr, h)
        ret_h = make_label(vwap_adj, h, lo, hi)
        Xtr_np, ytr_np, dtr_np = build_xy_seq(fl, ret_h, train_dates, factor_cols, universe_mask, clip=3.0, factor_series=factor_series)
        Xt = torch.as_tensor(Xtr_np, dtype=torch.float32, device=DEVICE)
        yt = torch.as_tensor(ytr_np, dtype=torch.float32, device=DEVICE)
        dtr = torch.as_tensor(dtr_np, dtype=torch.int64, device=DEVICE)
        in_dim = Xtr_np.shape[2]  # 【改动】特征维 = 序列第三维（158）
        print(f"\n--- h{h} train={tuple(Xt.shape)} (N,lookback,D) ---", flush=True)

        seed_alphas = []
        model_dir = RESULTS_DIR / "models" / f"h{h}"
        alpha_dir = RESULTS_DIR / "alphas" / f"h{h}"
        model_dir.mkdir(parents=True, exist_ok=True)
        alpha_dir.mkdir(parents=True, exist_ok=True)

        for si in range(N_SEED):
            best_state, best_ep, loss_hist = train_horizon(
                Xt, yt, dtr, Xv, yv, dv, Xte, yte, dte, in_dim, si,
                LR, MOMENTUM, WEIGHT_DECAY, EPOCHS, PATIENCE, BATCH_SIZE, SKIP,
                LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT)
            # 保存每 seed 的损失历史（供画图/分析）。
            pd.DataFrame(loss_hist).to_csv(
                LOSS_DIR / f"search_h{h}_seed{si}_loss.csv", index=False)
            for row in loss_hist:
                agg_loss_rows.append({"h": h, "seed": si, **row})

            torch.save(best_state, model_dir / f"model_seed{si}.pt")
            model = AlphaLSTM(in_dim, hidden_size=LSTM_HIDDEN, num_layers=LSTM_LAYERS,
                              dropout=LSTM_DROPOUT).to(DEVICE)
            model.load_state_dict(best_state)
            model.eval()
            alpha_s = predict_alpha_pit(model, fl, factor_cols, test_dates, universe_mask, factor_series=factor_series)
            alpha_s.to_parquet(alpha_dir / f"alpha_seed{si}.parquet")
            m = metrics_for_alpha(alpha_s, ret_eval)
            seed_alphas.append(alpha_s)
            print(f"    [seed{si}/{N_SEED}] best_ep={best_ep} IC={m['IC']:.4f} "
                  f"RankIC={m['RankIC']:.4f} (累计 {time.time()-t_all:.0f}s)", flush=True)
            del model

        alpha_avg = sum(seed_alphas) / len(seed_alphas)
        alpha_avg.to_parquet(alpha_dir / "alpha_agg.parquet")
        m_agg = metrics_for_alpha(alpha_avg, ret_eval)
        ic_rows.append({"h": h, **m_agg})  # 【改动】记录 IC 排名
        print(f"  h{h}: agg IC={m_agg['IC']:.4f} ICIR={m_agg['ICIR']:.4f} "
              f"RankIC={m_agg['RankIC']:.4f} RankICIR={m_agg['RankICIR']:.4f} "
              f"(累计 {time.time()-t_all:.0f}s)", flush=True)
        torch.cuda.empty_cache()

    # 保存跨 seed 的聚合损失（供 run_plot.py 画"无 BLO"损失曲线）。
    pd.DataFrame(agg_loss_rows).to_csv(LOSS_DIR / "search_loss_agg.csv", index=False)
    # 保存 IC 排名（供画图 / BLO 自动 R_AGG 参考）。
    pd.DataFrame(ic_rows).to_csv(LOSS_DIR / "search_ic_ranking.csv", index=False)
    print(f"\n*** 全部完成，总耗时 {time.time()-t_all:.0f}s ***")


if __name__ == "__main__":
    main()
