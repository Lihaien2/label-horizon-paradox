"""Data pipeline: factor preprocessing, label construction, universe, prediction.

Provides the shared data utilities used by both the brute-force search
(``run_search.py``) and the bi-level optimization (``run_blo.py``):

  - ``load_factors_tanhw``: read the preprocessed factor cache (69 tanh + 89 keep).
  - ``load_label_data``: read post-adjusted VWAP.
  - ``make_label``: build an h-horizon label (post-adjusted VWAP return, with
    optional quantile trimming).
  - ``compute_label_quantile``: training-set label quantile thresholds.
  - ``prepare_factor_series`` / ``build_xy_seq``: 【改动】把"单日截面"改为"回看
    LOOKBACK 天序列"，停牌缺失用前向填充(ffill)。
  - ``get_tradable`` / ``get_train_union_tradable``: tradable masks.
  - ``predict_alpha_pit``: point-in-time alpha prediction（【改动】序列打分）。
  - ``evaluate_mse_loss``: 计算模型 MSE（用于损失曲线）。
  - ``metrics_for_alpha``: daily IC / RankIC / ICIR / RankICIR.

All market data is read directly from ``data/a_share_data/`` (post-adjusted
parquet files); no database connection is needed at runtime.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (_HERE, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import torch

from src.config import DEVICE, RAW_DATA_DIR, PROCESSED_DIR, TRAIN_START, TRAIN_END, LOOKBACK
from factors_calculate.alpha158 import ALPHA158_NAMES


# ---- Factor intrinsic classification (69 tanh + 89 keep) ----
# 69 heavy-tailed factors (price ratios / MA / ROC etc.): cross-sectional
#   tanh(median / IQR*2) compressed to [-1, 1].
# 89 bounded factors (candlestick ratios / statistics etc.): kept as-is.
# KMID2/KSFT2/KUP2/KLOW2 are candlestick ratios (divided by h-l, bounded in
# [-1, 1]) and are moved to the keep class.
TANH_FACTOR = {
    'KMID', 'KLEN', 'KSFT', 'KUP', 'KLOW',
    'OPEN0', 'HIGH0', 'LOW0', 'VWAP0',
    'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60',
    'MA5', 'MA10', 'MA20', 'MA30', 'MA60',
    'STD5', 'STD10', 'STD20', 'STD30', 'STD60',
    'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60',
    'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60',
    'MAX5', 'MAX10', 'MAX20', 'MAX30', 'MAX60',
    'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60',
    'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30', 'QTLU60',
    'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60',
    'VMA5', 'VMA10', 'VMA20', 'VMA30', 'VMA60',
    'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60',
    'WVMA5', 'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60',
}


def load_factors_tanhw(use_cache=True):
    """Load factors with intrinsic-class preprocessing.

    89 factors kept as-is, 69 cross-sectional tanh(IQR*2); tanh-class outliers
    beyond the training-set q001/q999 are set to NaN.

    use_cache=True: read the preprocessed cache (PROCESSED_DIR/factors.parquet),
        saving several minutes. Cache spec: 69 tanh + 89 keep, NaN thresholds computed on
        TRAIN_START..TRAIN_END (look-ahead safe), keep-class not de-extremed.
    use_cache=False: process on the fly (thresholds use TRAIN_START..TRAIN_END).
    """
    factor_cols = ALPHA158_NAMES
    if use_cache:
        print("[1] Loading preprocessed factor cache (69 tanh + 89 keep)...")
        long_df = pd.read_parquet(PROCESSED_DIR / "factors.parquet")
        n_tanh = len([c for c in factor_cols if c in TANH_FACTOR])
        print(f"  tanh: {n_tanh} | keep: {len(factor_cols) - n_tanh} | shape={long_df.shape}")
        return long_df, factor_cols

    print("[1] Loading raw factor cache (on-the-fly, 69 tanh + 89 keep)...")
    long_df = pd.read_parquet(RAW_DATA_DIR / "factors.parquet")
    n_tanh = len([c for c in factor_cols if c in TANH_FACTOR])
    n_keep = len(factor_cols) - n_tanh
    print(f"  tanh: {n_tanh} | keep: {n_keep}")

    # ---- 1) Per-factor q001/q999 thresholds on the training set ----
    tr_start = pd.Timestamp(TRAIN_START)
    tr_end = pd.Timestamp(TRAIN_END)
    train_idx = (long_df.index.get_level_values("date") >= tr_start) & \
                (long_df.index.get_level_values("date") <= tr_end)
    lo = {}
    hi = {}
    for c in factor_cols:
        s = long_df.loc[train_idx, c].replace([np.inf, -np.inf], np.nan)
        q = s.quantile([0.001, 0.999])
        lo[c] = float(q.loc[0.001])
        hi[c] = float(q.loc[0.999])
    n_drop_total = 0
    for c in factor_cols:
        col = long_df[c].replace([np.inf, -np.inf], np.nan)
        mask = (col < lo[c]) | (col > hi[c])
        n_drop_total += int(mask.sum())
        long_df[c] = col.mask(mask)
    print(f"  Outliers set to NaN (train q001/q999 thresholds): {n_drop_total} points "
          f"({100 * n_drop_total / (len(long_df) * len(factor_cols)):.4f}%)")

    # ---- 2) Cross-sectional tanh / keep as-is ----
    for c in factor_cols:
        col = long_df[c]
        if c in TANH_FACTOR:
            g = col.groupby(level="date", group_keys=False)
            med = g.transform("median")
            q25 = g.transform(lambda x: x.quantile(0.25))
            q75 = g.transform(lambda x: x.quantile(0.75))
            iqr = (q75 - q25).replace(0, np.nan)
            z = (col - med) / (iqr * 2.0)  # IQR*2: 77% linear region, 23% saturated
            z = z.replace([np.inf, -np.inf], np.nan)
            long_df[c] = np.tanh(z)
        else:
            long_df[c] = col
    return long_df, factor_cols


def load_label_data():
    """Load post-adjusted VWAP (wide frame)."""
    return pd.read_parquet(RAW_DATA_DIR / "vwap_adj.parquet")


def make_label(vwap_adj, h, lo=None, hi=None):
    """Build an h-horizon label: post-adjusted VWAP return, optional quantile trim.

    label = vwap_adj[t+h+1] / vwap_adj[t+1] - 1
    Quantile trimming (prevents extremes from polluting the cross-sectional
    z-score statistics):
      - Samples outside [lo, hi] are set to NaN (thresholds from the training
        set, no look-ahead).
      - Thresholds are computed per h (distributions differ across horizons).
    """
    ret = vwap_adj.shift(-(h + 1)) / vwap_adj.shift(-1) - 1.0
    ret = ret.replace([np.inf, -np.inf], np.nan)
    if lo is not None and hi is not None:
        ret = ret.mask((ret < lo) | (ret > hi))
    ret.index.name = "date"
    return ret


def prepare_factor_series(factor_long, factor_cols):
    """【改动】把 (date, wind_code) 长表因子转成"按股票对齐 + 前向填充"的完整序列。

    返回 (all_dates, stock_arrs):
      - all_dates: 升序的全部交易日（因子数据覆盖到的日期），list[pd.Timestamp]。
      - stock_arrs: dict[wind_code -> np.ndarray (T_all, D) float32]。
        每只股票对齐到 all_dates 的全部交易日；停牌缺失日沿时间前向填充(ffill)；
        上市前的前导 NaN 保留（用于后续判断窗口是否凑满 lookback 天）。
    """
    all_dates = sorted({pd.Timestamp(d) for d in factor_long.index.get_level_values("date").unique()})
    idx = pd.DatetimeIndex(all_dates)
    stock_arrs = {}
    for code, g in factor_long.groupby(level="wind_code", sort=True):
        gf = g.droplevel("wind_code")[factor_cols].sort_index()
        aligned = gf.reindex(idx).ffill()  # 停牌缺失前向填充（因果）；上市前仍为 NaN
        stock_arrs[code] = aligned.to_numpy(dtype=np.float32)
    return all_dates, stock_arrs


def _window_slice(stock_arrs, code, p, lookback):
    """取股票 code 在全局日期位置 p 处的回看窗口 (lookback, D)。

    窗口内若有任何 NaN（上市前前导缺失，无法前向填充）则返回 None，表示该股票当日
    凑不齐完整窗口，应被丢弃（只做 ffill、不做 bfill 的因果约束）。
    """
    arr = stock_arrs.get(code)
    if arr is None:
        return None
    s = arr[p - lookback + 1:p + 1]
    if np.isnan(s).any():
        return None
    return s


def build_xy_seq(factor_long, ret, dates_set, factor_cols, mask, clip=3.0, lookback=LOOKBACK,
                 factor_series=None):
    """【改动】构建序列样本 (X, y, d)。

    X: (N, lookback, D)，每个样本为过去 lookback 个交易日的特征序列（停牌日已 ffill）。
    y: (N,)，当日截面 z-score + clip 后的 h-horizon 标签。
    d: (N,)，日期 id（用于按日聚合的 IC 计算）。

    与旧 build_xy_clip 的区别：
      - 输入从"单日截面 (N, D)"变为"回看序列 (N, lookback, D)"；
      - 停牌缺失改用前向填充(ffill)而非整行丢弃；
      - 上市前前导 NaN 仍无法填充，此类股票当日丢弃。

    Args:
        factor_series: 可选，precompute 好的 (all_dates, stock_arrs)（prepare_factor_series
            的返回值）。传入可避免对同一因子数据重复计算（大幅提速）。
    """
    if factor_series is None:
        all_dates, stock_arrs = prepare_factor_series(factor_long, factor_cols)
    else:
        all_dates, stock_arrs = factor_series
    pos = {d: i for i, d in enumerate(all_dates)}
    fac_dates_set = set(all_dates)
    ret_dates_set = set(ret.index)
    mask_dates_set = set(mask.index) if mask is not None else set()

    X_parts, y_parts, d_parts = [], [], []
    for tj in sorted(dates_set):
        if tj not in fac_dates_set or tj not in ret_dates_set:
            continue
        p = pos[tj]
        if p < lookback - 1:  # 回看窗口不足，跳过（会丢掉最早 lookback-1 个交易日）
            continue
        r_j = ret.loc[tj]
        codes = list(r_j.index)
        # 可交易掩码：仅保留当日非停牌股票（对齐旧逻辑的点时间口径）。
        if tj in mask_dates_set:
            in_idx = mask.loc[tj]
            in_idx = in_idx[in_idx].index
            codes = [c for c in codes if c in in_idx]
        r_j = r_j.loc[codes].replace([np.inf, -np.inf], np.nan).dropna()

        rows, keep = [], []
        for c in r_j.index:
            s = _window_slice(stock_arrs, c, p, lookback)
            if s is None:
                continue
            rows.append(s)
            keep.append(c)
        if len(keep) < 20:
            continue
        X_j = np.stack(rows, axis=0)  # (n, lookback, D)
        r_j = r_j.loc[keep]
        r_mean, r_std = r_j.mean(), r_j.std()
        if r_std < 1e-12:
            continue
        z = ((r_j - r_mean) / r_std).clip(-clip, clip)
        X_parts.append(X_j)
        y_parts.append(z.to_numpy(dtype=np.float32))
        d_parts.append(np.full(len(keep), hash(tj) % (2**31), dtype=np.int64))

    if not X_parts:
        return None, None, None
    X = np.concatenate(X_parts, axis=0)  # (N, lookback, D)
    y = np.concatenate(y_parts, axis=0)  # (N,)
    d = np.concatenate(d_parts, axis=0)  # (N,)
    return X, y, d


def compute_label_quantile(vwap_adj_train, h, q_lo=0.005, q_hi=0.995):
    """Quantile thresholds from the training-set label distribution (per h)."""
    r = (vwap_adj_train.shift(-(h + 1)) / vwap_adj_train.shift(-1) - 1.0)
    r = r.replace([np.inf, -np.inf], np.nan)
    s = r.stack()
    return float(s.quantile(q_lo)), float(s.quantile(q_hi))


def get_tradable(start, end):
    m = pd.read_parquet(RAW_DATA_DIR / "tradable_mask.parquet")
    return m.loc[pd.Timestamp(start):pd.Timestamp(end)]


def get_universe_mask(start, end):
    """【改动】沪深300 每日 PIT 股票池掩码（当日成分股 & 非停牌）。

    只做沪深300：股票池 = 沪深300当日成分（member_mask_csi300）∩ 非停牌（tradable_mask）。
    返回 (T x N) bool 宽表，index=date，columns=wind_code。
    """
    member = pd.read_parquet(RAW_DATA_DIR / "member_mask_csi300.parquet")
    tradable = get_tradable(start, end)
    # 对齐到可交易掩码的日期轴/股票列（成分与行情代码一致）。
    member = member.reindex(index=tradable.index, columns=tradable.columns).fillna(False)
    return (member & tradable).astype(bool)


def get_train_union_tradable(full_start, full_end, train_start, train_end):
    """Look-ahead-safe training universe: union of stocks tradable during the
    training period, broadcast to the full range.
    """
    train_mask = get_tradable(train_start, train_end)
    in_train = train_mask.any(axis=0)
    full_mask = get_tradable(full_start, full_end)
    out = full_mask.copy()
    out.loc[:, :] = False
    cols = in_train.index.intersection(out.columns)
    out[cols] = in_train[cols].values
    return out


def predict_alpha_pit(model, factor_long, factor_cols, test_dates, member_mask, lookback=LOOKBACK,
                      factor_series=None):
    """Point-in-time 序列打分（【改动】：回看 lookback 天窗口打分）。

    每个交易日 t：对当日可交易且能凑满 lookback 天窗口（窗口内特征已 ffill）的股票，
    用模型打分。停牌/窗口不完整的股票不被打分（alpha=NaN），与训练样本构造
    （build_xy_seq）口径一致——模型只训练过窗口完整的股票，因此只对同类股票打分。

    Args:
        model: AlphaLSTM 等序列模型，输入 [B, lookback, D]。
        factor_long: 预处理后的因子长表。
        factor_cols: 因子列名。
        test_dates: 需要打分的日期集合。
        member_mask: 可交易/成分掩码（run_search 传入的 universe_mask），排除停牌股票。
        lookback: 回看窗口天数。
        factor_series: 可选，prepare_factor_series 的返回值，避免重复计算。

    Returns:
        宽表 DataFrame（index=date, columns=wind_code），值为当日 alpha 打分。
    """
    if factor_series is None:
        all_dates, stock_arrs = prepare_factor_series(factor_long, factor_cols)
    else:
        all_dates, stock_arrs = factor_series
    pos = {d: i for i, d in enumerate(all_dates)}
    mm_by_day = {}
    if member_mask is not None:
        for t in member_mask.index:
            in_idx = member_mask.loc[t]
            mm_by_day[t] = in_idx[in_idx].index

    alpha_preds = {}
    model.eval()
    with torch.no_grad():
        for t in sorted(test_dates):
            p = pos.get(t)
            if p is None or p < lookback - 1:
                continue
            rows, keep = [], []
            for c in stock_arrs:
                # 可交易掩码：停牌股票当日不打分。
                if t in mm_by_day and c not in mm_by_day[t]:
                    continue
                s = _window_slice(stock_arrs, c, p, lookback)
                if s is None:
                    continue
                rows.append(s)
                keep.append(c)
            if len(keep) < 20:
                continue
            X_arr = np.stack(rows, axis=0)  # (n, lookback, D)
            pred = model(torch.as_tensor(X_arr, dtype=torch.float32, device=DEVICE)).cpu().numpy()
            alpha_preds[t] = pd.Series(pred, index=keep)
    return pd.DataFrame(alpha_preds).T


def evaluate_mse_loss(model, X, y, device=DEVICE, batch_size=8192):
    """【改动】分批计算模型在 (X, y) 上的 MSE（用于绘制损失曲线）。

    X 形状 (N, lookback, D)，y 形状 (N,)。返回标量 MSE；空数据返回 NaN。
    """
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = X[i:i + batch_size]
            yb = y[i:i + batch_size]
            pred = model(xb)
            total += float(((pred - yb) ** 2).sum().item())
            n += xb.shape[0]
    return total / n if n else float("nan")


def metrics_for_alpha(alpha, ret):
    """Daily IC / RankIC / ICIR / RankICIR for an alpha matrix.

    ``alpha`` and ``ret`` are (date x code) wide frames or Series.
    """
    ics, rankics = [], []
    for t in alpha.index.intersection(ret.index):
        a = alpha.loc[t].dropna()
        r = ret.loc[t]
        common = a.index.intersection(r.index)
        a = a.loc[common]
        r = r.loc[common].dropna()
        a = a.loc[r.index]
        if len(a) < 20:
            continue
        ics.append(a.corr(r))
        rankics.append(a.corr(r, method="spearman"))
    ic = np.array([x for x in ics if not np.isnan(x)])
    rc = np.array([x for x in rankics if not np.isnan(x)])
    return {
        "IC": ic.mean() if len(ic) else np.nan,
        "ICIR": ic.mean() / ic.std() if len(ic) and ic.std() > 0 else np.nan,
        "RankIC": rc.mean() if len(rc) else np.nan,
        "RankICIR": rc.mean() / rc.std() if len(rc) and rc.std() > 0 else np.nan,
    }
