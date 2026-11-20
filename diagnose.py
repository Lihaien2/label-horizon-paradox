import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import pandas as pd
from src.config import RAW_DATA_DIR
from src.data_utils import load_factors_tanhw, load_label_data, make_label, compute_label_quantile

fl, factor_cols = load_factors_tanhw(use_cache=True)
vwap = load_label_data()
lo, hi = compute_label_quantile(vwap.loc["2020-01-02":"2023-12-31"], 5)
ret = make_label(vwap, 5, lo, hi)
test_dates = ret.loc["2025-01-02":"2026-06-30"].index

fac = fl[factor_cols]
res = []
for c in factor_cols:
    f = fac[c].unstack("wind_code")            # date x code
    ics = []
    for t in test_dates:
        if t not in f.index:
            continue
        a = f.loc[t]
        b = ret.loc[t]
        cc = a.dropna().index.intersection(b.dropna().index)
        if len(cc) < 20:
            continue
        ics.append(a.loc[cc].corr(b.loc[cc], method="spearman"))
    if ics:
        res.append((c, float(np.nanmean(ics))))

res.sort(key=lambda x: -abs(x[1]))
print("单因子与 h5 收益的截面 RankIC（测试期，按 |RankIC| 排序）：")
for c, ic in res[:20]:
    print(f"  {c}: {ic:+.4f}")