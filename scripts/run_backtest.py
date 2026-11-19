"""解耦回测：从保存的 alpha parquet 跑"阶梯子账户"回测——沪深300，TopN=30，5个子账户。

（【改动】子账户 10 -> 5（持有期 10天 -> 5天，与目标 h=5 对齐）；TopN 默认 30；
股票池固定沪深300；只做 CSI300 基准对比。）

与训练完全解耦：只读 results/alphas/h{h}/ 下的 alpha parquet 与 data/ 行情数据，
不碰模型/因子代码。

回测配置：
  - n_ladders=5, rebalance_freq=5（每个子账户每 5 天调仓，5 个子账户错开）
  - DailyClose（t 收盘信号 -> t+1 成交）+ VWAPExec（次日 VWAP 成交）
  - CostModel（印花税 0.05% 单边卖 + 佣金 0.025% 双边）
  - Constraints(rollover=True)：涨停封单买入权重滚到替补股票；跌停/停牌冻结

输出（results/backtest/）：
  - backtest_{mode}_topn_{config}_h{h}.csv：每个 h 的 NAV 序列
  - backtest_{mode}_topn_summary.csv：汇总（sharpe/年化/回撤/换手）
  - log_backtest.txt

用法：
  python scripts/run_backtest.py --mode agg                        # 聚合 alpha，TopN=30
  python scripts/run_backtest.py --mode agg --topn 30 --h 1 3 5    # 指定 horizon
  python scripts/run_backtest.py --mode seed --topn 30 --h 5       # seed 模式单 h
"""
from __future__ import annotations
import sys, warnings, time, argparse
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

from engine import LadderBacktest, TopNOptimizer
from engine.base.execution import VWAPExec
from engine.base.timeline import DailyClose
from engine.base.costs import CostModel
from engine.base.constraints import Constraints
from src.config import RAW_DATA_DIR, UNIVERSE_NAME, UNIVERSE_INDEX_CODE

ALPHA_DIR = _ROOT / "results" / "alphas"
OUT_DIR = _ROOT / "results" / "backtest"
LOG = OUT_DIR / "log_backtest.txt"

# 【改动】子账户数 10 -> 5；调仓频率 10 -> 5（每个子账户持有 5 天）。
N_LADDERS = 5
REBALANCE_FREQ = 5
N_SEED = 15
HORIZONS = list(range(1, 6))
DEFAULT_TOPN = 30          # 【改动】TopN 默认 30 只
DEFAULT_UNIVERSE = "csi300"
_lines = []


def p(msg=""):
    print(msg, flush=True)
    _lines.append(str(msg))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG, "w") as f:
        f.write("\n".join(_lines))


def make_optimizer(topn: int, universe=None):
    """构造沪深300 TopN 优化器（每日从成分股中选 alpha 最高的 topn 只等权）。"""
    masks = []
    uni = universe or DEFAULT_UNIVERSE
    for name in uni.split(","):
        name = name.strip().lower()
        path = RAW_DATA_DIR / f"member_mask_{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"成分掩码不存在: {path}（请先运行 build_backtest_data.py）")
        masks.append(pd.read_parquet(path))
    return TopNOptimizer(n=topn, member_masks=masks), masks


def index_benchmarks():
    """沪深300 买入持有基准（测试期）。"""
    from src.config import TEST_START, TEST_END
    from performance import metrics as M
    ic = pd.read_parquet(RAW_DATA_DIR / "index_close.parquet")
    s, e = pd.Timestamp(TEST_START), pd.Timestamp(TEST_END)
    out = {}
    for name, code in {"CSI300": UNIVERSE_INDEX_CODE}.items():
        if code not in ic.columns:
            continue
        close = ic[code].loc[s:e].dropna()
        ret = close.pct_change().dropna()
        out[name] = {"sharpe": M.sharpe(ret), "ann": M.annualized_return(ret),
                     "mdd": M.max_drawdown(ret)}
    return out


def run_one(alpha: pd.DataFrame, optimizer, rollover: bool = True) -> dict:
    """跑一次阶梯回测，返回指标 dict。"""
    r = LadderBacktest(
        alpha_signals=alpha,
        optimizer=optimizer,
        n_ladders=N_LADDERS,
        rebalance_freq=REBALANCE_FREQ,
        timepoint=DailyClose(),
        execution=VWAPExec(),
        cost_model=CostModel(),
        constraints=Constraints(max_weight=None, participation_rate=None, residual="cash", rollover=rollover),
    ).run()
    return {
        "nav": r.nav,
        "sharpe": float(r.sharpe),
        "ann": float(r.annualized_return),
        "mdd": float(r.max_drawdown),
        "turnover": float(r.avg_turnover),
        "end_nav": float(r.nav.iloc[-1]) if len(r.nav) else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description="沪深300 阶梯回测（5 子账户，TopN=30）")
    ap.add_argument("--mode", choices=["agg", "seed"], default="agg",
                    help="agg=读聚合 alpha 回测一次；seed=逐 seed 回测后平均指标")
    ap.add_argument("--topn", type=int, default=DEFAULT_TOPN, help=f"选股数（默认 {DEFAULT_TOPN}）")
    ap.add_argument("--h", type=int, nargs="+", default=None, help="指定 horizon（默认 h1..5）")
    ap.add_argument("--universe", type=str, default=DEFAULT_UNIVERSE,
                    help="选股成分池（默认 csi300，逗号分隔）")
    ap.add_argument("--no-rollover", action="store_true",
                    help="关闭涨停封单买入权重滚到替补股票（默认开启）")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    horizons = args.h if args.h else HORIZONS
    uni_tag = f"_{'_'.join(u.strip().lower() for u in args.universe.split(','))}"
    config_name = f"topn{args.topn}{uni_tag}"

    idx_bm = index_benchmarks()

    p("=" * 90)
    p(f"沪深300 阶梯回测 | mode={args.mode} | topn={args.topn} universe={args.universe}")
    p(f"  n_ladders={N_LADDERS} rebalance_freq={REBALANCE_FREQ}（每子账户持有 5 天）"
      f" DailyClose+VWAP，印花税 0.05% 卖单 + 佣金 0.025% 双边，涨停买入滚到替补，跌停/停牌冻结")
    p(f"  horizons={horizons}")
    if idx_bm:
        p(f"  基准: " + " | ".join(f"{k} sharpe={v['sharpe']:.3f} ann={v['ann']*100:.2f}%" for k, v in idx_bm.items()))
    p("=" * 90)

    rows = []
    t_all = time.time()
    for h in horizons:
        h_dir = ALPHA_DIR / f"h{h}"
        if not h_dir.exists():
            p(f"\n--- h{h}: alpha 目录不存在，跳过 ---")
            continue

        if args.mode == "agg":
            path = h_dir / "alpha_agg.parquet"
            if not path.exists():
                p(f"\n--- h{h}: alpha_agg.parquet 不存在，跳过 ---")
                continue
            alpha = pd.read_parquet(path)
            alphas = [(alpha, 1)]
        else:  # seed
            seed_alphas = []
            for si in range(N_SEED):
                path = h_dir / f"alpha_seed{si}.parquet"
                if path.exists():
                    seed_alphas.append(pd.read_parquet(path))
            if not seed_alphas:
                p(f"\n--- h{h}: 无 seed alpha，跳过 ---")
                continue
            alphas = [(a, 1) for a in seed_alphas]

        optimizer, _ = make_optimizer(args.topn, universe=args.universe)
        t1 = time.time()
        res_list = [run_one(a, optimizer, rollover=not args.no_rollover) for a, _ in alphas]
        avg = {
            "sharpe": float(np.mean([r["sharpe"] for r in res_list])),
            "ann": float(np.mean([r["ann"] for r in res_list])),
            "mdd": float(np.mean([r["mdd"] for r in res_list])),
            "turnover": float(np.mean([r["turnover"] for r in res_list])),
            "end_nav": float(np.mean([r["end_nav"] for r in res_list])),
        }
        nav0 = res_list[0]["nav"]
        tag = f"{args.mode}_topn_{config_name}"
        p(f"--- h{h} [{tag}] sharpe={avg['sharpe']:.3f} ann={avg['ann']*100:.2f}% "
          f"mdd={avg['mdd']*100:.2f}% turnover={avg['turnover']*100:.2f}% (耗时 {time.time()-t1:.0f}s)")
        pd.DataFrame({"date": nav0.index, "nav": nav0.values}).to_csv(
            OUT_DIR / f"backtest_{tag}_h{h}.csv", index=False)
        rows.append({"h": h, "mode": args.mode, "pool": "topn", "config": config_name, **avg,
                     "n_seed": len(alphas) if args.mode == "seed" else 1})

        pd.DataFrame(rows).to_csv(OUT_DIR / f"backtest_{args.mode}_topn_summary.csv", index=False)

    # 汇总表
    p(f"\n{'='*90}")
    p(f"Summary: mode={args.mode} pool=topn（沪深300，TopN={args.topn}）")
    p(f"{'h':>3} | {'sharpe':>7} | {'ann':>8} | {'mdd':>7} | {'turnover':>7} | {'end_nav':>7}")
    p("-" * 60)
    for r in rows:
        p(f"{r['h']:>3} | {r['sharpe']:>7.3f} | {r['ann']*100:>7.2f}% | {r['mdd']*100:>6.2f}% | "
          f"{r['turnover']*100:>6.2f}% | {r['end_nav']:>7.4f}")
    if idx_bm:
        p(f"\n基准（测试期）:")
        for k, v in idx_bm.items():
            p(f"  {k}: sharpe={v['sharpe']:.3f} ann={v['ann']*100:.2f}% mdd={v['mdd']*100:.2f}%")
    p(f"\nNAV 保存至: backtest_{args.mode}_topn_{config_name}_h{{h}}.csv")
    p(f"汇总保存至: backtest_{args.mode}_topn_summary.csv")
    p(f"总耗时 {time.time()-t_all:.0f}s")
    p(f"\n*** Done ***")


if __name__ == "__main__":
    main()
