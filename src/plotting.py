"""【改动】绘图模块：为报告分析生成 PNG 图与汇总表。

使用 matplotlib（Agg 后端，不弹窗，只保存 PNG）。图表文字用英文（避免中文字体
缺失导致乱码）；代码注释用中文。

对外提供：
  - plot_loss_curves_blo_vs_noblo：核心图——"有 BLO vs 无 BLO"在训练集/测试集的
    四条损失曲线，画在同一张图里。
  - plot_ic_ranking：h=1..5 的 IC/RankIC 柱状图。
  - plot_lambda_distribution：BLO 学到的 λ 分布柱状图。
  - plot_backtest_nav：各 horizon 回测净值曲线（含 CSI300 基准）。
  - save_summary_md：把 IC 排名、λ、回测指标汇总成 Markdown 表格。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无窗口后端，仅保存图片
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (_HERE, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.config import FIGURES_DIR, RESULTS_DIR, LOSSES_DIR, TARGET_H, RAW_DATA_DIR

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# 全局样式。
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def _fig(title, xlabel, ylabel):
    """创建带标题/坐标轴的单个子图。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


def plot_loss_curves_blo_vs_noblo(search_loss_path=None, blo_loss_path=None, out_path=None):
    """【核心图】有 BLO vs 无 BLO 在**训练集/验证集**的四条损失曲线（一张图对比）。

    （按用户要求：loss 曲线看的是 train 与 valid，不再用 test。）
    无 BLO 取 h=TARGET_H 的搜索模型：跨 seed 对每个 epoch 的 train/valid MSE 取均值。
    有 BLO 取跨 seed/warmup 对每个 iter 的 train/valid MSE 取均值。
    若旧版 blo_loss_agg.csv 没有 valid_loss 列，会回退到 test_loss 并给出提示。
    """
    search_loss_path = Path(search_loss_path) if search_loss_path else LOSSES_DIR / "search_loss_agg.csv"
    blo_loss_path = Path(blo_loss_path) if blo_loss_path else RESULTS_DIR / "blo" / "blo_loss_agg.csv"
    out_path = Path(out_path) if out_path else FIGURES_DIR / "loss_curve_blo_vs_noblo.png"

    fig, ax = _fig("Training/Validation Loss: No-BLO (h=5) vs BLO (LSTM)",
                   "training progress (epoch for No-BLO / iter for BLO)", "MSE loss")

    plotted_any = False

    # ---- 无 BLO：搜索 h=5 的损失（按 epoch 对 seed 取均值，train + valid）----
    if search_loss_path.exists():
        df = pd.read_csv(search_loss_path)
        sub = df[df["h"] == TARGET_H]
        if not sub.empty:
            g = sub.groupby("epoch")[["train_loss", "valid_loss"]].mean().reset_index()
            ax.plot(g["epoch"], g["train_loss"], label="No-BLO train", color="tab:blue", marker="o", lw=2)
            ax.plot(g["epoch"], g["valid_loss"], label="No-BLO valid", color="tab:red", marker="o", lw=2)
            plotted_any = True
    else:
        print(f"  [plot] 未找到 {search_loss_path}（先运行 run_search.py）")

    # ---- 有 BLO：按 iter 对 seed/warmup 取均值（train + valid）----
    if blo_loss_path.exists():
        df = pd.read_csv(blo_loss_path)
        if not df.empty:
            if "valid_loss" in df.columns and df["valid_loss"].notna().any():
                g = df.groupby("iter")[["train_loss", "valid_loss"]].mean().reset_index()
                ax.plot(g["iter"], g["train_loss"], label="BLO train", color="tab:cyan", linestyle="--", lw=2)
                ax.plot(g["iter"], g["valid_loss"], label="BLO valid", color="tab:orange", linestyle="--", lw=2)
                plotted_any = True
            else:
                # 兼容旧文件：没有 valid_loss 时退回 test_loss，并明确提示需要重跑 run_blo。
                print("  [plot] 警告：blo_loss_agg.csv 缺少 valid_loss 列（旧版），"
                      "暂用 test_loss 代替；请重跑 run_blo.py 以得到验证集曲线。")
                g = df.groupby("iter")[["train_loss", "test_loss"]].mean().reset_index()
                ax.plot(g["iter"], g["train_loss"], label="BLO train", color="tab:cyan", linestyle="--", lw=2)
                ax.plot(g["iter"], g["test_loss"], label="BLO test (legacy)", color="tab:orange", linestyle="--", lw=2)
                plotted_any = True
    else:
        print(f"  [plot] 未找到 {blo_loss_path}（先运行 run_blo.py）")

    if not plotted_any:
        print("  [plot] 无损失数据，跳过 loss_curve 图。")
        plt.close(fig)
        return None

    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [plot] 损失曲线已保存：{out_path}")
    return out_path


def plot_ic_ranking(ic_path=None, out_path=None):
    """h=1..5 的 IC/RankIC 柱状图（展示"标签 horizon 悖论"）。"""
    ic_path = Path(ic_path) if ic_path else LOSSES_DIR / "search_ic_ranking.csv"
    out_path = Path(out_path) if out_path else FIGURES_DIR / "ic_ranking.png"
    if not ic_path.exists():
        print(f"  [plot] 未找到 {ic_path}，跳过 IC 排名图。")
        return None
    df = pd.read_csv(ic_path).sort_values("h")
    fig, ax = _fig("Brute-force search IC by horizon (target h=5)", "horizon h", "IC")
    x = np.arange(len(df))
    w = 0.38
    ax.bar(x - w/2, df["IC"], w, label="IC", color="tab:blue")
    if "RankIC" in df.columns:
        ax.bar(x + w/2, df["RankIC"], w, label="RankIC", color="tab:green")
    ax.set_xticks(x)
    ax.set_xticklabels([f"h{int(h)}" for h in df["h"]])
    ax.axhline(0, color="black", lw=0.8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [plot] IC 排名图已保存：{out_path}")
    return out_path


def plot_lambda_distribution(blo_results_path=None, out_path=None):
    """BLO 学到的 λ 分布（跨 warmup 平均）柱状图。"""
    blo_results_path = Path(blo_results_path) if blo_results_path else RESULTS_DIR / "blo" / "blo_results.csv"
    out_path = Path(out_path) if out_path else FIGURES_DIR / "lambda_distribution.png"
    if not blo_results_path.exists():
        print(f"  [plot] 未找到 {blo_results_path}，跳过 λ 分布图。")
        return None
    df = pd.read_csv(blo_results_path)
    if "ens_lambda" not in df.columns or df.empty:
        print("  [plot] blo_results.csv 无 ens_lambda，跳过 λ 分布图。")
        return None
    lam = np.array([json.loads(x) for x in df["ens_lambda"]]).mean(axis=0)
    hs = list(range(1, len(lam) + 1))
    fig, ax = _fig("BLO learned horizon weights lambda", "horizon h", "lambda")
    ax.bar([f"h{h}" for h in hs], lam, color="tab:purple")
    ax.axhline(0, color="black", lw=0.8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [plot] λ 分布图已保存：{out_path}")
    return out_path


def plot_backtest_nav(nav_dir=None, out_path=None, benchmark=True):
    """各 horizon 回测净值曲线（含 CSI300 基准，可选）。"""
    nav_dir = Path(nav_dir) if nav_dir else RESULTS_DIR / "backtest"
    out_path = Path(out_path) if out_path else FIGURES_DIR / "backtest_nav.png"
    if not nav_dir.exists():
        print(f"  [plot] 未找到 {nav_dir}，跳过回测净值图。")
        return None

    fig, ax = _fig("Backtest NAV by horizon (TopN=30, CSI300)", "date", "NAV")
    plotted_any = False
    # 找 agg 模式的 NAV csv：backtest_agg_topn_*_h{h}.csv
    for p in sorted(nav_dir.glob("backtest_agg_topn_*_h*.csv")):
        h = p.stem.rsplit("_h", 1)[-1]
        try:
            nav = pd.read_csv(p)
        except Exception:
            continue
        if "date" not in nav.columns or "nav" not in nav.columns or nav.empty:
            continue
        nav["date"] = pd.to_datetime(nav["date"])
        ax.plot(nav["date"], nav["nav"], label=f"h{h}", lw=1.5)
        plotted_any = True

    # 沪深300 基准（测试期）。
    if benchmark:
        try:
            from src.config import TEST_START, TEST_END
            ic = pd.read_parquet(RAW_DATA_DIR / "index_close.parquet")
            code = [c for c in ic.columns if "000300" in c or "399300" in c]
            if code:
                close = ic[code[0]].loc[pd.Timestamp(TEST_START):pd.Timestamp(TEST_END)].dropna()
                bm_nav = close / close.iloc[0]
                ax.plot(bm_nav.index, bm_nav.values, label="CSI300", color="black", lw=2)
        except Exception as ex:
            print(f"  [plot] 基准绘制失败（忽略）: {ex}")

    if not plotted_any:
        print("  [plot] 无回测 NAV 数据，跳过。")
        plt.close(fig)
        return None
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  [plot] 回测净值图已保存：{out_path}")
    return out_path


def _df_to_markdown(df: pd.DataFrame) -> str:
    """手动把 DataFrame 转成 Markdown 表格（避免依赖 tabulate）。"""
    df = df.copy()
    cols = list(df.columns)
    for c in cols:
        if df[c].dtype.kind in "fc":  # 数值列四舍五入保留 4 位
            df[c] = df[c].round(4)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False, name=None)]
    return "\n".join([header, sep] + rows)


def save_summary_md(out_path=None):
    """把 IC 排名、BLO λ、回测指标汇总成 Markdown 表格（供写报告）。"""
    out_path = Path(out_path) if out_path else FIGURES_DIR / "summary.md"
    lines = ["# 实验汇总（自动生成）\n"]

    ic_path = LOSSES_DIR / "search_ic_ranking.csv"
    if ic_path.exists():
        ic = pd.read_csv(ic_path).sort_values("h")
        lines.append("## 1. 暴力搜索 IC（h=1..5，目标 h=5）\n")
        lines.append(_df_to_markdown(ic))
        lines.append("")

    blo_path = RESULTS_DIR / "blo" / "blo_results.csv"
    if blo_path.exists() and blo_path.stat().st_size > 0:
        blo = pd.read_csv(blo_path)
        if "ens_lambda" in blo.columns and not blo.empty:
            lam = np.array([json.loads(x) for x in blo["ens_lambda"]]).mean(axis=0)
            lam_df = pd.DataFrame({"h": list(range(1, len(lam) + 1)), "lambda": lam})
            lines.append("## 2. BLO 学到的 λ（跨 warmup 平均）\n")
            lines.append(_df_to_markdown(lam_df))
            if "ens_rc" in blo.columns:
                lines.append(f"\n- 集成 rc 均值 = {blo['ens_rc'].mean():.4f}；"
                             f"argmax 众数 = h{int(blo['ens_argmax'].mode().iloc[0])}\n")

    bt_path = RESULTS_DIR / "backtest" / "backtest_agg_topn_summary.csv"
    if bt_path.exists():
        bt = pd.read_csv(bt_path)
        lines.append("## 3. 回测汇总（沪深300，TopN=30，5 子账户）\n")
        cols = [c for c in ["h", "sharpe", "ann", "mdd", "turnover", "end_nav"] if c in bt.columns]
        lines.append(_df_to_markdown(bt[cols]))
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [summary] 汇总表已保存：{out_path}")
    return out_path
