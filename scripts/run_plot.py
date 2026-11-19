"""【改动】生成报告用的图片与汇总表（训练/回测全部跑完后执行）。

一键调用 src.plotting 里的所有绘图函数，产出：
  - results/figures/loss_curve_blo_vs_noblo.png   （核心：有/无 BLO 的训练集+测试集损失曲线）
  - results/figures/ic_ranking.png                 （h=1..5 的 IC/RankIC）
  - results/figures/lambda_distribution.png        （BLO 学到的 λ）
  - results/figures/backtest_nav.png               （各 horizon 回测净值 + CSI300 基准）
  - results/figures/summary.md                     （Markdown 汇总表）

用法：
  python scripts/run_plot.py
"""
from __future__ import annotations
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (_HERE, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src import plotting as pl


def main():
    print("=== 生成报告图片与汇总表 ===")
    pl.plot_loss_curves_blo_vs_noblo()   # 必须项：四曲线损失图
    pl.plot_ic_ranking()
    pl.plot_lambda_distribution()
    pl.plot_backtest_nav()
    pl.save_summary_md()
    print("\n*** 完成，输出在 results/figures/ ***")


if __name__ == "__main__":
    main()
