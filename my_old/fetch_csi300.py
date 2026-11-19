"""
沪深300成分股数据获取与预处理脚本。

用法示例：
    python utils/fetch_CSI300.py --token YOUR_TOKEN
    python utils/fetch_CSI300.py --token YOUR_TOKEN \
        --start_date 20220101 --end_date 20241231 \
        --exit_offset 5 --ma_windows 5 10 20

输出（--out_dir 下，默认 ./dataset/CSI300）：
    panel.csv         多股票面板（核心产物）
    fetch_meta.json   抓取参数快照（供训练侧校验 target 语义）
    fetch_log.txt     运行日志

panel.csv 列结构：
    date                信号日（t 日收盘后产生信号）
    target_start_date   入场日 = t + entry_offset 的交易日
    target_end_date     出场日 = t + exit_offset 的交易日
    stock_code          股票代码
    open/high/low/close t 日行情
    volume/amount       t 日成交量、成交额
    pct_chg             t 日涨跌幅（%）
    turn                t 日换手率（%）
    ma5/ma20/...        按 --ma_windows 生成的因子
    target              close[target_end_date] / open[target_start_date] - 1
    is_constituent      t 日是否在指数成分内（唯一准入规则）

加因子：只在 compute_factors() 里改，见该函数注释。
"""

import os
import sys
import json
import time
import argparse
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import tushare as ts


# ============================================================
# 0. 全局配置
# ============================================================
DEFAULT_TOKEN = os.environ.get('TUSHARE_TOKEN', '')

# 基础行情列（不是因子）；因子由 compute_factors 动态生成
FEATURE_COLS_BASE = ['open', 'high', 'low', 'close',
                     'volume', 'amount', 'pct_chg', 'turn']


# ============================================================
# 1. 日志
# ============================================================
class Logger:
    def __init__(self, path):
        parent = os.path.dirname(path) or '.'
        os.makedirs(parent, exist_ok=True)
        self.path = path
        self.fp = open(path, 'w', encoding='utf-8')

    def log(self, msg):
        print(msg)
        self.fp.write(msg + '\n')
        self.fp.flush()

    def close(self):
        self.fp.close()


# ============================================================
# 2. 指数成分（月度快照近似）
# ============================================================
def get_index_constituents(pro, index_code, start_date, end_date, logger):
    """
    用 index_weight 月度快照近似重建成分区间。
    每个快照状态延续到下一个快照前一天；最后一个延续到 end_date。
    返回 memberships 列表，每项 = {stock_code, effective_start, effective_end, ...}。
    """
    logger.log('\n' + '=' * 60)
    logger.log(f'[Step 1] 获取指数 {index_code} 历史成分（月度快照近似）')
    logger.log('=' * 60)

    # 按自然月遍历
    start_ts = pd.to_datetime(start_date, format='%Y%m%d')
    end_ts = pd.to_datetime(end_date, format='%Y%m%d')
    month_starts = pd.date_range(start_ts.replace(day=1),
                                 end_ts.replace(day=1), freq='MS')

    # trade_date -> [con_code, ...]
    snapshots = {}

    for ms in month_starts:
        me = ms + pd.offsets.MonthEnd(0)
        lo = max(ms.strftime('%Y%m%d'), start_date)
        hi = min(me.strftime('%Y%m%d'), end_date)

        try:
            weights = pro.index_weight(
                index_code=index_code, start_date=lo, end_date=hi,
                fields='index_code,con_code,trade_date,weight')
        except Exception as e:
            raise RuntimeError(
                f'index_weight 查询失败（{lo}—{hi}）。'
                f'检查 Token、积分（需≥2000）和指数代码 {index_code}。') from e

        if weights is None or weights.empty:
            logger.log(f'  [WARN] {lo}—{hi} 无月度快照')
            continue

        required = {'con_code', 'trade_date', 'weight'}
        if not required.issubset(weights.columns):
            raise RuntimeError(
                f'index_weight 返回字段不完整: '
                f'{sorted(required - set(weights.columns))}')

        weights = weights.copy()
        weights['trade_date'] = (weights['trade_date'].astype(str)
                                 .str.replace('-', '', regex=False))
        weights['con_code'] = weights['con_code'].astype(str)

        for td, group in weights.groupby('trade_date'):
            codes = sorted(set(group['con_code'].dropna()))
            if codes:
                snapshots[td] = codes

    if not snapshots:
        raise RuntimeError(
            f'index_weight 在 {start_date}—{end_date} 无任何快照。'
            f'确认指数代码 {index_code}、Token、积分。')

    snapshot_dates = sorted(snapshots)
    logger.log(f'月度快照数: {len(snapshot_dates)}；'
               f'首末: {snapshot_dates[0]}—{snapshot_dates[-1]}')
    logger.log('限制：成员状态按快照日期阶梯延续，非交易所公告级精确生效日。')

    # 快照展开为 [effective_start, effective_end] 区间
    memberships = []
    for i, snap_date in enumerate(snapshot_dates):
        interval_start = max(start_date, snap_date)
        if i + 1 < len(snapshot_dates):
            next_dt = pd.to_datetime(snapshot_dates[i + 1], format='%Y%m%d')
            interval_end = (next_dt - pd.Timedelta(days=1)).strftime('%Y%m%d')
        else:
            interval_end = end_date
        interval_end = min(interval_end, end_date)
        if interval_start > interval_end:
            continue
        for code in snapshots[snap_date]:
            memberships.append({
                'stock_code': code,
                'effective_start': interval_start,
                'effective_end': interval_end,
                'snapshot_date': snap_date,
            })

    logger.log(f'区间记录: {len(memberships)}；'
               f'股票数: {len(set(m["stock_code"] for m in memberships))}')
    return memberships


# ============================================================
# 3. 因子计算（加因子只改这里）
# ============================================================
def compute_factors(df, ma_windows):
    """
    在 df 上就地计算所有衍生因子列，返回 (df, factor_cols)。

    加因子步骤：
      1. 在下方"因子区"加一行 df['新列'] = ...
      2. 把 '新列' append 到 factor_cols
    调用方会自动把 factor_cols 纳入 keep / required，无需改其他地方。
    """
    factor_cols = []

    # ---- 因子：移动平均 ----
    for w in ma_windows:
        col = f'ma{w}'
        df[col] = df['close'].rolling(int(w)).mean()
        factor_cols.append(col)

    # ================= 因子区：未来加在这里 =================
    # 示例（取消注释即可启用）：
    # df['rsi14'] = compute_rsi(df['close'], 14)
    # factor_cols.append('rsi14')
    #
    # df['vol20'] = df['pct_chg'].rolling(20).std()
    # factor_cols.append('vol20')
    # =========================================================

    return df, factor_cols


# ============================================================
# 4. 单只股票行情 + target
# ============================================================
def fetch_stock_data(pro, stock_code, start_date, end_date,
                     entry_offset, exit_offset, ma_windows):
    """
    抓取单只股票前复权日线 + 换手率，计算因子与 target。
    缺失值只允许 ffill（因果），禁止 bfill；首端缺失行最终 drop。
    """
    # ---- 4.1 前复权日线 ----
    try:
        df = ts.pro_bar(ts_code=stock_code, adj='qfq',
                        start_date=start_date, end_date=end_date)
    except Exception as e:
        print(f'  [ERROR] {stock_code} pro_bar 异常: {repr(e)}')
        return None
    if df is None or df.empty:
        print(f'  [WARN] {stock_code} pro_bar 返回空')
        return None
    df = df.sort_values('trade_date').reset_index(drop=True)

    # ---- 4.2 换手率 ----
    try:
        basic = pro.daily_basic(
            ts_code=stock_code, start_date=start_date, end_date=end_date,
            fields='trade_date,turnover_rate')
    except Exception as e:
        print(f'  [WARN] {stock_code} daily_basic 异常: {repr(e)}')
        basic = None
    if basic is None or basic.empty:
        print(f'  [SKIP] {stock_code} 无法获取换手率，跳过')
        return None

    df = df.merge(basic, on='trade_date', how='left')
    df = df.rename(columns={'turnover_rate': 'turn'})
    df = df.sort_values('trade_date').reset_index(drop=True)

    # ---- 4.3 日期格式化 ----
    df['date'] = pd.to_datetime(
        df['trade_date'].astype(str), format='%Y%m%d'
    ).dt.strftime('%Y-%m-%d')
    df = df.rename(columns={'vol': 'volume'})

    # ---- 4.4 pct_chg 兜底 ----
    if 'pct_chg' not in df.columns:
        df['pct_chg'] = df['close'].pct_change() * 100

    # ---- 4.5 因果前向填充 ----
    fill_cols = ['open', 'high', 'low', 'close',
                 'volume', 'amount', 'pct_chg', 'turn']
    for col in fill_cols:
        if col in df.columns:
            df[col] = df[col].ffill()

    # ---- 4.6 因子（集中计算） ----
    df, factor_cols = compute_factors(df, ma_windows)

    # ---- 4.7 target：t 信号，t+entry 开盘进场，t+exit 收盘出场 ----
    df['target_start_date'] = (pd.to_datetime(df['date'])
                               .shift(-entry_offset)
                               .dt.strftime('%Y-%m-%d'))
    df['target_end_date'] = (pd.to_datetime(df['date'])
                             .shift(-exit_offset)
                             .dt.strftime('%Y-%m-%d'))
    entry_price = df['open'].shift(-entry_offset)
    exit_price = df['close'].shift(-exit_offset)
    valid = (entry_price.notna()
             & exit_price.notna()
             & (entry_price > 0))
    df['target'] = np.where(valid, exit_price / entry_price - 1.0, np.nan)
    df['stock_code'] = stock_code

    # ---- 4.8 保留列（基础 + 因子 + target） ----
    keep = (['date', 'target_start_date', 'target_end_date', 'stock_code']
            + FEATURE_COLS_BASE + factor_cols + ['target'])
    keep = [c for c in keep if c in df.columns]
    df = df[keep]

    # ---- 4.9 删除缺失行（required 自动包含所有因子） ----
    required = (['target', 'target_start_date', 'target_end_date']
                + [c for c in fill_cols if c in df.columns]
                + factor_cols)
    df = df.dropna(subset=required).reset_index(drop=True)
    return df


# ============================================================
# 5. 构建面板
# ============================================================
def build_panel(pro, memberships, start_date, end_date, logger,
                entry_offset, exit_offset, ma_windows):
    logger.log('\n' + '=' * 60)
    logger.log('[Step 2] 获取行情并附加逐日成分标记')
    logger.log('=' * 60)

    # ---- 5.1 股票池 ----
    stock_codes = sorted({m['stock_code'] for m in memberships})

    # ---- 5.2 按股票预分组区间，避免逐行扫全表 ----
    intervals_by_code = defaultdict(list)
    for m in memberships:
        intervals_by_code[m['stock_code']].append(
            (m['effective_start'], m['effective_end']))

    # ---- 5.3 逐股票抓取 ----
    all_data, failed = [], []
    t0 = time.time()

    for i, code in enumerate(stock_codes, 1):
        df = fetch_stock_data(pro, code, start_date, end_date,
                              entry_offset, exit_offset, ma_windows)
        if df is None:
            failed.append((code, '行情或换手率获取失败'))
            logger.log(f'  [{i}/{len(stock_codes)}] {code} 获取失败')
            continue
        if len(df) < 100:
            failed.append((code, f'长度不足 ({len(df)})'))
            logger.log(f'  [{i}/{len(stock_codes)}] {code} 长度不足 ({len(df)})')
            continue

        # ---- 5.4 逐日打成分标记 ----
        df['date_key'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d')
        df['is_constituent'] = False
        for lo, hi in intervals_by_code.get(code, []):
            mask = df['date_key'].between(lo, hi)
            df.loc[mask, 'is_constituent'] = True
        df = df.drop(columns=['date_key'])

        all_data.append(df)
        n_const = int(df['is_constituent'].sum())
        logger.log(f'  [{i}/{len(stock_codes)}] {code}: '
                   f'{len(df)} 行，成分期 {n_const} 行')

    logger.log(f'获取完成，耗时 {time.time() - t0:.1f}s，'
               f'成功 {len(all_data)} 只，失败 {len(failed)} 只')
    for code, reason in failed:
        logger.log(f'  失败: {code}: {reason}')
    if not all_data:
        raise RuntimeError('没有任何股票数据获取成功。')

    # ---- 5.5 拼接、排序、落盘 ----
    panel = pd.concat(all_data, ignore_index=True)
    panel = panel.sort_values(['date', 'stock_code']).reset_index(drop=True)
    panel.to_csv(PANEL_PATH, index=False)
    logger.log(f'面板 CSV 已保存: {PANEL_PATH}')
    return panel


# ============================================================
# 6. 校验报告
# ============================================================
def validate_and_report(panel, logger):
    logger.log('\n' + '=' * 60)
    logger.log('[Step 3] 数据校验与统计报告')
    logger.log('=' * 60)

    n_rows = len(panel)
    n_stocks = panel['stock_code'].nunique()
    n_const = int(panel['is_constituent'].sum())

    logger.log(f'总行数        : {n_rows}')
    logger.log(f'股票数        : {n_stocks}')
    logger.log(f'日期范围      : {panel["date"].min()} ~ {panel["date"].max()}')
    logger.log(f'列数          : {panel.shape[1]}')
    logger.log(f'列名          : {list(panel.columns)}')
    logger.log(f'成分期行数    : {n_const} ({n_const / n_rows:.1%})')

    # 缺失值
    logger.log('\n--- 缺失值 ---')
    na_counts = panel.isna().sum()
    any_na = False
    for col, cnt in na_counts.items():
        if cnt > 0:
            logger.log(f'  {col}: {cnt}')
            any_na = True
    if not any_na:
        logger.log('  无缺失值')

    # 每股票行数
    logger.log('\n--- 每股票 总行数 / 成分期行数 ---')
    g = panel.groupby('stock_code').agg(
        n_rows=('date', 'size'), n_const=('is_constituent', 'sum'))
    logger.log(f'  总行数  最小:{g.n_rows.min()} 最大:{g.n_rows.max()} '
               f'中位:{int(g.n_rows.median())}')
    logger.log(f'  成分行  最小:{g.n_const.min()} 最大:{g.n_const.max()} '
               f'中位:{int(g.n_const.median())}')

    # 分布
    logger.log('\n--- pct_chg 分布 ---')
    pc = panel['pct_chg']
    logger.log(f'  min={pc.min():.3f} max={pc.max():.3f} '
               f'mean={pc.mean():.3f} std={pc.std():.3f}')

    logger.log('\n--- 价格分布 ---')
    for col in ['open', 'high', 'low', 'close']:
        v = panel[col]
        logger.log(f'  {col}: min={v.min():.3f} max={v.max():.3f} '
                   f'mean={v.mean():.3f} std={v.std():.3f}')

    logger.log('\n--- target 分布 ---')
    tg = panel['target']
    logger.log(f'  min={tg.min():.3f} max={tg.max():.3f} '
               f'mean={tg.mean():.3f} std={tg.std():.3f}')

    # 样例
    logger.log('\n--- 前 5 行 ---')
    logger.log(panel.head().to_string())
    logger.log('\n--- 随机 5 行 ---')
    logger.log(panel.sample(min(5, len(panel)), random_state=42).to_string())

    # 健康检查
    logger.log('\n--- 健康检查 ---')
    checks = {
        '行数 > 10000': n_rows > 10000,
        '股票数 >= 50': n_stocks >= 50,
        '无缺失值': na_counts.sum() == 0,
        '成分期占比 10%~90%': 0.1 <= n_const / n_rows <= 0.9,
        'pct_chg 有方差': pc.std() > 0.1,
        'target 无 NaN': tg.isna().sum() == 0,
    }
    all_ok = True
    for k, v in checks.items():
        flag = 'PASS' if v else 'FAIL'
        logger.log(f'  [{flag}] {k}')
        all_ok = all_ok and v
    logger.log('\n全部通过。' if all_ok else '\n部分未通过，请检查。')
    return all_ok


# ============================================================
# 7. 主函数
# ============================================================
def main():
    global OUTPUT_DIR, PANEL_PATH

    parser = argparse.ArgumentParser(
        description='沪深300成分股面板数据获取（Tushare）')
    parser.add_argument('--token', default=DEFAULT_TOKEN,
                        help='Tushare Token，默认读环境变量 TUSHARE_TOKEN')
    parser.add_argument('--index_code', default='399300.SZ',
                        help='指数代码：399300.SZ 或 000300.SH（均为沪深300）')
    parser.add_argument('--start_date', default='20200101',
                        help='开始日期 YYYYMMDD')
    parser.add_argument('--end_date', default='20251231',
                        help='结束日期 YYYYMMDD')
    parser.add_argument('--entry_offset', type=int, default=1,
                        help='信号日后第几个交易日开盘进场（默认 1）')
    parser.add_argument('--exit_offset', type=int, default=10,
                        help='信号日后第几个交易日收盘出场，即预测步长（默认 10）')
    parser.add_argument('--ma_windows', type=int, nargs='+', default=[5, 20],
                        help='移动平均窗口，如 --ma_windows 5 10 20')
    parser.add_argument('--out_dir', default='./dataset/CSI300',
                        help='输出目录')
    args = parser.parse_args()

    # ---- 参数检查 ----
    if not args.token:
        print('ERROR: 未提供 Tushare Token（环境变量或 --token）')
        sys.exit(1)
    if args.entry_offset < 1:
        print(f'ERROR: entry_offset 必须 >=1，当前 {args.entry_offset}')
        sys.exit(1)
    if args.exit_offset <= args.entry_offset:
        print(f'ERROR: exit_offset({args.exit_offset}) 必须 > '
              f'entry_offset({args.entry_offset})')
        sys.exit(1)
    if not args.ma_windows or any(int(w) <= 0 for w in args.ma_windows):
        print('ERROR: ma_windows 必须为正整数列表')
        sys.exit(1)
    for name, val in [('start_date', args.start_date),
                      ('end_date', args.end_date)]:
        try:
            pd.to_datetime(val, format='%Y%m%d')
        except ValueError:
            print(f'ERROR: {name}={val} 不是 YYYYMMDD')
            sys.exit(1)
    if args.start_date >= args.end_date:
        print(f'ERROR: start_date 必须早于 end_date')
        sys.exit(1)

    # ---- 输出路径 ----
    OUTPUT_DIR = args.out_dir
    PANEL_PATH = os.path.join(OUTPUT_DIR, 'panel.csv')
    LOG_PATH = os.path.join(OUTPUT_DIR, 'fetch_log.txt')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger = Logger(LOG_PATH)
    try:
        ts.set_token(args.token)
        pro = ts.pro_api()

        # Step 1: 成分区间
        memberships = get_index_constituents(
            pro, args.index_code, args.start_date, args.end_date, logger)

        # Step 2: 行情 + 成分标记
        panel = build_panel(pro, memberships, args.start_date, args.end_date,
                            logger, args.entry_offset, args.exit_offset,
                            args.ma_windows)

        # Step 3: 校验
        validate_and_report(panel, logger)

        # fetch_meta.json：供训练侧校验 target 语义
        ma_cols = [f'ma{w}' for w in args.ma_windows]
        feature_cols = FEATURE_COLS_BASE + ma_cols
        meta = {
            'index_code': args.index_code,
            'start_date': args.start_date,
            'end_date': args.end_date,
            'entry_offset': args.entry_offset,
            'exit_offset': args.exit_offset,
            'ma_windows': list(args.ma_windows),
            'feature_cols': feature_cols,
            'eligibility_col': 'is_constituent',
            'target_definition': (
                f'close[t+{args.exit_offset}] / '
                f'open[t+{args.entry_offset}] - 1'),
            'fetched_at': datetime.now().isoformat(timespec='seconds'),
        }
        meta_path = os.path.join(OUTPUT_DIR, 'fetch_meta.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logger.log(f'fetch_meta 已保存: {meta_path}')
        logger.log(f'target 定义: {meta["target_definition"]}')

    finally:
        logger.close()


if __name__ == '__main__':
    main()