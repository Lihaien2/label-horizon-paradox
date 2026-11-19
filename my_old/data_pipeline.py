"""
沪深300（CSI300）多股票面板数据 —— 预处理 + 特征工程 + DataLoader 统一入口。

输入文件（由 utils/fetch_csi300.py 生成，目录如 ./dataset/CSI300）：
    panel.csv        多股票面板

输出样本（每个样本）：
    seq_x      : (seq_len, n_features)   窗口内标准化后的特征
    seq_y      : (pred_len, 1)           标准化后的 target
    seq_x_mark : (seq_len, 4)            时间特征 [月, 日, 周几, 年内天数] 归一化
    seq_y_mark : (pred_len, 4)           同上，预测期的时间特征
    sid        : int                     股票 ID（仅 val/test 返回）

切分语义（只看 signal_date）：
    train: signal_date <= train_end
    val:   train_end < signal_date <= val_end
    test:  val_end   < signal_date <= test_end

加因子：只改 utils/fetch_csi300.py 的 compute_factors，重跑 fetch，
        本文件通过 fetch_meta.feature_cols 自动跟随。
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from utils.timefeatures import time_features


# ============================================================
# 0. 列/键定义
# ============================================================
# panel.csv 必须包含的列（其余由 fetch_meta 动态决定）
PANEL_REQUIRED_COLS = {
    'date', 'target_start_date', 'target_end_date',
    'stock_code', 'target', 'is_constituent',
}

# fetch_meta.json 必须包含的键（用于交叉校验）
META_REQUIRED_KEYS = {'entry_offset', 'exit_offset', 'feature_cols'}


# ============================================================
# 1. fetch_meta 加载与校验
# ============================================================
def load_fetch_meta(panel_path, args):
    """
    读取并校验 fetch_meta.json。

    校验项：
      - entry_offset / exit_offset 与 args 是否一致（args 为 None 则跳过）
      - feature_cols 中 args 要求的列是否都在 meta 里
    返回 meta dict；文件不存在时只 WARN，不中断。
    """
    # panel.csv 同目录下的 fetch_meta.json
    meta_path = os.path.join(os.path.dirname(panel_path), 'fetch_meta.json')

    # 文件缺失：向后兼容旧 panel
    if not os.path.exists(meta_path):
        print(f'[Dataset][WARN] 未找到 {meta_path}，'
              '跳过 target 语义一致性校验（建议补跑 fetch）')
        return None

    # 读入
    with open(meta_path, encoding='utf-8') as f:
        meta = json.load(f)

    # 必须键检查
    missing = META_REQUIRED_KEYS - set(meta.keys())
    if missing:
        raise ValueError(f'fetch_meta.json 缺少键: {sorted(missing)}')

    # entry/exit_offset 交叉校验（args 为 None 则跳过）
    for k in ('entry_offset', 'exit_offset'):
        arg_val = getattr(args, k, None)
        if arg_val is not None and int(arg_val) != int(meta[k]):
            raise ValueError(
                f'fetch/train 的 {k} 不一致：fetch={meta[k]}, train={arg_val}。'
                'target 语义会与回测假设错位，请统一后重跑 fetch。')

    # feature_cols 交叉校验
    arg_feats = getattr(args, 'feature_cols', None)
    if arg_feats:
        meta_feats = list(meta['feature_cols'])
        missing_feats = [c for c in arg_feats if c not in meta_feats]
        if missing_feats:
            raise ValueError(
                f'args.feature_cols 中 {missing_feats} 不在 fetch_meta.feature_cols '
                f'({meta_feats})；请重跑 fetch 或修正 args。')

    print(f'[Dataset] fetch_meta 校验通过: {meta.get("target_definition")}')
    return meta


# ============================================================
# 2. Dataset
# ============================================================
class Dataset_MultiStock(Dataset):
    """沪深300 多股票面板数据集。"""

    def __init__(self, args, root_path, data_path='panel.csv', flag='train',
                 size=None, features='MS', target='target',
                 scale=True, timeenc=0, freq='d',
                 stock_code_col='stock_code', window_stride=None):
        # ---- 基本参数 ----
        self.args = args
        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag
        if flag not in ('train', 'val', 'test'):
            raise ValueError(f'不支持的 flag: {flag}')
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]

        # ---- seq_len / label_len / pred_len ----
        if size is None:
            self.seq_len, self.label_len, self.pred_len = 60, 0, 1
        else:
            self.seq_len, self.label_len, self.pred_len = size

        # ---- 模型/任务参数 ----
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.stock_code_col = stock_code_col

        # ---- window_stride：优先参数，其次 args，最后 exit_offset ----
        if window_stride is None:
            window_stride = getattr(args, 'window_stride', None)
        if window_stride is None:
            window_stride = getattr(args, 'exit_offset', 10) or 10
        self.window_stride = max(1, int(window_stride))

        # ---- 切分日期 ----
        self.train_end = pd.Timestamp(self._get_arg('train_end', '2024-01-31'))
        self.val_end = pd.Timestamp(self._get_arg('val_end', '2024-08-31'))
        self.test_end = pd.Timestamp(self._get_arg('test_end', '2025-12-31'))

        # ---- 其他可选项 ----
        self.scale_target = bool(self._get_arg('scale_target', True))
        self.eligibility_col = self._get_arg('eligibility_col', 'is_constituent')

        # ---- multi_stock 约束 ----
        if self.label_len != 0:
            raise ValueError('Dataset_MultiStock 只支持 label_len=0')

        # ---- 进入主流程 ----
        self.__read_data__()

    # ============================================================
    # 2.1 工具方法
    # ============================================================
    def _get_arg(self, name, default):
        """安全读取 args 属性，缺失时返回 default。"""
        return getattr(self.args, name, default)

    @staticmethod
    def _is_true(value):
        """把 bool / str / 1 / 0 统一转成 bool。"""
        if isinstance(value, str):
            return value.strip().lower() in ('true', '1', 'yes', 'y')
        return bool(value) if pd.notna(value) else False

    def _make_stamps(self, dates):
        """
        生成时间特征矩阵，形状 (n, 4)。
        timeenc=0：手动构造 [month, day, weekday, day_of_year] 归一化
        timeenc=1：调 time_features（freq 决定维度）
        """
        dates_dt = pd.DatetimeIndex(pd.to_datetime(dates))

        if self.timeenc == 0:
            stamps = np.stack([
                np.asarray(dates_dt.month) / 12.0,
                np.asarray(dates_dt.day) / 31.0,
                np.asarray(dates_dt.weekday) / 6.0,
                np.asarray(dates_dt.dayofyear) / 366.0,
            ], axis=1).astype(np.float32)
        else:
            stamps = time_features(dates_dt, freq=self.freq).transpose(1, 0)
            stamps = stamps.astype(np.float32)
        return stamps

    # ============================================================
    # 2.2 读 CSV + 基础校验
    # ============================================================
    def _load_dataframe(self):
        """读 panel.csv，校验列，读 fetch_meta，确定 feature_cols。"""
        panel_path = os.path.join(self.root_path, self.data_path)

        # ---- 读 CSV ----
        df = pd.read_csv(panel_path)

        # ---- 必须列检查 ----
        missing = PANEL_REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(
                f'panel.csv 缺少必要列: {sorted(missing)}。'
                '请重新运行 utils/fetch_csi300.py。')

        # ---- 读 fetch_meta 并交叉校验 ----
        meta = load_fetch_meta(panel_path, self.args)

        # ---- 确定 feature_cols：args 优先，其次 meta ----
        arg_feats = getattr(self.args, 'feature_cols', None)
        if arg_feats:
            self.feature_cols = list(arg_feats)
        elif meta and meta.get('feature_cols'):
            self.feature_cols = list(meta['feature_cols'])
        else:
            raise ValueError('未找到 feature_cols（既不在 args 也不在 fetch_meta）')

        # ---- 特征列必须存在 ----
        miss_feat = [c for c in self.feature_cols if c not in df.columns]
        if miss_feat:
            raise ValueError(f'feature_cols 中以下列不在 panel.csv: {miss_feat}')

        self.n_features = len(self.feature_cols)
        self.scale_cols = list(self.feature_cols)
        return df

    def _parse_dates(self, df):
        """解析日期列，丢掉非法行，检查日期逻辑。"""
        df = df.copy()
        for col in ('date', 'target_start_date', 'target_end_date'):
            df[col] = pd.to_datetime(df[col], errors='coerce')

        # 三个日期或 target 任一缺失 → 无效样本
        df = df.dropna(
            subset=['date', 'target_start_date', 'target_end_date', self.target])

        # 日期逻辑：signal < start <= end
        invalid = (
            (df['target_start_date'] <= df['date'])
            | (df['target_end_date'] < df['target_start_date'])
        )
        if invalid.any():
            ex = df.loc[invalid, [self.stock_code_col, 'date',
                                  'target_start_date', 'target_end_date']].head(5)
            raise ValueError(
                '发现不合法的目标日期（要求 signal_date < target_start_date '
                '<= target_end_date）：\n' + ex.to_string(index=False))

        # 按股票 + 日期排序
        df = df.sort_values([self.stock_code_col, 'date']).reset_index(drop=True)
        return df

    # ============================================================
    # 2.3 全局交易日历与决策日
    # ============================================================
    def _build_calendar(self, df):
        """
        构造全局交易日历（所有股票日期去重排序）并派生决策日。
        决策日 = 从 seq_len-1 起、每 window_stride 取一个。
        """
        all_dates = pd.DatetimeIndex(sorted(df['date'].unique()))
        if len(all_dates) < self.seq_len:
            raise ValueError(
                f'全局交易日数 {len(all_dates)} < seq_len {self.seq_len}，'
                '无法构造窗口')

        # 决策日：保证每只股票至少有 seq_len 个全局历史观测
        decision_dates = all_dates[self.seq_len - 1::self.window_stride]
        self.decision_date_set = set(pd.Timestamp(d) for d in decision_dates)

        print(f'[Dataset] 全局交易日数={len(all_dates)}, '
              f'决策日数={len(decision_dates)}, '
              f'stride={self.window_stride}')
        print(f'[Dataset] 切分（按 signal_date）: '
              f'train<= {self.train_end.date()}, '
              f'val({self.train_end.date()}, {self.val_end.date()}], '
              f'test({self.val_end.date()}, {self.test_end.date()}]')
        return all_dates

    # ============================================================
    # 2.4 split 判断（只看 signal_date，无 purge）
    # ============================================================
    def _is_in_split(self, signal_date):
        """判断 signal_date 是否属于当前 split。"""
        if self.set_type == 0:      # train
            return signal_date <= self.train_end
        elif self.set_type == 1:    # val
            return self.train_end < signal_date <= self.val_end
        else:                       # test
            return self.val_end < signal_date <= self.test_end

    # ============================================================
    # 2.5 拟合 scalers
    # ============================================================
    def _fit_feature_scaler(self, df):
        """
        feature scaler：只用训练期所有行拟合。
        窗口内可含训练期非成分行（合法历史上下文），故拟合也用全部行。
        """
        train_rows = df[df['date'] <= self.train_end]
        if train_rows.empty:
            raise ValueError('训练期没有任何行，无法拟合 feature scaler')

        self.scaler = StandardScaler()
        if self.scale:
            self.scaler.fit(train_rows[self.scale_cols].astype(float).values)
            print(f'[Dataset] feature scaler 拟合: n_rows={len(train_rows)}, '
                  f'n_features={self.n_features}')
        else:
            print('[Dataset] 跳过 feature scaler（scale=False）')

    def _fit_target_scaler(self, df):
        """
        target scaler：只用实际会进入训练集的样本的 target 拟合。
        条件与 _generate_samples 完全一致（决策日 + 成分 + split）。
        """
        if not self.scale_target:
            self.target_mean, self.target_std = None, None
            print('[Dataset] 跳过 target scaler（scale_target=False）')
            return

        actual_targets = []

        # ---- 逐股票扫描 ----
        for _, g in df.groupby(self.stock_code_col, sort=False):
            g = g.sort_values('date').reset_index(drop=True)
            if len(g) < self.seq_len:
                continue

            dates_arr = g['date'].values
            consts_arr = g[self.eligibility_col].values
            y_arr = g[self.target].astype(float).values

            for i in range(self.seq_len - 1, len(g)):
                signal_date = pd.Timestamp(dates_arr[i])

                # 1) 必须是决策日
                if signal_date not in self.decision_date_set:
                    continue
                # 2) 必须是成分
                if not self._is_true(consts_arr[i]):
                    continue
                # 3) 必须属于当前 split
                if not self._is_in_split(signal_date):
                    continue

                actual_targets.append(y_arr[i])

        if not actual_targets:
            raise ValueError(
                '按实际训练样本筛选后没有可用 target，无法拟合 target scaler')

        t = np.asarray(actual_targets, dtype=np.float64)
        self.target_mean = float(np.mean(t))
        std = float(np.std(t, ddof=0))
        self.target_std = std if std > 0 else 1.0
        print(f'[Dataset] target scaler 拟合: n_samples={len(t)}, '
              f'mean={self.target_mean:.8f}, std={self.target_std:.8f}')

    # ============================================================
    # 2.6 生成样本
    # ============================================================
    def _generate_samples(self, df):
        """逐股票生成属于当前 split 的样本。"""
        self.data_x = []
        self.data_y = []
        self.data_stamp = []
        self.data_sid = []
        self.data_meta = []

        # ---- 股票代码 → 整数 ID（排序保证确定性） ----
        codes = sorted(df[self.stock_code_col].astype(str).unique())
        self.code_to_id = {c: i for i, c in enumerate(codes)}

        # ---- 逐股票处理 ----
        for code, g in df.groupby(self.stock_code_col, sort=False):
            g = g.sort_values('date').reset_index(drop=True)
            if len(g) < self.seq_len:
                continue

            # ---- 整只股票预处理 ----
            raw_x = g[self.feature_cols].astype(float).values
            x_all = self.scaler.transform(raw_x) if self.scale else raw_x
            stamps = self._make_stamps(g['date'].values)

            dates_arr = g['date'].values
            starts_arr = g['target_start_date'].values
            ends_arr = g['target_end_date'].values
            consts_arr = g[self.eligibility_col].values
            y_all = g[self.target].astype(float).values

            # ---- 遍历所有候选窗口终点 ----
            for i in range(self.seq_len - 1, len(g)):
                signal_date = pd.Timestamp(dates_arr[i])

                # 1) 必须是决策日
                if signal_date not in self.decision_date_set:
                    continue
                # 2) 必须是成分
                if not self._is_true(consts_arr[i]):
                    continue
                # 3) 必须属于当前 split
                if not self._is_in_split(signal_date):
                    continue

                # ---- 生成样本 ----
                x = x_all[i - self.seq_len + 1:i + 1]
                y = y_all[i]
                if self.target_mean is not None:
                    y = (y - self.target_mean) / self.target_std

                self.data_x.append(x.astype(np.float32))
                self.data_y.append(np.array([y], dtype=np.float32))
                self.data_stamp.append((
                    stamps[i - self.seq_len + 1:i + 1],
                    stamps[i:i + 1],
                ))
                self.data_sid.append(self.code_to_id[str(code)])
                self.data_meta.append({
                    'date': signal_date.strftime('%Y-%m-%d'),
                    'target_start_date': pd.Timestamp(starts_arr[i]).strftime('%Y-%m-%d'),
                    'target_end_date': pd.Timestamp(ends_arr[i]).strftime('%Y-%m-%d'),
                    'stock_code': str(code),
                })

        # ---- 空检查 ----
        if not self.data_x:
            raise ValueError(
                f'{("train", "val", "test")[self.set_type]} 分区没有任何样本；'
                '检查切分日期、seq_len、is_constituent。')

        # ---- 转 numpy ----
        self.data_x = np.asarray(self.data_x, dtype=np.float32).reshape(
            -1, self.seq_len, self.n_features)
        self.data_y = np.asarray(self.data_y, dtype=np.float32).reshape(-1, 1)
        self.data_sid = np.asarray(self.data_sid, dtype=np.int64)

    # ============================================================
    # 2.7 审计
    # ============================================================
    def _audit(self):
        """打印样本统计并校验 signal_date 未跨 split。"""
        audit = pd.DataFrame(self.data_meta)
        print(f'[Audit] split={("train", "val", "test")[self.set_type]}, '
              f'samples={len(audit)}')

        if len(audit):
            print(f'  date range: {audit["date"].min()} ~ {audit["date"].max()}')
            print(f'  n_stocks: {audit["stock_code"].nunique()}, '
                  f'n_dates: {audit["date"].nunique()}')

        # ---- signal_date 越界检查 ----
        audit['date'] = pd.to_datetime(audit['date'])
        if self.set_type == 0:
            bad = audit[audit['date'] > self.train_end]
        elif self.set_type == 1:
            bad = audit[(audit['date'] <= self.train_end)
                        | (audit['date'] > self.val_end)]
        else:
            bad = audit[(audit['date'] <= self.val_end)
                        | (audit['date'] > self.test_end)]
        if len(bad):
            print(f'[Audit][ERROR] {len(bad)} 条样本 signal_date 跨 split')
            print(bad.head(5).to_string(index=False))
            raise ValueError('样本 signal_date 跨 split 边界；停止。')

        # ---- 数据分布 ----
        print(f'[Dataset] data_x mean={self.data_x.mean():.4f}, '
              f'std={self.data_x.std():.4f}, '
              f'max_abs={np.abs(self.data_x).max():.2f}')
        print(f'[Dataset] data_y mean={self.data_y.mean():.4f}, '
              f'std={self.data_y.std():.4f}')

    # ============================================================
    # 2.8 主流程
    # ============================================================
    def __read_data__(self):
        # 1) 读 CSV + 校验列
        df = self._load_dataframe()
        # 2) 解析日期
        df = self._parse_dates(df)
        # 3) 全局日历 + 决策日
        self.all_dates = self._build_calendar(df)
        # 4) 拟合 feature scaler（训练期所有行）
        self._fit_feature_scaler(df)
        # 5) 拟合 target scaler（实际训练样本）
        self._fit_target_scaler(df)
        # 6) 生成当前 split 样本
        self._generate_samples(df)
        # 7) 审计
        self._audit()

    # ============================================================
    # 2.9 样本访问
    # ============================================================
    def __getitem__(self, index):
        x = self.data_x[index]
        # 保证 y 是 [pred_len, 1]
        y = self.data_y[index].reshape(self.pred_len, -1)
        xm, ym = self.data_stamp[index]

        # train 返回 4 元组；val/test 返回 5 元组（带 sid）
        if self.set_type == 0:
            return x, y, xm, ym
        return x, y, xm, ym, self.data_sid[index]

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        """反标准化 target。"""
        if self.target_mean is None:
            return data
        return data * self.target_std + self.target_mean


# ============================================================
# 3. collate：支持 4/5 元组
# ============================================================
def collate_multi_stock(batch):
    """把 list of 4/5 元组 batch 转成 tensor batch。"""
    n = len(batch[0])

    # 前 4 项总是存在
    xs, ys, xms, yms = zip(*[b[:4] for b in batch])
    x = torch.stack([torch.tensor(v) for v in xs])
    y = torch.stack([torch.tensor(v) for v in ys])
    xm = torch.stack([torch.tensor(v) for v in xms])
    ym = torch.stack([torch.tensor(v) for v in yms])

    if n == 4:
        return x, y, xm, ym
    if n == 5:
        sids = torch.stack([torch.tensor(b[4], dtype=torch.long) for b in batch])
        return x, y, xm, ym, sids
    raise ValueError(f'Unsupported batch tuple length: {n}')


# ============================================================
# 4. 统一入口
# ============================================================
def data_provider(args, flag):
    """multi_stock 数据入口。返回 (dataset, dataloader)。"""
    # timeenc 由 embed 参数决定
    timeenc = 0 if args.embed != 'timeF' else 1

    # 训练集 shuffle；val/test 不 shuffle（保证评估顺序可对齐）
    shuffle = (flag == 'train')

    # 构造 Dataset
    dataset = Dataset_MultiStock(
        args=args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        scale=True,
        timeenc=timeenc,
        freq=args.freq,
        stock_code_col=getattr(args, 'stock_code_col', 'stock_code'),
        window_stride=getattr(args, 'window_stride', None),
    )

    # 构造 DataLoader
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_multi_stock,
    )

    print(f'[{flag}] samples={len(dataset)}, n_features={dataset.n_features}, '
          f'seq_len={args.seq_len}, pred_len={args.pred_len}')

    return dataset, loader