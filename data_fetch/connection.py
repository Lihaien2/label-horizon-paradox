"""统一数据连接：Tushare（【改动】替代原 DolphinDB）。

Token 集中在这里管理；上层只通过 ``pro()`` 获取 Tushare 接口实例。

用法：
    from data_fetch.connection import pro
    api = pro()                     # Tushare Pro API 实例
    df = api.daily(ts_code="000001.SZ", start_date="20200101", end_date="20201231")

Token 从环境变量 ``TUSHARE_TOKEN`` 读取；运行构建脚本前先设置：
    set TUSHARE_TOKEN=你的token   （Windows）
    或 export TUSHARE_TOKEN=你的token （Linux/macOS）

注意：本项目只需要沪深300（000300.SH）成分股，用到 daily/adj_factor/
index_weight/suspend_d/trade_cal/index_daily 等接口，2000 积分即可覆盖
（index_weight / adj_factor / suspend_d 均 2000 积分起）。
"""
from __future__ import annotations

import os
import threading

import tushare as ts

__all__ = ["pro", "TUSHARE_TOKEN", "check_token"]


def _read_token() -> str:
    """从环境变量读取 Tushare Token（可在调用前动态设置）。"""
    return os.environ.get("TUSHARE_TOKEN", "")


# 模块级缓存：避免反复 set_token（Tushare 的 token 只需设置一次）。
_lock = threading.Lock()
_token_set: str | None = None


def check_token() -> str:
    """读取并校验 Token；为空时抛错（给出清晰提示）。"""
    tok = _read_token()
    if not tok:
        raise RuntimeError(
            "未配置 Tushare Token。请设置环境变量 TUSHARE_TOKEN（2000积分及以上），"
            "或改为 set TUSHARE_TOKEN=你的token 后再运行。"
        )
    return tok


def pro() -> "ts.pro_api":
    """获取 Tushare Pro API 实例（线程安全，token 只设置一次）。

    返回的 ``pro`` 可直接调用 ``daily`` / ``adj_factor`` / ``index_weight`` 等接口。
    """
    global _token_set
    tok = check_token()
    with _lock:
        if _token_set != tok:
            ts.set_token(tok)
            _token_set = tok
    return ts.pro_api()
