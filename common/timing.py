# -*- coding: utf-8 -*-
"""
common/timing.py
--------------------------------
通用计时工具，用于 v0/v1/v3 统计：
- 程序总时长（含准备阶段）
- 模拟操作时长（你控制开始/结束）
- 平均每个操作耗时（传入总操作数；可选分手势统计）
- v0 场景可从 monkey 的标准输出解析 "Events injected: N"

用法要点：
- 创建 Timing() 即启动“程序总时长”计时
- 在模拟操作开始处调用 .start_sim()
- 在模拟操作结束处调用 .stop_sim()
- 结束前调用 .set_ops_total(n)（或 .set_ops_by_kind({...})），然后打印 .summary_str()
"""

import time
import re
from typing import Optional, Dict, IO

_MONKEY_INJECTED_RE = re.compile(r"Events injected:\s*(\d+)", re.IGNORECASE)

class Timing:
    def __init__(self) -> None:
        self._t_prog_start = time.time()
        self._t_sim_start: Optional[float] = None
        self._t_sim_end: Optional[float] = None
        self._ops_total: Optional[int] = None
        self._ops_by_kind: Dict[str, int] = {}   # 可选：{"place":N, "drag":M, ...}

    # --- 时钟控制 ---
    def start_sim(self) -> None:
        if self._t_sim_start is None:
            self._t_sim_start = time.time()

    def stop_sim(self) -> None:
        if self._t_sim_start is not None and self._t_sim_end is None:
            self._t_sim_end = time.time()

    # --- 事件数设置 ---
    def set_ops_total(self, n: int) -> None:
        self._ops_total = max(0, int(n))

    def set_ops_by_kind(self, d: Dict[str, int]) -> None:
        # 允许只传有意义的 key，例如 {"drag": 120, "pinch": 0}
        clean = {}
        for k, v in (d or {}).items():
            try:
                clean[str(k)] = max(0, int(v))
            except Exception:
                pass
        self._ops_by_kind = clean
        # 若未设置总数，自动合计
        if self._ops_total is None:
            self._ops_total = sum(clean.values())

    # --- 从 monkey 输出解析注入事件数（v0 专用，可选） ---
    @staticmethod
    def parse_monkey_injected_from_stream(stream: IO[str]) -> Optional[int]:
        """
        读取 monkey 的 stdout（文本流），找到 'Events injected: N'。
        若未找到返回 None。你可以一边读取一边把每行丢给本函数解析。
        """
        for line in stream:
            m = _MONKEY_INJECTED_RE.search(line)
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    pass
        return None

    # --- 统计取值 ---
    def total_runtime(self) -> float:
        return time.time() - self._t_prog_start

    def sim_runtime(self) -> float:
        if self._t_sim_start is None:
            return 0.0
        end = self._t_sim_end if self._t_sim_end is not None else time.time()
        return max(0.0, end - self._t_sim_start)

    def avg_time_per_op(self) -> float:
        ops = (self._ops_total or 0)
        return 0.0 if ops <= 0 else self.sim_runtime() / float(ops)

    def avg_time_per_kind(self) -> Dict[str, float]:
        if not self._ops_by_kind:
            return {}
        rt = self.sim_runtime()
        return {k: (0.0 if v <= 0 else rt / float(v)) for k, v in self._ops_by_kind.items()}

    # --- 输出 ---
    def summary_dict(self) -> Dict[str, object]:
        out = {
            "total_runtime_sec": round(self.total_runtime(), 6),
            "sim_runtime_sec": round(self.sim_runtime(), 6),
            "ops_total": int(self._ops_total or 0),
            "avg_time_per_op_sec": round(self.avg_time_per_op(), 6),
        }
        if self._ops_by_kind:
            out["ops_by_kind"] = {k: int(v) for k, v in self._ops_by_kind.items()}
            out["avg_time_per_kind_sec"] = {k: round(v, 6) for k, v in self.avg_time_per_kind().items()}
        return out

    def summary_str(self) -> str:
        d = self.summary_dict()
        lines = []
        lines.append("[timing]")
        lines.append("  total_runtime_sec       = {:.3f}".format(d["total_runtime_sec"]))
        lines.append("  sim_runtime_sec         = {:.3f}".format(d["sim_runtime_sec"]))
        lines.append("  ops_total               = {}".format(d["ops_total"]))
        lines.append("  avg_time_per_op_sec     = {:.6f}".format(d["avg_time_per_op_sec"]))
        if "ops_by_kind" in d:
            lines.append("  ops_by_kind             = {}".format(d["ops_by_kind"]))
            lines.append("  avg_time_per_kind_sec   = {}".format(d["avg_time_per_kind_sec"]))
        return "\n".join(lines)
