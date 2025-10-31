# -*- coding: utf-8 -*-
"""
common/verify_motion.py
--------------------------------
- 解析 TAG=AR_OP 的单行 JSON（place/drag/pinch/rotate/tap）
- 计算 per-kind 与 overall 成功率
- overall 的口径可配置：overall_basis ∈ {"place","tap"}（默认 "place"）
  * "place": overall = place + drag + pinch + rotate（tap 不计，避免 place+tap 双计）
  * "tap":   overall = tap   + drag + pinch + rotate（place 完全当作 tap 的特例，不计入 overall）

注：place 在 app 里发生时通常也会有一条 tap；若选 "tap" 作为口径，place 仍单独打印，但不会进 overall。
"""

import json
import re
import time
import threading
import subprocess
from collections import Counter
from typing import Optional, Dict, Any, List

TAG = "AR_OP"
JSON_RE = re.compile(r'\bAR_OP\b[^{]*({.*})')  # 兼容 "AR_OP   :" / "AR_OP(1234):" 等

KINDS_ALL = ("place_start", "place_ok", "place_fail", "drag", "pinch", "rotate", "tap")

def _safe_rate(ok: int, total: int) -> float:
    return 0.0 if total <= 0 else ok / float(total)

def parse_line(line: str) -> Optional[Dict[str, Any]]:
    if "AR_OP" not in line:
        return None
    m = JSON_RE.search(line)
    if not m:
        return None
    try:
        evt = json.loads(m.group(1))
        k = evt.get("kind")
        if isinstance(k, str):
            evt["kind"] = k.strip().lower()
        return evt
    except Exception:
        return None

class MotionStats:
    """线程安全计数与成功率。"""
    def __init__(self, overall_basis: str = "place") -> None:
        self._cnt = Counter()
        self._t0 = time.time()
        self._lock = threading.Lock()
        self._pending_place = 0
        self._overall_basis = overall_basis if overall_basis in ("place","tap") else "place"

    def set_overall_basis(self, basis: str) -> None:
        if basis not in ("place","tap"):
            basis = "place"
        with self._lock:
            self._overall_basis = basis

    def reset(self) -> None:
        with self._lock:
            self._cnt.clear()
            self._t0 = time.time()
            self._pending_place = 0

    def feed_event(self, evt: Dict[str, Any]) -> None:
        kind = str(evt.get("kind", "")).strip().lower()
        ok = bool(evt.get("ok", False))
        if kind not in KINDS_ALL:
            return
        with self._lock:
            if kind == "place_start":
                self._cnt["place_start_total"] += 1
                self._pending_place += 1
                return
            if kind == "place_ok":
                if self._pending_place > 0:
                    self._cnt["place_ok_total"] += 1
                    self._pending_place -= 1
                return
            if kind == "place_fail":
                if self._pending_place > 0:
                    self._pending_place -= 1
                self._cnt["place_fail_total"] += 1
                return
            # 其余：drag/pinch/rotate/tap
            self._cnt[f"{kind}_total"] += 1
            if ok:
                self._cnt[f"{kind}_ok"] += 1

    def feed_line(self, line: str) -> None:
        evt = parse_line(line)
        if evt:
            self.feed_event(evt)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            c = dict(self._cnt)
            elapsed = time.time() - self._t0
            basis = self._overall_basis

        place_ok  = c.get("place_ok_total", 0)
        place_try = c.get("place_start_total", 0)
        drag_ok, drag_total = c.get("drag_ok", 0), c.get("drag_total", 0)
        pin_ok,  pin_total  = c.get("pinch_ok", 0), c.get("pinch_total", 0)
        rot_ok,  rot_total  = c.get("rotate_ok", 0), c.get("rotate_total", 0)
        tap_ok,  tap_total  = c.get("tap_ok", 0),  c.get("tap_total", 0)

        if basis == "tap":
            ok_sum  = tap_ok + drag_ok + pin_ok + rot_ok
            try_sum = tap_total + drag_total + pin_total + rot_total
        else:  # "place"
            ok_sum  = place_ok + drag_ok + pin_ok + rot_ok
            try_sum = place_try + drag_total + pin_total + rot_total

        return {
            "elapsed_sec": elapsed,
            "counts": c,
            "place":   {"ok": place_ok, "total": place_try, "rate": _safe_rate(place_ok, place_try)},
            "drag":    {"ok": drag_ok,  "total": drag_total, "rate": _safe_rate(drag_ok, drag_total)},
            "pinch":   {"ok": pin_ok,   "total": pin_total,  "rate": _safe_rate(pin_ok, pin_total)},
            "rotate":  {"ok": rot_ok,   "total": rot_total,  "rate": _safe_rate(rot_ok, rot_total)},
            "tap":     {"ok": tap_ok,   "total": tap_total,  "rate": _safe_rate(tap_ok, tap_total)},
            "overall": {"ok": ok_sum,   "total": try_sum,    "rate": _safe_rate(ok_sum, try_sum),
                        "basis": basis},
        }

    def summary_str(self) -> str:
        s = self.snapshot()
        lines = []
        lines.append(f"[verify_motion] elapsed={s['elapsed_sec']:.1f}s")
        lines.append("--------------------------------------------------------")
        lines.append(f"{'place':6s}  ok={s['place']['ok']:4d}  total={s['place']['total']:4d}  succ={s['place']['rate']:6.3f}")
        lines.append(f"{'drag':6s}  ok={s['drag']['ok']:4d}  total={s['drag']['total']:4d}  succ={s['drag']['rate']:6.3f}")
        lines.append(f"{'pinch':6s}  ok={s['pinch']['ok']:4d}  total={s['pinch']['total']:4d}  succ={s['pinch']['rate']:6.3f}")
        lines.append(f"{'rotate':6s} ok={s['rotate']['ok']:4d}  total={s['rotate']['total']:4d}  succ={s['rotate']['rate']:6.3f}")
        lines.append(f"{'tap':6s}   ok={s['tap']['ok']:4d}  total={s['tap']['total']:4d}  succ={s['tap']['rate']:6.3f}")
        lines.append("--------------------------------------------------------")
        lines.append(f"overall(basis={s['overall']['basis']})  ok={s['overall']['ok']:4d}  total={s['overall']['total']:4d}  succ={s['overall']['rate']:6.3f}")
        return "\n".join(lines)

class _LogcatThread(threading.Thread):
    def __init__(self, stats: MotionStats, serial: Optional[str] = None, extra_args: Optional[List[str]] = None):
        super().__init__(daemon=True)
        self._stats = stats
        self._serial = serial
        self._extra = extra_args or []
        self._proc: Optional[subprocess.Popen] = None
        self._stop = threading.Event()

    def run(self):
        cmd = ["adb"]
        if self._serial:
            cmd += ["-s", self._serial]
        cmd += ["logcat", "-s", f"{TAG}:D"] + self._extra
        try:
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        except FileNotFoundError:
            return
        try:
            assert self._proc.stdout is not None
            for line in self._proc.stdout:
                if self._stop.is_set():
                    break
                self._stats.feed_line(line)
        finally:
            self._terminate()

    def stop(self):
        self._stop.set()
        self._terminate()

    def _terminate(self):
        try:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
        except Exception:
            pass

class MotionVerifier:
    def __init__(self, overall_basis: str = "place") -> None:
        self.stats = MotionStats(overall_basis=overall_basis)
        self._th: Optional[_LogcatThread] = None

    def set_overall_basis(self, basis: str) -> None:
        self.stats.set_overall_basis(basis)

    def feed_line(self, line: str) -> None:
        self.stats.feed_line(line)

    def feed_event(self, evt: Dict[str, Any]) -> None:
        self.stats.feed_event(evt)

    def start_from_adb(self, serial: Optional[str] = None, extra_logcat_args: Optional[List[str]] = None) -> None:
        self.stop()
        self._th = _LogcatThread(self.stats, serial=serial, extra_args=extra_logcat_args)
        self._th.start()

    def stop(self) -> None:
        if self._th is not None:
            self._th.stop()
            self._th = None

    def reset(self) -> None:
        self.stats.reset()

    def snapshot(self) -> Dict[str, Any]:
        return self.stats.snapshot()

    def summary_str(self) -> str:
        return self.stats.summary_str()
