# src/policy/policy.py
# -*- coding: utf-8 -*-
"""
N/M policy for deciding whether an op is "supported" on a target.

Usage:
    policy = NMPolicy.from_cfg(cfg)
    support = policy.decide_support(results)  # results: list[bool]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class NMPolicy:
    N: int = 10
    M: int = 2

    @classmethod
    def from_cfg(cls, cfg: Dict[str, Any]) -> "NMPolicy":
        pc = cfg.get("policy", {})
        return cls(
            N=int(pc.get("N", 10)),
            M=int(pc.get("M", 2)),
        )

    def decide_support(self, results: List[bool]) -> bool:
        """
        Args:
            results: list of booleans for one (target, op) pair.

        Returns:
            True if the op is considered supported (>= M successes),
            False otherwise.
        """
        if not results:
            return False
        ok_count = sum(1 for r in results if r)
        return ok_count >= self.M
