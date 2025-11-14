# policy/policy.py
from typing import Dict, Any, List

class NMPolicy:
    def __init__(self, N: int = 10, M: int = 2, rng=None):
        self.N = N
        self.M = M
        self.rng = rng

    def decide_support(self, op_type: str, trial_results: List[bool]) -> bool:
        return sum(trial_results) >= self.M

    def sample_params(self, op_type: str) -> Dict[str, Any]:
        """
        统一随机化操作参数的入口：
          tap:   press_ms, jitter半径
          drag:  方向(任意角)、长度、速度、多指偏移
          rotate:角度(正/负)、半径、双指间距/速度
        """
        # TODO: 使用 self.rng 生成可重复的随机
        return {}
