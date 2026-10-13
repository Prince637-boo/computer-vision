import math
from typing import Optional


class OneEuro:
    """Adaptive low-pass filter with lower lag during fast movement."""

    def __init__(
        self, min_cutoff: float = 1.2, beta: float = 0.02, d_cutoff: float = 1.0
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x: Optional[float] = None
        self.dx: Optional[float] = None
        self.t: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float) -> float:
        if self.x is None or self.dx is None or self.t is None:
            self.x, self.dx, self.t = x, 0.0, t
            return x

        dt = max(t - self.t, 1e-3)
        derivative = (x - self.x) / dt
        self.dx += self._alpha(self.d_cutoff, dt) * (derivative - self.dx)
        cutoff = self.min_cutoff + self.beta * abs(self.dx)
        self.x += self._alpha(cutoff, dt) * (x - self.x)
        self.t = t
        return self.x

    def reset(self) -> None:
        self.x = None
        self.dx = None
        self.t = None
