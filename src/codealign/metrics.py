from __future__ import annotations

import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from the Codex paper (Chen et al., 2021).

    Given n samples of which c are correct, the probability that at least
    one of k randomly drawn samples is correct: 1 - C(n-c, k) / C(n, k).
    """
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))
