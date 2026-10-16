"""Reliability calibration for noisy simulated XOR APUFs."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from pypuf.simulation.base import LTFArray


def latent_value(puf, challenges: np.ndarray) -> np.ndarray:
    """Evaluate an LTF array without dispatching to its noisy ``ltf_eval``."""
    transformed = puf.transform(challenges, puf.weight_array.shape[0])
    chain_values = LTFArray.ltf_eval(puf, transformed)
    return np.asarray(puf.combiner(chain_values))


def latent_response(puf, challenges: np.ndarray) -> np.ndarray:
    """Return the deterministic polar response of a noisy-capable PUF."""
    return np.sign(latent_value(puf, challenges))


def initial_noisiness_for_ber(k: int, target_ber: float) -> float:
    """Invert the independent Gaussian chain-noise approximation.

    PyPUF adds Gaussian noise with standard deviation ``sqrt(n) * noisiness``
    to each chain score. The expression is exact for independent, unit-variance
    Gaussian chain scores and is used only to initialize empirical calibration.
    """
    if k < 1 or not 0 <= target_ber < 0.5:
        raise ValueError("Require k >= 1 and target_ber in [0, 0.5).")
    if target_ber == 0:
        return 0.0
    chain_ber = (1 - (1 - 2 * target_ber) ** (1 / k)) / 2
    return float(np.tan(np.pi * chain_ber))


def calibrate_noisiness(
    factory: Callable[[float], object],
    challenges: np.ndarray,
    target_ber: float,
    *,
    iterations: int = 12,
) -> tuple[float, float]:
    """Calibrate PyPUF noisiness against output BER on fixed challenges.

    ``factory`` must return a fresh PUF with deterministic weight and noise
    seeds for each candidate. Returning a fresh object makes each comparison
    use common random numbers and prevents calibration from consuming the
    random stream used by the actual experiment.
    """
    if target_ber == 0:
        return 0.0, 0.0
    if iterations < 1:
        raise ValueError("Calibration iterations must be positive.")

    latent = latent_response(factory(0.0), challenges)

    def measured(candidate: float) -> float:
        observed = np.asarray(factory(candidate).eval(challenges))
        return float(np.mean(observed != latent))

    initial = initial_noisiness_for_ber(factory(0.0).k, target_ber)
    low, high = 0.0, max(initial * 2, 1e-4)
    high_ber = measured(high)
    while high_ber < target_ber and high < 8:
        low, high = high, high * 2
        high_ber = measured(high)

    candidates = [(0.0, 0.0), (high, high_ber)]
    for _ in range(iterations):
        midpoint = (low + high) / 2
        midpoint_ber = measured(midpoint)
        candidates.append((midpoint, midpoint_ber))
        if midpoint_ber < target_ber:
            low = midpoint
        else:
            high = midpoint
    return min(candidates, key=lambda item: abs(item[1] - target_ber))


def classification_noise_penalty(ber: float) -> float:
    """Reference sample-complexity multiplier for symmetric label noise."""
    if not 0 <= ber < 0.5:
        return float("inf")
    return float(1 / (1 - 2 * ber) ** 2)
