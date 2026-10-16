import itertools
import logging
from typing import Callable, Tuple

import numpy as np
import numpy.typing as npt
from numpy.random import default_rng
from pypuf.simulation import ArbiterPUF, XORArbiterPUF

from .models import FiLMLayer, PUFModel, PUFNetwork

__all__ = [
    "CorrelatedArbiterPUF",
    "CorrelatedXORArbiterPUF",
    "FiLMLayer",
    "Grid",
    "PUFModel",
    "PUFNetwork",
    "generate_weights_bias",
    "logger",
    "min_entropy",
    "random_crps",
    "shannon_entropy",
    "suffix_parity_transform",
]

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False  # prevent duplicate logs via root

if not logger.handlers:  # avoid re-adding on reload
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def shannon_entropy(p: npt.ArrayLike) -> npt.NDArray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -((p * np.log2(p)) + ((1 - p) * np.log2(1 - p)))


def min_entropy(p: npt.ArrayLike) -> npt.NDArray:
    p = np.clip(np.asarray(p), 1e-9, 1 - 1e-9)
    return -np.log2(np.maximum(p, 1 - p))


class Grid:
    def __init__(self, **kwargs):
        self.params = kwargs

    def __len__(self) -> int:
        return np.prod([len(v) for v in self.params.values()])

    def __dict__(self):
        return self.params

    def __iter__(self):
        keys = self.params.keys()
        for x in itertools.product(*self.params.values()):
            yield dict(zip(keys, x))


def suffix_parity_transform(X: npt.NDArray) -> npt.NDArray:
    """Transform the polar array X using the Suffix Parity Transform

    ALWAYS use k=1 transform for the features, even for k-XOR

    Arguments:
        X:

    Returns:
        The Suffix Parity Transformed Array with shape (len(X), -1)
    """
    return ArbiterPUF.transform_atf(X, k=1).reshape(len(X), -1)


def random_crps(
    n_samples: int, n_bits: int, seed: int | None = None, as_polar: bool = True
) -> npt.NDArray:
    """Creates a random 2d array of binary or polar responses
    Arguments:
        n_samples: Number of samples (rows)
        n_bits: Number of bits (columns)
        seed: Optional seed for the RNG
        as_polar: If True, return {-1, 1}, else {0, 1}
    """
    rng = np.random.default_rng(seed)
    bits = rng.binomial(1, 0.5, (n_samples, n_bits)).astype(np.int8)
    if as_polar:
        return (2 * bits) - 1
    return bits


def generate_weights_bias(
    n_bits: int,
    weight_loc: float = 0,
    weight_scale: float = 1,
    bias_loc: float = 0,
    bias_scale: float = 0.5,
    k: int = 1,
    beta: float = 0.0,
    seed: int | None = None,
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Generate a shared systematic k-XOR APUF template.

    Arguments:
        n_bits: Number of inputs bits of the APUF.
        weight_loc: Location (mean) of the Normal distribution for the weights. Defaults to 0.
        weight_scale: Scale of the Normal distribution for the weights. Defaults to 1.
        bias_loc: Location (mean) of the Normal distribution for the biases. Defaults to 0.
        bias_scale: Scale of the Normal distribution for the biases. Defaults to 0.5.
        k: Defaults to 1.
        beta: Controls symmetric intra-die chain-to-chain correlation through a latent
    macro component. It only has an effect for k > 1.
        seed: Seed used for the RNG. Defaults to no seed.

    Raises:
        ValueError: If k is smaller than 1
        ValueError: If beta is not in the range [0, 1]

    Returns:

    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if not 0 <= beta <= 1:
        raise ValueError("beta must be in [0, 1]")

    rng = default_rng(seed=seed)

    weight_scales = np.concatenate(
        (np.array([weight_scale / 2]), np.full(max(0, n_bits - 1), weight_scale))
    )
    independent_weights = rng.normal(loc=0, scale=weight_scales, size=(k, n_bits))
    independent_bias = rng.normal(loc=0, scale=bias_scale, size=(k, 1))

    if k == 1 or beta == 0:
        weights = independent_weights + weight_loc
        bias = independent_bias + bias_loc
        return np.concatenate((weights, bias), axis=1), bias

    latent_weights = rng.normal(loc=0, scale=weight_scales, size=(1, n_bits))
    latent_bias = rng.normal(loc=0, scale=bias_scale, size=(1, 1))

    residual_scale = np.sqrt(1 - beta**2)
    base_weights = (
        weight_loc + beta * latent_weights + residual_scale * independent_weights
    )
    base_bias = bias_loc + beta * latent_bias + residual_scale * independent_bias

    return np.concatenate((base_weights, base_bias), axis=1), base_bias


class CorrelatedArbiterPUF(ArbiterPUF):
    """
    An Arbiter PUF that shares a nominal underlying delay distribution with other instances.
    Parameters:
        alpha: Blend the unique weights with the shared nominal weights
               alpha = 0.0 -> 100% independent (Standard pypuf behavior)
               alpha = 1.0 -> 100% identical to shared base (Perfectly correlated)
    """

    def __init__(
        self,
        n: int,
        shared_weights: np.ndarray,
        shared_bias: np.ndarray,
        alpha: float,
        seed: int | None = None,
        transform: Callable | str | None = None,
        noisiness: float = 0,
    ):
        # Initialize as a completely independent, random APUF (generates W_unique)
        super().__init__(n=n, seed=seed, transform=transform, noisiness=noisiness)  # type: ignore[invalid-argument-type]

        shared_weights = np.asarray(shared_weights).reshape(self.weight_array.shape)
        shared_bias = np.asarray(shared_bias).reshape(self.bias.shape)

        self.weight_array = (alpha * shared_weights) + (
            np.sqrt(1 - alpha**2) * self.weight_array
        )

        # PyPUF evaluates the appended coefficient, not a later assignment to bias.
        self.bias = self.weight_array[:, -1:].copy()


class CorrelatedXORArbiterPUF(XORArbiterPUF):
    """
    XOR APUF population model.

    alpha controls inter-die correlation by blending a shared systematic template
    with the device-unique random residual. generate_weights_bias(..., beta=...)
    creates the symmetric chain-correlated shared template.
    """

    def __init__(
        self,
        n: int,
        k,
        shared_weights: np.ndarray,
        shared_bias: np.ndarray,
        alpha: float,
        seed: int | None = None,
        transform: Callable | str | None = None,
        noisiness: float = 0,
    ):
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")

        # Initialize standard independent XOR APUF
        super().__init__(n=n, k=k, seed=seed, transform=transform, noisiness=noisiness)  # type: ignore[invalid-argument-type]

        shared_weights = np.asarray(shared_weights).reshape(self.weight_array.shape)
        shared_bias = np.asarray(shared_bias).reshape(self.bias.shape)

        # The blend works exactly the same, as long as shared_weights is shape (k, n)
        self.weight_array = (alpha * shared_weights) + (
            np.sqrt(1 - alpha**2) * self.weight_array
        )
        self.bias = self.weight_array[:, -1:].copy()
