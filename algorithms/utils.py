"""Utility functions and classes
"""

from typing import Optional, List, Tuple
import numpy as np

SMALL_POSITIVE = 1e-10
DEFAULT_VALUE = 1

def project(x):
    """Project `x` to simplex, enforces minimum value for numerical stability"""
    if isinstance(x, dict):
        prob = np.array(list(x.values()))
        assert np.all(np.logical_or(prob > 0, np.isclose(prob, 0)))
        prob = np.maximum(prob, SMALL_POSITIVE)
        prob /= prob.sum()
        return {a: prob[i] for i, a in enumerate(list(x.keys()))}
    else:
        assert np.all(np.logical_or(x > 0, np.isclose(x, 0)))
        x = np.maximum(x, SMALL_POSITIVE)
        return x / x.sum()


def project_alpha(alpha):
    assert isinstance(alpha, np.ndarray)
    if min(alpha) > 0:
        return alpha / alpha.sum()
    sz = alpha.size
    alpha += abs(min(alpha))
    if alpha.sum() == 0:
        alpha = 1. / sz * np.ones(sz)
    else:
        alpha /= alpha.sum()
    return alpha

def project_energy_to_positive(energy_dict):
    min_energy = min(energy_dict.values())
    if min_energy > 0:
        return energy_dict
    return {
        a: energy_dict[a] + abs(min_energy)
        for a in energy_dict.keys()
    }



def kl(
    x: np.ndarray,
    y: Optional[np.ndarray],
) -> float:
    if y is None:
        return DEFAULT_VALUE
    assert x.shape == y.shape
    cum = 0
    for x_, y_ in zip(x.flatten(), y.flatten()):
        if np.isclose(x_, 0):
            continue
        if np.isclose(y_, 0):
            cum += x_ * np.log(x_ / SMALL_POSITIVE)
        else:
            cum += x_ * np.log(x_ / y_)
    return float(cum / np.prod(x.shape[:-1]))


def is_power_of_2(n: int) -> bool:
    """Return whether `n` is power of 2"""
    return (n & (n - 1) == 0) and n != 0


def schedule(upper_lim: int, power_sc: bool=True) -> List[Tuple[int, bool]]:
    ls: List[Tuple[int, bool]] = []
    if power_sc:
        for i in range(upper_lim):
            ls.append((i, i == 0 or is_power_of_2(i) or i == upper_lim - 1))
    else:
        for i in range(upper_lim):
            ls.append((i, i == 0 or i % 20 == 0 or i == upper_lim - 1))
    return ls


def convert_returns(returns, game_str):
    new_returns = np.copy(returns)

    # Game A: [0, 1] vs 2, Game B: [0, 2] vs 1
    if game_str in ["mix_kuhn_a", "mix_goofspiel"]:
        new_returns[0] = new_returns[1] = (returns[0] + returns[1]) / 2
    elif game_str in ["mix_kuhn_b"]:
        new_returns[0] = new_returns[2] = (returns[0] + returns[2]) / 2

    return np.array(new_returns)
