"""Boundary learning: DP initialisation and local search.

Components:
    is_valid_segmentation      — check block size constraints
    get_block_sizes            — compute block sizes from boundary positions
    optimize_boundaries_dp     — DP shortest-path for optimal segmentation
    local_search               — shift/merge/split/swap moves with SMC feedback
"""

import numpy as np
from typing import Callable, Awaitable, Optional


def is_valid_segmentation(
    boundaries: list[int],
    T: int,
    min_block: int = 2,
    max_block: int = 20,
) -> bool:
    """Check that all blocks have sizes in [min_block, max_block] and end at T.

    Args:
        boundaries: Sorted list of cumulative boundary positions, ending at T.
        T: Total sequence length.
        min_block: Minimum block size.
        max_block: Maximum block size.
    """
    if not boundaries or boundaries[-1] != T:
        return False
    prev = 0
    for b in boundaries:
        block_len = b - prev
        if block_len < min_block or block_len > max_block:
            return False
        prev = b
    return True


def get_block_sizes(boundaries: list[int]) -> list[int]:
    """Compute block sizes from boundary positions.

    Args:
        boundaries: Sorted list of cumulative boundary positions.

    Returns:
        List of block sizes.
    """
    prev = 0
    sizes = []
    for b in boundaries:
        sizes.append(b - prev)
        prev = b
    return sizes


def optimize_boundaries_dp(
    disc_cache: dict[tuple[int, int], float],
    T: int,
    min_block: int = 2,
    max_block: int = 20,
    lam: float = 0.1,
) -> list[int]:
    """Dynamic programming for optimal segmentation.

    Shortest-path on a DAG: node t = "covered positions 1..t".
    Edge (s, t) = block spanning (s, t] with cost = -disc(s,t) + λ.

    Recurrence:
        V(t) = max_{s} [ V(s) + disc(s,t) - λ ]
        V(0) = 0

    Maximises total discrimination with penalty λ per boundary.

    Args:
        disc_cache: Precomputed discrimination scores keyed by (start, end).
        T: Total sequence length.
        min_block: Minimum block size.
        max_block: Maximum block size.
        lam: Per-boundary penalty (larger → fewer boundaries).

    Returns:
        Sorted boundary positions ending at T.
    """
    V = {0: 0.0}
    back = {0: None}

    for t in range(min_block, T + 1):
        best_val = -1e12
        best_s = None
        for s in range(max(0, t - max_block), t - min_block + 1):
            if s not in V:
                continue
            key = (s, t)
            if key not in disc_cache:
                continue
            val = V[s] + disc_cache[key] - lam
            if val > best_val:
                best_val = val
                best_s = s
        if best_val > -1e12:
            V[t] = best_val
            back[t] = best_s

    # Trace back from T
    if T not in back:
        return []
    boundaries = []
    pos = T
    while pos is not None and pos > 0:
        boundaries.append(pos)
        pos = back.get(pos)
    return sorted(boundaries)


async def local_search(
    boundaries: list[int],
    eval_fn: Callable[[list[int]], Awaitable[float]],
    T: int,
    min_block: int = 2,
    max_block: int = 20,
    n_steps: int = 100,
    seed: int = 42,
) -> tuple[list[int], float]:
    """Local search over segmentations with shift/merge/split/swap moves.

    Uses simulated annealing: always accept improvements, sometimes
    accept worse solutions to escape local optima.

    Args:
        boundaries: Starting segmentation (sorted positions ending at T).
        eval_fn: Async function mapping boundaries → score (higher is better).
        T: Total sequence length.
        min_block: Minimum block size.
        max_block: Maximum block size.
        n_steps: Number of search steps.
        seed: Random seed.

    Returns:
        (best_boundaries, best_score)
    """
    rng = np.random.default_rng(seed)

    current = boundaries.copy()
    current_score = await eval_fn(current)
    best = current.copy()
    best_score = current_score

    for step in range(n_steps):
        r = rng.random()

        if r < 0.40:
            # SHIFT: move one interior boundary by ±1
            if len(current) <= 1:
                continue
            idx = rng.integers(0, len(current) - 1)  # don't shift T
            candidate = current.copy()
            candidate[idx] += rng.choice([-1, 1])
            candidate = sorted(candidate)

        elif r < 0.65:
            # MERGE: remove one interior boundary
            if len(current) <= 2:
                continue
            idx = rng.integers(0, len(current) - 1)
            candidate = current[:idx] + current[idx + 1:]

        elif r < 0.85:
            # SPLIT: add a boundary inside a large block
            sizes = get_block_sizes(current)
            splittable = [k for k, s in enumerate(sizes) if s >= 2 * min_block]
            if not splittable:
                continue
            k = rng.choice(splittable)
            start = current[k - 1] if k > 0 else 0
            end = current[k]
            valid_splits = [
                sp for sp in range(start + min_block, end - min_block + 1)
                if (sp - start) <= max_block and (end - sp) <= max_block
            ]
            if not valid_splits:
                continue
            sp = rng.choice(valid_splits)
            candidate = sorted(current + [sp])

        else:
            # SWAP: remove one boundary, add another
            if len(current) <= 2:
                continue
            idx = rng.integers(0, len(current) - 1)
            temp = current[:idx] + current[idx + 1:]
            if is_valid_segmentation(temp, T, min_block, max_block):
                candidate = temp
            else:
                sizes = get_block_sizes(temp) if temp and temp[-1] == T else []
                splittable = [k for k, s in enumerate(sizes) if s >= 2 * min_block]
                if not splittable:
                    continue
                k = rng.choice(splittable)
                start = temp[k - 1] if k > 0 else 0
                end = temp[k]
                valid_splits = [
                    sp for sp in range(start + min_block, end - min_block + 1)
                    if (sp - start) <= max_block and (end - sp) <= max_block
                ]
                if not valid_splits:
                    continue
                sp = rng.choice(valid_splits)
                candidate = sorted(temp + [sp])

        if not is_valid_segmentation(candidate, T, min_block, max_block):
            continue

        score = await eval_fn(candidate)

        # Simulated annealing acceptance
        temperature = max(0.5, 5.0 * (1 - step / n_steps))
        delta = score - current_score
        if delta > 0 or rng.random() < np.exp(delta / max(temperature, 1e-8)):
            current = candidate
            current_score = score
            if score > best_score:
                best = candidate.copy()
                best_score = score

    return best, best_score


def make_initial_boundaries(
    T: int,
    interval: int = 5,
    min_block: int = 2,
    max_block: int = 20,
) -> list[int]:
    """Create initial evenly-spaced boundaries.

    Generates boundaries at every `interval` tokens, adjusting the last
    block to reach exactly T.

    Args:
        T: Total sequence length.
        interval: Target block size.
        min_block: Minimum block size.
        max_block: Maximum block size.

    Returns:
        Sorted boundary positions ending at T.
    """
    boundaries = []
    pos = interval
    while pos < T:
        boundaries.append(pos)
        pos += interval
    boundaries.append(T)

    # Validate — if last block is too small, merge with previous
    if len(boundaries) > 1:
        last_size = boundaries[-1] - boundaries[-2]
        if last_size < min_block:
            boundaries.pop(-2)

    return boundaries
