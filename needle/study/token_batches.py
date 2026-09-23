"""Deterministic response-token batches for paired recovery runs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TokenCursor:
    """Position in the cyclic, seeded example order."""

    example: int = 0
    target: int = 0


def next_target_batches(
    target_masks: np.ndarray,
    order: np.ndarray,
    cursor: TokenCursor,
    target_tokens: int = 8192,
    rows_per_batch: int = 16,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], TokenCursor]:
    """Select exactly ``target_tokens`` supervised positions, without dropping any.

    The last example of one update may continue in the next. Padding rows have
    an all-zero mask and are never counted by the loss. The same seed, masks,
    and starting cursor yield identical microbatches for every architecture.
    """
    if target_tokens <= 0 or rows_per_batch <= 0:
        raise ValueError("target_tokens and rows_per_batch must be positive")
    masks = np.asarray(target_masks)
    order = np.asarray(order, dtype=np.int32)
    if masks.ndim != 2 or len(order) != len(masks) or len(order) == 0:
        raise ValueError("order must contain one index per nonempty mask row")
    if sorted(order.tolist()) != list(range(len(order))):
        raise ValueError("order must be a permutation of mask row indices")
    counts = np.count_nonzero(masks[:, 1:], axis=1)
    if np.any(counts == 0):
        raise ValueError("each example needs at least one next-token target")
    if not 0 <= cursor.example < len(order):
        raise ValueError("example cursor is out of range")

    remaining = target_tokens
    example_pos, target_pos = cursor.example, cursor.target
    batches: list[tuple[np.ndarray, np.ndarray]] = []
    row_ids: list[int] = []
    selected_masks: list[np.ndarray] = []

    def flush() -> None:
        if not row_ids:
            return
        pad_row = row_ids[-1]
        while len(row_ids) < rows_per_batch:
            row_ids.append(pad_row)
            selected_masks.append(np.zeros(masks.shape[1], dtype=np.float32))
        batches.append((np.asarray(row_ids, dtype=np.int32),
                        np.stack(selected_masks, axis=0)))
        row_ids.clear()
        selected_masks.clear()

    while remaining:
        row = int(order[example_pos])
        positions = np.flatnonzero(masks[row, 1:]) + 1
        if target_pos >= len(positions):
            raise ValueError("target cursor is out of range")
        take = min(remaining, len(positions) - target_pos)
        selected = np.zeros(masks.shape[1], dtype=np.float32)
        selected[positions[target_pos:target_pos + take]] = 1.0
        row_ids.append(row)
        selected_masks.append(selected)
        remaining -= take
        target_pos += take
        if target_pos == len(positions):
            example_pos = (example_pos + 1) % len(order)
            target_pos = 0
        if len(row_ids) == rows_per_batch:
            flush()
    flush()
    return batches, TokenCursor(example_pos, target_pos)
