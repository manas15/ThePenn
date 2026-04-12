"""
ml/features.py

Feature engineering and data augmentation for accelerometer sequences.
"""

import math
import random

import numpy as np


def compute_features(samples_xyz):
    """
    Convert raw [[x,y,z], ...] to a (T, 10) feature matrix.

    Features per timestep:
        0-2:  raw x, y, z
        3-5:  first-order deltas (dx, dy, dz)
        6:    L2 norm of first-order delta
        7-9:  second-order deltas (ddx, ddy, ddz)
    """
    arr = np.array(samples_xyz, dtype=np.float32)
    T = len(arr)

    deltas = np.zeros_like(arr)
    if T > 1:
        deltas[1:] = arr[1:] - arr[:-1]

    l2 = np.sqrt((deltas ** 2).sum(axis=1, keepdims=True))

    ddeltas = np.zeros_like(arr)
    if T > 2:
        ddeltas[2:] = deltas[2:] - deltas[1:-1]

    return np.concatenate([arr, deltas, l2, ddeltas], axis=1)


def augment(samples_xyz, rng=None):
    """
    Apply random augmentations to raw [[x,y,z], ...] data.
    Returns augmented copy (does not modify input).
    """
    if rng is None:
        rng = random.Random()

    arr = np.array(samples_xyz, dtype=np.float32)

    # Time warping: resample at 0.8x-1.2x speed
    if rng.random() < 0.5:
        factor = rng.uniform(0.8, 1.2)
        new_len = max(3, int(len(arr) * factor))
        old_t = np.linspace(0, 1, len(arr))
        new_t = np.linspace(0, 1, new_len)
        arr = np.stack([np.interp(new_t, old_t, arr[:, i]) for i in range(3)], axis=1)

    # Gaussian noise
    if rng.random() < 0.5:
        noise = np.random.default_rng(rng.randint(0, 2**31)).normal(0, 0.005, arr.shape)
        arr = arr + noise.astype(np.float32)

    # Amplitude scaling
    if rng.random() < 0.5:
        arr = arr * rng.uniform(0.9, 1.1)

    # Random Z-axis rotation
    if rng.random() < 0.5:
        angle = math.radians(rng.uniform(-15, 15))
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rotated = arr.copy()
        rotated[:, 0] = arr[:, 0] * cos_a - arr[:, 1] * sin_a
        rotated[:, 1] = arr[:, 0] * sin_a + arr[:, 1] * cos_a
        arr = rotated

    # Time shift: trim 0-10% from ends
    if rng.random() < 0.5 and len(arr) > 6:
        max_trim = max(1, len(arr) // 10)
        trim_start = rng.randint(0, max_trim)
        trim_end = rng.randint(0, max_trim)
        end_idx = len(arr) - trim_end
        if end_idx - trim_start >= 3:
            arr = arr[trim_start:end_idx]

    return arr.tolist()
