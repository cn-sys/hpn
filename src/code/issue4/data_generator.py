"""
确定性数据生成器 — NCCL 确定性诊断。

使用固定 seed 按 rank 生成输入张量，确保:
  - 每个 rank 有不同但确定的输入（暴露归约顺序敏感性）
  - 相同 seed 始终生成相同张量（支持跨运行比对）

核心设计决策: 每个 rank 使用不同 seed，使得归约顺序影响结果。
若所有 rank 输入相同，任何累加顺序都会得到相同结果——
诊断工具将永远捕捉不到非确定性。
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data layout constants (matching nccl-tests buffer semantics)
# ---------------------------------------------------------------------------

# For AllReduce: each rank contributes `count` elements, output is `count` elements.
# For ReduceScatter: each rank contributes `nranks * count` elements,
#   output per rank is `count` elements.
# nccl-tests "size" = nranks × count × sizeof(dtype) for ReduceScatter.
# nccl-tests "size" = count × sizeof(dtype) for AllReduce.

DTYPE_NP_MAP: dict[str, type] = {
    "float32":  np.float32,
    "float16":  np.float16,
    "bfloat16": np.float16,  # numpy doesn't have bf16 natively; use fp16 as proxy for shape
}

DTYPE_BYTES: dict[str, int] = {
    "float32":  4,
    "float16":  2,
    "bfloat16": 2,
}


@dataclass
class DataGenerator:
    """为 NCCL 诊断运行生成确定性输入张量。

    Parameters
    ----------
    dtype : str
        One of 'float32', 'float16', 'bfloat16'.
    base_seed : int
        Root seed. Per-rank seeds are derived as base_seed + rank.
    collective : str
        'allreduce' or 'reducescatter' — affects element count calculation.
    """

    dtype: str = "float32"
    base_seed: int = 42
    collective: str = "allreduce"

    # ------------------------------------------------------------------
    # ---- 元素计数（匹配 nccl-tests 约定） ----
    # ------------------------------------------------------------------

    def elem_count(self, size_bytes: int, nranks: int) -> int:
        """根据给定总字节数计算每 rank 的元素数。

        Matching nccl-tests convention:
          - AllReduce:    count = size_bytes / sizeof(dtype)
          - ReduceScatter: count = size_bytes / (sizeof(dtype) * nranks)
        """
        type_bytes = DTYPE_BYTES[self.dtype]
        if self.collective == "allreduce":
            count = size_bytes // type_bytes
        elif self.collective == "reducescatter":
            count = size_bytes // (type_bytes * nranks)
        else:
            raise ValueError(f"Unknown collective: {self.collective}")

        if count <= 0:
            raise ValueError(
                f"size_bytes={size_bytes} too small for dtype={self.dtype} "
                f"({type_bytes} bytes/elem) × nranks={nranks} "
                f"(collective={self.collective})"
            )
        return count

    # ------------------------------------------------------------------
    # ---- 数据生成 ----
    # ------------------------------------------------------------------

    def generate(self, rank: int, size_bytes: int, nranks: int) -> np.ndarray:
        """为单个 rank 生成确定性数据。

        Uses SHA-256 seeded PRNG: seed → deterministic bits → normalized to [−1, +1] range.
        This avoids any platform-dependent RNG behavior across runs.

        For AllReduce: produces `count` elements (the per-rank contribution).
        For ReduceScatter: produces `nranks * count` elements (the per-rank send buffer,
        of which `count` elements are scattered to this rank after reduction).
        """
        count = self.elem_count(size_bytes, nranks)
        # ReduceScatter: each rank contributes nranks * recvcount elements as send buffer
        if self.collective == "reducescatter":
            count = count * nranks
        seed = self.base_seed + rank
        data = _deterministic_rand(count, seed)

        np_dtype = DTYPE_NP_MAP[self.dtype]
        return data.astype(np_dtype)

    def generate_all(self, size_bytes: int, nranks: int) -> list[np.ndarray]:
        """为所有 rank 生成输入。 Returns list[np.ndarray] indexed by rank."""
        return [self.generate(r, size_bytes, nranks) for r in range(nranks)]

    # ------------------------------------------------------------------
    # Reference computation (CPU ground truth for validation)
    # ------------------------------------------------------------------

    def reference_allreduce(self, inputs: list[np.ndarray]) -> np.ndarray:
        """Compute the reference AllReduce result on CPU in float64.

        This is the "ideal" result — a single-precision floating sum of all
        rank inputs. Not bitwise-identical to any GPU result but serves as a
        sanity bound for diff magnitude.
        """
        acc = np.zeros_like(inputs[0], dtype=np.float64)
        for arr in inputs:
            acc += arr.astype(np.float64)
        return acc


# ---------------------------------------------------------------------------
# Internal: deterministic pseudo-random generator using SHA-256
# ---------------------------------------------------------------------------

def _deterministic_rand(count: int, seed: int) -> np.ndarray:
    """Generate count floats in [−1, 1] using SHA-256 as the entropy source.

    This is fully deterministic across all platforms and Python versions
    because SHA-256 is a standardized cryptographic hash.
    """
    data = np.empty(count, dtype=np.float32)
    generated = 0
    block = 0

    while generated < count:
        # Hash (seed || block) → 32 bytes → 8× float32
        h = hashlib.sha256()
        h.update(struct.pack("<II", seed, block))
        digest = h.digest()

        for i in range(0, 28, 4):  # 7 floats per 32-byte digest
            if generated >= count:
                break
            # Interpret 4 bytes as uint32, scale to [−1, 1]
            val = struct.unpack("<I", digest[i:i + 4])[0]
            data[generated] = (val / 0xFFFFFFFF) * 2.0 - 1.0
            generated += 1

        block += 1

    return data


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def validate_data_divergence(inputs: list[np.ndarray]) -> bool:
    """检查各 rank 输入是否不同（有意义诊断的必要条件）。

    Returns True if at least one rank has different data from another.
    """
    if len(inputs) < 2:
        return False
    ref = inputs[0]
    for arr in inputs[1:]:
        if not np.array_equal(ref, arr):
            return True
    return False
