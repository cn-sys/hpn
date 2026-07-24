"""Pure-Python comparison and reporting primitives for NCCL run captures.

This module is deliberately free of PyTorch / CUDA imports so that:
- unit tests run on CPU-only machines,
- existing capture files can be compared offline,
- the comparison logic is auditable without a GPU environment.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Schema version — bump when the capture format changes.
# ---------------------------------------------------------------------------
SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TensorCapture:
    """A single (call, rank) output captured as raw bytes."""

    call: int
    rank: int
    dtype: str
    shape: list[int]
    # Raw bytes stored as hex string to keep JSON portable.
    payload_hex: str

    @property
    def payload(self) -> bytes:
        """Return the raw bytes of this capture."""
        return bytes.fromhex(self.payload_hex)

    @property
    def sha256(self) -> str:
        """SHA-256 hash of the raw bytes, for integrity and fingerprinting."""
        return hashlib.sha256(self.payload).hexdigest()


@dataclass(frozen=True)
class Difference:
    """A single bitwise difference between two TensorCaptures."""

    run: int              # which run (1-indexed, baseline is run 0)
    call: int             # which collective call within the run
    rank: int             # which rank's output differs
    first_byte: int       # byte offset of first difference
    first_element: int    # element index of first difference
    changed_bytes: int    # how many bytes differ
    changed_bits: int     # how many bits differ
    max_abs_error: float | None  # maximum absolute error (None for non-float)
    max_ulp_error: int | None    # maximum ULP distance (None for non-float)
    baseline_sha256: str
    candidate_sha256: str


# ---------------------------------------------------------------------------
# Float helpers — map dtype string to (struct format, integer format, width)
# ---------------------------------------------------------------------------

def _float_format(dtype: str) -> tuple[str, str, int] | None:
    return {
        "torch.float16": ("e", "H", 2),
        "torch.float32": ("f", "I", 4),
        "torch.float64": ("d", "Q", 8),
    }.get(dtype)


def _ordered_int(bits: int, width: int) -> int:
    """Map IEEE 754 sign-magnitude bit patterns to monotonically ordered integers.

    This makes subtraction meaningful: ULP distance between two floats equals
    |ordered(a) - ordered(b)|.
    """
    sign = 1 << (width * 8 - 1)
    mask = (1 << (width * 8)) - 1
    # If the sign bit is set (negative), flip all bits to reverse order.
    # If the sign bit is clear (positive), set the sign bit.
    return (~bits & mask) if bits & sign else (bits | sign)


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------

def compare_capture(
    baseline: TensorCapture,
    candidate: TensorCapture,
    run: int,
) -> Difference | None:
    """Compare two TensorCaptures and return a Difference if they diverge.

    Args:
        baseline: The reference capture (usually run 0).
        candidate: The capture to compare against the baseline.
        run: 1-indexed run number of the candidate.

    Returns:
        A Difference object, or None if the captures are bitwise identical.

    Raises:
        ValueError: if metadata (dtype, shape, length) doesn't match.
    """
    # Metadata must match — otherwise comparison is meaningless.
    if (baseline.dtype, baseline.shape) != (candidate.dtype, candidate.shape):
        raise ValueError(
            f"capture metadata changed at call={baseline.call}, rank={baseline.rank}"
        )

    left, right = baseline.payload, candidate.payload
    if len(left) != len(right):
        raise ValueError("capture payload lengths differ")

    # Fast path: identical bytes.
    if left == right:
        return None

    # --- Bitwise difference analysis ---
    # XOR each byte pair: non-zero bytes indicate differences.
    xor = bytes(a ^ b for a, b in zip(left, right))

    # First byte that differs.
    first_byte = next(i for i, byte in enumerate(xor) if byte)

    # How many bytes / bits changed.
    changed_bytes = sum(bool(byte) for byte in xor)
    changed_bits = sum(byte.bit_count() for byte in xor)

    # --- Floating-point error analysis (only for known float dtypes) ---
    fmt = _float_format(baseline.dtype)
    max_abs: float | None = None
    max_ulp: int | None = None
    element_size = fmt[2] if fmt else 1

    if fmt:
        float_code, int_code, width = fmt
        count = len(left) // width
        # Unpack as floats and as raw integers.
        left_values = struct.unpack(f"<{count}{float_code}", left)
        right_values = struct.unpack(f"<{count}{float_code}", right)
        left_bits = struct.unpack(f"<{count}{int_code}", left)
        right_bits = struct.unpack(f"<{count}{int_code}", right)

        # Absolute error (skip NaN pairs, where abs(NaN - NaN) is meaningless).
        abs_errors = [
            abs(a - b)
            for a, b in zip(left_values, right_values)
            if not (math.isnan(a) and math.isnan(b))
        ]
        max_abs = max(abs_errors, default=0.0)

        # ULP distance via ordered integer representation.
        max_ulp = max(
            abs(_ordered_int(a, width) - _ordered_int(b, width))
            for a, b in zip(left_bits, right_bits)
        )

    return Difference(
        run=run,
        call=baseline.call,
        rank=baseline.rank,
        first_byte=first_byte,
        first_element=first_byte // element_size,
        changed_bytes=changed_bytes,
        changed_bits=changed_bits,
        max_abs_error=max_abs,
        max_ulp_error=max_ulp,
        baseline_sha256=baseline.sha256,
        candidate_sha256=candidate.sha256,
    )


# ---------------------------------------------------------------------------
# Run-level orchestration
# ---------------------------------------------------------------------------

def load_capture(path: Path) -> dict[str, Any]:
    """Load a capture JSON file and validate its schema version."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported capture schema in {path}")
    return data


def compare_runs(run_paths: Iterable[Path]) -> dict[str, Any]:
    """Compare multiple independent run captures and produce a report.

    The first run is treated as the baseline. Every other run is compared
    against it call-by-call and rank-by-rank.

    Args:
        run_paths: Paths to capture JSON files, at least 2.

    Returns:
        A report dict with keys: schema_version, metadata, run_files,
        bitwise_identical, first_divergence, differences.

    Raises:
        ValueError: if fewer than 2 runs, or metadata doesn't match.
    """
    paths = list(run_paths)
    if len(paths) < 2:
        raise ValueError("at least two independent runs are required")

    runs = [load_capture(path) for path in paths]

    # --- Validate that all runs share the same workload definition ---
    # These keys define the workload; if they differ the runs are not comparable.
    workload_keys = ("op", "dtype", "elements", "calls", "world_size", "seed")
    baseline_meta = {key: runs[0]["metadata"][key] for key in workload_keys}
    for index, run in enumerate(runs[1:], 1):
        candidate_meta = {key: run["metadata"][key] for key in workload_keys}
        if candidate_meta != baseline_meta:
            raise ValueError(f"run {index} is not comparable to the baseline")

    # --- Build baseline index: (call, rank) -> TensorCapture ---
    baseline = {
        (item["call"], item["rank"]): TensorCapture(**item)
        for item in runs[0]["captures"]
    }

    # --- Compare each subsequent run ---
    differences: list[Difference] = []
    for run_index, run in enumerate(runs[1:], 1):
        candidate = {
            (item["call"], item["rank"]): TensorCapture(**item)
            for item in run["captures"]
        }
        if candidate.keys() != baseline.keys():
            raise ValueError(f"run {run_index} has an incomplete capture set")

        for key in sorted(baseline):
            diff = compare_capture(baseline[key], candidate[key], run_index)
            if diff:
                differences.append(diff)

    # --- Find the first divergence (by call, then run, then rank) ---
    first = min(
        differences,
        key=lambda item: (item.call, item.run, item.rank),
        default=None,
    )

    # --- Summarize how differences evolve across calls (iteration) ---
    evolution_by_call: dict[int, dict[str, int | float]] = {}
    for d in differences:
        c = d.call
        if c not in evolution_by_call:
            evolution_by_call[c] = {
                "diff_ranks": 0,
                "max_abs_error": 0.0,
                "max_ulp": 0,
            }
        evolution_by_call[c]["diff_ranks"] += 1
        if d.max_abs_error is not None:
            evolution_by_call[c]["max_abs_error"] = max(
                float(evolution_by_call[c]["max_abs_error"]), d.max_abs_error
            )
        if d.max_ulp_error is not None:
            evolution_by_call[c]["max_ulp"] = max(
                int(evolution_by_call[c]["max_ulp"]), d.max_ulp_error
            )

    return {
        "schema_version": SCHEMA_VERSION,
        # Preserve the complete measured environment in the report.
        "metadata": runs[0]["metadata"],
        "run_files": [str(path) for path in paths],
        "bitwise_identical": not differences,
        "first_divergence": asdict(first) if first else None,
        # Evolution of differences across collective calls.
        # Key = call index, value = {diff_ranks, max_abs_error, max_ulp}.
        "evolution_by_call": {
            str(k): v for k, v in sorted(evolution_by_call.items())
        },
        "differences": [asdict(item) for item in differences],
    }
