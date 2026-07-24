"""One independent distributed NCCL capture. Launch with torchrun.

This worker is the smallest unit of the diagnostic pipeline. Every invocation
creates a fresh NCCL communicator, runs the requested collective several times,
captures every output as raw bytes, and writes a single JSON file.

Usage (not invoked directly — use diagnose.py or sweep.py):
    torchrun --standalone --nproc-per-node=N worker.py \
        --op all_reduce --elements 1048576 --calls 5 \
        --dtype float32 --seed 2026 --output run_00.json

Architecture note:
    Output gathering uses a Gloo control group instead of NCCL. This prevents
    the instrumentation's internal AllGather from interfering with the
    collective under test (e.g. when NCCL_ALGO=Tree would be invalid for
    gather_object).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

import torch
import torch.distributed as dist

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="Path for the capture JSON file.")
    parser.add_argument("--op", choices=("all_reduce", "reduce_scatter"),
                        default="all_reduce")
    parser.add_argument("--elements", type=int, default=1 << 18,
                        help="Number of elements per rank (default: 256K).")
    parser.add_argument("--calls", type=int, default=10,
                        help="How many collective calls to record in this run.")
    parser.add_argument("--dtype", choices=("float16", "float32", "float64"),
                        default="float32")
    parser.add_argument("--seed", type=int, default=2026,
                        help="Base seed; rank offset is added automatically.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def nccl_version_string() -> str:
    """Return the NCCL version as a dotted string (e.g. '2.27.3')."""
    version = torch.cuda.nccl.version()
    if isinstance(version, tuple):
        return ".".join(map(str, version))
    return str(version)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # --- Basic validation ---
    if args.elements <= 0 or args.calls <= 0:
        raise ValueError("--elements and --calls must be positive")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # reduce_scatter divides output across ranks.
    if args.op == "reduce_scatter" and args.elements % world_size:
        raise ValueError(
            "--elements must be divisible by world_size for reduce_scatter"
        )

    # --- Init ---
    torch.cuda.set_device(local_rank)
    # The NCCL communicator under test.
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))

    # A separate Gloo group solely for gathering payloads to rank 0.
    # This avoids NCCL_ALGO/NCCL_PROTO side-effects on instrumentation.
    control_group = dist.new_group(backend="gloo")

    dtype = getattr(torch, args.dtype)

    # --- Deterministic per-rank input generation ---
    # Each rank gets a fixed but rank-distinct input. The seed is rank-dependent
    # so that different world sizes don't silently share inputs.
    generator = torch.Generator(device="cpu").manual_seed(args.seed + rank)
    source = torch.randn(args.elements, dtype=dtype, generator=generator).cuda(local_rank)

    captures: list[dict[str, object]] = []

    # --- Run the collective `calls` times ---
    for call in range(args.calls):
        # Perturb the input slightly between calls so that each call's output is
        # independently meaningful. The perturbation is deterministic:
        # call * epsilon added to every element.
        input_tensor = source + call * torch.finfo(dtype).eps

        if args.op == "all_reduce":
            output = input_tensor.clone()
            dist.all_reduce(output)
        else:  # reduce_scatter
            output = torch.empty(
                args.elements // world_size, dtype=dtype, device=local_rank
            )
            dist.reduce_scatter_tensor(output, input_tensor.contiguous())

        torch.cuda.synchronize()

        # Serialize output to raw bytes — this is what we compare later.
        raw = (
            output.detach()
            .cpu()
            .contiguous()
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
        captures.append({
            "call": call,
            "rank": rank,
            "dtype": str(dtype),
            "shape": list(output.shape),
            "payload_hex": raw.hex(),
        })

    # --- Gather all rank captures to rank 0 via Gloo ---
    gathered: list[list[dict[str, object]] | None] | None = (
        [None] * world_size if rank == 0 else None
    )
    dist.gather_object(captures, gathered, dst=0, group=control_group)

    # --- Write capture file (rank 0 only) ---
    if rank == 0:
        # Record environment metadata for reproducibility.
        metadata = {
            "op": args.op,
            "dtype": str(dtype),
            "elements": args.elements,
            "calls": args.calls,
            "world_size": world_size,
            "seed": args.seed,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "nccl_version": nccl_version_string(),
            "gpu": torch.cuda.get_device_name(local_rank),
            "host": platform.node(),
            # Record which NCCL_ALGO / NCCL_PROTO / NCCL_TOPO_FILE were
            # in effect (if any).  This is critical for experiment traceability.
            "nccl_env": {
                key: os.environ[key]
                for key in ("NCCL_ALGO", "NCCL_PROTO", "NCCL_TOPO_FILE")
                if key in os.environ
            },
        }

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "metadata": metadata,
                    # Flatten: gather_object returns a list of per-rank lists.
                    "captures": [
                        item
                        for rank_items in (gathered or [])
                        for item in (rank_items or [])
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # --- Cleanup ---
    dist.destroy_process_group(control_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
