"""Run independent NCCL jobs and compare their output bytes.

This is the main orchestration script.  For a given configuration it:
1. launches `--runs` independent `torchrun --standalone` process groups,
2. each group runs `worker.py` to capture collective outputs,
3. compares all runs via `core.compare_runs`,
4. writes a `report.json` and prints the first divergence.

Usage:
    # NCCL automatic selection
    python diagnose.py --nproc-per-node 4 --runs 5 \
        --op all_reduce --elements 1048576 --calls 20 \
        --output-dir results/default

    # Pin a specific algorithm/protocol
    python diagnose.py --nproc-per-node 4 --runs 5 \
        --algo Ring --proto Simple --op all_reduce \
        --elements 1048576 --calls 20 --output-dir results/ring-simple
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# core.py is in the same directory.
from core import compare_runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nproc-per-node", type=int, required=True,
                        help="Number of GPUs to use.")
    parser.add_argument("--runs", type=int, default=3,
                        help="How many independent launches (>=2).")
    parser.add_argument("--op", choices=("all_reduce", "reduce_scatter"),
                        default="all_reduce")
    parser.add_argument("--elements", type=int, default=1 << 18,
                        help="Elements per rank (default: 256K).")
    parser.add_argument("--calls", type=int, default=10,
                        help="Collective calls per launch.")
    parser.add_argument("--dtype", choices=("float16", "float32", "float64"),
                        default="float32")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--algo",
                        help="NCCL_ALGO value, e.g. Ring or Tree")
    parser.add_argument("--proto",
                        help="NCCL_PROTO value, e.g. Simple, LL, or LL128")
    parser.add_argument("--topo-file", type=Path,
                        help="Path to NCCL_TOPO_FILE XML (optional).")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("diagnostic-output"))
    parser.add_argument("--keep-payloads", action="store_true",
                        help="Keep per-run capture files (they are large).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.runs < 2 or args.nproc_per_node < 2:
        raise ValueError("--runs and --nproc-per-node must both be at least 2")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Resolve worker.py path ---
    worker = Path(__file__).with_name("worker.py")

    # --- Build environment ---
    env = os.environ.copy()
    # Set NCCL overrides if requested; unset if not.
    for key, value in (
        ("NCCL_ALGO", args.algo),
        ("NCCL_PROTO", args.proto),
        ("NCCL_TOPO_FILE", str(args.topo_file.resolve()) if args.topo_file else None),
    ):
        if value:
            env[key] = value
        else:
            env.pop(key, None)

    # --- Launch independent runs ---
    captures: list[Path] = []
    for run in range(args.runs):
        capture = args.output_dir / f"run-{run:02d}.json"
        cmd = [
            "torchrun",
            "--standalone",
            f"--nproc-per-node={args.nproc_per_node}",
            str(worker),
            "--output", str(capture),
            "--op", args.op,
            "--elements", str(args.elements),
            "--calls", str(args.calls),
            "--dtype", args.dtype,
            "--seed", str(args.seed),
        ]
        print(f"[run {run + 1}/{args.runs}] {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, env=env, check=True)
        captures.append(capture)

    # --- Compare all runs ---
    report = compare_runs(captures)
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # --- Summary ---
    print(json.dumps(report["first_divergence"], indent=2))
    print(f"bitwise_identical={report['bitwise_identical']}  report={report_path}")

    # --- Clean up payloads (they can be large) ---
    if not args.keep_payloads:
        for capture in captures:
            capture.unlink(missing_ok=True)

    # Exit 0 = identical, 2 = divergence detected.
    return 0 if report["bitwise_identical"] else 2


if __name__ == "__main__":
    sys.exit(main())
