"""Execute a controlled NCCL algorithm/protocol/size matrix and summarize it.

Runs `diagnose.py` for every combination of `--sizes × --algos × --protos`,
collects the results, and writes a `summary.json` that captures differences
across message sizes (scale), algorithms, and protocols.

Usage:
    python sweep.py --nproc-per-node 4 --runs 5 \
        --sizes 128,1024,16384,131072,1048576 \
        --algos Ring,Tree --protos Simple,LL,LL128 \
        --calls 20
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nproc-per-node", type=int, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--op", choices=("all_reduce", "reduce_scatter"),
                        default="all_reduce")
    parser.add_argument("--calls", type=int, default=10)
    parser.add_argument("--dtype", choices=("float16", "float32", "float64"),
                        default="float32")
    parser.add_argument("--sizes",
                        default="128,1024,16384,131072,1048576,8388608,67108864",
                        help=("Comma-separated message sizes in elements "
                              "(FP32: 128B=32el, 1KB=256el, 16KB=4096el, "
                              "128KB=32768el, 1MB=262144el, 8MB=2097152el, "
                              "64MB=16777216el)."))
    parser.add_argument("--algos", default="Ring,Tree",
                        help="Comma-separated NCCL_ALGO values.")
    parser.add_argument("--protos", default="Simple,LL,LL128",
                        help="Comma-separated NCCL_PROTO values.")
    parser.add_argument("--topo-file", type=Path,
                        help="Path to NCCL_TOPO_FILE XML (optional).")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("sweep-output"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # diagnose.py lives next to this script.
    driver = Path(__file__).with_name("diagnose.py")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []

    # --- Triple-nested scan: size × algo × proto ---
    for size_str in args.sizes.split(","):
        elements = int(size_str)
        for algo in args.algos.split(","):
            for proto in args.protos.split(","):
                slug = f"size{elements}-{algo.lower()}-{proto.lower()}"
                output = args.output_dir / slug

                cmd = [
                    sys.executable,
                    str(driver),
                    "--nproc-per-node", str(args.nproc_per_node),
                    "--runs", str(args.runs),
                    "--op", args.op,
                    "--elements", str(elements),
                    "--calls", str(args.calls),
                    "--dtype", args.dtype,
                    "--output-dir", str(output),
                ]
                if algo != "default":
                    cmd.extend(("--algo", algo))
                if proto != "default":
                    cmd.extend(("--proto", proto))
                if args.topo_file:
                    cmd.extend(("--topo-file", str(args.topo_file)))

                print(f"\n=== SIZE={elements}el ALGO={algo} PROTO={proto} ===",
                      flush=True)
                result = subprocess.run(cmd, env=os.environ.copy())
                report_path = output / "report.json"

                row: dict[str, object] = {
                    "elements": elements,
                    "algo": algo,
                    "proto": proto,
                    "exit_code": result.returncode,
                    "report": str(report_path),
                }
                if report_path.exists():
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    row["bitwise_identical"] = report["bitwise_identical"]
                    row["first_divergence"] = report["first_divergence"]
                    # Record evolution_by_call so the final summary shows how
                    # differences accumulate across iterations.
                    row["evolution_by_call"] = report.get("evolution_by_call")
                else:
                    row["error"] = "launch failed; inspect console/NCCL logs"
                rows.append(row)

    # --- Write summary ---
    summary = {
        "experiment": {
            "op": args.op,
            "calls": args.calls,
            "runs": args.runs,
            "world_size": args.nproc_per_node,
            "dtype": args.dtype,
        },
        "results": rows,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nsummary={summary_path}")

    # Non-zero if any configuration failed.
    return 1 if any("error" in row for row in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
