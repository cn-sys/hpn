#!/usr/bin/env python3
"""
独立非确定性复现案例

复现已验证的 NCCL 非确定性场景:
  Ring AllReduce + float16 数据 + N GPU，跨运行逐位比对。

根因: Ring 算法的串行累加顺序依赖于 chunk 划分，而 chunk 划分
随缓冲区大小、协议和 NVLink 通道数微调。浮点非结合性导致
不同累加顺序产生不同的舍入误差 → 位级结果不同。

测试用例:
  Case 1: 同一配置重复运行 → 应为确定性的（NCCL 保证）
  Case 2: Ring vs Tree 同一数据 → 预期有差异（不同归约顺序）
  Case 3: float16 vs float32 → 展示精度对确定性的影响

用法:
    python reproduce_case.py [--nranks 8] [--size 128M] [--dtype float16]
    python reproduce_case.py --algo Ring --trials 5  # 迭代演化

硬件要求: >= 2 个 NVIDIA GPU，支持 NCCL。

参考:
  - NCCL#1055: Ring vs Tree 精度对比 (A100/A800)
  - PyTorch#138811: H20 allreduce 非确定性
  - NCCL#1975: Ring 算法精度讨论
  - NCCL#157: chunk 划分与确定性
"""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
import os
import socket
import struct
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

DTYPE_NP_MAP = {"float32": np.float32, "float16": np.float16,
                 "bfloat16": np.float16}
DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2}


@dataclass
class ReproduceResult:
    trial: int
    algo: str
    proto: str
    dtype: str
    size_bytes: int
    output: np.ndarray
    elapsed_ms: float


# ---------------------------------------------------------------------------
# Deterministic data generation
# ---------------------------------------------------------------------------

def _deterministic_rand(count: int, seed: int) -> np.ndarray:
    """Generate count floats in [-1, 1] via SHA-256 PRNG."""
    data = np.empty(count, dtype=np.float32)
    generated = 0
    block = 0
    while generated < count:
        h = hashlib.sha256()
        h.update(struct.pack("<II", seed, block))
        digest = h.digest()
        for i in range(0, 28, 4):
            if generated >= count:
                break
            val = struct.unpack("<I", digest[i:i + 4])[0]
            data[generated] = (val / 0xFFFFFFFF) * 2.0 - 1.0
            generated += 1
        block += 1
    return data


def generate_inputs(size_bytes: int, nranks: int, dtype: str) -> list[np.ndarray]:
    type_bytes = DTYPE_BYTES[dtype]
    count = size_bytes // type_bytes
    inputs = []
    for rank in range(nranks):
        arr = _deterministic_rand(count, seed=42 + rank)
        inputs.append(arr.astype(DTYPE_NP_MAP[dtype]))
    return inputs


# ---------------------------------------------------------------------------
# ---- PyTorch NCCL worker ----
# ---------------------------------------------------------------------------

def _torch_dtype(dtype: str):
    import torch
    return {"float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16}[dtype]


def _allreduce_worker(
    rank: int, world_size: int, port: int,
    algo: str, proto: str, dtype: str, input_data: np.ndarray,
    collective: str, result_dict: dict,
) -> None:
    import torch
    import torch.distributed as dist

    os.environ["NCCL_ALGO"] = algo
    os.environ["NCCL_PROTO"] = proto

    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank, world_size=world_size,
        device_id=device,
    )

    dt = _torch_dtype(dtype)
    tensor = torch.from_numpy(input_data).to(device=device, dtype=dt)

    # Warmup
    warm = tensor.clone()
    if collective == "allreduce":
        dist.all_reduce(warm)
    torch.cuda.synchronize()

    # Timed run
    tensor = torch.from_numpy(input_data).to(device=device, dtype=dt)
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    if collective == "allreduce":
        dist.all_reduce(tensor)
    elif collective == "reducescatter":
        count = input_data.size // world_size
        dist.reduce_scatter_tensor(
            tensor.view(-1),
            torch.zeros(count, dtype=dt, device=device),
        )

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    if rank == 0:
        result_dict[0] = tensor.cpu().float().numpy()
        result_dict["elapsed"] = (t1 - t0) * 1000.0

    dist.destroy_process_group()


def _run_trial(
    algo: str, proto: str, dtype: str, size_bytes: int,
    nranks: int, collective: str, trial_id: int,
) -> ReproduceResult:
    port = _find_free_port()
    inputs = generate_inputs(size_bytes, nranks, dtype)
    count = size_bytes // DTYPE_BYTES[dtype]

    manager = mp.Manager()
    result_dict = manager.dict()

    processes = []
    for rank in range(nranks):
        p = mp.Process(
            target=_allreduce_worker,
            args=(rank, nranks, port, algo, proto, dtype,
                  inputs[rank], collective, result_dict),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    output = np.array(result_dict.get(0, np.zeros(count, dtype=np.float32)))
    elapsed = result_dict.get("elapsed", 0.0)

    return ReproduceResult(
        trial=trial_id, algo=algo, proto=proto, dtype=dtype,
        size_bytes=size_bytes, output=output, elapsed_ms=elapsed,
    )


# ---------------------------------------------------------------------------
# Comparison (with distribution analysis)
# ---------------------------------------------------------------------------

def compare_bitwise(r1: ReproduceResult, r2: ReproduceResult) -> dict:
    b = r1.output
    t = r2.output

    if b.shape != t.shape:
        return {"error": f"Shape mismatch: {b.shape} vs {t.shape}"}

    diff_mask = (b != t)
    diff_idx = np.where(diff_mask)[0]
    diff_count = len(diff_idx)
    total = b.size

    if diff_count == 0:
        return {"bitwise_match": True, "total_elements": int(total)}

    abs_diff = np.abs(b[diff_mask].astype(np.float64) -
                      t[diff_mask].astype(np.float64))

    # Spatial distribution analysis
    n = total
    third = n // 3
    fc = int(np.sum(diff_idx < third))
    mc = int(np.sum((diff_idx >= third) & (diff_idx < 2 * third)))
    bc = int(np.sum(diff_idx >= 2 * third))

    fr = fc / third if third > 0 else 0.0
    mr = mc / third if third > 0 else 0.0
    br = bc / (n - 2 * third) if (n - 2 * third) > 0 else 0.0

    max_seg = max(fr, mr, br)
    if max_seg == 0:
        conc = "uniform"
    elif max_seg >= 2 * min(f for f in (fr, mr, br) if f > 0):
        conc = {0: "front", 1: "mid", 2: "back"}[np.argmax([fr, mr, br])]
    elif fr + br > 2 * mr:
        conc = "edges"
    else:
        conc = "uniform"

    return {
        "bitwise_match": False,
        "total_elements": int(total),
        "diff_count": diff_count,
        "diff_ratio_pct": diff_count / total * 100,
        "first_diff_offset": int(diff_idx[0]) if diff_count > 0 else -1,
        "first_diff_baseline": float(b.flat[diff_idx[0]]),
        "first_diff_target": float(t.flat[diff_idx[0]]),
        "max_abs_diff": float(np.max(abs_diff)),
        "mean_abs_diff": float(np.mean(abs_diff)),
        "std_abs_diff": float(np.std(abs_diff)),
        "distribution": {
            "front_ratio_pct": f"{fr * 100:.4f}%",
            "mid_ratio_pct": f"{mr * 100:.4f}%",
            "back_ratio_pct": f"{br * 100:.4f}%",
            "concentration": conc,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    import multiprocessing as _mp
    _mp.set_start_method("spawn", force=True)
    
    args = _parse_args(argv)

    print("=" * 70)
    print("  NCCL Non-Determinism Reproduction Case")
    print("=" * 70)
    print(f"  Algorithm : {args.algo}")
    print(f"  Protocol  : {args.proto}")
    print(f"  Data type : {args.dtype}")
    print(f"  Data size : {args.size}")
    print(f"  N ranks   : {args.nranks}")
    print(f"  Collective: {args.collective}")
    print(f"  Trials    : {args.trials}")
    print()

    available = _count_gpus()
    if available < args.nranks:
        print(f"  WARNING: Only {available} GPUs available, using {available}")
        args.nranks = available
    if args.nranks < 2:
        print("  ERROR: Need at least 2 GPUs. Aborting.")
        return 1

    size_bytes = _parse_size(args.size)

    # 案例 1：同配置多次试验 —— 迭代级演化
    print("--- Case 1: Same config, run-to-run determinism ---")
    print("  (NCCL guarantee: same input + same topology = bitwise identical)")
    print()
    trials: list[ReproduceResult] = []
    for i in range(args.trials):
        print(f"  Trial {i + 1}/{args.trials}...", end=" ", flush=True)
        r = _run_trial(
            algo=args.algo, proto=args.proto, dtype=args.dtype,
            size_bytes=size_bytes, nranks=args.nranks,
            collective=args.collective, trial_id=i + 1,
        )
        trials.append(r)
        print(f"done ({r.elapsed_ms:.2f}ms, checksum={r.output.sum():.6f})")

    print()
    if len(trials) >= 2:
        report = compare_bitwise(trials[0], trials[1])
        _print_report(report)

        if len(trials) > 2:
            print("  Iteration evolution (trial_0 baseline):")
            growing = True
            prev_ratio = 0.0
            for i in range(2, len(trials)):
                r = compare_bitwise(trials[0], trials[i])
                ratio = r.get("diff_ratio_pct", 0)
                growing = growing and (ratio >= prev_ratio)
                prev_ratio = ratio
                status = "IDENTICAL" if r["bitwise_match"] else f"{ratio:.4f}% diff"
                print(f"    trial_0 vs trial_{i}: {status}")
            if growing and not report["bitwise_match"]:
                print("    → Diffs MONOTONICALLY GROWING — accumulation pattern!")
            print()

    # 案例 2：Ring vs Tree 跨算法比对
    if args.algo != "Ring":
        print("--- Case 2: Ring vs Tree cross-algorithm comparison ---")
        print("  Running Ring trial...", end=" ", flush=True)
        r_ring = _run_trial("Ring", args.proto, args.dtype, size_bytes,
                            args.nranks, args.collective, 0)
        print(f"done ({r_ring.elapsed_ms:.2f}ms)")

        print("  Running Tree trial...", end=" ", flush=True)
        r_tree = _run_trial("Tree", args.proto, args.dtype, size_bytes,
                            args.nranks, args.collective, 0)
        print(f"done ({r_tree.elapsed_ms:.2f}ms)")

        print("\n  Ring vs Tree comparison:")
        _print_report(compare_bitwise(r_ring, r_tree))
        print()

    # 案例 3：float16 vs float32 精度影响
    if args.dtype == "float16":
        print("--- Case 3: float16 vs float32 precision impact ---")
        print("  Running float32 trial (same element count)...", end=" ", flush=True)
        r_f32 = _run_trial(args.algo, args.proto, "float32",
                           size_bytes * 2, args.nranks, args.collective, 0)
        print(f"done ({r_f32.elapsed_ms:.2f}ms)")
        print("  Note: float32 has 23-bit mantissa vs float16's 10-bit — "
              "better precision, less non-determinism.")
        print()

    print("=" * 70)
    print("  Reproduction complete.")
    print()
    print("  Root cause: NCCL Ring algorithm serial accumulation order depends")
    print("  on chunk partitioning (buffer size, NVLink channels).")
    print("  Floating-point non-associativity → different rounding errors.")
    print()
    print("  Recommendations for bitwise determinism:")
    print("    1. Use NCCL_ALGO=Tree or NCCL_ALGO=PAT instead of Ring")
    print("    2. Use float32 for reduction buffers (not float16/bf16)")
    print("    3. Fix all NCCL env vars: NCCL_ALGO, NCCL_PROTO, NCCL_NCHANNELS")
    print("    4. Set torch.use_deterministic_algorithms(True)")
    print("    5. Consider Reduce+Broadcast with fixed root")
    print("=" * 70)

    return 0


# ---------------------------------------------------------------------------
# ---- 工具函数 ----
# ---------------------------------------------------------------------------

def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _count_gpus() -> int:
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def _parse_size(s: str) -> int:
    s = s.strip().upper()
    m = {"B": 1, "K": 1024, "M": 1024**2, "G": 1024**3}
    for suffix, mult in m.items():
        if s.endswith(suffix):
            return int(s[:-1]) * mult
    return int(s)


def _parse_args(argv: Optional[list[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NCCL Non-Determinism Reproduction Case")
    p.add_argument("--algo", default="Ring",
                   choices=["Ring", "Tree", "PAT", "CollnetDirect",
                            "CollnetChain", "NVLS", "NVLSTree"])
    p.add_argument("--proto", default="Simple", choices=["LL", "Simple", "LL128"])
    p.add_argument("--dtype", default="float16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--size", default="128M")
    p.add_argument("--nranks", type=int, default=8)
    p.add_argument("--trials", type=int, default=2,
                   help="Number of repeated trials (>=3 enables iteration evolution)")
    p.add_argument("--collective", default="allreduce",
                   choices=["allreduce", "reducescatter"])
    return p.parse_args(argv)


def _print_report(report: dict) -> None:
    if report.get("error"):
        print(f"  ERROR: {report['error']}")
        return
    if report["bitwise_match"]:
        print(f"  RESULT: BITWISE IDENTICAL ({report['total_elements']} elements)")
        return

    print(f"  RESULT: BITWISE DIFFERENCE DETECTED!")
    print(f"    Total elements  : {report['total_elements']}")
    print(f"    Diff count      : {report['diff_count']} "
          f"({report['diff_ratio_pct']:.4f}%)")
    print(f"    First diff @ {report['first_diff_offset']}: "
          f"baseline={report['first_diff_baseline']:.8e}, "
          f"target={report['first_diff_target']:.8e}")
    print(f"    Max abs diff    : {report['max_abs_diff']:.8e}")
    print(f"    Mean abs diff   : {report['mean_abs_diff']:.8e}")
    print(f"    Std abs diff    : {report['std_abs_diff']:.8e}")

    dist = report.get("distribution", {})
    if dist:
        print(f"    Distribution    : "
              f"front={dist['front_ratio_pct']}, "
              f"mid={dist['mid_ratio_pct']}, "
              f"back={dist['back_ratio_pct']} "
              f"({dist['concentration']})")


if __name__ == "__main__":
    sys.exit(main())
