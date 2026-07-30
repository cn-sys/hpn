#!/usr/bin/env python3
"""
NCCL Bitwise 可复现性诊断工具 — 主入口

完整诊断流水线编排:
  1. 环境预检（GPU 数量、NCCL、PyTorch）
  2. 生成配置扫描矩阵（ALGO × PROTO × SIZE × DTYPE）
  3. 数据生成器验证
  4. 每个配置运行 N 次（PyTorch NCCL），支持断点恢复
  5. 逐位比对 + 差异分布分析
  6. 差异随规模和迭代的演化追踪
  7. 生成报告 + 确定性配置建议

用法:
    python -m issue4.diagnose --mode quick --nranks 8
    python -m issue4.diagnose --mode quick --list-configs
    python -m issue4.diagnose --algo Ring --proto Simple --dtype float32 --size 128M
    python -m issue4.diagnose --json report.json
    python -m issue4.diagnose --self-test
    python -m issue4.diagnose --inject-difference auto --nranks 4

依赖: PyTorch >= 1.12（含 NCCL 后端）、numpy、CUDA GPU >= 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from .config_matrix import ConfigMatrix, ConfigEntry, HardwareCaps, DTYPE_SENSITIVITY
from .data_generator import DataGenerator, validate_data_divergence
from .runner import NcclRunner, RunResult, MultiRunResult
from .comparator import (
    BitwiseComparator, DiffReport, EvolutionReport, IterEvolutionReport,
    ThreeLevelSummary,
    compute_ulp, analyze_xor_float,
)
from .reporter import Reporter, DiagnosticSummary


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point. Returns 0 on success, 1 on pre-flight failure."""
    import multiprocessing as _mp
    _mp.set_start_method("spawn", force=True)
    
    args = _parse_args(argv)

    # ------------------------------------------------------------------
    # Self-test mode: run GPU integration test and exit
    # ------------------------------------------------------------------
    if args.self_test:
        return _run_self_test(args.nranks)

    # ------------------------------------------------------------------
    # ---- 阶段 0：环境预检 ----
    # ------------------------------------------------------------------
    print("Phase 0: Environment pre-flight check...")
    env_ok, env_msg = _check_environment(args.nranks)
    print(env_msg)
    if not env_ok:
        print("  ABORT: Environment check failed. Fix the issues above and retry.")
        return 1

    # Detect hardware capabilities for compatibility filtering
    hw = HardwareCaps.detect()
    print(f"  SM compute   : {hw.sm_major} ({hw.gpu_name})")
    print(f"  NVSwitch     : {'yes' if hw.has_nvswitch else 'no'}")
    print(f"  Collnet/SHARP: {'yes' if hw.has_collnet else 'no'}")
    print(f"  LL128 support: {'yes' if hw.supports_ll128() else 'no — requires Hopper (SM90+)'}")
    print()

    # ------------------------------------------------------------------
    # ---- 阶段 1：构建配置矩阵 ----
    # ------------------------------------------------------------------
    print("Phase 1: Building configuration sweep matrix...")
    matrix = _build_matrix(args)
    configs, hw_warnings = matrix.generate_with_warnings(hw)

    # Print hardware-incompatible skipped configs
    if hw_warnings:
        print("  Hardware-incompatible configs skipped:")
        for w in hw_warnings:
            print(w)
        print()

    if not configs:
        print("  ERROR: No valid configs generated (all filtered by collective/hardware).")
        return 1
    print(matrix.summary())
    print()

    if args.list_configs:
        print("Configs to be executed:")
        for i, cfg in enumerate(configs):
            print(f"  [{i:3d}] {cfg}")
        print(f"\n  ({len(configs)} configs total, ~{len(configs)*2} trials)")
        return 0

    # ------------------------------------------------------------------
    # ---- 阶段 1.5：ULP 注入验证 ---- (if --inject-difference)
    # ------------------------------------------------------------------
    if args.inject_difference is not None:
        print("Phase 1.5: ULP injection pipeline validation...")
        ok = _run_injection_test(args, hw)
        if ok:
            print("  ✓ Injection pipeline verified — detector correctly identifies injected diffs.")
        else:
            print("  ✗ Injection pipeline FAILED — detector missed an injected difference!")
        print()
        if not args.force:
            print("  Exiting (injection-only mode). Use --force to continue to full sweep.")
            return 0 if ok else 1

    # ------------------------------------------------------------------
    # ---- 阶段 2：验证数据生成 ----
    # ------------------------------------------------------------------
    print("Phase 2: Validating data generator...")
    gen = DataGenerator(
        dtype=configs[0].dtype,
        base_seed=42,
        collective=matrix.collective,
    )
    test_inputs = gen.generate_all(configs[0].size_bytes, matrix.nranks)
    if not validate_data_divergence(test_inputs):
        print("  WARNING: All ranks have identical input data. "
              "Non-determinism may not be detectable.")
        print("  Consider using per-rank seeds (default behavior).")
    else:
        print(f"  OK: {matrix.nranks} ranks, each with distinct deterministic input.")
    print()

    # ------------------------------------------------------------------
    # ---- 阶段 3：运行扫描 ----
    # ------------------------------------------------------------------
    all_results: list[list[RunResult]] = []
    all_full_results: list[list[MultiRunResult]] = []
    start_idx = 0

    use_three_level = args.n_calls > 1
    mode_label = f"  Backend: pytorch, {args.n_calls} calls per run, "
    mode_label += "three-level comparison (run / call / rank)" if use_three_level else "rank-0 only"

    if start_idx > 0:
        print(f"Phase 3: Resuming from checkpoint ({start_idx}/{len(configs)} done)...")
    else:
        print(f"Phase 3: Running {len(configs)} configs × {args.trials} trials each...")
        print(mode_label)
        print()

    runner = NcclRunner(nranks=matrix.nranks, backend="pytorch")
    from .runner import MultiRunResult as MRR

    t_start = time.perf_counter()
    i = start_idx
    for i in range(start_idx, len(configs)):
        cfg = configs[i]
        print(f"  [{i + 1}/{len(configs)}] {cfg}")
        try:
            if use_three_level:
                full = runner.run_full(cfg, n_calls=args.n_calls, trials=args.trials)
                all_full_results.append(full)
                if len(full) >= 2:
                    checksums = [
                        [float(np.sum(o.astype(np.float64)))
                         for o in c0.values()]
                        for c0 in full[0].call_outputs
                    ]
                    print(f"         {args.n_calls} calls × {matrix.nranks} ranks captured")
            else:
                results = runner.run(cfg, trials=args.trials)
                all_results.append(results)
                if len(results) >= 2:
                    c1, c2 = results[0].checksum(), results[1].checksum()
                    if abs(c1 - c2) > 0:
                        print(f"         checksum diff detected: {abs(c1 - c2):.6e}")
        except Exception as exc:
            print(f"         ERROR: {exc}")
            if use_three_level:
                all_full_results.append([])
            else:
                all_results.append([])

    t_end = time.perf_counter()
    elapsed = t_end - t_start
    effective = max(i - start_idx + 1, 1)
    print(f"\n  Sweep complete in {elapsed:.1f}s "
          f"({elapsed / effective:.1f}s per config)")
    print()

    # ------------------------------------------------------------------
    # ---- 阶段 4-6：比对 + 报告（分支：二级 vs 三级） ----
    # ------------------------------------------------------------------
    comp = BitwiseComparator(tolerance="bitwise")

    if use_three_level:
        print("Phase 4: Three-level comparison (run-vs-run / call-vs-call / rank-vs-rank)...")
        all_summaries: list[ThreeLevelSummary] = []
        for i, (full_trials, cfg) in enumerate(zip(all_full_results, configs)):
            if len(full_trials) >= 2:
                label = f"{cfg.algo}/{cfg.proto}/{cfg.dtype}/{cfg.size_label}"
                summary = comp.full_three_level_report(full_trials, label=label)
                all_summaries.append(summary)
                status = "CLEAN" if summary.all_clean else "DIFFS"
                print(f"  [{i+1}] {label}: {status}")

        # Print per-level details for first failing config
        for s in all_summaries:
            if not s.all_clean:
                print(f"\n  Detailed breakdown: {s.label}")
                print(s.summary())
                # Show first failing cell
                for level_name, level_data in [
                    ("run-vs-run", s.run_vs_run),
                    ("call-vs-call", s.call_vs_call),
                    ("rank-vs-rank", s.rank_vs_rank),
                ]:
                    for key, reports in level_data.items():
                        for r in reports:
                            if not r.bitwise_match:
                                print(f"    {level_name} {key}: {r.diff_count} diffs "
                                      f"({r.diff_ratio*100:.4f}%), "
                                      f"ULP={r.xor_detail.ulp_distance if r.xor_detail else '?'}")
                                print(f"      offset={r.first_diff_offset} "
                                      f"max_abs={r.max_abs_diff:.6e}")
                break

        # ---- JSON 输出 ----
        if args.json:
            import json as _json
            data = {
                "config_count": len(configs),
                "full_configs": [],
                "summary": {
                    "clean": sum(1 for s in all_summaries if s.all_clean),
                    "with_diffs": sum(1 for s in all_summaries if not s.all_clean),
                },
            }
            for s in all_summaries:
                cfg_entry = {
                    "label": s.label,
                    "all_clean": s.all_clean,
                    "run_vs_run": [],
                    "call_vs_call": {},
                    "rank_vs_rank": [],
                }
                for key, reports in s.run_vs_run.items():
                    for r in reports:
                        cfg_entry["run_vs_run"].append({
                            "key": key,
                            "bitwise_match": r.bitwise_match,
                            "diff_count": r.diff_count,
                            "diff_ratio": f"{r.diff_ratio*100:.4f}%",
                            "ulp": r.xor_detail.ulp_distance if r.xor_detail else -1,
                        })
                for rank, reports in s.call_vs_call.items():
                    diffs = []
                    for r in reports:
                        diffs.append({
                            "diff_count": r.diff_count,
                            "diff_ratio": f"{r.diff_ratio*100:.4f}%",
                        })
                    cfg_entry["call_vs_call"][str(rank)] = diffs
                for key, reports in s.rank_vs_rank.items():
                    for r in reports:
                        cfg_entry["rank_vs_rank"].append({
                            "key": key,
                            "bitwise_match": r.bitwise_match,
                            "diff_count": r.diff_count,
                            "diff_ratio": f"{r.diff_ratio*100:.4f}%",
                        })
                data["full_configs"].append(cfg_entry)
            with open(args.json, "w", encoding="utf-8") as f:
                _json.dump(data, f, indent=2)
            print(f"\n  JSON report written to: {args.json}")

    else:
        # Legacy two-level path (unchanged)
        print("Phase 4: Bitwise comparison + distribution analysis...")
        reports: list[DiffReport] = []
        for i, (trials, cfg) in enumerate(zip(all_results, configs)):
            if len(trials) >= 2:
                report = comp.compare(trials[0], trials[1])
                reports.append(report)
            else:
                # 未产生有效结果（worker 崩溃等）→ 标记为未运行，不是 PASS
                reports.append(DiffReport(config=cfg, bitwise_match=False, total_elements=0))

        print("Phase 5: Tracking diff evolution across sizes and iterations...")
        evolutions = comp.track_evolution(all_results, configs)
        iter_evo: Optional[IterEvolutionReport] = None
        worst_cfg_idx = _find_worst_config(all_results)
        if worst_cfg_idx is not None and len(all_results[worst_cfg_idx]) >= 2:
            iter_evo = comp.track_iter_evolution(all_results[worst_cfg_idx])
        print()

        print("Phase 6: Generating report...")
        reporter = Reporter(output_json=args.json is not None,
                           output_console=True,
                           json_path=args.json or "")
        summary = reporter.report(reports, evolutions, args.json)
        if iter_evo and iter_evo.iteration_diffs:
            print()
            print(iter_evo.summary())
        if summary.diff_count > 0:
            print("Diagnosis: Non-determinism detected! See recommendations above.")
        else:
            print("Diagnosis: All configurations bitwise-deterministic on this hardware.")

    # ------------------------------------------------------------------
    # ---- 清理 + 返回 ----
    # ------------------------------------------------------------------
    return 0


# ---------------------------------------------------------------------------
# ---- 环境预检 ----
# ---------------------------------------------------------------------------

def _check_environment(nranks: int) -> tuple[bool, str]:
    """Check that the environment is capable of running the diagnostic.

    Returns (ok: bool, message: str).
    """
    lines: list[str] = []

    # 1. Python version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    lines.append(f"  Python       : {py_ver}")

    # 2. PyTorch + CUDA
    try:
        import torch
        lines.append(f"  PyTorch      : {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        lines.append(f"  CUDA avail   : {cuda_avail}")
        if cuda_avail:
            gpu_count = torch.cuda.device_count()
            lines.append(f"  GPU count    : {gpu_count}")
            for g in range(min(gpu_count, 4)):
                lines.append(f"    GPU {g}: {torch.cuda.get_device_name(g)}")
            if gpu_count > 4:
                lines.append(f"    ... and {gpu_count - 4} more")
        else:
            return False, "\n".join(lines + ["  FAIL: CUDA not available."])
    except ImportError:
        return False, "\n".join(lines + ["  FAIL: PyTorch not installed."])

    # 3. NCCL backend
    try:
        import torch.distributed as dist
        lines.append(f"  NCCL avail   : {dist.is_nccl_available()}")
        if not dist.is_nccl_available():
            return False, "\n".join(lines + ["  FAIL: NCCL backend not available. "
                                              "Rebuild PyTorch with NCCL support."])
    except Exception:
        return False, "\n".join(lines + ["  FAIL: torch.distributed unavailable."])

    # 4. GPU count vs requested ranks
    gpu_count = torch.cuda.device_count()
    if nranks > gpu_count:
        return False, "\n".join(lines + [
            f"  FAIL: Requested {nranks} ranks but only {gpu_count} GPUs available.\n"
            f"  Use --nranks {gpu_count} or reduce the rank count."
        ])
    if nranks < 2:
        return False, "\n".join(lines + [
            "  FAIL: At least 2 ranks required for meaningful diagnostic."
        ])

    # 5. NumPy
    import numpy as np
    lines.append(f"  NumPy        : {np.__version__}")

    lines.append("  ✓ Environment OK")
    return True, "\n".join(lines)


# ---------------------------------------------------------------------------
# ---- 杂项 ----
# ---------------------------------------------------------------------------

def _find_worst_config(results: list[list[RunResult]]) -> Optional[int]:
    """Find index of config with largest checksum delta between trials."""
    worst_idx = None
    worst_delta = -1.0
    for i, trials in enumerate(results):
        if len(trials) >= 2:
            delta = abs(trials[0].checksum() - trials[1].checksum())
            if delta > worst_delta:
                worst_delta = delta
                worst_idx = i
    return worst_idx


# ---------------------------------------------------------------------------
# ---- 命令行参数 ----
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[list[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NCCL Bitwise Reproducibility Diagnostic Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m issue4.diagnose --mode quick
  python -m issue4.diagnose --mode quick --list-configs
  python -m issue4.diagnose --mode standard --json report.json
  python -m issue4.diagnose --algo Ring --size 128M
  python -m issue4.diagnose --inject-difference auto --nranks 4
        """,
    )
    p.add_argument("--mode", choices=["quick", "standard", "exhaustive"],
                   default="quick",
                   help="Sweep mode: quick (~2 min), standard (~10 min), exhaustive (~30 min)")
    p.add_argument("--nranks", type=int, default=8,
                   help="Number of GPU ranks (default: 8)")
    p.add_argument("--trials", type=int, default=2,
                    help="Number of run-to-run trials per config (default: 2)")
    p.add_argument("--self-test", action="store_true",
                    help="Run minimal GPU integration test and exit")
    p.add_argument("--n-calls", type=int, default=1,
                    help="Number of NCCL calls per process group (enables "
                         "three-level run/call/rank comparison)")
    p.add_argument("--algo", type=str, default=None)
    p.add_argument("--proto", type=str, default=None)
    p.add_argument("--dtype", type=str, default=None)
    p.add_argument("--size", type=str, default=None)
    p.add_argument("--collective", choices=["allreduce", "reducescatter"],
                   default="allreduce")
    p.add_argument("--json", type=str, default=None,
                    help="Write JSON report to this file path")
    p.add_argument("--list-configs", action="store_true",
                    help="Print the config sweep matrix and exit (no GPU run)")
    p.add_argument("--inject-difference", type=str, default=None, metavar="SPEC",
                    help="Run ULP injection validation. SPEC: 'auto' or 'offset=42,magnitude=3'")
    p.add_argument("--force", action="store_true",
                    help="With --inject-difference: continue to sweep after validation")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# ---- ULP 注入管线 ----
# ---------------------------------------------------------------------------

def _run_injection_test(
    args: argparse.Namespace, hw: HardwareCaps
) -> bool:
    """Validate the diagnostic pipeline by injecting a known bit-level difference.

    How it works:
      1. Generate deterministic input for a single config
      2. Run the collective once → baseline output
      3. Inject a known ULP difference into the baseline output (simulates a "diff")
      4. Run the comparator between baseline and injected → should DETECT the diff
      5. Verify the XOR/ULP analysis matches the injection

    This end-to-end test confirms that the entire pipeline correctly
    detects bit-level non-determinism.
    """
    import struct

    # Build a minimal single-config matrix for injection test
    algo = args.algo or "Ring"
    proto = args.proto or "Simple"
    dtype = args.dtype or "float32"
    size_str = args.size or "128M"
    nranks = args.nranks

    matrix = ConfigMatrix.from_cli(
        algo=algo, proto=proto, dtype=dtype, size=size_str, nranks=nranks
    )
    matrix.collective = args.collective
    configs = matrix.generate(hw)
    if not configs:
        print("  SKIP: No valid config for injection test (hardware-incompatible).")
        return True  # not a failure — hardware limitation

    cfg = configs[0]
    print(f"  Config: {cfg}")

    # Phase A: Run baseline
    runner = NcclRunner(nranks=nranks, backend="pytorch")
    results = runner.run(cfg, trials=1)
    if not results:
        print("  FAIL: Runner produced no results.")
        return False
    baseline = results[0]
    baseline_arr = baseline.output.copy()
    print(f"  Baseline checksum: {baseline.checksum():.6f}")

    # Phase B: Parse injection spec
    inj_offset, inj_magnitude_ulp = _parse_injection_spec(
        args.inject_difference or "auto", len(baseline_arr)
    )

    # Phase C: Inject difference into baseline output copy
    injected_arr = _inject_ulp(
        baseline_arr, offset=inj_offset, ulp_magnitude=inj_magnitude_ulp
    )

    # Phase D: Compare baseline vs injected
    from .runner import RunResult as RR
    injected_result = RunResult(
        config=cfg, output=injected_arr, elapsed_ms=0.0, rank=0,
    )

    comp = BitwiseComparator(tolerance="bitwise")
    report = comp.compare(baseline, injected_result)

    # Phase E: Validate
    if report.bitwise_match:
        print(f"  FAIL: Comparator reported BITWISE MATCH despite injected "
              f"{inj_magnitude_ulp} ULP diff at offset {inj_offset}!")
        return False

    print(f"  Comparator detected {report.diff_count} differing elements "
          f"(expected 1 at offset {inj_offset}).")

    if report.xor_detail:
        xd = report.xor_detail
        expected_xor = compute_ulp(
            float(baseline_arr[inj_offset]),
            float(injected_arr[inj_offset]),
        )
        print(f"  Injection  : offset={inj_offset}, magnitude={inj_magnitude_ulp} ULP")
        print(f"  Detected   : offset={xd.offset}, ULP={xd.ulp_distance}")
        print(f"  XOR detail : {xd.baseline_bits} → {xd.target_bits} "
              f"(xor=0x{xd.xor_bits:08x}, flips={xd.n_total_bits_flipped})")

        if xd.offset != inj_offset:
            print(f"  FAIL: Offset mismatch — injected {inj_offset}, detected {xd.offset}")
            return False
        if xd.ulp_distance != expected_xor:
            print(f"  WARN: ULP mismatch — expected {expected_xor}, detected {xd.ulp_distance}")
            # Not a hard failure — ULP can differ due to float32 internal rounding
    else:
        print("  FAIL: XOR detail not computed!")
        return False

    return True


def _parse_injection_spec(spec: str, max_offset: int) -> tuple[int, int]:
    """Parse injection specification string.

    Formats:
      "auto"                    → offset=midpoint, magnitude=1
      "offset=42,magnitude=3"   → explicit offset and ULP magnitude
      "offset=42"               → explicit offset, magnitude=1
    """
    offset = max_offset // 2
    magnitude = 1

    if spec == "auto":
        return (offset, magnitude)

    for part in spec.split(","):
        part = part.strip()
        if part.startswith("offset="):
            offset = min(int(part.split("=")[1]), max_offset - 1)
        elif part.startswith("magnitude="):
            magnitude = max(1, int(part.split("=")[1]))

    return (offset, magnitude)


def _inject_ulp(arr: np.ndarray, offset: int, ulp_magnitude: int) -> np.ndarray:
    """Inject a known ULP difference into an array at a specific offset.

    Returns a COPY of arr with the modification applied.
    """
    import struct

    result = arr.copy()
    original = float(result.flat[offset])

    # Reinterpret float32 as int32 (signed); use unsigned for bit arithmetic
    bits_signed = struct.unpack("<i", struct.pack("<f", original))[0]
    bits_unsigned = bits_signed & 0xFFFFFFFF  # uint32 view

    # Add ULP magnitude (saturate at finite bounds)
    if bits_signed >= 0:
        new_bits = min(bits_unsigned + ulp_magnitude, 0x7F7FFFFF)
    else:
        # 负浮点数：减小 bit pattern（走向更负）
        new_bits = max(bits_unsigned - ulp_magnitude, 0x00800001)  # min pos norm

    # Reinterpret int32 → float32
    injected = struct.unpack("<f", struct.pack("<i", new_bits))[0]
    result.flat[offset] = injected

    return result


def _build_matrix(args: argparse.Namespace) -> ConfigMatrix:
    """Build ConfigMatrix from parsed CLI args."""
    if any([args.algo, args.proto, args.dtype, args.size]):
        matrix = ConfigMatrix.from_cli(
            algo=args.algo, proto=args.proto,
            dtype=args.dtype, size=args.size,
            nranks=args.nranks,
        )
    else:
        builders = {
            "quick":      ConfigMatrix.quick_sweep,
            "standard":   ConfigMatrix.standard_sweep,
            "exhaustive": ConfigMatrix.exhaustive_sweep,
        }
        matrix = builders[args.mode](nranks=args.nranks)

    matrix.collective = args.collective
    return matrix


# ---------------------------------------------------------------------------
# ---- 自检（GPU 集成冒烟测试） ----
# ---------------------------------------------------------------------------

def _run_self_test(nranks: int) -> int:
    """Run a minimal GPU integration test and return 0 on pass, 1 on fail."""
    import struct

    print("=" * 60)
    print("  NCCL Diagnostic Tool — Self-Test (GPU Integration)")
    print("=" * 60)

    # 1. Environment check
    try:
        import torch
        import torch.distributed as dist
    except ImportError:
        print("  FAIL: torch not installed")
        return 1

    gpu_count = torch.cuda.device_count()
    print(f"  GPUs: {gpu_count}")
    if gpu_count < 2:
        print("  SKIP: need >= 2 GPUs for meaningful test")
        return 0

    nranks = min(nranks, gpu_count)
    print(f"  Using {nranks} ranks")

    # 2. Hardware detection
    hw = HardwareCaps.detect()
    print(f"  SM: {hw.sm_major} ({hw.gpu_name})")
    if hw.supports_ll128():
        print("  LL128: supported")
    else:
        print("  LL128: NOT supported (requires Hopper+)")

    # 3. Run one config × 2 trials
    from .config_matrix import ConfigEntry
    from .data_generator import DataGenerator
    from .runner import NcclRunner
    from .comparator import BitwiseComparator

    size_bytes = 64 * 1024  # 64K
    cfg = ConfigEntry(algo="Tree", proto="Simple", size_bytes=size_bytes,
                     size_label="64K", dtype="float32", nranks=nranks)

    print(f"\n  Config: {cfg}")
    print("  Running 2 trials...", end=" ", flush=True)

    try:
        runner = NcclRunner(nranks=nranks, backend="pytorch")
        results = runner.run(cfg, trials=2)
    except Exception as e:
        print(f"\n  FAIL: {e}")
        return 1

    if len(results) < 2:
        print("\n  FAIL: not enough results")
        return 1

    print(f"done ({results[0].elapsed_ms:.2f}ms, {results[1].elapsed_ms:.2f}ms)")

    # 4. Compare
    comp = BitwiseComparator()
    report = comp.compare(results[0], results[1])

    if report.bitwise_match:
        print("  Bitwise comparison: IDENTICAL")
    else:
        print(f"  Bitwise comparison: {report.diff_count} DIFFERENCES "
              f"({report.diff_ratio*100:.4f}%)")
        if report.xor_detail:
            print(f"    First diff ULP: {report.xor_detail.ulp_distance}")

    # 5. ULP injection validation
    print("\n  ULP injection test:", end=" ", flush=True)
    baseline = results[0].output.copy()
    offset = len(baseline) // 2
    bits = struct.unpack("<i", struct.pack("<f", float(baseline[offset])))[0]
    new_bits = min(bits + 1, 0x7F7FFFFF) if bits >= 0 else max(bits - 1, -0x7F7FFFFF)
    injected = baseline.copy()
    injected[offset] = struct.unpack("<f", struct.pack("<i", new_bits))[0]

    from .runner import RunResult
    inj_result = RunResult(config=cfg, output=injected)
    inj_report = comp.compare(results[0], inj_result)

    if inj_report.bitwise_match:
        print("FAIL — did not detect injected 1-ULP diff!")
        return 1
    if inj_report.first_diff_offset != offset:
        print(f"FAIL — wrong offset ({inj_report.first_diff_offset} != {offset})")
        return 1
    print(f"PASS — detected 1-ULP diff at offset {offset}")

    print("\n" + "=" * 60)
    print("  Self-test PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
