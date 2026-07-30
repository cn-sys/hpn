"""
逐位比对器 — NCCL 确定性诊断。

对 RunResult 输出进行逐元素比对，提供:
  - 首次位级差异出现的偏移位置
  - 差异量级统计（最大值、均值、标准差）
  - 差异元素比例
  - 差异随数据规模 / 迭代次数的演化追踪
  - 差异空间分布分析（前/中/后三区聚类 + 直方图）
  - 逐位 XOR 分解 + ULP 距离

核心比对使用 numpy.array_equal — 零容忍度。
这能捕捉到最小的位级非确定性。
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config_matrix import ConfigEntry
from .runner import RunResult


# ---------------------------------------------------------------------------
# ---- 报告数据类 ----
# ---------------------------------------------------------------------------

@dataclass
class XorDetail:
    """Byte-level XOR + ULP analysis for a single differing float32 element.

    Interprets two float32 values as IEEE-754 bit patterns, computes:
      - XOR of the raw 32-bit patterns → reveals which bits flipped
      - Sign / exponent / mantissa sub-patterns
      - ULP (Units in Last Place) distance — the canonical measure of
        floating-point difference independent of magnitude.
    """

    offset: int = 0               # element index in tensor
    baseline_bits: str = ""        # hex representation, e.g. "0x40490fdb"
    target_bits: str = ""
    xor_bits: int = 0              # raw XOR of the two bit-patterns
    sign_diff: bool = False        # did the sign bit flip?
    exp_diff: int = 0              # exponent field difference
    mantissa_diff: int = 0         # mantissa (fraction) field difference
    ulp_distance: int = 0          # ULP distance between the two values
    n_mantissa_bits_flipped: int = 0  # count of flipped mantissa bits
    n_total_bits_flipped: int = 0     # total bit flips in the 32-bit word

    def summary(self) -> str:
        return (
            f"XOR @ offset {self.offset}: "
            f"baseline={self.baseline_bits} target={self.target_bits}\n"
            f"       XOR=0x{self.xor_bits:08x}  "
            f"ULP={self.ulp_distance}  "
            f"sign={'FLIP' if self.sign_diff else 'ok'}  "
            f"exp_diff={self.exp_diff}  "
            f"mantissa_flips={self.n_mantissa_bits_flipped}/{23}"
        )


def compute_ulp(a: float, b: float) -> int:
    """计算两个 float32 值的 ULP 距离。

    ULP (Units in Last Place): the number of representable float32 values
    between `a` and `b`. For IEEE-754 binary32:
      - Reinterpret both as int32
      - Convert to sign-magnitude (handle the two's complement quirk)
      - Absolute difference = ULP distance

    Reference: "Comparing Floating Point Numbers, 2012 Edition" — Random ASCII

    Examples:
      compute_ulp(1.0, 1.0 + 1e-7) ≈ 1    (adjacent floats = 1 ULP)
      compute_ulp(1.0, 2.0) ≈ 2**23         (~8 million ULPs)
      compute_ulp(0.0, -0.0) == 0           (IEEE-754: +0 and −0 are equal)
    """
    # 通过 struct 将 float32 重新解释为 int32，避免 numpy 依赖
    a_bits = struct.unpack("<i", struct.pack("<f", float(a)))[0]
    b_bits = struct.unpack("<i", struct.pack("<f", float(b)))[0]

    # 处理 +0 vs -0 和 NaN vs NaN（IEEE-754：值相等但位模式可能不同）
    # 同时处理一般的值相等（位模式相同或浮点值相等）
    if float(a) == float(b):
        return 0

    # 二进制补码 → 符号-幅度转换，确保正确排序：
    # 二进制补码中负数是"反向"的 — e.g., -1 is 0xBF800000
    # while -2 is 0xC0000000. For ULP distance we need the signed-integer ordering.
    def _to_signed(x: int) -> int:
        # IEEE-754 二进制补码转符号-幅度：负数翻转位序以保证 ULP 距离单调
        if x & 0x80000000:  # negative (or -0)
            return 0x80000000 - (x & 0x7FFFFFFF)
        return x

    a_signed = _to_signed(a_bits)
    b_signed = _to_signed(b_bits)
    return abs(a_signed - b_signed)


def analyze_xor_float(a: float, b: float, offset: int = 0) -> XorDetail:
    """对两个 float32 值进行字节级 XOR 分解。

    IEEE-754 binary32 layout: [sign:1][exponent:8][mantissa:23]
    """
    a_bits = struct.unpack("<I", struct.pack("<f", float(a)))[0]
    b_bits = struct.unpack("<I", struct.pack("<f", float(b)))[0]
    xor = a_bits ^ b_bits

    # 提取各字段
    sign_mask = 0x80000000
    exp_mask = 0x7F800000
    mantissa_mask = 0x007FFFFF

    sign_diff = (xor & sign_mask) != 0
    exp_diff = abs(
        ((a_bits & exp_mask) >> 23) - ((b_bits & exp_mask) >> 23)
    )
    mant_diff = abs(
        (a_bits & mantissa_mask) - (b_bits & mantissa_mask)
    )

    # 统计尾数区域翻转的位数
    mant_xor = xor & mantissa_mask
    n_mantissa_flips = mant_xor.bit_count()

    # 统计总翻转位数
    n_total_flips = xor.bit_count()

    ulp = compute_ulp(a, b)

    return XorDetail(
        offset=offset,
        baseline_bits=f"0x{a_bits:08x}",
        target_bits=f"0x{b_bits:08x}",
        xor_bits=xor,
        sign_diff=sign_diff,
        exp_diff=exp_diff,
        mantissa_diff=mant_diff,
        ulp_distance=ulp,
        n_mantissa_bits_flipped=n_mantissa_flips,
        n_total_bits_flipped=n_total_flips,
    )


@dataclass
class DiffReport:
    """单个配置对的逐位比对详细报告。"""

    config: ConfigEntry
    bitwise_match: bool
    first_diff_offset: int = -1          # -1 if no diff
    first_diff_baseline: float = 0.0
    first_diff_target: float = 0.0
    diff_count: int = 0                   # number of differing elements
    diff_ratio: float = 0.0               # diff_count / total_elements
    max_abs_diff: float = 0.0
    mean_abs_diff: float = 0.0
    std_abs_diff: float = 0.0
    total_elements: int = 0
    diff_indices: Optional[np.ndarray] = None  # indices of all diff positions
    distribution: Optional[DiffDistribution] = None  # spatial clustering
    xor_detail: Optional[XorDetail] = None   # XOR/ULP for first-diff element

    def summary(self) -> str:
        if self.bitwise_match:
            return (
                f"[PASS] {self.config}\n"
                f"       {self.total_elements} elements — bitwise identical"
            )

        lines = [
            f"[FAIL] {self.config}",
            f"       Total: {self.total_elements} elements, "
            f"diff: {self.diff_count} ({self.diff_ratio * 100:.4f}%)",
            f"       First diff @ offset {self.first_diff_offset}: "
            f"baseline={self.first_diff_baseline:.8e}, "
            f"target={self.first_diff_target:.8e}",
            f"       max_abs_diff={self.max_abs_diff:.8e}, "
            f"mean_abs_diff={self.mean_abs_diff:.8e}",
        ]
        if self.distribution:
            lines.append(f"       {self.distribution}")
        if self.xor_detail:
            lines.append(f"       {self.xor_detail.summary()}")
        return "\n".join(lines)


@dataclass
class DiffDistribution:
    """Spatial distribution of differences within the output tensor."""

    front_ratio: float = 0.0   # diff ratio in first 1/3 of tensor
    mid_ratio: float = 0.0     # diff ratio in middle 1/3
    back_ratio: float = 0.0    # diff ratio in last 1/3
    diff_concentration: str = "uniform"  # "front" / "mid" / "back" / "uniform" / "edges"

    # 差异量级直方图（log10 分桶）
    histogram_buckets: dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"Dist: front={self.front_ratio * 100:.3f}% "
            f"mid={self.mid_ratio * 100:.3f}% "
            f"back={self.back_ratio * 100:.3f}% "
            f"concentration={self.diff_concentration}"
        )

    def to_dict(self) -> dict:
        return {
            "front_ratio_pct": f"{self.front_ratio * 100:.4f}%",
            "mid_ratio_pct": f"{self.mid_ratio * 100:.4f}%",
            "back_ratio_pct": f"{self.back_ratio * 100:.4f}%",
            "concentration": self.diff_concentration,
            "histogram_buckets": self.histogram_buckets,
        }


@dataclass
class EvolutionReport:
    """追踪差异随规模或迭代的演化。"""

    config_label: str                     # e.g. "Ring/Simple/float32"
    entries: list[tuple[int, DiffReport]] = field(default_factory=list)
    # (size_bytes, diff_report)

    def add(self, size_bytes: int, report: DiffReport) -> None:
        self.entries.append((size_bytes, report))

    def diff_ratio_curve(self) -> dict[str, list]:
        """Return {size_labels: [...], diff_ratios: [...]} for plotting."""
        sizes: list[str] = []
        ratios: list[float] = []
        for sz, r in self.entries:
            sizes.append(_format_size(sz))
            ratios.append(r.diff_ratio)
        return {"size_labels": sizes, "diff_ratios": ratios}

    def summary(self) -> str:
        lines = [f"Evolution: {self.config_label}"]
        for sz, r in self.entries:
            status = "IDENTICAL" if r.bitwise_match else f"{r.diff_ratio * 100:.4f}% diff"
            lines.append(f"  {_format_size(sz):>6s} → {status}")
        return "\n".join(lines)


@dataclass
class IterEvolutionReport:
    """Tracks diff accumulation across iterations within a single run.

    Key for the issue requirement: "差异随迭代的演化".
    In real training, differences accumulate step-by-step. This tracks
    how the diff between a fixed baseline and each iteration grows.
    """

    config_label: str
    iteration_diffs: list[dict] = field(default_factory=list)
    # Each entry: {"iter": N, "diff_count": ..., "diff_ratio": ..., "max_abs_diff": ...}

    def add(self, iteration: int, report: DiffReport) -> None:
        self.iteration_diffs.append({
            "iter": iteration,
            "diff_count": report.diff_count,
            "diff_ratio": report.diff_ratio,
            "max_abs_diff": report.max_abs_diff,
        })

    def growing(self) -> bool:
        """Check if diffs are monotonically growing (accumulation pattern)."""
        if len(self.iteration_diffs) < 2:
            return False
        ratios = [d["diff_ratio"] for d in self.iteration_diffs]
        return all(ratios[i] <= ratios[i + 1] for i in range(len(ratios) - 1))

    def summary(self) -> str:
        lines = [f"Iter Evolution: {self.config_label}"]
        for d in self.iteration_diffs:
            lines.append(
                f"  iter {d['iter']:3d}: diff={d['diff_count']} "
                f"({d['diff_ratio'] * 100:.4f}%)  max={d['max_abs_diff']:.6e}"
            )
        if self.growing():
            lines.append("  → Diffs are MONOTONICALLY GROWING (accumulation)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Comparator
# ---------------------------------------------------------------------------

@dataclass
class BitwiseComparator:
    """以逐位精度比对 NCCL 运行输出。

    Parameters
    ----------
    tolerance : str
        'bitwise' — strict element equality (default for diagnostic)
        'relative' — use relative tolerance (for sanity checks)
    rtol : float
        Relative tolerance for 'relative' mode.
    """

    tolerance: str = "bitwise"
    rtol: float = 1e-5

    # ------------------------------------------------------------------
    # ---- 单次比对 ----
    # ------------------------------------------------------------------

    def compare(
        self, baseline: RunResult, target: RunResult
    ) -> DiffReport:
        """逐元素比对两个运行结果。"""
        b = baseline.output
        t = target.output

        if b.shape != t.shape:
            raise ValueError(
                f"Shape mismatch: baseline {b.shape} vs target {t.shape}"
            )
        total = b.size

        if self.tolerance == "bitwise":
            diff_mask = (b != t)
        elif self.tolerance == "relative":
            diff_mask = ~np.isclose(b, t, rtol=self.rtol, atol=0)
        else:
            raise ValueError(f"Unknown tolerance mode: {self.tolerance}")

        diff_indices = np.where(diff_mask)[0]
        diff_count = len(diff_indices)

        report = DiffReport(
            config=target.config,
            bitwise_match=(diff_count == 0),
            total_elements=int(total),
            diff_count=diff_count,
            diff_ratio=diff_count / total if total > 0 else 0.0,
            diff_indices=diff_indices if diff_count > 0 else None,
        )

        if diff_count > 0:
            abs_diff = np.abs(b[diff_mask].astype(np.float64) -
                              t[diff_mask].astype(np.float64))
            first_idx = diff_indices[0]
            report.first_diff_offset = int(first_idx)
            report.first_diff_baseline = float(b.flat[first_idx])
            report.first_diff_target = float(t.flat[first_idx])
            report.max_abs_diff = float(np.max(abs_diff))
            report.mean_abs_diff = float(np.mean(abs_diff))
            report.std_abs_diff = float(np.std(abs_diff))

            # --- Spatial distribution analysis ---
            report.distribution = self.analyze_distribution(
                diff_indices, abs_diff, int(total)
            )

            # --- XOR / ULP analysis of first diff ---
            report.xor_detail = analyze_xor_float(
                report.first_diff_baseline,
                report.first_diff_target,
                offset=report.first_diff_offset,
            )

        return report

    # ------------------------------------------------------------------
    # ---- 分布分析 ----
    # ------------------------------------------------------------------

    def analyze_distribution(
        self, diff_indices: np.ndarray, abs_diffs: np.ndarray, total_elements: int
    ) -> DiffDistribution:
        """分析差异在张量中的空间聚类。

        This matters because:
          - Front-heavy diffs → initialization / first-chunk issue
          - Back-heavy diffs  → tail-of-chunk rounding (most common in NCCL)
          - Uniform diffs     → systemic non-determinism (algorithm-level)
          - Edges             → chunk-boundary effects
        """
        n = total_elements
        third = n // 3

        # Partition mask
        front_mask = diff_indices < third
        mid_mask = (diff_indices >= third) & (diff_indices < 2 * third)
        back_mask = diff_indices >= 2 * third

        front_count = int(np.sum(front_mask))
        mid_count = int(np.sum(mid_mask))
        back_count = int(np.sum(back_mask))

        # Per-segment ratios (normalized by segment size)
        front_ratio = front_count / third if third > 0 else 0.0
        mid_ratio = mid_count / third if third > 0 else 0.0
        back_ratio = back_count / (n - 2 * third) if (n - 2 * third) > 0 else 0.0

        # Concentration heuristic
        max_seg = max(front_ratio, mid_ratio, back_ratio)
        if max_seg == 0:
            concentration = "uniform"
        elif max_seg >= 2 * min(f for f in (front_ratio, mid_ratio, back_ratio) if f > 0):
            # One segment has 2× more diffs than another → concentrated
            concentration = {
                0: "front", 1: "mid", 2: "back"
            }[np.argmax([front_ratio, mid_ratio, back_ratio])]
        elif front_ratio + back_ratio > 2 * mid_ratio:
            concentration = "edges"
        else:
            concentration = "uniform"

        # --- Histogram of diff magnitudes (log10 buckets) ---
        histogram: dict[str, int] = {}
        if len(abs_diffs) > 0:
            log_abs = np.log10(abs_diffs + 1e-40)
            bins = [-12, -10, -8, -6, -4, -2, 0]
            for lo, hi in zip(bins, bins[1:]):
                count = int(np.sum((log_abs >= lo) & (log_abs < hi)))
                if count > 0:
                    histogram[f"1e{lo}~1e{hi}"] = count

        return DiffDistribution(
            front_ratio=front_ratio,
            mid_ratio=mid_ratio,
            back_ratio=back_ratio,
            diff_concentration=concentration,
            histogram_buckets=histogram,
        )

    # ------------------------------------------------------------------
    # ---- 多试验 & 跨配置比对 ----
    # ------------------------------------------------------------------

    def compare_trials(self, results: list[RunResult]) -> list[DiffReport]:
        """将多次试验与首次（基准）比对。"""
        if len(results) < 2:
            return []
        baseline = results[0]
        reports: list[DiffReport] = []
        for target in results[1:]:
            reports.append(self.compare(baseline, target))
        return reports

    def compare_configs(
        self, results: list[list[RunResult]]
    ) -> list[DiffReport]:
        """对每个配置（含 2+ 试验），比对试验 0 与试验 1。

        Returns one DiffReport per config.
        """
        reports: list[DiffReport] = []
        for trials in results:
            if len(trials) < 2:
                continue
            reports.append(self.compare(trials[0], trials[1]))
        return reports

    # ------------------------------------------------------------------
    # ---- 跨规模的演化追踪 ----
    # ------------------------------------------------------------------

    def track_evolution(
        self, results: list[list[RunResult]], matrix: list[ConfigEntry]
    ) -> dict[str, EvolutionReport]:
        """按 (algo/proto/dtype) 分组，追踪差异率 vs 数据规模。

        Returns {label: EvolutionReport} keyed by config label.
        """
        evolutions: dict[str, EvolutionReport] = {}

        for i, (trials, cfg) in enumerate(zip(results, matrix)):
            label = f"{cfg.algo}/{cfg.proto}/{cfg.dtype}"
            if label not in evolutions:
                evolutions[label] = EvolutionReport(config_label=label)

            if len(trials) >= 2:
                report = self.compare(trials[0], trials[1])
                evolutions[label].add(cfg.size_bytes, report)

        return evolutions

    # ------------------------------------------------------------------
    # Iteration-level evolution (NEW — "差异随迭代的演化")
    # ------------------------------------------------------------------

    def track_iter_evolution(
        self, results: list[RunResult]
    ) -> IterEvolutionReport:
        """追踪单配置内跨迭代的差异累积。

        Each result represents a separate allreduce invocation with the same
        input. The baseline is iteration 0; subsequent iterations are compared
        against it.

        This reveals whether non-determinism is:
          - Accumulating (diff ratio grows with iter) → training drift risk
          - Stable (diff ratio constant) → single-precision loss, not compounding
        """
        if len(results) < 2:
            cfg = results[0].config if results else None
            label = f"{cfg.algo}/{cfg.proto}/{cfg.dtype}" if cfg else "unknown"
            return IterEvolutionReport(config_label=label)

        baseline = results[0]
        label = f"{baseline.config.algo}/{baseline.config.proto}/{baseline.config.dtype}"

        report = IterEvolutionReport(config_label=label)
        for i, target in enumerate(results[1:], start=1):
            diff = self.compare(baseline, target)
            report.add(i, diff)

        return report

    # ------------------------------------------------------------------
    # ---- 三级比对（运行 / 调用 / rank） ----
    # ------------------------------------------------------------------

    def compare_run_vs_run(
        self, run0: "MultiRunResult", run1: "MultiRunResult"
    ) -> dict[str, list[DiffReport]]:
        """Compare two runs: for each (call_idx, rank), trial0 vs trial1.

        Returns {"call0_rank1": DiffReport, ...} keyed by "call{k}_rank{r}".
        """
        reports: dict[str, list[DiffReport]] = {}
        n_calls = min(len(run0.call_outputs), len(run1.call_outputs))
        for c in range(n_calls):
            ranks0 = run0.call_outputs[c]
            ranks1 = run1.call_outputs[c]
            for r in sorted(set(ranks0) & set(ranks1)):
                key = f"call{c}_rank{r}"
                b = _make_runresult(run0.config, ranks0[r], r)
                t = _make_runresult(run1.config, ranks1[r], r)
                reports.setdefault(key, []).append(self.compare(b, t))
        return reports

    def compare_call_vs_call(
        self, run: "MultiRunResult"
    ) -> dict[int, list[DiffReport]]:
        """Within one run, compare call 0 vs each subsequent call, per rank.

        Returns {rank: [DiffReport(call0_vs_call1), DiffReport(call0_vs_call2), ...]}.
        """
        reports: dict[int, list[DiffReport]] = {}
        calls = run.call_outputs
        if len(calls) < 2:
            return reports
        baseline_call = calls[0]
        for rank in sorted(baseline_call):
            baseline = _make_runresult(run.config, baseline_call[rank], rank)
            for ci in range(1, len(calls)):
                if rank in calls[ci]:
                    tgt = _make_runresult(run.config, calls[ci][rank], rank)
                    reports.setdefault(rank, []).append(self.compare(baseline, tgt))
        return reports

    def compare_rank_vs_rank(
        self, run: "MultiRunResult", call_idx: int = 0
    ) -> dict[str, list[DiffReport]]:
        """Within one run and one call, compare rank 0 vs each other rank.

        NCCL guarantees all ranks receive identical results in a single
        collective call. This method VERIFIES that guarantee.

        Returns {"rank0_vs_rank1": DiffReport, ...}.
        """
        reports: dict[str, list[DiffReport]] = {}
        if call_idx >= len(run.call_outputs):
            return reports
        per_rank = run.call_outputs[call_idx]
        ranks = sorted(per_rank)
        if len(ranks) < 2:
            return reports
        baseline = _make_runresult(run.config, per_rank[ranks[0]], ranks[0])
        for r in ranks[1:]:
            key = f"rank{ranks[0]}_vs_rank{r}"
            tgt = _make_runresult(run.config, per_rank[r], r)
            reports.setdefault(key, []).append(self.compare(baseline, tgt))
        return reports

    def full_three_level_report(
        self, results: list["MultiRunResult"], label: str = ""
    ) -> "ThreeLevelSummary":
        """Run all three comparison levels and return a structured summary."""
        summary = ThreeLevelSummary(label=label or results[0].config.algo)
        if len(results) >= 2:
            summary.run_vs_run = self.compare_run_vs_run(results[0], results[1])
        if results:
            summary.call_vs_call = self.compare_call_vs_call(results[0])
            summary.rank_vs_rank = self.compare_rank_vs_rank(results[0], call_idx=0)
        return summary


# ---------------------------------------------------------------------------
# ---- 三级摘要 ----
# ---------------------------------------------------------------------------

@dataclass
class ThreeLevelSummary:
    """Structured report from three-level comparison."""

    label: str = ""
    run_vs_run: dict[str, list[DiffReport]] = field(default_factory=dict)
    call_vs_call: dict[int, list[DiffReport]] = field(default_factory=dict)
    rank_vs_rank: dict[str, list[DiffReport]] = field(default_factory=dict)

    @property
    def all_clean(self) -> bool:
        for reports in self.run_vs_run.values():
            if any(not r.bitwise_match for r in reports):
                return False
        for reports in self.call_vs_call.values():
            if any(not r.bitwise_match for r in reports):
                return False
        for reports in self.rank_vs_rank.values():
            if any(not r.bitwise_match for r in reports):
                return False
        return True

    def summary(self) -> str:
        lines = [f"Three-Level Comparison: {self.label}"]
        lines.append(f"  Run-vs-Run   : {self._summarize_level(self.run_vs_run)}")
        lines.append(f"  Call-vs-Call : {self._summarize_level(self.call_vs_call)}")
        lines.append(f"  Rank-vs-Rank : {self._summarize_level(self.rank_vs_rank)}")
        return "\n".join(lines)

    @staticmethod
    def _summarize_level(reports: dict) -> str:
        if not reports:
            return "NO DATA"
        total = sum(len(v) for v in reports.values())
        failing = sum(
            sum(1 for r in v if not r.bitwise_match) for v in reports.values()
        )
        if failing == 0:
            return f"all {total} identical"
        return f"{failing}/{total} with differences"


def _make_runresult(config: "ConfigEntry", output: np.ndarray,
                    rank: int) -> "RunResult":
    """Helper to construct a RunResult on-the-fly for comparison."""
    from .runner import RunResult as RR
    return RR(config=config, output=output, rank=rank)


# ---------------------------------------------------------------------------

def _format_size(nbytes: int) -> str:
    if nbytes < 1024:
        return f"{nbytes}B"
    if nbytes < 1024 * 1024:
        return f"{nbytes // 1024}K"
    return f"{nbytes // (1024 * 1024)}M"
