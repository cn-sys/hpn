"""
结果报告器 — NCCL 确定性诊断。

以多种格式输出诊断结果:
  - 控制台表格（人类可读）
  - JSON 报告（机器可读，适用于 CI / 自动化分析）
  - 基于差异模式的配置建议

诊断逻辑:
  Ring → 非确定性风险最高（串行累加）
  Tree → 风险较低（平衡二叉树归约）
  PAT  → 风险最低（并行聚合树）
  float16/bf16 → 高敏感度（低尾数精度）
  float32      → 中等敏感度
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Optional

from .comparator import DiffReport, EvolutionReport
from .config_matrix import ConfigEntry


@dataclass
class DiagnosticSummary:
    """Top-level diagnostic summary with recommendations."""

    total_configs: int = 0
    bitwise_match_count: int = 0
    diff_count: int = 0

    # Key findings
    failing_configs: list[DiffReport] = field(default_factory=list)
    passing_configs: list[DiffReport] = field(default_factory=list)

    # Worst case
    worst_diff: Optional[DiffReport] = None

    # Recommendations
    recommendations: list[str] = field(default_factory=list)
    deterministic_config: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "total_configs": self.total_configs,
            "bitwise_match_count": self.bitwise_match_count,
            "diff_count": self.diff_count,
            "worst_diff": _diff_to_dict(self.worst_diff) if self.worst_diff else None,
            "failing_configs": [_diff_to_dict(r) for r in self.failing_configs],
            "passing_configs": [_diff_to_dict(r) for r in self.passing_configs],
            "recommendations": self.recommendations,
            "deterministic_config": self.deterministic_config,
        }


@dataclass
class Reporter:
    """格式化并输出诊断结果。"""

    output_json: bool = True
    output_console: bool = True
    json_path: str = ""

    # ------------------------------------------------------------------
    # Main entry: generate report from comparison results
    # ------------------------------------------------------------------

    def report(
        self,
        reports: list[DiffReport],
        evolutions: Optional[dict[str, EvolutionReport]] = None,
        json_path: Optional[str] = None,
    ) -> DiagnosticSummary:
        """生成并输出完整诊断报告。"""
        summary = self._build_summary(reports)

        if self.output_console:
            self._console_report(summary, evolutions)

        if self.output_json:
            path = json_path or self.json_path or "diagnostic_report.json"
            self._json_report(summary, evolutions, path)

        return summary

    # ------------------------------------------------------------------
    # Summary construction
    # ------------------------------------------------------------------

    def _build_summary(self, reports: list[DiffReport]) -> DiagnosticSummary:
        s = DiagnosticSummary()
        s.total_configs = len(reports)
        s.passing_configs = [r for r in reports if r.bitwise_match]
        s.failing_configs = [r for r in reports if not r.bitwise_match]
        s.bitwise_match_count = len(s.passing_configs)
        s.diff_count = len(s.failing_configs)

        # Worst case: highest diff_ratio
        if s.failing_configs:
            s.worst_diff = max(s.failing_configs, key=lambda r: r.diff_ratio)

        # Generate recommendations
        s.recommendations = self._generate_recommendations(reports)
        s.deterministic_config = self._find_deterministic_config(reports)

        return s

    # ------------------------------------------------------------------
    # ---- 控制台输出 ----
    # ------------------------------------------------------------------

    def _console_report(
        self, summary: DiagnosticSummary,
        evolutions: Optional[dict[str, EvolutionReport]] = None,
    ) -> None:
        self._print_header("NCCL Bitwise Reproducibility Diagnostic Report")
        self._print_section("Overview")
        print(f"  Configs tested  : {summary.total_configs}")
        print(f"  Bitwise match   : {summary.bitwise_match_count}")
        print(f"  Bitwise diff    : {summary.diff_count}")

        if summary.failing_configs:
            self._print_section("Failing Configurations (non-bitwise-deterministic)")
            for r in sorted(summary.failing_configs, key=lambda x: -x.diff_ratio):
                print(f"  {_format_diff_line(r)}")

        if summary.passing_configs:
            self._print_section("Passing Configurations (bitwise-deterministic)")
            for r in summary.passing_configs:
                print(f"  [PASS] {_format_config_line(r.config)}")

        if summary.worst_diff:
            self._print_section("Worst Non-Determinism Case")
            print(f"  Config: {summary.worst_diff.config}")
            print(f"  Diff ratio: {summary.worst_diff.diff_ratio * 100:.4f}%")
            print(f"  Max abs diff: {summary.worst_diff.max_abs_diff:.8e}")
            if summary.worst_diff.first_diff_offset >= 0:
                print(f"  First diff @ offset {summary.worst_diff.first_diff_offset}: "
                      f"baseline={summary.worst_diff.first_diff_baseline:.8e}, "
                      f"target={summary.worst_diff.first_diff_target:.8e}")

        if evolutions:
            self._print_section("Diff Evolution Across Data Sizes")
            for label, evo in evolutions.items():
                print(f"  {label}:")
                for sz, r in evo.entries:
                    status = "IDENTICAL" if r.bitwise_match else f"{r.diff_ratio * 100:6.2f}% diff"
                    print(f"    {_format_size_static(sz):>6s} → {status}")

        self._print_section("Recommendations for Bitwise Determinism")
        for i, rec in enumerate(summary.recommendations, 1):
            print(f"  {i}. {rec}")

        if summary.deterministic_config:
            self._print_section("Deterministic Configuration")
            print(f"  {summary.deterministic_config}")

        self._print_footer()

    # ------------------------------------------------------------------
    # ---- JSON 输出 ----
    # ------------------------------------------------------------------

    def _json_report(
        self, summary: DiagnosticSummary,
        evolutions: Optional[dict[str, EvolutionReport]],
        path: str,
    ) -> None:
        data = summary.to_dict()

        if evolutions:
            data["evolution"] = {}
            for label, evo in evolutions.items():
                data["evolution"][label] = evo.diff_ratio_curve()

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\n  JSON report written to: {path}")

    # ------------------------------------------------------------------
    # ---- 诊断建议引擎 ----
    # ------------------------------------------------------------------

    def _generate_recommendations(self, reports: list[DiffReport]) -> list[str]:
        recs: list[str] = []

        # Analyze failing configs by algo
        algo_fails: dict[str, list[DiffReport]] = {}
        for r in reports:
            if not r.bitwise_match:
                algo_fails.setdefault(r.config.algo, []).append(r)

        if "Ring" in algo_fails and len(algo_fails["Ring"]) >= len([r for r in reports if r.config.algo == "Ring"]):
            recs.append(
                "NCCL_ALGO=Ring consistently produces non-bitwise-deterministic results. "
                "Use NCCL_ALGO=Tree or NCCL_ALGO=PAT instead. "
                "Ring's serial accumulation order is inherently sensitive to chunk partitioning."
            )

        if "Tree" in algo_fails:
            recs.append(
                "Even Tree algorithm showed non-determinism. Try increasing precision: "
                "use float32 instead of float16/bf16, or enable float64 reduction if supported."
            )

        # Analyze by dtype
        dtype_fails: dict[str, list[DiffReport]] = {}
        for r in reports:
            if not r.bitwise_match:
                dtype_fails.setdefault(r.config.dtype, []).append(r)

        if "float16" in dtype_fails or "bfloat16" in dtype_fails:
            recs.append(
                "float16 / bfloat16 amplify non-determinism due to low mantissa precision "
                "(10-bit / 7-bit). Use float32 for reduction buffers when bitwise reproducibility "
                "is required."
            )

        # Passing configs → suggest them
        passing = [r for r in reports if r.bitwise_match]
        if passing:
            best = min(passing, key=lambda r: r.config.size_bytes)  # smallest size that works
            recs.append(
                f"At least one configuration achieved bitwise determinism: "
                f"NCCL_ALGO={best.config.algo}, NCCL_PROTO={best.config.proto}, "
                f"dtype={best.config.dtype}."
            )

        # General guidance
        recs.append(
            "For production determinism: set CUBLAS_WORKSPACE_CONFIG=:4096:8, "
            "torch.use_deterministic_algorithms(True), torch.backends.cudnn.benchmark=False, "
            "and torch.backends.cudnn.deterministic=True."
        )

        recs.append(
            "To guarantee bitwise reproducibility: use Reduce+Broadcast (fixed root) "
            "instead of AllReduce. NCCL_ALGO=PAT (Parallel Aggregated Trees, NCCL 2.23+) "
            "also provides better determinism than Ring."
        )

        return recs

    def _find_deterministic_config(self, reports: list[DiffReport]) -> Optional[str]:
        """Find the most robust deterministic config."""
        passing = [r for r in reports if r.bitwise_match]
        if not passing:
            return None

        # Tree/PAT 优先，float32 优于 float16
        scored = []
        for r in passing:
            score = 0
            if r.config.algo in ("PAT", "Tree"):
                score += 2
            if r.config.dtype == "float32":
                score += 1
            scored.append((score, r))

        best = max(scored, key=lambda x: x[0])
        return (
            f"NCCL_ALGO={best[1].config.algo} "
            f"NCCL_PROTO={best[1].config.proto} "
            f"dtype={best[1].config.dtype}"
        )

    # ------------------------------------------------------------------
    # ---- 格式化工具 ----
    # ------------------------------------------------------------------

    @staticmethod
    def _print_header(title: str) -> None:
        print(f"\n{'=' * 68}")
        print(f"  {title}")
        print(f"{'=' * 68}")

    @staticmethod
    def _print_section(title: str) -> None:
        print(f"\n--- {title} ---")

    @staticmethod
    def _print_footer() -> None:
        print(f"\n{'=' * 68}\n")


def _format_config_line(cfg: ConfigEntry) -> str:
    return f"algo={cfg.algo:>14s}  proto={cfg.proto:>7s}  dtype={cfg.dtype:>8s}  size={cfg.size_label:>6s}"


def _format_diff_line(r: DiffReport) -> str:
    cfg = r.config
    return (
        f"[DIFF] {_format_config_line(cfg)}  "
        f"diff={r.diff_count}/{r.total_elements} ({r.diff_ratio * 100:.4f}%)  "
        f"max_abs={r.max_abs_diff:.6e}"
    )


def _diff_to_dict(r: DiffReport) -> dict:
    return {
        "algo": r.config.algo,
        "proto": r.config.proto,
        "dtype": r.config.dtype,
        "size_bytes": r.config.size_bytes,
        "size_label": r.config.size_label,
        "bitwise_match": r.bitwise_match,
        "total_elements": r.total_elements,
        "diff_count": r.diff_count,
        "diff_ratio": f"{r.diff_ratio * 100:.6f}%",
        "max_abs_diff": f"{r.max_abs_diff:.8e}" if r.max_abs_diff > 0 else "0",
        "mean_abs_diff": f"{r.mean_abs_diff:.8e}" if r.mean_abs_diff > 0 else "0",
        "first_diff_offset": r.first_diff_offset,
        "first_diff_baseline": f"{r.first_diff_baseline:.8e}",
        "first_diff_target": f"{r.first_diff_target:.8e}",
    }


def _format_size_static(nbytes: int) -> str:
    if nbytes < 1024:
        return f"{nbytes}B"
    if nbytes < 1024 * 1024:
        return f"{nbytes // 1024}K"
    return f"{nbytes // (1024 * 1024)}M"
