"""Generate configuration recommendations from a sweep summary.

Reads the `summary.json` produced by sweep.py and outputs actionable
recommendations in Markdown format, suitable for inclusion in ANALYSIS.md.

Usage:
    python recommend.py --summary sweep-output/summary.json
    python recommend.py --summary sweep-output/summary.json --format json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True,
                        help="Path to summary.json from sweep.py.")
    parser.add_argument("--format", choices=("md", "json"), default="md",
                        help="Output format: markdown (default) or json.")
    return parser.parse_args()


def _score_from_row(row: dict[str, object]) -> float:
    """Convert a result row to a consistency score (1.0 = identical, 0.0 = diverged)."""
    if row.get("error"):
        return -1.0  # launch failure
    if row.get("bitwise_identical") is True:
        return 1.0
    return 0.0


def recommend(summary_path: Path) -> dict:
    """Analyse summary.json and return structured recommendations."""
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    results = data["results"]

    # --- Categorize rows ---
    good = [r for r in results if _score_from_row(r) == 1.0]
    bad = sorted(
        [r for r in results if _score_from_row(r) == 0.0],
        key=lambda r: r.get("elements", 0),
    )
    failed = [r for r in results if _score_from_row(r) == -1.0]

    # --- Group by factor ---
    def _group(rows, key):
        d = defaultdict(list)
        for r in rows:
            d[r.get(key, "?")].append(r)
        return dict(d)

    by_algo = _group(results, "algo")
    by_proto = _group(results, "proto")
    by_size = _group(results, "elements")

    # Per-factor summary
    factor_summary = {}
    for name, groups in [("algo", by_algo), ("proto", by_proto)]:
        factor_summary[name] = {
            k: {
                "total": len(v),
                "identical": sum(1 for r in v if _score_from_row(r) == 1.0),
                "diverged": sum(1 for r in v if _score_from_row(r) == 0.0),
            }
            for k, v in sorted(groups.items())
        }

    return {
        "recommended": good,
        "avoid": bad,
        "failed": failed,
        "factor_summary": factor_summary,
        "by_size": {
            str(k): {
                "total": len(v),
                "identical": sum(1 for r in v if _score_from_row(r) == 1.0),
            }
            for k, v in sorted(by_size.items())
        },
    }


def render_markdown(rec: dict) -> str:
    """Render recommendations as Markdown."""
    lines: list[str] = []

    lines.append("## 推荐配置 (bitwise 一致)")
    lines.append("")
    if rec["recommended"]:
        lines.append("| algo | proto | elements |")
        lines.append("|------|-------|----------|")
        for r in rec["recommended"]:
            lines.append(f"| {r['algo']} | {r['proto']} | {r.get('elements', '?')} |")
    else:
        lines.append("（未发现 bitwise 一致的配置）")
    lines.append("")

    lines.append("## 应避免的配置")
    lines.append("")
    if rec["avoid"]:
        lines.append("| algo | proto | elements | first_divergence |")
        lines.append("|------|-------|----------|------------------|")
        for r in rec["avoid"]:
            fd = r.get("first_divergence") or {}
            fd_desc = (
                f"call={fd.get('call','?')}, rank={fd.get('rank','?')}"
                if fd else "detected"
            )
            lines.append(
                f"| {r['algo']} | {r['proto']} | {r.get('elements','?')} "
                f"| {fd_desc} |"
            )
    else:
        lines.append("（所有有效配置均 bitwise 一致）")
    lines.append("")

    if rec["failed"]:
        lines.append("## 不兼容的配置")
        lines.append("")
        lines.append("| algo | proto | elements |")
        lines.append("|------|-------|----------|")
        for r in rec["failed"]:
            lines.append(f"| {r['algo']} | {r['proto']} | {r.get('elements','?')} |")
        lines.append("")

    lines.append("## 按因素统计")
    lines.append("")
    for factor, groups in rec["factor_summary"].items():
        lines.append(f"### {factor}")
        lines.append("")
        lines.append("| 值 | 总数 | 一致 | 不一致 |")
        lines.append("|----|:----:|:----:|:------:|")
        for k, v in groups.items():
            lines.append(
                f"| {k} | {v['total']} | {v['identical']} | {v['diverged']} |"
            )
        lines.append("")

    lines.append("## 按消息大小统计")
    lines.append("")
    lines.append("| 元素数 | 总数 | 一致 |")
    lines.append("|:------:|:----:|:----:|")
    for size, v in rec["by_size"].items():
        lines.append(f"| {size} | {v['total']} | {v['identical']} |")

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    rec = recommend(args.summary)

    if args.format == "json":
        print(json.dumps(rec, indent=2, default=str))
    else:
        print(render_markdown(rec))

    return 0 if not rec["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
