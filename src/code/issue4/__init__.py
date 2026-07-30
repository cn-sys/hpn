"""
NCCL 集合通信位级可复现性诊断工具。

在大规模分布式训练中诊断 NCCL 集合通信操作（AllReduce / Reduce-Scatter）
的逐位非确定性问题。

根因：浮点加法不满足结合律。NCCL 的分块策略随数据规模、算法、
协议和拓扑变化，导致归约累加顺序不同 → 逐位结果不同。

提供：
  1. 配置矩阵扫描（算法 × 协议 × 数据规模 × 精度）
  2. SHA-256 确定性逐 rank 差异化数据生成
  3. 逐位比对与差异统计（XOR / ULP 位级分解）
  4. 差异随规模 / 迭代的演化追踪
  5. 诊断建议与确定性配置推荐
"""

__version__ = "0.1.0"
__author__ = "NCCL Determinism Diagnostic Tool Contributors"

from .config_matrix import ConfigMatrix, ConfigEntry, HardwareCaps
from .data_generator import DataGenerator
from .runner import NcclRunner, RunResult, MultiRunResult
from .comparator import (
    BitwiseComparator,
    DiffReport,
    DiffDistribution,
    XorDetail,
    EvolutionReport,
    IterEvolutionReport,
    ThreeLevelSummary,
    compute_ulp,
    analyze_xor_float,
)
from .reporter import Reporter, DiagnosticSummary

__all__ = [
    "ConfigMatrix", "ConfigEntry", "HardwareCaps",
    "DataGenerator",
    "NcclRunner", "RunResult", "MultiRunResult",
    "BitwiseComparator",
    "DiffReport", "DiffDistribution", "XorDetail",
    "EvolutionReport", "IterEvolutionReport",
    "ThreeLevelSummary",
    "compute_ulp", "analyze_xor_float",
    "Reporter", "DiagnosticSummary",
]
