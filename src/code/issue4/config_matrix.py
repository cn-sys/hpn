"""
配置矩阵生成器 — NCCL 确定性诊断。

对影响位级可复现性的 NCCL 参数（算法、协议、数据规模、数据类型）
生成笛卡尔积，用于诊断扫描。

参考:
  - NCCL_ALGO: Ring, Tree, CollnetDirect, CollnetChain, NVLS, NVLSTree, PAT
  - NCCL_PROTO: LL, LL128, Simple
  - NCCL 环境变量文档: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# 可用算法 / 协议定义（来自 NCCL tuning.cc 和文档）
# ---------------------------------------------------------------------------

# Algorithm → supported collectives matrix (NCCL 2.19+, Table III from ETH paper)
ALGO_COLLECTIVE_SUPPORT: dict[str, set[str]] = {
    "Ring":            {"allreduce", "reducescatter", "allgather", "broadcast", "reduce"},
    "Tree":            {"allreduce", "reducescatter", "allgather", "broadcast", "reduce"},
    "CollnetDirect":   {"allreduce"},
    "CollnetChain":    {"allreduce"},
    "NVLS":            {"allreduce", "reducescatter", "allgather"},
    "NVLSTree":        {"allreduce"},
    "PAT":             {"allreduce", "reducescatter"},
}

# 协议 → 硬件要求
# LL128 requires Hopper (SM90+) or later
PROTO_REQUIREMENTS: dict[str, str] = {
    "LL":     "All platforms",
    "LL128":  "Hopper+ (SM90+) — enabling on unsupported HW causes silent corruption",
    "Simple": "All platforms",
}

# 协议 → 最低 SM 主版本号（None = 无限制）
PROTO_MIN_SM: dict[str, int | None] = {
    "LL":     None,
    "LL128":  90,     # Hopper (SM 9.0)
    "Simple": None,
}

# 算法 → 硬件要求
# Some algorithms require NVSwitch, SHARP-capable network, or specific topology.
ALGO_HW_REQUIREMENTS: dict[str, str] = {
    "Ring":           "All platforms (may silently corrupt on A800 with many NVLink channels — see NCCL#1055)",
    "Tree":           "All platforms",
    "CollnetDirect":  "Requires NVSwitch + SHARP-capable network",
    "CollnetChain":   "Requires SHARP-capable network (collnet support)",
    "NVLS":           "Requires NVSwitch (H100/B200/Blackwell) — single-node with NVSwitch fabric",
    "NVLSTree":       "Requires NVSwitch (H100/B200/Blackwell)",
    "PAT":            "NCCL 2.23+ — all platforms",
}

# 算法 → 需要 NVSwitch？
ALGO_NEEDS_NVSWITCH: set[str] = {"NVLS", "NVLSTree", "CollnetDirect"}

# 算法 → 需要 SHARP/collnet？
ALGO_NEEDS_COLLNET: set[str] = {"CollnetDirect", "CollnetChain"}

# 数据类型 → 对归约顺序的精度敏感度
DTYPE_SENSITIVITY: dict[str, str] = {
    "float32":   "Low — 23-bit mantissa, good tolerance",
    "float16":   "High — 10-bit mantissa, susceptible to order-dependent rounding",
    "bfloat16":  "High — 7-bit mantissa + larger exponent range, prone to non-determinism",
}


# ---------------------------------------------------------------------------
# Hardware capability record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HardwareCaps:
    """检测到的硬件能力，用于兼容性过滤。"""

    sm_major: int           # e.g. 80 (A100), 90 (H100), 100 (B200)
    gpu_name: str           # e.g. "NVIDIA A100-SXM4-80GB"
    gpu_count: int          # total GPU count visible
    has_nvswitch: bool      # NVSwitch fabric present?
    has_collnet: bool       # SHARP/collnet-capable network?
    nccl_version: str = ""  # e.g. "2.21.5"

    def supports_ll128(self) -> bool:
        """LL128 requires Hopper (SM90+) or later."""
        return self.sm_major >= 90

    def supports_nvswitch_algos(self) -> bool:
        """NVLS/NVLSTree/CollnetDirect require NVSwitch."""
        return self.has_nvswitch

    def supports_collnet_algos(self) -> bool:
        """CollnetChain requires SHARP-capable network."""
        return self.has_collnet

    def compatible_algos(self) -> set[str]:
        """Return set of algorithm names supported on this hardware."""
        algos = {"Ring", "Tree", "PAT"}
        if self.has_nvswitch:
            algos.update({"NVLS", "NVLSTree"})
        if self.has_nvswitch and self.has_collnet:
            algos.add("CollnetDirect")
        if self.has_collnet:
            algos.add("CollnetChain")
        return algos

    def compatible_protos(self) -> set[str]:
        """Return set of protocol names supported on this hardware."""
        protos = {"LL", "Simple"}
        if self.supports_ll128():
            protos.add("LL128")
        return protos

    def check_and_warn(self, algo: str, proto: str) -> list[str]:
        """Return list of warnings for an algo/proto combo on this hardware."""
        warnings: list[str] = []
        if proto == "LL128" and not self.supports_ll128():
            warnings.append(
                f"LL128 requires Hopper+ (SM90+). "
                f"Detected SM{self.sm_major} ({self.gpu_name}). "
                f"Enabling LL128 on unsupported HW causes SILENT DATA CORRUPTION. "
                f"SKIPPED."
            )
        if algo in ALGO_NEEDS_NVSWITCH and not self.has_nvswitch:
            warnings.append(
                f"Algorithm '{algo}' requires NVSwitch ({ALGO_HW_REQUIREMENTS[algo]}). "
                f"NVSwitch not detected. SKIPPED."
            )
        if algo in ALGO_NEEDS_COLLNET and not self.has_collnet:
            warnings.append(
                f"Algorithm '{algo}' requires SHARP/collnet-capable network. "
                f"Not detected. SKIPPED."
            )
        return warnings

    @classmethod
    def detect(cls, require_nvswitch: bool = False,
               require_collnet: bool = False) -> "HardwareCaps":
        """Auto-detect hardware capabilities via PyTorch / pynvml."""
        sm_major = 0
        gpu_name = "unknown"
        gpu_count = 0

        try:
            import torch
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                props = torch.cuda.get_device_properties(0)
                sm_major = props.major
                gpu_name = props.name
        except Exception:
            pass

        # NVSwitch 检测：数据中心 GPU (H100/B200/A100) 通常有 NVSwitch，
        # 消费级 GPU (RTX/GeForce) 一定没有。用 GPU 名称精确判断。
        _datacenter_gpus = {"H100", "H200", "H800", "B100", "B200", "A100", "A800",
                           "GH200", "GB200"}
        _consumer_gpus = {"RTX", "GeForce", "GTX", "TITAN", "Quadro"}

        has_nvswitch = False
        if sm_major >= 80 and gpu_count >= 2:
            gpu_upper = gpu_name.upper()
            is_consumer = any(pat.upper() in gpu_upper for pat in _consumer_gpus)
            is_datacenter = any(pat.upper() in gpu_upper for pat in _datacenter_gpus)
            if is_datacenter and not is_consumer:
                has_nvswitch = True
        has_nvswitch = has_nvswitch or require_nvswitch

        # Collnet/SHARP: typically InfiniBand with SHARP-capable switches.
        # Default to False unless explicitly enabled.
        has_collnet = require_collnet

        return cls(
            sm_major=sm_major,
            gpu_name=gpu_name,
            gpu_count=gpu_count,
            has_nvswitch=has_nvswitch,
            has_collnet=has_collnet,
        )


@dataclass(frozen=True)
class ConfigEntry:
    """诊断扫描中的单个配置点。"""

    algo: str          # e.g. "Ring"
    proto: str         # e.g. "Simple"
    size_bytes: int    # e.g. 1048576
    size_label: str    # human-readable, e.g. "1M"
    dtype: str         # e.g. "float32"
    nranks: int = 8    # number of ranks
    collective: str = "allreduce"  # allreduce or reducescatter

    def env_dict(self) -> dict[str, str]:
        """返回此配置的 os.environ 风格字典。"""
        return {
            "NCCL_ALGO": self.algo,
            "NCCL_PROTO": self.proto,
        }

    def __str__(self) -> str:
        return (
            f"Config(algo={self.algo:>14s}, proto={self.proto:>7s}, "
            f"size={self.size_label:>6s}, dtype={self.dtype:>8s}, nranks={self.nranks})"
        )


@dataclass
class ConfigMatrix:
    """生成 NCCL 确定性相关参数的完整笛卡尔积。"""

    algos: list[str] = field(default_factory=lambda: ["Ring", "Tree", "PAT"])
    protos: list[str] = field(default_factory=lambda: ["LL", "Simple"])
    dtypes: list[str] = field(default_factory=lambda: ["float32", "float16"])
    size_bytes: list[int] = field(default_factory=lambda: [
        1 * 1024,            #   1K — single-chunk region
        4 * 1024,            #   4K — chunk transition boundary
        64 * 1024,           #  64K
        1 * 1024 * 1024,     #   1M — multi-chunk region
        16 * 1024 * 1024,    #  16M — nranks chunks
        128 * 1024 * 1024,   # 128M — full chunk saturation
    ])
    nranks: int = 8
    collective: str = "allreduce"

    # ------------------------------------------------------------------
    # 工厂方法：根据不同场景构建合理的默认配置
    # ------------------------------------------------------------------

    @classmethod
    def quick_sweep(cls, nranks: int = 8) -> "ConfigMatrix":
        """Minimal sweep — Ring vs Tree, float32, two sizes. ~2 min."""
        return cls(
            algos=["Ring", "Tree"],
            protos=["Simple"],
            dtypes=["float32"],
            size_bytes=[16 * 1024 * 1024, 128 * 1024 * 1024],
            nranks=nranks,
        )

    @classmethod
    def standard_sweep(cls, nranks: int = 8) -> "ConfigMatrix":
        """Standard diagnostic sweep — all algos × protocols × 2 dtypes. ~10 min."""
        return cls(
            algos=["Ring", "Tree", "PAT"],
            protos=["LL", "Simple"],
            dtypes=["float32", "float16"],
            size_bytes=[4 * 1024, 64 * 1024, 1 * 1024 * 1024,
                        16 * 1024 * 1024, 128 * 1024 * 1024],
            nranks=nranks,
        )

    @classmethod
    def exhaustive_sweep(cls, nranks: int = 8) -> "ConfigMatrix":
        """Exhaustive sweep — all combinations including bfloat16. ~30 min."""
        return cls(
            algos=["Ring", "Tree", "PAT"],
            protos=["LL", "Simple"],
            dtypes=["float32", "float16", "bfloat16"],
            size_bytes=[1 * 1024, 4 * 1024, 64 * 1024,
                        1 * 1024 * 1024, 16 * 1024 * 1024,
                        128 * 1024 * 1024],
            nranks=nranks,
        )

    @classmethod
    def from_cli(cls, algo: Optional[str] = None, proto: Optional[str] = None,
                 dtype: Optional[str] = None, size: Optional[str] = None,
                 nranks: int = 8) -> "ConfigMatrix":
        """Build matrix from CLI overrides."""
        algos = [algo] if algo else ["Ring", "Tree", "PAT"]
        protos = [proto] if proto else ["LL", "Simple"]
        dtypes = [dtype] if dtype else ["float32", "float16"]

        if size:
            sizes = [_parse_size(size)]
        else:
            sizes = [4 * 1024, 64 * 1024, 1 * 1024 * 1024,
                     16 * 1024 * 1024, 128 * 1024 * 1024]

        return cls(algos=algos, protos=protos, dtypes=dtypes,
                   size_bytes=sizes, nranks=nranks)

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def generate(self, hw: HardwareCaps | None = None) -> list[ConfigEntry]:
        """Generate all config combinations, filtered by collective support
        and optionally by hardware compatibility.

        Parameters
        ----------
        hw : HardwareCaps, optional
            If provided, unsupported algo/proto combos are silently skipped.
        """
        entries: list[ConfigEntry] = []
        for algo, proto, dtype, sz in itertools.product(
            self.algos, self.protos, self.dtypes, self.size_bytes
        ):
            # Skip combinations unsupported by this collective
            if self.collective not in ALGO_COLLECTIVE_SUPPORT.get(algo, set()):
                continue

            # Hardware compatibility filter
            if hw is not None:
                warnings = hw.check_and_warn(algo, proto)
                if warnings:
                    # Skipping incompatible config
                    continue

            entries.append(ConfigEntry(
                algo=algo,
                proto=proto,
                size_bytes=sz,
                size_label=_format_size(sz),
                dtype=dtype,
                nranks=self.nranks,
                collective=self.collective,
            ))
        return entries

    def generate_with_warnings(
        self, hw: HardwareCaps
    ) -> tuple[list[ConfigEntry], list[str]]:
        """Like generate(), but also returns a list of skipped-config warnings."""
        entries: list[ConfigEntry] = []
        skipped_warnings: list[str] = []
        for algo, proto, dtype, sz in itertools.product(
            self.algos, self.protos, self.dtypes, self.size_bytes
        ):
            if self.collective not in ALGO_COLLECTIVE_SUPPORT.get(algo, set()):
                continue

            hw_warnings = hw.check_and_warn(algo, proto)
            if hw_warnings:
                skipped_warnings.extend(
                    f"  Skipped {algo}/{proto}/{dtype}/{_format_size(sz)}: {w}"
                    for w in hw_warnings
                )
                continue

            entries.append(ConfigEntry(
                algo=algo, proto=proto, size_bytes=sz,
                size_label=_format_size(sz), dtype=dtype,
                nranks=self.nranks, collective=self.collective,
            ))
        return entries, skipped_warnings

    def __len__(self) -> int:
        return len(self.generate())

    def __iter__(self):
        return iter(self.generate())

    def summary(self) -> str:
        """Human-readable summary of the sweep space."""
        entries = self.generate()
        return (
            f"Config sweep: {len(entries)} combinations\n"
            f"  Algorithms : {self.algos}\n"
            f"  Protocols  : {self.protos}\n"
            f"  Data types : {self.dtypes}\n"
            f"  Size range : {_format_size(min(self.size_bytes))} ~ "
            f"{_format_size(max(self.size_bytes))} "
            f"({len(self.size_bytes)} points)\n"
            f"  N ranks    : {self.nranks}\n"
            f"  Collective : {self.collective}"
        )


# ---------------------------------------------------------------------------
# ---- 工具函数 ----
# ---------------------------------------------------------------------------

def _format_size(nbytes: int) -> str:
    """Format byte count as human-readable string."""
    if nbytes < 1024:
        return f"{nbytes}B"
    if nbytes < 1024 * 1024:
        return f"{nbytes // 1024}K"
    return f"{nbytes // (1024 * 1024)}M"


def _parse_size(s: str) -> int:
    """Parse human-readable size string to bytes. e.g. '128M' → 134217728."""
    s = s.strip().upper()
    multipliers = {"B": 1, "K": 1024, "M": 1024**2, "G": 1024**3}
    for suffix, mult in multipliers.items():
        if s.endswith(suffix):
            return int(s[:-1]) * mult
    return int(s)
