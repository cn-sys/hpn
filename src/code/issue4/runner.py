"""
NCCL 集合通信执行器。

提供两种后端:
  1. PyTorch NCCL — 使用 torch.distributed（推荐，无需额外依赖）
  2. nccl-tests  — 包装 all_reduce_perf / reduce_scatter_perf 二进制（原始 NCCL）

执行器在集合操作完成后捕获输出张量，用于后续逐位比对。
每次运行通过独立进程（每 rank 一个）来避免 NCCL 环境变量缓存问题。

设计说明:
  nccl-tests 不导出原始缓冲区数据，因此 nccl-tests 后端需打补丁或在
  CUDA 侧保存缓冲区。在此之前，PyTorch 后端是主要路径。
"""

from __future__ import annotations

import multiprocessing as mp
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config_matrix import ConfigEntry
from .data_generator import DataGenerator, DTYPE_BYTES


@dataclass
class RunResult:
    """单次 NCCL 集合通信调用的输出。"""

    config: ConfigEntry
    output: np.ndarray           # result tensor (shape depends on collective)
    elapsed_ms: float = 0.0      # GPU-side wall time
    rank: int = 0                # which rank's output this is

    def checksum(self) -> float:
        """浮点校验和，用于快速漂移检测。"""
        return float(np.sum(self.output.astype(np.float64)))

    def save(self, path: str) -> None:
        """Save output as raw binary (float32) for cross-run comparison."""
        self.output.astype(np.float32).tofile(path)

    @classmethod
    def load(cls, path: str, config: ConfigEntry) -> "RunResult":
        """从原始二进制文件加载输出。"""
        count = config.size_bytes // DTYPE_BYTES.get(config.dtype, 4)
        if config.collective == "reducescatter":
            count = count // config.nranks
        data = np.fromfile(path, dtype=np.float32, count=count)
        return cls(config=config, output=data)


@dataclass
class MultiRunResult:
    """Output of N calls × R ranks within one process-group run.

    call_outputs[call_idx][rank] = np.ndarray (rank's output for that call).
    """

    config: ConfigEntry
    call_outputs: list[dict[int, np.ndarray]]  # [call_idx] → {rank: output}
    elapsed_ms: list[float]                     # per-call latency


@dataclass
class NcclRunner:
    """执行 NCCL 集合通信操作并捕获输出张量。

    Parameters
    ----------
    nranks : int
        Number of GPU ranks to use.
    backend : str
        'pytorch' (default) — torch.distributed
        'nccl-tests' — raw NCCL via nccl-tests binaries (needs buffer-dump patch)
    nccl_tests_dir : str, optional
        Path to nccl-tests build directory. Required if backend='nccl-tests'.
    scratch_dir : str, optional
        Directory for temporary output files. Uses system temp if None.
    """

    nranks: int = 8
    backend: str = "pytorch"
    nccl_tests_dir: str = ""
    scratch_dir: str = ""

    _data_gen: Optional[DataGenerator] = field(default=None, init=False)

    # ------------------------------------------------------------------
    # ---- 公共 API ----
    # ------------------------------------------------------------------

    def run(self, config: ConfigEntry, trials: int = 2) -> list[RunResult]:
        """Run the collective `trials` times under the same config.

        Each trial spawns a fresh process group to guarantee clean NCCL state.
        Returns list of RunResult, one per trial (rank-0 only, backward-compat).
        """
        results: list[RunResult] = []
        for _ in range(trials):
            result = self._run_once(config)
            results.append(result)
        return results

    def run_full(self, config: ConfigEntry, n_calls: int,
                 trials: int = 2) -> list[MultiRunResult]:
        """Run `n_calls` collective calls inside each of `trials` process groups.

        Returns one MultiRunResult per trial. Each MultiRunResult contains
        all call outputs for all ranks — suitable for three-level comparison:
          - run-vs-run:   trial0[call_k][rank_r] vs trial1[call_k][rank_r]
          - call-vs-call: trial_t[call_i][rank_r] vs trial_t[call_j][rank_r]
          - rank-vs-rank: trial_t[call_k][rank_a] vs trial_t[call_k][rank_b]
        """
        results: list[MultiRunResult] = []
        for _ in range(trials):
            mrr = self._run_full_once(config, n_calls)
            results.append(mrr)
        return results

    def sweep(
        self, configs: list[ConfigEntry], trials: int = 2
    ) -> list[list[RunResult]]:
        """运行所有配置。 Returns [[trial_0, trial_1, ...], ...] per config."""
        all_results: list[list[RunResult]] = []
        for cfg in configs:
            results = self.run(cfg, trials=trials)
            all_results.append(results)
        return all_results

    # ------------------------------------------------------------------
    # ---- 完整模式：进程组内多调用 / 全部 rank ----
    # ------------------------------------------------------------------

    def _run_full_once(self, config: ConfigEntry, n_calls: int) -> MultiRunResult:
        """Execute `n_calls` collective invocations within one process group."""
        self._data_gen = DataGenerator(
            dtype=config.dtype, base_seed=42, collective=config.collective,
        )
        port = _find_free_port()
        count = self._data_gen.elem_count(config.size_bytes, config.nranks)
        inputs = self._data_gen.generate_all(config.size_bytes, config.nranks)

        with mp.Manager() as manager:
            result_dict = manager.dict()
            processes = []
            for rank in range(config.nranks):
                p = mp.Process(
                    target=self._pytorch_worker_full,
                    args=(rank, config.nranks, port, config, count, n_calls,
                          inputs[rank], result_dict),
                )
                p.start()
                processes.append(p)
            for p in processes:
                p.join()

            # Collect: result_dict[f"{call_idx}_{rank}"] = np.ndarray
            call_outputs: list[dict[int, np.ndarray]] = []
            elapsed_ms: list[float] = []
            for call_idx in range(n_calls):
                per_rank: dict[int, np.ndarray] = {}
                for rank in range(config.nranks):
                    key = f"{call_idx}_{rank}"
                    if key not in result_dict:
                        raise RuntimeError(
                            f"Worker did not produce output for call={call_idx} "
                            f"rank={rank}. Check GPU memory / NCCL."
                        )
                    per_rank[rank] = np.array(result_dict[key])
                call_outputs.append(per_rank)
                elapsed_ms.append(float(result_dict.get(f"_elapsed_{call_idx}", 0.0)))

        return MultiRunResult(
            config=config, call_outputs=call_outputs, elapsed_ms=elapsed_ms,
        )

    @staticmethod
    def _pytorch_worker_full(
        rank: int, world_size: int, port: int, config: ConfigEntry,
        count: int, n_calls: int, input_data: np.ndarray, result_dict: dict,
    ) -> None:
        """Worker: init NCCL once, run collective `n_calls` times."""
        import torch
        import torch.distributed as dist

        os.environ.update(config.env_dict())
        torch.cuda.set_device(rank % torch.cuda.device_count())
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank, world_size=world_size,
            device_id=device,
        )

        dt = _torch_dtype(config.dtype)

        # Warmup
        warm = torch.from_numpy(input_data).to(device=device, dtype=dt)
        if config.collective == "allreduce":
            dist.all_reduce(warm)
        elif config.collective == "reducescatter":
            wc = input_data.size // world_size
            dist.reduce_scatter_tensor(
                torch.zeros(wc, dtype=dt, device=device), warm,
            )
        torch.cuda.synchronize()

        # Timed calls
        for call_idx in range(n_calls):
            tensor = torch.from_numpy(input_data).to(device=device, dtype=dt)
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            if config.collective == "allreduce":
                dist.all_reduce(tensor)
                out = tensor
            elif config.collective == "reducescatter":
                recv_count = input_data.size // world_size
                out = torch.zeros(recv_count, dtype=dt, device=device)
                dist.reduce_scatter_tensor(out, tensor)
            else:
                raise ValueError(f"Unknown collective: {config.collective}")

            torch.cuda.synchronize()
            t1 = time.perf_counter()

            result_dict[f"{call_idx}_{rank}"] = out.cpu().float().numpy()
            result_dict[f"_elapsed_{call_idx}"] = (t1 - t0) * 1000.0

        dist.destroy_process_group()

    # ------------------------------------------------------------------
    # ---- 内部：单次运行编排 ---- (backward-compat)
    # ------------------------------------------------------------------

    def _run_once(self, config: ConfigEntry) -> RunResult:
        """执行一次集合通信调用并返回输出。"""
        self._data_gen = DataGenerator(
            dtype=config.dtype,
            base_seed=42,
            collective=config.collective,
        )

        if self.backend == "pytorch":
            return self._run_pytorch(config)
        elif self.backend == "nccl-tests":
            return self._run_nccl_tests(config)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    # ------------------------------------------------------------------
    # ---- PyTorch NCCL 后端 ----
    # ------------------------------------------------------------------

    def _run_pytorch(self, config: ConfigEntry) -> RunResult:
        """通过独立进程中的 torch.distributed 运行。"""
        port = _find_free_port()

        # 提前生成各 rank 输入（CPU 侧）
        count = self._data_gen.elem_count(config.size_bytes, config.nranks)
        inputs = self._data_gen.generate_all(config.size_bytes, config.nranks)

        with mp.Manager() as manager:
            result_dict = manager.dict()

            processes = []
            for rank in range(config.nranks):
                p = mp.Process(
                    target=self._pytorch_worker,
                    args=(rank, config.nranks, port, config, count,
                          inputs[rank], result_dict),
                )
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            # 提取 rank-0 输出 (all ranks produce identical result in a
            # single AllReduce invocation — NCCL guarantee)
            if 0 not in result_dict:
                raise RuntimeError(
                    f"Rank-0 worker did not produce output. "
                    f"Check GPU memory and NCCL availability for config: {config}"
                )
            output = np.array(result_dict[0])
            elapsed = result_dict.get("elapsed", 0.0)

        return RunResult(config=config, output=output, elapsed_ms=elapsed, rank=0)

    @staticmethod
    def _pytorch_worker(
        rank: int, world_size: int, port: int, config: ConfigEntry,
        count: int, input_data: np.ndarray, result_dict: dict,
    ) -> None:
        """逐 rank 的工作函数（在独立进程中运行）。

        Uses Gloo backend for barrier/control-plane operations (so NCCL_ALGO
        doesn't affect rank synchronization) and NCCL exclusively for the
        collective operation being tested.
        """
        import torch
        import torch.distributed as dist

        os.environ.update(config.env_dict())

        torch.cuda.set_device(rank % torch.cuda.device_count())
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

        # --- Primary NCCL group (for collectives only) ---
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=world_size,
            device_id=device,
        )

        # --- Gloo subgroup for barrier / control-plane ---
        # This isolates NCCL_ALGO changes from synchronization ops.
        # NCCL's barrier is a no-op reduction but Gloo is TCP-based
        # and completely unaffected by NCCL tuning variables.
        all_ranks = list(range(world_size))
        gloo_group: dist.ProcessGroup | None = None
        try:
            gloo_group = dist.new_group(
                ranks=all_ranks,
                backend="gloo",
            )
        except Exception:
            # Gloo may not be available in some builds; fall back to NCCL barrier
            gloo_group = None

        def _barrier() -> None:
            """Use Gloo if available, else fall back to NCCL barrier."""
            if gloo_group is not None:
                dist.barrier(group=gloo_group)
            else:
                dist.barrier()

        # --- Warmup (on NCCL) ---
        dt = _torch_dtype(config.dtype)
        tensor = torch.from_numpy(input_data).to(device=device, dtype=dt)

        warm = tensor.clone()
        if config.collective == "allreduce":
            dist.all_reduce(warm)
        elif config.collective == "reducescatter":
            # ReduceScatter: sendbuf has nranks*count elements, recvbuf has count
            recv_count = input_data.size // world_size
            dist.reduce_scatter_tensor(
                torch.zeros(recv_count, dtype=dt, device=device),
                warm,
            )
        torch.cuda.synchronize()
        _barrier()

        # --- Timed run (NCCL collective only) ---
        tensor = torch.from_numpy(input_data).to(device=device, dtype=dt)
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        if config.collective == "allreduce":
            dist.all_reduce(tensor)
        elif config.collective == "reducescatter":
            dist.reduce_scatter_tensor(
                torch.zeros(count, dtype=dt, device=device),
                tensor,
            )
        else:
            raise ValueError(f"Unknown collective: {config.collective}")

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        # --- Store result ---
        if rank == 0:
            result_dict[0] = tensor.cpu().float().numpy()
            result_dict["elapsed"] = (t1 - t0) * 1000.0

        _barrier()
        if gloo_group is not None:
            dist.destroy_process_group(gloo_group)
        dist.destroy_process_group()

    # ------------------------------------------------------------------
    # ---- nccl-tests 后端（需 buffer-dump 补丁） ----
    # ------------------------------------------------------------------

    def _run_nccl_tests(self, config: ConfigEntry) -> RunResult:
        """通过 nccl-tests 的 all_reduce_perf 二进制运行。

        WARNING: nccl-tests does NOT export raw tensor data from GPU buffers.
        This backend requires one of:
          a) A patched nccl-tests binary that dumps sendbuff/recvbuff to disk
          b) A CUDA-side LD_PRELOAD interposer that captures buffer contents
          c) Using nvprof/nsys to capture memory states

        Until one of these is in place, use the 'pytorch' backend instead.
        """
        if not self.nccl_tests_dir:
            raise RuntimeError(
                "nccl_tests_dir required for nccl-tests backend.\n"
                "Use the 'pytorch' backend (default) for bitwise comparison.\n"
                "Example: NcclRunner(nranks=8, backend='pytorch')"
            )

        binary_map = {
            "allreduce":      "all_reduce_perf",
            "reducescatter":  "reduce_scatter_perf",
        }
        binary = os.path.join(self.nccl_tests_dir, binary_map[config.collective])

        if not os.path.isfile(binary):
            raise FileNotFoundError(
                f"nccl-tests binary '{binary}' not found.\n"
                f"Build nccl-tests first:\n"
                f"  git clone https://github.com/NVIDIA/nccl-tests\n"
                f"  cd nccl-tests && make MPI=1\n"
                f"  Then pass --nccl-tests-dir ./build to the diagnostic tool."
            )

        dtype_map = {"float32": "float", "float16": "half", "bfloat16": "bfloat16"}
        cmd = [
            binary,
            "-b", str(config.size_bytes),
            "-e", str(config.size_bytes),
            "-g", str(config.nranks),
            "-n", "1",
            "-w", "1",
            "-d", dtype_map.get(config.dtype, "float"),
            "-c", "0",
            "--blocking", "1",
        ]

        env = os.environ.copy()
        env.update(config.env_dict())

        try:
            result = subprocess.run(
                cmd, env=env, capture_output=True, text=True, timeout=120,
                cwd=self.nccl_tests_dir,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"nccl-tests timed out for config: {config}")

        if result.returncode != 0:
            raise RuntimeError(
                f"nccl-tests failed (rc={result.returncode}):\n{result.stderr}"
            )

        raise NotImplementedError(
            "nccl-tests backend requires a buffer-dump patch to capture raw "
            "output data for bitwise comparison. The nccl-tests binary does "
            "not expose GPU buffer contents.\n\n"
            "Workaround: use the 'pytorch' backend instead:\n"
            "  NcclRunner(nranks=8, backend='pytorch')\n\n"
            "Future: implement a CUDA LD_PRELOAD interposer or contribute a "
            "--dump-buffers flag to nccl-tests."
        )


# ---------------------------------------------------------------------------
# ---- 工具函数 ----
# ---------------------------------------------------------------------------

def _find_free_port() -> int:
    """在本地查找空闲 TCP 端口，同时确保 port+1 也空闲（NCCL 内部占用）。

    快速连续创建进程组时，上次的 port+1 可能还没释放——这里做显式检查。
    """
    for _ in range(20):  # 最多试 20 次
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        # NCCL 内部也占 port+1 — 检查相邻端口是否空闲
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2:
                s2.bind(("127.0.0.1", port + 1))
                s2.close()
            return port
        except OSError:
            continue
    raise RuntimeError("无法找到连续两个空闲端口 (NCCL 需要 port 和 port+1)")


def _torch_dtype(dtype: str):
    """将 dtype 字符串转换为 torch.dtype。"""
    import torch
    return {
        "float32":  torch.float32,
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]
