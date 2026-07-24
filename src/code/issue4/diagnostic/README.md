# NCCL Bitwise 可复现性诊断工具

对集合通信（AllReduce / Reduce-Scatter）进行 **run-to-run** bitwise 一致性诊断。  
每次 `torchrun --standalone` 启动一个全新的 NCCL communicator，确保拓扑发现、算法选择等非确定性来源在每次 run 中被重新触发。

## 文件结构

```
diagnostic/
├── core.py          ← 纯 Python 比对引擎（XOR、ULP、SHA-256）
├── worker.py        ← torchrun 启动的单次 NCCL 捕获
├── diagnose.py      ← 编排多次独立 run + 比对 + 报告
├── sweep.py         ← algo × proto 矩阵批量扫描
└── README.md        ← 本文件
```

## 环境要求

- Linux, Python 3.10+
- PyTorch with NCCL
- 至少 2 张 NVIDIA GPU
- `torchrun` 可用

## 快速开始

```bash
cd src/code/issue4/diagnostic

# 默认 NCCL 自动选择，跑 5 次独立 run
python diagnose.py --nproc-per-node 4 --runs 5 \
    --op all_reduce --elements 1048576 --calls 20 \
    --output-dir results/default

# 固定算法/协议
python diagnose.py --nproc-per-node 4 --runs 5 \
    --algo Ring --proto Simple --op all_reduce \
    --elements 1048576 --calls 20 --output-dir results/ring-simple

# 批量扫描 algo × proto 矩阵
python sweep.py --nproc-per-node 4 --runs 5 \
    --algos default,Ring,Tree --protos default,Simple,LL \
    --elements 1048576 --calls 20
```

## 输出

每次运行 `diagnose.py` 会在 `--output-dir` 下生成 `report.json`：

```json
{
  "bitwise_identical": false,
  "first_divergence": {
    "run": 1,
    "call": 0,
    "rank": 0,
    "first_byte": 24,
    "first_element": 6,
    "changed_bytes": 94656,
    "changed_bits": 192767,
    "max_abs_error": 9.5367431640625e-07,
    "max_ulp_error": 49152,
    "baseline_sha256": "c5797c...",
    "candidate_sha256": "3f3470..."
  }
}
```

`sweep.py` 额外生成 `summary.json`，汇总所有 algo × proto 组合的结果。

## 退出码

| 退出码 | 含义 |
|:------:|------|
| 0 | 所有 run bitwise 一致 |
| 2 | 检测到位级差异 |
| 非零 | 配置无效或启动失败 |

## 保留原始数据

默认情况下比对完成后会删除各个 run 的中间文件（它们可能很大）。
如需保留供后续离线比对，使用 `--keep-payloads`：

```bash
python diagnose.py ... --keep-payloads
```

然后可以离线比对已有的 capture 文件：

```bash
python core.py run-00.json run-01.json run-02.json --output comparison.json
```

## 设计要点

1. **真正的 run-to-run**：每次 `torchrun --standalone` 创建全新进程组，而非在同一 communicator 内重复调用。
2. **Gloo 控制通道**：payload 通过 Gloo `gather_object` 收集，避免 NCCL 的 instrumentation 流量干扰被测集体通信。
3. **不支持组合不隐藏**：不合法的 algo/proto 组合将直接被 NCCL 拒绝并记录为错误，不会静默回退。
4. **不做假设**：不对算法名（Ring/Tree/NVLS 等）做确定性推断，仅报告实测结果。
