# NCCL 集合通信位级可复现性诊断工具

> 定位 AllReduce / Reduce-Scatter 中由 NCCL 算法/协议选择引入的逐位非确定性。

## 快速开始

```bash
cd hpn/src/code

# 自检 — 验证环境是否能跑 NCCL + 管线是否正常
PYTHONPATH=. python -m issue4.diagnose --self-test --nranks 4

# 扫描 — 遍历多种配置，输出报告
PYTHONPATH=. python -m issue4.diagnose --mode standard --nranks 4 --json report.json

# 复现 — 跨算法对比，定位非确定性根因
PYTHONPATH=. python issue4/reproduce_case.py --algo Tree --dtype float16 --size 128M --nranks 4
```

## CLI 参数

**diagnose.py** — 主入口：

| 参数 | 作用 | 示例 |
|---|---|---|
| `--mode` | 扫描模式：`quick` / `standard` / `exhaustive` | `--mode standard` |
| `--nranks` | 参与通信的 GPU 数量 | `--nranks 4` |
| `--n-calls` | 单进程组内 NCCL 调用次数（>1 启用三级对比） | `--n-calls 5` |
| `--json` | 输出 JSON 报告路径 | `--json report.json` |
| `--list-configs` | 只打印配置矩阵，不执行 | `--list-configs` |
| `--inject-difference` | ULP 注入验证（`auto` 或 `offset=42,magnitude=3`） | `--inject-difference auto` |
| `--force` | 配合 `--inject-difference`：注入后继续完整扫描 | `--force` |
| `--self-test` | 跑一次最小 GPU 冒烟测试并退出 | `--self-test` |
| `--algo/--proto/--dtype/--size` | 限定单个配置（不扫矩阵） | `--algo Ring --size 128M` |

**reproduce_case.py** — 复现脚本：

| 参数 | 作用 |
|---|---|
| `--algo/--proto/--dtype/--size/--nranks` | 指定测试配置 |
| `--trials` | 重复运行次数（≥3 启用迭代演化检测） |
| `--collective` | `allreduce`（默认）或 `reducescatter` |

## 扫描模式

| 模式 | 算法 | 协议 | 精度 | 数据规模 | 配置总数 |
|---|---|---|---|---|---|
| `quick` | Ring, Tree | Simple | fp32 | 16M, 128M | 4 |
| `standard` | Ring, Tree, PAT | LL, Simple | fp32, fp16 | 4K~128M（5 档） | 60 |
| `exhaustive` | Ring, Tree, PAT | LL, Simple | fp32, fp16, bf16 | 1K~128M（6 档） | 108（含 LL128 过滤后更少） |

PAT 和 LL128 受硬件兼容性过滤，不支持时自动跳过。

## 硬件环境变量

4090D / 5090（无 NVLink / 无 P2P）需要在所有命令前加：

```bash
NCCL_P2P_DISABLE=1
```

4090 公版、A100、H100 等有 NVLink/P2P 的卡不需要。

## 验证结果（4×RTX 5090）

```
Case 1: Tree/Simple/float16 — run-to-run BITWISE IDENTICAL
Case 2: Ring vs Tree 跨算法比对:
  Diff count: 23751354 / 67108864 (35.39%)
  Max abs diff: 3.91e-03
  Distribution: uniform → 算法级根因
```

## 目录结构

```
issue4/
├── diagnose.py          主入口 8 阶段流水线
├── config_matrix.py     硬件检测 + 配置矩阵生成
├── data_generator.py    SHA-256 确定性数据生成
├── runner.py            NCCL 执行器
├── comparator.py        逐位比对 + XOR/ULP + 三级对比
├── reporter.py          报告输出 + 诊断建议引擎
├── reproduce_case.py    独立复现脚本
├── requirements.txt
└── docs/                5 张 Mermaid 架构图
```

## 硬件兼容性

| 算法 | 要求 |
|---|---|
| Ring, Tree, PAT | 所有 GPU |
| NVLS, NVLSTree | NVSwitch（H100 / B200） |
| CollnetDirect | NVSwitch + SHARP |
| CollnetChain | SHARP 网络 |

| 协议 | 要求 |
|---|---|
| LL, Simple | 所有 GPU |
| LL128 | SM90+（Hopper / Blackwell）。非 Hopper 启用**静默数据损坏** |

| GPU | P2P 可用 | LL128 | 注意事项 |
|---|---|---|---|
| 4090D / 5090 | ❌ 需 `NCCL_P2P_DISABLE=1` | ✅（仅 5090） | 5090 NCCL ≥ 2.27 |
| 4090 公版 | ✅ | ❌（SM89） | — |
| A100 / H100 | ✅ | ❌（仅 H100） | 有 NVSwitch，可跑 NVLS |

## 参考

- NCCL#1975: AllReduce determinism
- NCCL#1055: Ring vs Tree precision on A100/A800
- NCCL#157: Chunk partitioning and determinism
