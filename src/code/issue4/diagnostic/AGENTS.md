# AGENTS.md — Issue 4 项目接手指南

## 项目概述

**Isssue 4**: 集合通信 Bitwise 可复现性诊断工具  
**目标硬件**: RTX 5090 (Blackwell, PCIe only, 无 NVSwitch/NVLink)  
**分支**: `issue4-sol`  
**代码位置**: `src/code/issue4/diagnostic/`

## 快速上手

### 环境检查

```bash
python -c "import torch; print(torch.__version__, torch.cuda.nccl.version())"
nvidia-smi
# 确认至少 2 张 GPU 可用
```

### 跑第一个测试

```bash
cd src/code/issue4/diagnostic

# 快速验证（小数据，2 GPU, 2 runs）
python diagnose.py --nproc-per-node 2 --runs 2 \
    --algo Tree --proto LL --elements 32 --calls 5 \
    --output-dir results/test --keep-payloads
```

## 完整实验流程

### 步骤 1: 全量三维扫描（size × algo × proto）

```bash
python sweep.py --nproc-per-node 4 --runs 10 \
    --sizes 32,256,4096,32768,262144,2097152,16777216 \
    --algos Ring,Tree --protos Simple,LL,LL128 \
    --calls 10 --output-dir sweep-output
```

### 步骤 2: 自动生成配置建议

```bash
python recommend.py --summary sweep-output/summary.json
# 或输出 Markdown 格式：
python recommend.py --summary sweep-output/summary.json --format md >> ANALYSIS.md
```

### 步骤 3: Non-determinism 案例

```bash
# 跨算法比对（100% 可复现）
python diagnose.py --nproc-per-node 4 --runs 5 \
    --algo Ring --proto Simple --elements 262144 --calls 20 \
    --output-dir results/ring-simple

python diagnose.py --nproc-per-node 4 --runs 5 \
    --algo Tree --proto Simple --elements 262144 --calls 20 \
    --output-dir results/tree-simple
```

### 步骤 4: 拓扑固定对照实验

```bash
# 先 dump 拓扑文件
export NCCL_TOPO_DUMP_FILE=/tmp/nccl_topo.xml
python -c "import torch; import torch.distributed as dist; dist.init_process_group('nccl'); dist.destroy_process_group()"

# 用固定拓扑跑
python diagnose.py --nproc-per-node 4 --runs 10 \
    --algo Tree --proto LL --elements 32 \
    --topo-file /tmp/nccl_topo.xml \
    --output-dir results/tree-ll-fixed-topo \
    --calls 10
```

## 验收标准覆盖清单

| 验收标准 | 对应文件/命令 |
|----------|-------------|
| 首次差异调用点 | `report.json` → `first_divergence.call, .rank, .first_byte` |
| 差异量级 | `report.json` → `max_abs_error, max_ulp, changed_bytes/bits` |
| 差异随迭代演化 | `report.json` → `evolution_by_call` |
| 差异随规模演化 | `sweep.py --sizes` → `summary.json` 按 elements 分组 |
| Non-determinism 案例 | Ring vs Tree 跨算法比对 |
| 配置建议 | `recommend.py` 自动生成 |

## 源码参考

```
workspace/tenct/
├── nccl/              ← NCCL 源码（理解初始化流程和拓扑发现）
│   └── src/
│       ├── init.cc    ← initTransportsRank(): 拓扑扫描入口
│       ├── graph/     ← Ring/Tree 通信图构建
│       └── enqueue.cc ← topoGetAlgoInfo(): 算法/协议选择
├── nccl-tests/        ← NCCL 官方测试（参考正确性校验方式）
└── hpn/               ← 本项目
    └── src/
        ├── code/issue4/diagnostic/  ← 诊断工具代码
        └── test/issue4/             ← Issue 文档
```

## 5090 硬件限制

- ❌ 不支持 NVLS(4), PAT(5), CollNetDirect(2)
- ❌ 无 NVLink / NVSwitch
- ❌ `torch.use_deterministic_algorithms(True)` 会直接报错
- ✅ 仅 Tree(0) 和 Ring(1) 可用
- ✅ PCIe 拓扑 → 不确定性更容易暴露（利好消息大小扫描）

## 关键设计决策

1. **torchrun --standalone**: 每次 run 创建全新进程组 → 真正的 run-to-run
2. **Gloo 控制通道**: `gather_object` 用 Gloo 而非 NCCL → 避免 NCCL_ALGO 设置污染 instrumentation
3. **payload_hex**: 原始字节存 hex → JSON 可移植，支持离线跨机器比对
4. **默认删中间文件**: `--keep-payloads` 保留，否则比对后自动清理（省磁盘）
5. **不隐藏不兼容组合**: Tree + ReduceScatter 直接报 ncclInvalidUsage，不静默回退

## 依赖

```
Python 3.10+
PyTorch with NCCL
torchrun (包含在 PyTorch 中)
至少 2 张 NVIDIA GPU
```

## 文件的职责

| 文件 | 职责 | 能独立运行？ |
|------|------|:---:|
| `core.py` | 比对引擎 (XOR, ULP, SHA-256) + `compare_runs()` | ✅ CLI 离线比对 |
| `worker.py` | torchrun 启动的单次 NCCL 捕获 | ❌ 需 torchrun |
| `diagnose.py` | 多 run 编排 + 报告生成 | ✅ |
| `sweep.py` | 三维扫描驱动 (size × algo × proto) | ✅ |
| `recommend.py` | 从 summary.json 提取建议 | ✅ |
| `ANALYSIS.md` | 分析报告模板 (run-and-fill) | — |
