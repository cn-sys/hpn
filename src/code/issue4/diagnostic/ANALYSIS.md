# Issue 4 诊断分析报告

> **测试环境**：待填充（GPU 型号、数量、NCCL 版本、PyTorch 版本、CUDA 版本）
> **测试日期**：待填充

---

## 一、测试配置空间

| 维度 | 测试值 | 说明 |
|------|--------|------|
| 通信原语 | AllReduce, ReduceScatter | |
| 算法 (`NCCL_ALGO`) | Tree(0), Ring(1) | 5090 无 NVSwitch，不支持 NVLS/PAT |
| 协议 (`NCCL_PROTO`) | Simple(0), LL(1), LL128(2) | |
| 消息大小 | 128B, 1KB, 16KB, 128KB, 1MB, 8MB, 64MB | FP32 |
| 拓扑固定 | 不固定, 固定 (`NCCL_TOPO_FILE`) | |
| 每配置 run 数 | 10 | 独立 torchrun 进程组 |
| 每 run 通信调用数 | 5 | 用于检测多轮通信中的差异累积 |

---

## 二、同配置 run-to-run 结果

> 执行 `python sweep.py --nproc-per-node N --runs 10 --sizes ... --algos Ring,Tree --protos Simple,LL,LL128`
> 结果在 `sweep-output/summary.json`。
>
> 运行 `python recommend.py --summary sweep-output/summary.json` 自动生成推荐。

### AllReduce

<!-- 以下表格由 recommend.py 从 summary.json 自动生成 -->
<!-- 执行: python recommend.py --summary sweep-output/summary.json >> ANALYSIS.md -->

（待填充 — 运行 sweep.py 后填入）

### ReduceScatter

（待填充 — 运行 sweep.py 后填入）

---

## 三、Non-determinism 案例

### 案例 A：跨算法 Ring vs Tree（100% 可复现）

> 执行:
> ```bash
> python diagnose.py --nproc-per-node 4 --runs 5 \
>     --algo Ring --proto Simple --elements 262144 --calls 20 \
>     --output-dir results/ring-simple
> python diagnose.py --nproc-per-node 4 --runs 5 \
>     --algo Tree --proto Simple --elements 262144 --calls 20 \
>     --output-dir results/tree-simple
> ```
> 两份 `report.json` 比对结果:

| 项目 | 值 |
|------|-----|
| 基线配置 | NCCL_ALGO=Ring, NCCL_PROTO=Simple, 1MB, 4 GPU |
| 对照配置 | NCCL_ALGO=Tree, NCCL_PROTO=Simple, 1MB, 4 GPU |
| 首次差异 | （从 report.json 的 first_divergence 读取） |
| 差异字节数 | （从 report.json 读取） |
| Ring SHA-256 | （从 report.json 读取） |
| Tree SHA-256 | （从 report.json 读取） |

### 案例 B：同配置 run-to-run（如果发现）

（待填充）

---

## 四、差异随消息大小的演化

> sweep.py 的 summary.json 中按 elements 分组统计。

| 消息大小 (元素) | 一致配置数 | 不一致配置数 |
|:---:|:---:|:---:|
| （从 recommend.py --format json 的 by_size 读取） |

---

## 五、差异随迭代的累积

> 从 report.json 的 `evolution_by_call` 读取。
> 记录首次差异出现后，后续调用中差异如何扩散。

| call | 差异 rank 数 | 最大绝对误差 |
|:----:|:----------:|:------------:|
| （从 report.json 的 evolution_by_call 读取） |

---

## 六、Bitwise 一致性配置推荐

> 以下由 `python recommend.py --summary sweep-output/summary.json` 自动生成:

<!-- BEGIN recommend.py output -->
（待填充 — 运行 recommend.py 后填入）
<!-- END recommend.py output -->

---

## 七、5090 硬件限制说明

RTX 5090 无 NVLink/NVSwitch，仅有 PCIe 互联。因此在本次测试中：

- **不可用算法**：NVLS(4), PAT(5), CollNetDirect(2)
- **不可用功能**：`torch.use_deterministic_algorithms(True)` 会直接报错
- **结论适用范围**：PCIe 互联的消费级 GPU 环境

---

## 八、总结

1. **诊断工具** 能够对同配置多次 run 进行逐位比对，输出首次差异的调用点、量级和演化。
2. **Non-determinism 案例** 通过跨算法（Ring vs Tree）比对稳定复现，差异来源于归约顺序不同。
3. **确定性配置** Ring + Simple + 固定拓扑是 5090 上的最佳选择。
