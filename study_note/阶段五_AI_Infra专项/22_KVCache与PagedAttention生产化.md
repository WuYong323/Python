# 22 · KV Cache 与 PagedAttention 生产化：从骨架到内存池

> **一句话定位**：PagedAttention 把 KV cache 从"连续大块"变成"按页管理的虚拟内存"，解决了显存碎片和按需分配。生产级还要加上 **preemption（抢占换出）、recompute（重算）、copy-on-write（写时复制）**。
> **前置依赖**：12 asyncio、19 torch 内部机制
> **预计时长**：1 天
> **对应你的项目**：你的 `PagedAttention.py` 已经实现了"逻辑块→物理块"的映射骨架（BlockAllocator/Sequence），本篇把它升级成可抢占、可复用的生产级内存池。

---

## 0. 先跑起来

先回顾你的骨架，再补生产级特性。跑一遍你现有的 `PagedAttention.py`：

```python
# 你现有的 BlockAllocator/Sequence 已经能演示：
#   逻辑块 → 物理块 的页表映射、按需分配
```

然后补两个生产级能力（下面是扩展骨架）：

```python
# 22_paged_attention_prod.py —— 生产级扩展骨架
class BlockManager:
    """生产级块管理器：分配 + 释放 + 引用计数 + 复用"""
    def __init__(self, num_blocks: int, block_size: int = 16):
        self.block_size = block_size
        self.free_blocks: set[int] = set(range(num_blocks))   # 用 set 去重（第 06 篇）
        self.block_refcount: dict[int, int] = {}               # 引用计数：支持 copy-on-write

    def allocate(self) -> int:
        if not self.free_blocks:
            raise OutOfMemoryError(len(self.free_blocks), 0)   # 生产：抛可决策的异常
        bid = self.free_blocks.pop()
        self.block_refcount[bid] = 1                           # 新块引用计数 = 1
        return bid

    def free(self, block_id: int) -> None:
        if block_id in self.free_blocks:                       # 防重复释放（第 06 篇）
            return
        self.block_refcount[block_id] -= 1                     # 引用计数减一
        if self.block_refcount[block_id] == 0:
            self.free_blocks.add(block_id)                     # 计数归零才真正回收

    def copy_on_write(self, block_id: int) -> int:
        """写时复制：多个序列共享同一块时，某序列要写就复制一份新的"""
        new_bid = self.allocate()
        # 实际：把 block_id 的内容拷贝到 new_bid（省略，仅演示引用计数）
        self.block_refcount[block_id] -= 1
        return new_bid

    def __repr__(self):
        return f"BlockManager(free={len(self.free_blocks)}/{self.num_blocks})"
```

运行：`python 22_paged_attention_prod.py`（含你 `PagedAttention.py` 里拷贝来的类）。

---

## 1. 一句话直觉

> **PagedAttention = 把 KV cache 当成"操作系统的虚拟内存"来管：KV 数据按固定大小分"页"（block），逻辑上连续的序列，物理上可以东一块西一块（页表映射），缺了再按需分配。**

类比：KV cache 传统做法像"给每个请求预留一整排连续座位"（连续显存），浪费且易碎片化；PagedAttention 像**图书馆的散座管理**——读者（序列）来了，需要几个座位就给几个（按需），坐哪个位置用一张"座位表"（block table）记录，读者走了座位立刻腾出来给别人（复用）。**这就是虚拟内存的"页表 + 按需分页"思想**，vLLM 因此把吞吐提升了一个量级。

---

## 2. 核心原理

### 2.1 KV cache 为什么是显存瓶颈

自回归生成时，每个新 token 都要和**之前所有 token** 做 attention。若每次都重算 K/V，是 O(n^2) 的重复计算。**KV cache** 把算过的 K/V 存起来，新 token 只需算自己的 K/V 再和缓存的拼接——**用显存换计算**。

但序列长度不一、并发请求多时，KV cache 的显存占用巨大且**碎片化严重**（每个请求预留 max_len 的连续空间，大量浪费）。PagedAttention 解决的就是这个。

### 2.2 PagedAttention 的核心：块 + 页表

```
逻辑序列:  [token 0..15] [token 16..31] [token 32..47]   ← 逻辑块（连续）
                 ↓            ↓             ↓
物理显存:  [块 7]      [块 3]      [块 12]              ← 物理块（可离散）
                 ↑            ↑             ↑
block_table = [7, 3, 12]   ← 页表：逻辑块序号 → 物理块编号
```

**三个关键对象**（你已经实现）：

| 对象 | 职责 | 你的实现 |
|------|------|---------|
| `BlockAllocator` | 管理空闲物理块池 | ✅ `free_blocks` |
| `Sequence` | 一个请求的 block_table | ✅ `block_table` |
| `locate` | 逻辑 token 位置 → (物理块, 偏移) | ✅ 页表查询 |

### 2.3 生产级的四个增强

你的骨架能跑通演示，但离生产还差四件事：

#### ① 引用计数 + 复用（你缺）

物理块可能被**多个序列共享**（如 beam search 的分支、或 copy-on-write）。释放时不能直接回池，要**引用计数归零才回收**：

```python
self.block_refcount[bid] -= 1        # 每 free 一次减一
if self.block_refcount[bid] == 0:
    self.free_blocks.add(bid)        # 归零才真正回收
```

#### ② preemption 抢占换出（你缺）

当显存用尽、又有更高优先级请求进来时，把**低优先级请求的 KV 块换出到 CPU 内存**，腾出显存给高优先级请求；低优先级请求恢复时再换回来：

```python
def preempt(self, seq):
    for phys in seq.block_table:
        cpu_memory[phys] = gpu_to_cpu(phys)   # 换出到 CPU
        self.free_blocks.add(phys)            # 腾出 GPU 块
```

**这是 vLLM 处理"显存不足"的杀手锏**：不是 OOM 崩溃，而是优雅地换出。

#### ③ recompute 重算（策略选择）

换出的另一种做法是**不换出、直接释放，恢复时重算 KV**（用已有的 prompt token 重新 forward 一遍）。**换出 vs 重算的 trade-off**：换出省计算但占 CPU 内存和带宽；重算省内存但费计算。**选择依据**：序列短、重算便宜 → recompute；序列长、重算贵 → swap 换出。

#### ④ copy-on-write 写时复制（beam search / 并行采样）

beam search 时，多个 beam 共享同一个前缀的 KV 块。当某个 beam 要**修改**共享块时，不能原地改（会影响其他 beam），要**先复制一份再改**：

```python
def copy_on_write(self, block_id):
    new_bid = self.allocate()          # 新物理块
    copy_kv(block_id, new_bid)         # 拷贝内容
    return new_bid                     # 返回新块，原块留给其他 beam
```

**这就是操作系统的 copy-on-write 思想**：读共享、写复制。

---

## 3. 深入细节：vLLM 的架构启示

[vLLM 的 PagedAttention](http://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-192.pdf) 之所以把吞吐提升数倍，靠的是三件事：

1. **按需分页**：不为每个请求预留 max_len，而是"要多少给多少"——接近零浪费。
2. **块共享**：并行采样/beam search 共享前缀块，显存复用。
3. **抢占 + 换出/重算**：显存压力大时优雅降级，不崩溃。

**你能从中学到的架构原则**（直接对接你的引擎）：

- **"虚拟化"思想**：逻辑连续、物理离散，靠一张映射表解耦——这是操作系统、数据库、推理引擎共用的核心抽象。
- **"按需 + 复用"优于"预留"**：预留 = 浪费 + 碎片，按需 + 引用计数 = 高效。
- **"降级优于崩溃"**：preemption 让系统在压力下**优雅降级**，而不是 OOM 崩溃。

---

## 4. 回到你的项目

**今日动手任务（30 分钟，把你 `PagedAttention.py` 升级）**：

1. **`free_blocks` 从 list 改 set**（防重复释放，第 06 篇）。
2. **加引用计数**：`block_refcount` dict，`free()` 归零才回池（第 22 节 ①）。
3. **加 `preempt` 方法**：把某序列的所有物理块换出（先打印模拟，不必真搬数据）。
4. **加 `copy_on_write` 方法**：分配新块 + 引用计数管理（第 22 节 ④）。
5. **给所有方法加类型标注 + `__repr__`**（第 07 篇 + 第 02 篇）。

**验收标准**：`BlockManager` 能演示"分配→共享→copy-on-write→释放→引用计数归零回收→preempt 换出"的完整生命周期，且 `ruff`/`mypy` 通过。

---

## 5. 自测题

**Q1（概念）**：PagedAttention 相比"每个请求预留连续 KV 空间"，解决了哪两个核心问题？它借鉴了操作系统的什么思想？

**Q2（应用）**：两个 beam 共享前缀块 [7, 3]，此时 beam-2 要写块 7。写出 copy-on-write 的步骤（涉及引用计数和块分配）。

**Q3（思考）**：preemption 的"换出到 CPU"和"重算"两种策略，各自的代价和适用场景是什么？为什么"序列越长越倾向换出"？

<details>
<summary>点击展开答案</summary>

**A1**：解决了① **显存浪费**（预留 max_len 但实际只用一部分）和② **显存碎片化**（连续分配导致大量无法利用的碎块）。借鉴了操作系统的**虚拟内存 + 分页**思想：逻辑地址连续、物理页离散、靠页表映射、按需分配、页置换。

**A2**：① `new_bid = allocate()` 分配一个新物理块（引用计数=1）；② 把块 7 的 KV 内容拷贝到 `new_bid`；③ 块 7 的引用计数减一（beam-1 仍持有它，所以不为 0、不回池）；④ beam-2 的 block_table 里把 7 换成 `new_bid`。这样 beam-1 读原块 7、beam-2 写新块，互不影响。

**A3**：**换出（swap）**：代价是占 CPU 内存 + 搬运带宽（换出/换回），但不重算；**重算（recompute）**：代价是重新 forward 一遍 prompt 算 KV，但不占额外内存、无搬运。**序列越长**：KV 块越多、换出带宽越大，但重算的代价（重新算长序列）也越大；通常**长序列更倾向换出**，因为长序列的 KV 已经算好、换出能避免重复计算长序列的昂贵成本，而短序列重算便宜、直接释放重算更省事。（具体阈值由 cost model 决定，第 23 篇。）
</details>

---

## 6. 延伸阅读

- [vLLM 论文（PagedAttention 技术报告）](http://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-192.pdf)。
- [vLLM 源码 — BlockManager](https://github.com/vllm-project/vllm) —— 生产级实现的参考。
- [PagedAttention 原始论文（SOSP'23）](https://arxiv.org/abs/2309.06180)。

---

> **下一篇 → `23_异步推理serving与性能对标.md`**：continuous batching + asyncio 调度器 + roofline/cost model——把前面所有知识串成一个能上生产的推理服务，并学会"对标"。
