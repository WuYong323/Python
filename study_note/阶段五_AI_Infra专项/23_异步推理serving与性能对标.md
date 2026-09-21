# 23 · 异步推理 serving 与性能对标：把引擎变成服务

> **一句话定位**：把前面所有知识（asyncio + 调度 + PagedAttention + 算子）串起来，搭一个能上生产的**异步推理服务**；再用 **roofline / cost model** 学会"我的引擎到底多快、还能快多少"。
> **前置依赖**：12 asyncio、22 PagedAttention
> **预计时长**：1 天
> **对应你的项目**：你的 `benchmark/`、`bench_harness.py`、`W2_Day6_cost_model_v0` 已经在做对标，本篇把"serving 调度"和"性能建模"系统化。

---

## 0. 先跑起来

```python
# 23_async_serving_demo.py —— 异步推理调度的最小骨架
import asyncio
import itertools

class AsyncEngine:
    """简化推理引擎：每个请求 = 一个协程，异步推进"""
    def __init__(self):
        self._counter = itertools.count()

    async def generate(self, prompt: str, max_tokens: int) -> str:
        result = prompt
        for _ in range(max_tokens):
            await asyncio.sleep(0.1)          # 模拟"等 GPU 算一个 token"（真实里是等 kernel）
            result += f"<t{next(self._counter)}>"
        return result

class ServingScheduler:
    """调度器：接收请求 → 排队 → 并发执行 → 返回结果"""
    def __init__(self, engine: AsyncEngine, max_concurrent: int = 3):
        self.engine = engine
        self.sem = asyncio.Semaphore(max_concurrent)   # 信号量：限制并发数（模拟显存上限）

    async def handle(self, req_id: int, prompt: str):
        async with self.sem:                           # 拿到"名额"才进
            print(f"[{req_id}] 开始")
            out = await self.engine.generate(prompt, max_tokens=5)
            print(f"[{req_id}] 完成: {out}")
            return out

    async def serve(self, requests):
        tasks = [asyncio.create_task(self.handle(i, p)) for i, p in requests]
        return await asyncio.gather(*tasks)

async def main():
    engine = AsyncEngine()
    sched = ServingScheduler(engine, max_concurrent=3)
    requests = [(i, f"req{i}") for i in range(8)]      # 8 个请求，最多 3 个并发
    t0 = asyncio.get_event_loop().time()
    await sched.serve(requests)
    print(f"总耗时: {asyncio.get_event_loop().time() - t0:.1f}s")

asyncio.run(main())
```

运行：`python 23_async_serving_demo.py`。观察：8 个请求、并发 3，如何通过 `Semaphore` 排队限流。

---

## 1. 一句话直觉

> **continuous batching = "公交车随上随下"**：传统 static batching 是一车人坐满、到终点才一起下；continuous batching 是每站有人上、有人下，车一直开，空座立刻给下一个人。

类比：推理 serving 的吞吐瓶颈是"GPU 一次能算多少"。static batching 让"先来的长请求"卡住"后来的短请求"（等最慢的那个）；continuous batching 让**每个 token 步都重新组批**——谁准备好谁上，谁完成谁下，GPU 永远满载。这是 vLLM 等高吞吐引擎的核心调度思想。

---

## 2. 核心原理

### 2.1 static batching vs continuous batching

```
static batching（低效）:
  batch = [A(100 tokens), B(2 tokens)]
  要等 A 和 B 都生成完，才组下一批。B 早早完成，却占着显存干等 A。

continuous batching（高效）:
  每生成一个 token，就检查：谁完成了？有新请求吗？
  → 完成的立即移出，新请求立即加入，显存块立即复用。
```

**continuous batching 的三个关键动作**：
1. **加入**：新请求到达，若显存够，立即分配 KV 块加入当前 batch。
2. **移除**：请求生成完（或 EOS），立即释放 KV 块、移出 batch。
3. **抢占**：显存不够时，换出低优先级请求（第 22 篇 preemption）。

### 2.2 异步调度器的三层结构

```
请求到达 → 请求队列(asyncio.Queue) → 调度器 → 引擎(step) → 响应
              ↑                              ↓
              └──── 显存块管理器(BlockManager) ┘
```

| 层 | 职责 | 你已学的对应篇 |
|----|------|--------------|
| **请求队列** | 收请求、背压、排队 | 12（asyncio.Queue） |
| **调度器** | 组批、限流、抢占决策 | 12 + 本篇 |
| **块管理器** | KV 块分配/复用/换出 | 22（PagedAttention） |
| **引擎** | 真正的算子计算（torch/CUDA） | 19/20/21 |

**核心调度循环**（伪代码）：

```python
async def scheduling_loop(self):
    while True:
        # 1. 从队列取新请求，能塞进 batch 就塞
        new_reqs = self.drain_queue()
        for r in new_reqs:
            if self.block_manager.can_allocate(r.max_tokens):
                self.running.append(r)
            else:
                self.waiting.append(r)   # 显存不够，等着

        # 2. 所有 running 请求各推进一个 token
        tokens = [r.next_token() for r in self.running]
        outputs = await self.engine.step(tokens)   # 异步等 GPU

        # 3. 完成的移除、释放块，腾出空间
        for r in finished(self.running):
            self.block_manager.free_all(r)
            self.running.remove(r)

        # 4. 有空间了，唤醒 waiting 里的请求
        self.promote_waiting()
```

### 2.3 性能对标：roofline 模型

**Roofline** 回答："我的算子到底是被**算力**卡住，还是被**显存带宽**卡住？"

```
性能上限 = min(峰值算力, 算术强度 × 峰值带宽)

算术强度 = FLOPs / 访存字节数
```

```
性能
 ↑        ┌───────────────  峰值算力（compute-bound 上限）
 │       ╱
 │      ╱  ← 斜线：带宽 × 算术强度（memory-bound 上限）
 │     ╱
 │    ╱  ridge point（转折点）
 │   ╱
 └──┴──────────────────→ 算术强度
```

- **算术强度低**（每个字节只算几下）→ **memory-bound**，优化方向是**减少访存**（融合、复用、共享内存）。
- **算术强度高**（每个字节算很多）→ **compute-bound**，优化方向是**提高算力利用率**（向量化、tensor core）。

**你的对标实践**：你 `cuda_study` 里做的"访存优化四板斧""occupancy 与配置扫描"本质就是**在 memory-bound 区优化访存**；你 `W2_Day6_cost_model_v0` 是在建**cost model**（预测算子耗时）。roofline 是 cost model 的理论底座。

### 2.4 cost model：预测算子耗时

```
预估耗时 ≈ max(计算量 / 峰值算力, 访存量 / 峰值带宽)
```

**用途**：
1. **选 kernel 配置**：不实际跑，先估哪个 BLOCK 大小更快（`triton.autotune` 就是自动干这个）。
2. **调度决策**：preemption 时，"换出 vs 重算"哪个便宜（第 22 篇 Q3）。
3. **容量规划**：一台 GPU 能同时服务多少请求。

> 你已经建了 `cost_model_v0`，方向完全正确。生产级的 cost model 要**用真实硬件参数标定**（实测峰值算力/带宽，而不是查手册），再和 profiler 数据互相对照。

---

## 3. 深入细节：serving 的三个工程要点

### 要点 1：信号量限流 = 显存保护

```python
sem = asyncio.Semaphore(max_concurrent)   # max_concurrent = 显存能容纳的请求数
async with sem:                           # 超过就排队，防止显存被打爆
    ...
```

**这对应显存管理的"准入控制"**：`max_concurrent` 由"显存总量 / 单请求峰值 KV 占用"算出，而不是拍脑袋。

### 要点 2：一个请求的完整生命周期追踪

```python
req_log = log.bind(request_id=req_id)     # structlog 绑定（第 09 篇）
req_log.info("queued")
req_log.info("scheduled", batch_size=n)
req_log.info("finished", tokens=count)
```

**价值**：高并发下，靠 `request_id` 把每个请求的日志串起来，排障才不抓瞎。

### 要点 3：GPU 是异步的，调度器也是异步的

引擎的 `step` 应该是 `async`（等 GPU kernel 完成），而不是同步阻塞。这样调度器在"等 GPU"时能去收新请求、做调度决策——**异步贯穿到底**，才有高吞吐。

---

## 4. 回到你的项目

**今日动手任务（30 分钟，把引擎接上调度器）**：

1. 用第 0 节的 `ServingScheduler` 骨架，把你的真实引擎（`model.generate` 的一个 step）包进 `AsyncEngine`（用 `asyncio.to_thread` 包同步 forward，见第 12 篇）。
2. 加 `Semaphore` 限流，`max_concurrent` 用一个"显存可容纳请求数"的估算值。
3. 用 structlog 给每个请求绑定 `request_id`，记录 queued/scheduled/finished。
4. （进阶）把你的 `BlockManager`（第 22 篇升级版）接进调度循环，实现"完成即释放、新请求即分配"。

**验收标准**：8 个请求能并发跑、按显存限额排队、日志按 request_id 可追踪。

---

## 5. 自测题

**Q1（概念）**：continuous batching 相比 static batching 的核心优势是什么？它为什么能显著提升吞吐？

**Q2（应用）**：一个算子算术强度很低（memory-bound），按 roofline 模型，优化方向应该是什么？举两个具体手段。

**Q3（思考）**：为什么 serving 的调度器、引擎的 `step`、以及 GPU 计算都要"异步"？如果调度器在等 GPU 时是同步阻塞的，会发生什么？

<details>
<summary>点击展开答案</summary>

**A1**：continuous batching 在**每个 token 步动态调整 batch**：完成的请求立即移出、新请求立即加入，GPU 几乎**时刻满载**；而 static batching 必须等**整批所有请求都完成**才能组下一批，短请求会被长请求"拖住"，GPU 大量时间空转等待。前者消除"长尾请求拖累整批"的浪费，所以吞吐显著提升。

**A2**：memory-bound 说明瓶颈在**访存**而非算力，优化方向是**减少数据搬运**：① **算子融合**（多个 kernel 合成一个，中间结果留寄存器/共享内存，不往返显存）；② **提高数据复用**（共享内存、tiling 分块，让已加载的数据被多次使用）；③ **合并访存**（coalesced access，让线程一次读连续内存）。你 `cuda_study` 的"访存优化四板斧"正是这些。

**A3**：因为 GPU 计算是异步的（提交后要等待），若调度器在"等 GPU 结果"时**同步阻塞**，这期间它**无法收新请求、无法做调度决策、无法释放已完成请求的显存**——整个服务吞吐坍缩成"一次只处理一个请求"。全链路异步让调度器在"GPU 忙"的间隙去干"收请求、调度、释放显存"这些事，把等待时间也利用起来，这才是高吞吐的本质。
</details>

---

## 6. 延伸阅读

- [vLLM — continuous batching 与调度](https://docs.vllm.ai/)。
- [Roofline 模型](https://en.wikipedia.org/wiki/Roofline_model) —— 计算/访存上限分析。
- [continuous batching 论文](https://aclanthology.org/2024.emnlp-demo.30.pdf)。
- 你的 `W2_Day6_cost_model_v0.md` 与 `benchmark/bench_utils.py` —— 自己的对标实践。

---

> **下一篇 → `24_综合实战_重构与面试要点.md`**：把 24 篇全部所学，端到端重构你的推理引擎项目，并提炼成面试能讲的亮点和路线。
