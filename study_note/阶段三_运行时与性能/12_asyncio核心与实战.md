# 12 · asyncio 核心与实战：单线程扛起高并发

> **一句话定位**：asyncio 用"单线程 + 事件循环 + 协程"，让一个进程优雅地调度成千上万个 I/O 任务。这是推理 serving、异步调度器的核心技术。
> **前置依赖**：11 并发模型
> **预计时长**：1 天
> **对应你的项目**：你第 23 篇要搭的异步推理调度器，本篇就是它的语法和心智地基。

---

## 0. 先跑起来

```python
# 12_asyncio_demo.py
import asyncio
import time

# ── 1. async/await 基础：协程 ──
async def say(name: str, delay: float) -> str:
    await asyncio.sleep(delay)      # await = 让出控制权，去等 delay 秒
    return f"{name} 完成"

async def main():
    t0 = time.perf_counter()
    # 顺序 await：一个接一个（慢）
    r1 = await say("A", 1)
    r2 = await say("B", 1)
    print("顺序:", r1, r2, f"{time.perf_counter()-t0:.2f}s")   # ~2s

    # 并发 gather：同时发起（快）
    t0 = time.perf_counter()
    results = await asyncio.gather(say("A", 1), say("B", 1))
    print("并发:", results, f"{time.perf_counter()-t0:.2f}s")  # ~1s

asyncio.run(main())

# ── 2. 阻塞 vs 非阻塞：asyncio.sleep 不阻塞，time.sleep 阻塞 ──
async def blocking_demo():
    t0 = time.perf_counter()
    # 错：time.sleep 阻塞整个事件循环，别的协程全卡住
    time.sleep(1)
    time.sleep(1)
    print("time.sleep 阻塞:", f"{time.perf_counter()-t0:.2f}s")  # ~2s

    t0 = time.perf_counter()
    await asyncio.gather(asyncio.sleep(1), asyncio.sleep(1))
    print("asyncio.sleep 并发:", f"{time.perf_counter()-t0:.2f}s")  # ~1s

asyncio.run(blocking_demo())

# ── 3. create_task：后台任务 + 等待 ──
async def task_demo():
    async def worker(i: int):
        await asyncio.sleep(1)
        return i * i

    tasks = [asyncio.create_task(worker(i)) for i in range(5)]  # 立即调度
    # 中间还能干别的...
    results = await asyncio.gather(*tasks)
    print("task 结果:", results)    # [0, 1, 4, 9, 16]

asyncio.run(task_demo())

# ── 4. 生产者-消费者：asyncio.Queue ──
async def producer(q: asyncio.Queue, n: int):
    for i in range(n):
        await asyncio.sleep(0.1)        # 模拟"生成一个任务要 0.1s"
        await q.put(i)
        print(f"  生产 {i}")
    await q.put(None)                   # 哨兵：告诉消费者"没了"

async def consumer(q: asyncio.Queue, name: str):
    while True:
        item = await q.get()
        if item is None:
            break
        await asyncio.sleep(0.3)        # 模拟"处理一个任务要 0.3s"
        print(f"{name} 消费 {item}")

async def pipeline():
    q: asyncio.Queue = asyncio.Queue(maxsize=3)   # maxsize 限流（背压）
    await asyncio.gather(producer(q, 6), consumer(q, "worker-1"), consumer(q, "worker-2"))

asyncio.run(pipeline())

# ── 5. 超时：asyncio.wait_for ──
async def timeout_demo():
    try:
        await asyncio.wait_for(asyncio.sleep(10), timeout=1.0)
    except asyncio.TimeoutError:
        print("超时了！1 秒没完成就被中断")

asyncio.run(timeout_demo())
```

运行：`python 12_asyncio_demo.py`。

**重点观察**：顺序 await 是 2s、gather 是 1s——**这就是"并发"和"串行"的区别，全程只有一个线程**。

---

## 1. 一句话直觉

> **asyncio = 一个"大管家"（事件循环）+ 一群"会主动让位的工人"（协程）。工人在 `await` 处主动说"我这事儿要等一会儿，你先忙别的"，管家就切去服务下一个工人。**

类比：协程像**餐厅服务员**——一个服务员可以同时服务多桌客人：给 A 桌点完菜（`await` 等厨房），不干站着，转去给 B 桌倒水，再回来看 A 桌菜好了没。**一个服务员（单线程）就能服务很多桌（高并发）**，秘诀就是"在等待时不闲着，去干别的"。

---

## 2. 核心原理

### 2.1 协程、事件循环、await

| 概念 | 是什么 | 类比 |
|------|--------|------|
| **协程** coroutine | `async def` 定义的函数，调用它不执行、返回协程对象 | 一个"任务单" |
| **事件循环** event loop | 调度所有协程的"引擎"，`asyncio.run()` 启动它 | 大管家 |
| **`await`** | 挂起当前协程、把控制权交还事件循环，去执行别的协程 | "我先让位" |

```python
async def f():
    await asyncio.sleep(1)   # 让位：事件循环此刻去跑别的协程
    return "done"

asyncio.run(f())             # 启动事件循环并运行 f 直到完成
```

**关键**：`await` 后面必须跟一个 **awaitable**（协程、Task、Future）。`await asyncio.sleep(1)` 不是"干等 1 秒"，而是"**挂起自己，告诉事件循环 1 秒后再唤醒我**"——这 1 秒里事件循环能跑成百上千个其他协程。

### 2.2 阻塞 vs 非阻塞：asyncio 的第一大坑

```python
await asyncio.sleep(1)   # ✅ 非阻塞：让出控制权，事件循环继续
time.sleep(1)            # ❌ 阻塞：整个事件循环卡死 1 秒，所有协程全停
```

**铁律：协程里绝不能调用阻塞函数**（`time.sleep`、同步 `requests`、同步文件 I/O、裸 `torch.cuda.synchronize()` 等）。阻塞函数会卡死整个事件循环，让所有"并发"瞬间变成"串行"。

**怎么办**：把阻塞操作丢到线程池，用 `await asyncio.to_thread(...)` 包一层：

```python
import asyncio, time

async def main():
    # 把阻塞的 time.sleep 放进线程池，不卡事件循环
    await asyncio.to_thread(time.sleep, 1)   # ✅ 事件循环继续跑别的
```

### 2.3 `gather` vs `create_task` vs `wait_for`

| API | 作用 | 语义 |
|-----|------|------|
| `await coro` | 顺序执行 | 等这一个完成才继续 |
| `asyncio.gather(a, b)` | 并发执行多个 | 全部完成后返回结果列表 |
| `asyncio.create_task(coro)` | 创建后台任务 | 立即调度，不等待（拿回 Task 对象） |
| `asyncio.wait_for(coro, t)` | 带超时 | 超时抛 `TimeoutError` |
| `asyncio.as_completed(...)` | 按完成顺序迭代 | 谁先完成先处理谁 |

```python
# create_task：启动后台任务，主协程继续干别的
task = asyncio.create_task(long_work())     # 立即开始调度
# ... 主协程干别的 ...
result = await task                          # 需要结果时再等它
```

> `create_task` 是"**fire-and-forget 但要保留句柄**"——不 `await` 它，任务也会在后台跑，但**强烈建议保留 task 引用**，否则任务可能被垃圾回收、异常被静默吞掉。

### 2.4 生产者-消费者：`asyncio.Queue`

这是**推理 serving 调度器的核心模式**：

```python
q: asyncio.Queue = asyncio.Queue(maxsize=3)   # maxsize 实现背压（限流）

async def producer(q):                        # 接收请求，放进队列
    while True:
        req = await recv_request()            # 从网络收请求
        await q.put(req)

async def consumer(q):                        # 从队列取请求，调度到 GPU
    while True:
        req = await q.get()
        result = await run_inference(req)     # 异步推理
        await send_response(result)
```

- **背压**：`maxsize` 满了，`put` 会阻塞，防止"生产太快撑爆内存"。
- **多消费者**：`gather(consumer(q, 1), consumer(q, 2))` 实现并行处理。
- **哨兵值**：第 0 节用 `None` 作为"结束信号"，让消费者优雅退出。

> 你第 23 篇的 continuous batching 调度器，就是在这个骨架上加"攒 batch""抢占""KV cache 管理"。

---

## 3. 深入细节：三个易错点

### 坑 1：忘写 `await`，协程根本没执行

```python
async def f(): return 42

async def main():
    f()                 # ❌ 没 await：只是创建了协程对象，没执行，还报 "never awaited" 警告
    return await f()    # ✅ 正确
```

### 坑 2：`create_task` 后不保留引用，任务被 GC、异常被吞

```python
async def main():
    for i in range(1000):
        asyncio.create_task(work(i))   # ❌ 引用丢失，可能被 GC，异常静默
    # 正确：存进集合
    tasks = {asyncio.create_task(work(i)) for i in range(1000)}
    await asyncio.gather(*tasks)
```

### 坑 3：把 CPU 密集计算直接放进协程，卡死循环

```python
async def main():
    heavy_compute()          # ❌ 纯 Python CPU 密集，不 await 不释放，卡死事件循环
    await asyncio.to_thread(heavy_compute)   # ✅ 丢线程池
```

---

## 4. 回到你的项目

1. **把 `generate()` 改造成"可异步步进"**：现在的 `generate` 是同步 `for _ in range(max_new_tokens)` 一口气生成完。asyncio 化的方向（第 23 篇详做）：

```python
class AsyncEngine:
    async def step(self, tokens: torch.Tensor) -> torch.Tensor:
        # 生成一个 token；GPU 计算期间用 await 让事件循环服务别的请求
        logits = await asyncio.to_thread(self.model.forward, tokens)  # 简化示意
        return sample(logits)
```

2. **用 `asyncio.Queue` 管请求**：将来 serving 的请求队列、KV cache 块池的"等待队列"，都用 `asyncio.Queue`。

3. **警惕阻塞调用**：你的引擎里 `torch.cuda.synchronize()`、`torch.load()`（磁盘 I/O）如果放进协程，会卡死事件循环——要用 `asyncio.to_thread` 包裹。

**今日动手任务（20 分钟）**：把第 0 节的"生产者-消费者"demo 改成你自己的场景：`producer` 模拟"收到请求（带 seq_len）"，`consumer` 模拟"用 `await asyncio.to_thread(time.sleep, seq_len*0.1)` 异步推理"，观察两个 consumer 如何交替处理请求。

---

## 5. 自测题

**Q1（概念）**：`await asyncio.sleep(1)` 和 `time.sleep(1)` 在协程里的本质区别是什么？

**Q2（应用）**：写出一个"同时请求 3 个模型后端、取最快返回结果"的函数（提示：`asyncio.wait` 的 `FIRST_COMPLETED` 或 `as_completed`）。

**Q3（思考）**：为什么 asyncio 的"单线程协程"比"多线程"更适合海量 I/O 并发？从内存开销、切换开销、竞态条件三个角度回答。

<details>
<summary>点击展开答案</summary>

**A1**：`await asyncio.sleep(1)` 是**非阻塞**——挂起当前协程、把控制权交还事件循环，让事件循环去执行其他协程，1 秒后再唤醒；`time.sleep(1)` 是**阻塞**——让当前**线程**睡 1 秒，期间事件循环完全停摆，所有协程都卡住，并发退化成串行。

**A2**（示例）：
```python
async def fastest(*coros):
    tasks = [asyncio.create_task(c) for c in coros]
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for p in pending:
        p.cancel()                 # 取消未完成的
    return done.pop().result()     # 返回最快的结果
```

**A3**：① **内存开销**：每个线程有独立栈（约 1MB+），一万线程 = 上万 MB；协程是轻量对象（KB 级），一万协程开销极小。② **切换开销**：线程切换要进内核态、保存/恢复上下文，昂贵；协程切换是事件循环内的用户态切换，几乎零成本。③ **竞态条件**：多线程共享内存，处处要加锁、易出 race condition；单线程协程只在 `await` 处切换，天然避免了大部分共享内存竞态，心智负担小得多。
</details>

---

## 6. 延伸阅读

- [asyncio 官方文档](https://docs.python.org/3/library/asyncio.html) —— 事件循环、Task、Queue、同步原语。
- [Python 官方 asyncio 教程](https://docs.python.org/3/library/asyncio-task.html)。
- [realpython — asyncio 指南](https://realpython.com/async-io-python/)（时效性较好）。

---

> **下一篇 → `13_性能剖析.md`**：`cProfile`/`py-spy`/`line_profiler`/`snakeviz` 怎么用，如何找到你引擎里真正的热点函数——性能优化的第一步是"测量"，而不是"感觉"。
