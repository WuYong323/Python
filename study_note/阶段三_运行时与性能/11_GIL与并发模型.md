# 11 · GIL 与并发模型：线程 / 进程 / 协程怎么选

> **一句话定位**：GIL 是 CPython 的"全局解释器锁"，它决定了"为什么 Python 多线程对 CPU 密集任务没用、对 I/O 密集却很好"。理解它，你才能在推理 serving、数据加载、算子调度里选对并发模型。
> **前置依赖**：01 对象模型
> **预计时长**：1 天
> **对应你的项目**：你的引擎现在单请求同步推理，将来上 serving 必须并发——本篇是选型地基。

---

## 0. 先跑起来

```python
# 11_concurrency_demo.py
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def cpu_bound(n: int) -> int:
    """CPU 密集：纯 Python 计算（GIL 全程锁死）"""
    s = 0
    for i in range(n):
        s += i * i
    return s

def io_bound(seconds: float) -> None:
    """I/O 密集：模拟等待网络/磁盘（等待时 GIL 被释放）"""
    time.sleep(seconds)

def run(label: str, fn, args, workers: int, pool_cls):
    t0 = time.perf_counter()
    with pool_cls(max_workers=workers) as pool:
        list(pool.map(fn, args))
    print(f"{label}: {time.perf_counter() - t0:.2f}s")

N = 10_000_000
# ── CPU 密集：多线程 vs 多进程 ──
run("CPU 密集-单线程", cpu_bound, [N], 1, ThreadPoolExecutor)
run("CPU 密集-4线程 ", cpu_bound, [N]*4, 4, ThreadPoolExecutor)   # 几乎不会变快！
run("CPU 密集-4进程 ", cpu_bound, [N]*4, 4, ProcessPoolExecutor)  # 明显变快

# ── I/O 密集：多线程 vs 多进程 ──
run("I/O 密集-4线程 ", io_bound, [0.5]*4, 4, ThreadPoolExecutor)  # ~0.5s，线程很合适
```

运行：`python 11_concurrency_demo.py`（Windows 上多进程要放在 `if __name__ == "__main__":` 里，见文件末尾说明）。

**观察三个关键现象**：
1. CPU 密集：4 线程 ≈ 单线程（GIL 让线程"轮流用一个核"），4 进程才真正并行加速。
2. I/O 密集：4 线程 ≈ 单次耗时（因为 sleep 时 GIL 被释放，4 个线程同时等）。

---

## 1. 一句话直觉

> **GIL 像"教室里唯一的麦克风"——同一时刻只有一个线程能拿着麦克风讲 Python 字节码；但线程去"做作业不发言"（I/O 等待、C 扩展计算）时，会主动放下麦克风，别人就能用。**

类比：CPython 执行 Python 字节码，就像全班讨论要轮流用麦克风（GIL）。CPU 密集任务 = 一直拿着麦克风讲个不停，多线程也没用（还是一个人在讲）；I/O 密集任务 = 讲一句就去查资料（sleep/网络/磁盘），把麦克风让给别人，多线程就能高效交替。

---

## 2. 核心原理

### 2.1 并发（concurrency）≠ 并行（parallelism）

| 概念 | 含义 | 关键词 |
|------|------|--------|
| **并发** | 同时"处理"多件事（快速切换，宏观同时） | 交替执行 |
| **并行** | 同时"执行"多件事（真正多核同时算） | 同时执行 |

GIL 允许**并发**（多线程交替），但**限制 CPU 密集任务的并行**（同一时刻只有一个线程执行 Python 字节码）。

### 2.2 GIL 到底锁什么、不锁什么

| 场景 | GIL 是否限制 | 原因 |
|------|-------------|------|
| 纯 Python 循环计算 | ✅ 限制 | 全程执行 Python 字节码，锁死 |
| `time.sleep` / 网络 / 磁盘 I/O | ❌ 不限制 | 等待时线程**主动释放 GIL** |
| numpy / torch 的矩阵运算 | ❌ 不限制 | C 扩展内部释放 GIL 做多线程计算 |
| CUDA kernel 执行 | ❌ 不限制 | GPU 计算在设备上，不占 GIL |

> **关键推论**：你推理引擎里的 `torch.matmul`、CUDA kernel，**在 C/CUDA 层执行时会释放 GIL**。所以"Python 多线程 + torch GPU 计算"是可行的——但业界 serving 主流仍选 asyncio，原因见下。

### 2.3 三种并发模型对比

| 模型 | 原理 | 适合 | 不适合 | 开销 |
|------|------|------|--------|------|
| **多线程** | 多个线程共享内存，GIL 限制 CPU | I/O 密集（网络/文件/DB） | CPU 密集纯 Python | 线程切换开销小，共享内存 |
| **多进程** | 每个进程独立 GIL、独立内存 | CPU 密集（绕过 GIL） | 频繁共享大数据（要 IPC） | 进程启动/通信开销大 |
| **asyncio** | 单线程 + 事件循环 + 协程 | 海量 I/O 并发（高吞吐） | CPU 密集（会阻塞循环） | 极低，无线程切换 |

### 2.4 决策口诀

```
任务 CPU 密集（纯 Python 算）   → 多进程 或 交给 C 扩展/numpy/torch
任务 I/O 密集（网络/文件/DB）   → 多线程 或 asyncio
任务海量并发 I/O（成千上万连接）→ asyncio（首选，单线程就能扛）
任务 GPU 计算（torch/CUDA）     → asyncio + 异步推理（主流）或 多线程
```

### 2.5 为什么推理 serving 用 asyncio？

一个推理服务要同时服务**成百上千个请求**，每个请求大部分时间在"等 GPU 算完 / 等下一个 token"。这是典型的 **I/O 密集 + 高并发**：

- **多线程**：上千线程 = 上千个线程栈 + 频繁切换 + 共享状态要加锁，复杂且容易出 bug。
- **asyncio**：**单线程 + 事件循环**，用协程表达"等 GPU 结果"这个等待，一个进程就能优雅地调度上千请求，开销极小。

> 这正是 vLLM、TGI 等推理框架用 async 调度器的根本原因。你第 23 篇会亲手搭一个 asyncio 推理调度器，本篇先把模型选对。

### 2.6 自由线程（no-GIL）：仍在走向"正式支持"

Python 3.13 引入**实验性**的"自由线程"构建（`python3.13t`），去掉 GIL 让多线程真正并行；3.14 延续该工作，并新增 [PEP 779](https://peps.python.org/pep-0779/) 明确"达到正式支持状态的标准"。**当前（截至 3.14）现状**：

- 仍属**实验性/未完全支持**，单线程有性能损失，部分 C 扩展（numpy/torch 的某些路径）尚未完全适配。
- **生产环境仍不建议默认开启**，但值得持续关注。

> 结论：**现阶段写生产代码仍按"有 GIL"来设计和选型**（多进程绕 CPU、asyncio 扛 I/O）。一旦自由线程达"正式支持"，选型逻辑会随之改变。

---

## 3. 深入细节：多线程的共享状态与锁

多线程共享内存，所以要防"竞态条件"（race condition）：

```python
import threading
counter = 0
lock = threading.Lock()

def inc():
    global counter
    for _ in range(100000):
        with lock:              # 加锁：读-改-写 变成原子操作
            counter += 1

threads = [threading.Thread(target=inc) for _ in range(4)]
[t.start() for t in threads]
[t.join() for t in threads]
print(counter)                  # 400000（不加锁会少于这个数，出现丢更新）
```

**为什么 `counter += 1` 需要锁**：它实际是"读 counter → 加 1 → 写回"三步，多线程可能交错执行，导致"丢更新"。`with lock:` 保证这三步互斥。

> 这就是 asyncio 的另一个优势：**单线程内协程只在 `await` 处切换，天然避免了大部分共享内存竞态**，心智负担小得多。你第 12 篇会体会到。

---

## 4. 回到你的项目

1. **认清你的负载类型**：你的推理引擎是 **GPU 计算 + I/O 等待** 混合。纯算子在 torch/CUDA 层（释放 GIL），但**调度、等待、请求管理**是 I/O 密集——**结论：asyncio 是正确选择**。
2. **`generate()` 目前是同步阻塞的**：一个请求 `model.generate()` 会占住整个进程，别的请求只能排队。改造方向（第 23 篇展开）：把"生成一个 token"变成可 `await` 的异步步进，让事件循环在等待 GPU 时去服务别的请求。
3. **数据加载可以用多线程/多进程**：你 `get_batch` 是一次性构造 batch。将来大数据集训练，`torch.utils.data.DataLoader(num_workers=4)` 内部就是多进程加载（绕 GIL 加速数据准备）。

**今日动手任务（10 分钟）**：跑第 0 节的 demo（多进程部分记得放 `if __name__ == "__main__":` 里），亲眼确认"CPU 密集 4 线程 ≈ 单线程、4 进程才加速"。

---

## 5. 自测题

**Q1（概念）**：GIL 导致 Python 多线程对 CPU 密集任务无效，但为什么 `time.sleep()` 的多线程却能"同时等待"？

**Q2（应用）**：以下场景各选什么并发模型？① 爬 1000 个网页；② 对 100 万张图做纯 Python 图像处理；③ 一个推理服务同时接 500 个请求；④ 训练时加载大 dataset。

**Q3（思考）**：为什么 torch 的 `torch.matmul` 在 GPU 上执行时，Python 的多线程不会因为 GIL 而变慢？这暗示了 AI Infra 里"重计算下沉到 C/CUDA 层"的什么设计原则？

<details>
<summary>点击展开答案</summary>

**A1**：`time.sleep` 是**阻塞式系统调用**，线程在等待期间不执行 Python 字节码，**主动释放 GIL**，所以其他线程能拿到 GIL 继续跑，多个线程看起来"同时等待"。GIL 只锁"执行 Python 字节码"这一件事，不锁"等待 I/O"。

**A2**：① asyncio（或线程池）——海量网络 I/O；② 多进程——纯 Python CPU 密集，绕 GIL；③ asyncio——高并发 I/O + GPU 等待；④ `DataLoader(num_workers=4)` 多进程——数据准备是 CPU 密集，绕过 GIL。

**A3**：`torch.matmul` 在 GPU 上执行时，Python 只负责"发起调用"，真正的矩阵运算在 **C++/CUDA 层**，那部分会**释放 GIL**（甚至在 GPU 上完全不占用 CPU 解释器）。这暗示 AI Infra 的核心设计原则：**把重计算下沉到 C/CUDA 层，Python 只做"胶水"和调度**——Python 的 GIL 和解释器开销就不再是瓶颈。你的 pybind11 + CUDA kernel 正是这一原则的实践。
</details>

---

## 6. 延伸阅读

- [Python GIL 官方术语表](https://docs.python.org/3/glossary.html#term-global-interpreter-lock) 与 [threading 模块](https://docs.python.org/3/library/threading.html)。
- [Python 自由线程（no-GIL）HOWTO](https://docs.python.org/3.14/howto/free-threading-python.html) 与 [PEP 779（正式支持标准）](https://peps.python.org/pep-0779/)。
- [PEP 703 — 使 GIL 可选](https://peps.python.org/pep-0703/)。

---

> **下一篇 → `12_asyncio核心与实战.md`**：事件循环、`async/await`、`gather`/`Task`/`Queue`/超时——亲手写出异步的生产者-消费者调度器，为你的推理 serving 打底。
