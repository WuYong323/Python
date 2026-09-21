# 21 · torch.compile 与 Triton：把性能优化自动化

> **一句话定位**：`torch.compile` 用"图捕获 + guard + 代码生成"自动把你的 PyTorch 代码编译成高效 kernel；Triton 让你用 Python 写 GPU kernel。两者结合，是 2025 年 AI Infra 的"性能自动化"主流路径。
> **前置依赖**：19 torch 内部机制、20 自定义算子
> **预计时长**：1 天
> **对应你的项目**：你已经手写过 CUDA/Triton kernel（`cuda_study/`），本篇教你用 `torch.compile` 把"手写 kernel"和"自动编译"接起来。

---

## 0. 先跑起来

```python
# 21_torch_compile_demo.py
import torch
import time

def fn(x, w):
    # 一个"组合式"函数：会被拆成多个 kernel
    return (x @ w.t()).relu().sum(dim=-1)

x = torch.randn(64, 128, device="cuda" if torch.cuda.is_available() else "cpu")
w = torch.randn(256, 128, device=x.device)

# ── 1. 直接跑（eager）：每个操作一个 kernel ──
fn(x, w)                       # 先热身
t0 = time.perf_counter()
for _ in range(100): fn(x, w)
eager_t = time.perf_counter() - t0

# ── 2. torch.compile 编译后跑 ──
compiled = torch.compile(fn)   # 一行！自动图捕获 + 代码生成 + 融合
compiled(x, w)                 # 第一次调用会编译（慢），之后快
t0 = time.perf_counter()
for _ in range(100): compiled(x, w)
compiled_t = time.perf_counter() - t0

print(f"eager: {eager_t:.4f}s  compiled: {compiled_t:.4f}s")

# ── 3. 看它生成了什么（可选） ──
# import torch._dynamo as dynamo
# dynamo.explain(fn)(x, w)   # 查看图捕获/重编译原因
```

运行：`python 21_torch_compile_demo.py`（有 GPU 效果更明显，CPU 也能跑）。

---

## 1. 一句话直觉

> **`torch.compile` = "JIT 编译器 + 自动优化器"：它先"偷看"你的代码画一张图（图捕获），然后自动把图里能合并的算子融合成更少的 kernel（代码生成），以后遇到形状一样的输入就直接用编译好的版本（guard 缓存）。**

类比：`torch.compile` 像**翻译官 + 打包员**——把你说的"多句话"（多个算子）听一遍（图捕获），翻译成"一句高效的话"（融合成少数 kernel）并打包好；下次你说同样的话（guard 命中），直接放录音（缓存），不再重新翻译。

---

## 2. 核心原理

### 2.1 torch.compile 的三阶段

```
① 图捕获（Dynamo）  → ② 图优化 + 代码生成（Inductor） → ③ 缓存 + guard 重放
```

| 阶段 | 组件 | 做什么 |
|------|------|--------|
| 图捕获 | **TorchDynamo** | 运行 Python 字节码，"偷看"实际执行，记录成 FX 图 |
| 代码生成 | **TorchInductor** | 把 FX 图转成 Triton/C++ kernel，做算子融合 |
| 重放 | **guard** | 记住"什么条件会破坏编译结果"，条件变就重编译 |

### 2.2 guard 与重编译：torch.compile 的第一大坑

`torch.compile` 会记录**输入的形状、dtype、device、stride** 等作为 guard。**一旦这些变了，就触发重编译**（慢）。

```python
compiled = torch.compile(fn)
compiled(torch.randn(64, 128))    # 编译（shape=64x128）
compiled(torch.randn(64, 128))    # guard 命中，直接用缓存（快）
compiled(torch.randn(128, 128))   # shape 变了 → 重编译！（慢）
```

**工业教训**：
1. **推理时固定 batch/seq 形状**，或做 padding 到固定长度，避免频繁重编译。
2. **用 `dynamic=True`** 允许动态形状（但性能略降）：

```python
compiled = torch.compile(fn, dynamic=True)   # 容忍形状变化，少重编译
```

3. **生产模式用 `reduce-overhead`**：减少每次调用的 Python 开销，适合"小算子反复调"的推理场景：

```python
compiled = torch.compile(fn, mode="reduce-overhead")
```

### 2.3 三种 mode 怎么选

| mode | 特点 | 适用 |
|------|------|------|
| `default` | 平衡 | 通用 |
| `reduce-overhead` | 减 Python 开销，内存多用点 | **推理**（小算子反复调） |
| `max-autotune` | 编译时自动调优（最慢编译、最快运行） | 训练、大算子 |

> 你的推理引擎，`mode="reduce-overhead"` 通常是最合适的起手式。

### 2.4 为什么 torch.compile 能加速：算子融合

你 `TorchBackend.fused_ffn` 里的 `gate * up` 再 `@ w_down`，eager 模式是**多个 kernel**，每个 kernel 都读写一遍显存（memory-bound）。`torch.compile` 能把它们**融合成更少的 kernel**，中间结果留在寄存器/共享内存，**少读写显存 = 更快**。

这就是你 `cuda_study` 里"访存优化四板斧"（合并访存、共享内存、减少显存往返）在编译器层面的**自动化版本**——你手写过一遍，现在能理解 `torch.compile` 在替你做什么。

### 2.5 Triton：用 Python 写 GPU kernel

你已经在 `cuda_study` 学过 Triton，这里只点**工业级关键点**：

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)
```

**Triton vs 手写 CUDA**：Triton 用 Python DSL 写 kernel，**自动处理** block 调度、向量化、部分优化；比 CUDA 好写、可移植（NVIDIA/AMD），性能接近手写 CUDA。**2025 年，除非极致性能，新算子优先 Triton**。

**`@triton.autotune`**：自动扫描 BLOCK 大小等配置，找最快的：

```python
@triton.autotune(configs=[triton.Config({"BLOCK": 128}), triton.Config({"BLOCK": 256})],
                 key=["N"])
@triton.jit
def kernel(...): ...
```

### 2.6 用户 Triton kernel 接入 torch.compile

这是"手写 kernel"和"自动编译"的结合点（官方 recipe 的核心）：

```python
# 1. 写 Triton kernel（如上 add_kernel）
# 2. 包装成普通 Python 函数
def triton_add(x, y):
    out = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]),)
    add_kernel[grid](x, y, out, N, BLOCK=256)
    return out

# 3. 让 torch.compile 追踪这个函数（配合第 20 篇的 register_fake）
compiled_add = torch.compile(triton_add)
```

**关键**：要让 `torch.compile` 能吃下你的 Triton kernel，需要（第 20 篇的）`register_fake`，让编译器在 meta 模式下知道你的 kernel 输出什么形状。**第 20 篇和第 21 篇在这里闭环**。

---

## 3. 深入细节：torch.compile 的三个高频坑

### 坑 1：第一次调用很慢（编译开销）

```python
compiled(x, w)   # 第一次：要编译，可能比 eager 还慢！
# 之后的调用才快。所以：先"热身"一次，再测性能。
```

**对策**：测量性能前先 `compiled(x, w)` 跑一次热身；生产环境启动时预热。

### 坑 2：图断裂（graph break）

某些 Python 结构（动态控制流依赖张量值、`print`、某些库调用）会让 Dynamo **捕获失败**，退化成"部分编译 + eager"，性能打折：

```python
def bad(x):
    if x.sum() > 0:      # 依赖张量值的 if —— 可能图断裂
        return x * 2
    return x
```

**排查**：用 `torch._dynamo.explain(fn)(x)` 看哪里断裂。**尽量让被编译的函数"张量友好"**（纯张量操作、少动态控制流）。

### 坑 3：`torch.compile` 不是万能的

- 它**不能**把"算法上低效"的代码变快（比如 O(n^2) 的朴素 attention 还是 O(n^2)，只是常数变小）。
- 它**不能**替代"算法级优化"（FlashAttention 是算法优化，torch.compile 是工程优化）。**先算法，后工程**。

> 你的引擎里，`TorchBackend.attention` 是朴素 O(n^2) 注意力，`torch.compile` 能让它常数级变快，但**无法**达到 FlashAttention 的量级——那需要算法级的 kernel（你已经在做 CUDA/Triton 版了）。理解这个边界，才知道何时该 compile、何时该手写 kernel。

---

## 4. 回到你的项目

1. **给 `TorchBackend` 的三个算子加 `torch.compile` 对比**：`torch.compile(be.rmsnorm)` 前后各测一次，看融合带来的提升（尤其 `fused_ffn` 这种多算子组合）。
2. **给你的 Triton kernel 补 `register_fake` 并接 `torch.compile`**：这是第 20 篇任务的自然延伸，验证"手写 kernel 能被自动编译吃掉"。
3. **记录重编译**：如果你的推理有变长序列，观察 `torch.compile` 是否频繁重编译，决定要不要 `dynamic=True` 或 padding。

**今日动手任务（25 分钟）**：用 `torch.compile(mode="reduce-overhead")` 编译你的 `fused_ffn`，对比 eager 的耗时（先热身再测），记录提升倍数；再用 `torch._dynamo.explain` 看有没有图断裂。

---

## 5. 自测题

**Q1（概念）**：torch.compile 的三阶段（图捕获、代码生成、重放）各自做什么？guard 的作用是什么？

**Q2（应用）**：为什么"先热身再测性能"对 torch.compile 特别重要？如果直接测第一次调用会得出什么错误结论？

**Q3（思考）**：torch.compile 能把朴素 O(n^2) attention 加速，但为何不能替代 FlashAttention？这揭示了"工程优化"和"算法优化"的什么区别？

<details>
<summary>点击展开答案</summary>

**A1**：① **图捕获（Dynamo）**：运行 Python 字节码，记录实际执行的算子，构建 FX 计算图；② **代码生成（Inductor）**：把 FX 图转成 Triton/C++ kernel，做算子融合等优化；③ **重放（guard）**：记录"输入的形状/dtype/device/stride 等条件"，下次调用时检查这些条件，没变就直接用缓存的编译结果，变了就重编译。guard 就是"缓存有效性检查"。

**A2**：torch.compile 的**第一次调用要编译**（图捕获 + 代码生成），耗时远超 eager，可能慢几十倍。若直接测第一次调用，会得出"torch.compile 更慢"的**错误结论**。正确做法：先 `compiled(x)` 跑一次触发编译（热身），再循环测后续调用的稳态性能。

**A3**：torch.compile 是**工程优化**——它减少 kernel 数量、减少显存读写、减少 Python 开销，但**不改变算法复杂度**：朴素 attention 的 O(n^2) 计算量和显存占用不变，只是常数变小。FlashAttention 是**算法优化**——它通过分块 + online softmax 把显存占用从 O(n^2) 降到 O(n)，从根本上改变了复杂度。**先算法（降复杂度）、后工程（降常数）**，两者是不同层次：torch.compile 无法把一个 O(n^2) 算法变成 O(n)。
</details>

---

## 6. 延伸阅读

- [torch.compile 官方文档](https://pytorch.org/docs/stable/torch.compiler.html)。
- [用户 Triton kernel 接入 torch.compile（官方 recipe）](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html)。
- [Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)。
- [TorchInductor 深入](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)。

---

> **下一篇 → `22_KVCache与PagedAttention生产化.md`**：把你 `PagedAttention.py` 的"骨架演示"升级成 vLLM 级别的生产级内存池——preemption、recompute、copy-on-write。
