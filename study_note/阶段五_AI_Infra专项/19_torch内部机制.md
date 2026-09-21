# 19 · torch 内部机制：Tensor 布局 / autograd / buffer

> **一句话定位**：从"会用 torch"到"懂 torch"，关键在三件事：**Tensor 的内存布局（stride/contiguous）**、**autograd 计算图**、**parameter vs buffer**。这决定了你写算子和模型时的正确性与性能。
> **前置依赖**：01 对象模型、07 类型
> **预计时长**：1 天
> **对应你的项目**：`model.py` 里的 `register_buffer(persistent=False)`、`weight tying`、`transpose(1,2).contiguous()`、`_init_weights` 全是本篇内容。

---

## 0. 先跑起来

```python
# 19_torch_internals_demo.py
import torch

# ── 1. stride 与 contiguous：同一个数据，多种"看法" ──
x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
print("x.shape:", x.shape, "stride:", x.stride())   # (4, 1)：第0维跨4个元素，第1维跨1个
print("x 是否连续:", x.is_contiguous())             # True

y = x.t()                    # 转置是"视图"：不复制数据，只改 stride
print("y.shape:", y.shape, "stride:", y.stride())   # (1, 4)：stride 反了
print("y 是否连续:", y.is_contiguous())             # False！
print("y 与 x 共享存储:", y.data_ptr() == x.data_ptr())  # True（同一块内存）

# view 要求连续；对不连续的 y 会报错
# y.view(-1)                 # RuntimeError!
z = y.contiguous()           # 复制成连续布局（新内存）
print("z 是否连续:", z.is_contiguous(), "z.data_ptr==x:", z.data_ptr() == x.data_ptr())

# ── 2. dtype：float32 / float16 / bfloat16 ──
f32 = torch.tensor([1.0])
f16 = f32.half()             # float16：省显存但精度低、范围小
bf16 = f32.bfloat16()        # bfloat16：和 f32 同指数范围，精度介于两者（LLM 训练/推理主流）
print("f32 bits:", f32.element_size()*8, "| f16:", f16.element_size()*8, "| bf16:", bf16.element_size()*8)

# ── 3. autograd：计算图与梯度 ──
a = torch.tensor(2.0, requires_grad=True)   # 需要梯度
b = a * a                                   # b = a^2
b.backward()                                # 自动求 db/da = 2a = 4
print("a.grad:", a.grad)                    # tensor(4.)

with torch.no_grad():                       # 推理：关梯度，省显存省时间
    c = a * 3
print("c 是否要梯度:", c.requires_grad)      # False

# ── 4. parameter vs buffer ──
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(4))          # 可训练参数（进 optimizer）
        self.register_buffer("freqs", torch.arange(4))  # buffer：随模型搬，但不训练
        self.register_buffer("cache", torch.zeros(4), persistent=False)  # 不进 state_dict

m = M()
print("参数:", list(m.named_parameters()))    # 只有 w
print("buffer:", list(m.named_buffers()))     # freqs + cache
print("state_dict 键:", list(m.state_dict().keys()))   # w + freqs（cache 被 persistent=False 排除）
```

运行：`python 19_torch_internals_demo.py`。

---

## 1. 一句话直觉

> **Tensor = 一块连续内存 + 一个"如何解读这块内存"的元信息（shape/stride/dtype）。`view`/`transpose` 只改"解读方式"不复制数据；`contiguous()` 才真正复制。**

类比：Tensor 像**同一卷胶卷**，`stride` 是"播放顺序"。转置不是"重新洗照片"，只是"换个播放顺序"（同一卷胶卷倒着放）；`contiguous()` 才是"真的重新冲洗一卷按新顺序排好的胶卷"。

---

## 2. 核心原理

### 2.1 stride / view / reshape / transpose

- **底层存储**：Tensor 的数据存在一块**一维连续内存**（`storage`）里。
- **stride**：告诉 torch "第 i 维上，相邻两个元素在内存里隔多少个元素"。
- **`view`**：只改 shape/stride，**要求原 tensor 连续**；不连续会报错。
- **`reshape`**：**优先用 view，不行就复制**（安全但可能隐式拷贝）。
- **`transpose`/`permute`**：**视图**，改 stride，结果**非连续**。

**工业意义**：频繁的 `transpose + contiguous` 会**多一次显存拷贝**，浪费显存和带宽。你 `model.py` 里：

```python
q, k, v = (t.transpose(1, 2) for t in (q, k, v))   # 转成 [B,H,T,D]，非连续
y = self.backend.attention(q, k, v, causal=True)
y = y.transpose(1, 2).contiguous().view(B, T, C)    # 又转回来 + 强制连续
```

这里的 `contiguous()` 就触发了一次拷贝。理解 stride 后，你能判断"哪些操作真的需要 contiguous、哪些可以省"。高性能 kernel 里，**避免不必要的 `contiguous()` 是省显存的基本功**。

### 2.2 dtype 与混合精度

| dtype | 位数 | 特点 | 用途 |
|-------|------|------|------|
| float32 | 32 | 精度高、慢、占显存 | 默认、调试 |
| float16 | 16 | 省显存、快，但**范围小**（易溢出） | 老式混合精度 |
| **bfloat16** | 16 | 和 fp32 **同指数范围**（不易溢出），精度低 | **LLM 训练/推理主流** |
| int8/fp8 | 8 | 极致省显存 | 量化推理 |

**混合精度**：权重/激活用 fp16/bf16 省显存提速，关键计算（如 softmax、norm）升回 fp32 保精度。你 `backend.py` 的 `rmsnorm` 里：

```python
in_dtype = x.dtype
x = x.float()          # 先升 fp32 算，保精度
...
return (x_norm.to(in_dtype)) * weight   # 再降回原 dtype
```

**这就是混合精度的标准写法**——你已经做对了，只是可能没意识到"为什么"：因为 norm 涉及平方求和，低精度会累积误差，升 fp32 再降回来是正确性需要。

### 2.3 autograd：计算图

```python
a = torch.tensor(2.0, requires_grad=True)
b = a * a            # 记录：b = a^2（在计算图上加一个 Mul 节点）
b.backward()         # 从 b 反向传播，算出 db/da
a.grad               # 4
```

- `requires_grad=True` 的张量参与运算时，torch 会**动态构建计算图**（记录每个操作）。
- `backward()` 沿图反向求梯度，存到各张量的 `.grad`。
- `with torch.no_grad():` 关闭图构建，**推理必用**（省显存、省时间）——你 `generate` 上的 `@torch.no_grad()` 就是这个。

**为什么推理要 no_grad**：训练要算梯度，所以每个中间张量都保留、记录操作；推理不需要梯度，`no_grad` 让 torch 不构建图、不保留中间张量，**显存和速度都显著改善**。

### 2.4 parameter vs buffer

| | `nn.Parameter` | `register_buffer` |
|---|---|---|
| 是否可训练 | ✅ 进 optimizer | ❌ 不训练 |
| 是否随模型 `.to(device)` | ✅ | ✅ |
| 是否进 `state_dict` | ✅ | 默认 ✅，`persistent=False` 则 ❌ |
| 典型用途 | 权重、bias | RoPE 因子、均值方差、缓存 |

**你 `model.py` 的两个关键用法**：

```python
# ① RoPE 因子注册成 buffer，persistent=False
self.register_buffer("freqs_cis", precompute_freqs_cis(...), persistent=False)
# 含义：freqs_cis 要随模型搬到 GPU（buffer 特性），但它可复算（persistent=False），
#      所以不进 checkpoint，省存储。你注释里已经写对了原理！

# ② 权重绑定 weight tying
self.lm_head.weight = self.tok_embeddings.weight
# 含义：让 lm_head 和 tok_embeddings 共享同一个 Parameter 对象（同一块显存），
#      省一半 embedding 显存。这是"对象模型"（01 篇）在 torch 里的直接应用：两个引用指向同一张量。
```

---

## 3. 深入细节：三个高频坑

### 坑 1：`view` 报错 = 不连续

```python
x = torch.randn(3, 4)
x.t().view(-1)     # RuntimeError: view size is not compatible
# 原因：转置后 stride 变了，view 要求连续
x.t().contiguous().view(-1)   # ✅ 或直接 .reshape(-1)
```

### 坑 2：原地操作破坏 autograd

```python
a = torch.tensor(2.0, requires_grad=True)
b = a * 2
b = b + 1          # ✅ 非原地，安全
# b += 1           # ❌ 原地操作，可能破坏计算图（报 "leaf variable" 或梯度错）
```

**规则**：**需要梯度的叶子张量，别用原地操作**（`+=`、`x[0]=...`）。你 `_init_weights` 里的 `nn.init.normal_(...)` 是原地操作，但**那是初始化（还没进计算图），所以安全**——时机是关键。

### 坑 3：`state_dict` 的键不对会静默失败

```python
model.load_state_dict(ckpt)          # 若键不匹配，默认报错
model.load_state_dict(ckpt, strict=False)   # strict=False：忽略多余/缺失键（危险，可能静默丢权重）
```

**建议**：加载时用 `strict=True`（默认），键对不上就报错，逼你排查。你 `model.py` 的 `model.load_state_dict(ckpt["model"])` 是默认 strict，✅。

---

## 4. 回到你的项目

1. **审查所有 `transpose` + `contiguous`**：`model.py` 的 attention 前后转换，用 `is_contiguous()` 打印确认哪些转换真的需要 `contiguous()`，哪些可以省掉（省一次显存拷贝）。
2. **理解你的 `rmsnorm` 混合精度写法**：现在你能说清"为什么先 `.float()` 再 `.to(in_dtype)`"了（见 2.2）。
3. **理解 `persistent=False` 和 weight tying**：它们分别是"buffer 不进 checkpoint"和"两个引用共享一张量"，本质都是省显存——这是 AI Infra 的核心素养。
4. **给 `_init_weights` 加注释**：标明"这是初始化期，原地操作安全"。

**今日动手任务（15 分钟）**：在 `model.py` 里加几行调试打印，确认 `freqs_cis` 是否在 `state_dict()` 里（应不在，因 persistent=False）、`lm_head.weight` 和 `tok_embeddings.weight` 是否共享同一 `data_ptr()`（应在，因 weight tying）。

---

## 5. 自测题

**Q1（概念）**：`view`、`reshape`、`transpose`、`contiguous` 四者的区别是什么？哪些是"视图"（不复制）、哪些可能复制？

**Q2（应用）**：判断下面代码是否报错，为什么？

```python
x = torch.randn(2, 3, 4)
y = x.permute(1, 0, 2)   # shape (3,2,4)
z = y.view(3, 8)         # 会报错吗？
```

**Q3（思考）**：为什么 LLM 推理用 `bfloat16` 而不是 `float16`？结合两者的"指数范围"差异，说明 bfloat16 在"大数值求和（如 attention score）"场景的优势。

<details>
<summary>点击展开答案</summary>

**A1**：`view` 是视图（不复制），但**要求原 tensor 连续**；`reshape` 是"能 view 就 view、不能就复制"的**安全版**；`transpose`/`permute` 是视图（不复制），但结果**非连续**；`contiguous()` 是**复制**成连续布局。视图共享内存（`data_ptr` 相同），复制产生新内存。

**A2**：会报错。`permute(1,0,2)` 后 `y` 的 stride 变成 (4, 12, 1)，是**非连续**的，`view` 要求连续，所以 `y.view(3,8)` 抛 `RuntimeError`。应改成 `y.contiguous().view(3, 8)` 或 `y.reshape(3, 8)`。

**A3**：`float16` 的**指数范围小**（最大约 65504），大数值求和时**极易溢出**成 inf；`bfloat16` 用 8 位指数（和 float32 相同），**指数范围大**（最大约 3.4e38），大数值相加不易溢出，代价是**尾数精度低**（7 位 vs 10 位）。attention score、norm 求和等场景数值范围大，bfloat16 的"大范围 + 低精度"正好匹配（求和不需要很高尾数精度，但需要防溢出），所以 LLM 训练/推理主流用 bfloat16。
</details>

---

## 6. 延伸阅读

- [PyTorch Tensor 与 stride](https://pytorch.org/docs/stable/tensors.html) 与 [contiguous 语义](https://pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html)。
- [autograd 机制](https://pytorch.org/docs/stable/notes/autograd.html)。
- [nn.Module 与 buffer/parameter](https://pytorch.org/docs/stable/notes/modules.html)。
- [bfloat16 说明](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)。

---

> **下一篇 → `20_自定义算子工业级.md`**：`TORCH_LIBRARY` + dispatcher + `register_fake` + 自定义 autograd——把你的 `bindings.cpp` 升级成"meta 安全、autograd 正确、能被 torch.compile 吃掉"的生产级自定义算子。
