# 04 · 类与 OOP 工业实践：从"能跑"到"可维护"

> **一句话定位**：把面向对象从"会写 class"升级到"会用 dataclass/Enum/property/ABC 写出稳、可读、可扩展的工业类"。
> **前置依赖**：01 对象模型、03 装饰器
> **预计时长**：1 天
> **对应你的项目**：`Config` 已经用了 `@dataclass`、`Backend` 用了 `ABC`，但 `self.mlp._is_residual_proj=True` 这种动态补丁说明 OOP 设计还能更稳。

---

## 0. 先跑起来

```python
# 04_oop_demo.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto

# ── 1. dataclass：自动生成 __init__/__repr__/__eq__ ──
@dataclass
class Config:
    vocab_size: int = 50257
    n_embd: int = 768
    n_layer: int = 12
    dropout: float = 0.1

c1 = Config()
c2 = Config(n_embd=256, n_layer=4)
print(c1)                    # Config(vocab_size=50257, n_embd=768, ...)  ← 自动可读 repr
print(c1 == Config())        # True  ← 自动值比较
print(c2)                    # Config(vocab_size=50257, n_embd=256, n_layer=4, dropout=0.1)

# ── 2. 可变默认值的坑 & field(default_factory) ──
@dataclass
class BlockState:
    phys_ids: list[int] = field(default_factory=list)   # ✅ 每个实例独立的空 list
    # 若写 phys_ids: list[int] = []  ← ❌ 所有实例共享同一个 list（经典大坑）

a, b = BlockState(), BlockState()
a.phys_ids.append(7)
print(a.phys_ids, b.phys_ids)   # [7] []  ← 互不影响（正确）

# ── 3. Enum：类型安全的常量 ──
class Device(Enum):
    CPU = auto()
    CUDA = auto()

def to_torch(dev: Device):
    return "cuda" if dev is Device.CUDA else "cpu"

print(to_torch(Device.CUDA))    # 'cuda'
# to_torch("cuda")              # 传字符串会怎样？见自测题 Q1

# ── 4. @property：方法变"属性"，封装内部逻辑 ──
class Sequence:
    def __init__(self, block_size: int):
        self.block_size = block_size
        self._length = 0

    @property
    def length(self):                  # 读：seq.length（像属性，不用加括号）
        return self._length

    @property
    def num_blocks(self):              # 惰性计算：用时才算
        return (self._length + self.block_size - 1) // self.block_size

    def append(self):
        self._length += 1

s = Sequence(16)
s.append(); s.append()
print(s.length, s.num_blocks)          # 2 1
# s.length = 5                          # AttributeError: can't set attribute（只读保护）

# ── 5. ABC：强制子类实现接口 ──
class Backend(ABC):
    @abstractmethod
    def rmsnorm(self, x): ...

class TorchBackend(Backend):
    def rmsnorm(self, x): return x
    # 若漏掉某个 @abstractmethod，实例化时会报 TypeError

TorchBackend()   # OK，因为实现了所有抽象方法
```

运行：`python 04_oop_demo.py`。

---

## 1. 一句话直觉

> **`dataclass` 是"数据容器"的标配模板，`Enum` 是"有限选项"的安全锁，`@property` 是"给属性装上安检门"，`ABC` 是"接口的合同"。**

类比：写类就像开公司——`dataclass` 帮你自动办好营业执照和门面（`__init__`/`__repr__`/`__eq__`）；`Enum` 规定员工只能选几个固定岗位（防拼写错、防魔法字符串）；`@property` 是在财务室门口设个门卫（想改数据得走流程）；`ABC` 是劳动合同（说好必须会哪些技能，不会就别入职）。

---

## 2. 核心原理

### 2.1 类的快速回顾：实例属性 vs 类属性

```python
class Model:
    device = "cpu"          # 类属性：所有实例共享，通过类或实例都能访问
    def __init__(self, name):
        self.name = name    # 实例属性：每个实例独立

m1, m2 = Model("a"), Model("b")
print(m1.device, m2.device)   # cpu cpu  ← 共享
Model.device = "cuda"         # 改类属性，影响所有实例
print(m1.device, m2.device)   # cuda cuda
```

**坑**：`m1.device = "x"` 不会改类属性，而是给 `m1` 新建一个实例属性，把类属性"遮住"了（读 `m1.device` 得到 x，`m2.device` 还是 cuda）。理解"实例属性遮蔽类属性"，能少踩很多坑。

### 2.2 继承与 MRO（方法解析顺序）

单继承简单；**多继承**的坑在于"同名方法到底调谁的"。Python 用 **C3 线性化算法**算出 MRO，顺序存在 `类.__mro__`：

```python
class A: ...
class B(A): ...
class C(A): ...
class D(B, C): ...     # 菱形继承

print(D.__mro__)   # (D, B, C, A, object)
```

`super()` 就是**沿着 MRO 顺序找下一个**：

```python
class A:
    def f(self): print("A")
class B(A):
    def f(self): print("B"); super().f()   # 调 MRO 里 B 后面的下一个 = A
class C(A):
    def f(self): print("C"); super().f()
class D(B, C):
    def f(self): print("D"); super().f()   # D → B → C → A（C3 决定的顺序）

D().f()   # 打印 D B C A
```

**工业建议**：
- **能用组合就不用多继承**（`class Model: self.backend = Backend()` 而不是 `class Model(Backend)`）。你的项目里 `LlamaModel` 持有 `self.backend` 而非继承 backend——**这是对的**，叫"组合优于继承"。
- 必须用多继承时（如 `class MyOp(torch.autograd.Function)`），用 `__mro__` 排查顺序。
- `super()` 统一用无参形式 `super().__init__()`（3.x 的规范写法）。

### 2.3 `@property`：封装 + 惰性计算

```python
@property
def x(self): return self._x        # 读
@x.setter
def x(self, v): self._x = v        # 写（可选）
@x.deleter
def x(self): del self._x           # 删（可选）
```

**三个经典用途**：
1. **只读属性**：外部能读 `seq.length`，但改不了（防止误改内部状态）。
2. **惰性计算**：`num_blocks` 用时才算，不存冗余状态（状态变了自动跟着变，不会"缓存失效"）。
3. **校验**：setter 里 `if v < 0: raise ValueError(...)`，把非法值挡在门外。

> 你的 `PagedAttention.py` 里 `Sequence.length` 现在是个裸属性 `self.length=0`，外部可以随便 `seq.length = -5` 搞坏状态。改成 `@property`（内部 `_length`），就是工业化的第一步。

### 2.4 `@classmethod` vs `@staticmethod`

| | 第一个参数 | 能访问类吗 | 典型用途 |
|---|-----------|-----------|---------|
| `@classmethod` | `cls`（类本身） | 能 | 工厂方法、类级配置 |
| `@staticmethod` | 无 | 不能 | 纯工具函数，只是"寄居"在类里 |

```python
class Config:
    def __init__(self, d): self.d = d

    @classmethod
    def from_json(cls, path):          # 工厂方法：从 json 造 Config
        import json
        return cls(json.load(open(path)))

    @staticmethod
    def is_valid(vocab): return vocab > 0   # 与实例无关的纯函数

cfg = Config.from_json("config.json")      # 不用先造实例再填
```

> 你的 `Config` 目前用 `Config(**ckpt['model_args'])` 从字典构造。给它加个 `@classmethod from_dict(cls, d)`，能集中处理"哪些字段可选、默认值、校验"，比散落在 `__main__` 里的裸 dict 干净得多。

### 2.5 `dataclass`：数据容器的工业标配

`@dataclass` 自动生成 `__init__`/`__repr__`/`__eq__`，还能定制：

```python
from dataclasses import dataclass, field

@dataclass(frozen=True)      # frozen=True：实例不可变（线程安全、可哈希）
class ModelArgs:
    n_embd: int
    n_layer: int = 12
    tags: list[str] = field(default_factory=list)   # 可变默认值必须用 factory
```

**三个必记点**：
1. **可变默认值必须 `field(default_factory=...)`**，否则所有实例共享同一个 list（见第 0 节示例 2）。
2. `frozen=True` 让实例**不可变**：适合当配置对象、dict 的 key，且天然线程安全。你 `model.py` 的 `Config` 建议加 `frozen=True`。
3. 需要**复杂校验/转换**（如"n_head 必须整除 n_embd"）时，用 `__post_init__`：

```python
@dataclass(frozen=True)
class Config:
    n_embd: int
    n_head: int

    def __post_init__(self):
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd 必须能被 n_head 整除")
```

> 你代码里 `assert n_embd % n_head == 0` 就是干这个的，但 `assert` 会被 `python -O` 优化掉、且报错信息弱。移到 `__post_init__` 用 `raise ValueError` 才是工业做法（第 09 篇细讲）。

### 2.6 `Enum`：消灭魔法字符串

```python
class BackendKind(Enum):
    TORCH = auto()
    TRITON = auto()
    CUDA = auto()

def build(kind: BackendKind) -> Backend:
    match kind:
        case BackendKind.TORCH:  return TorchBackend()
        case BackendKind.TRITON: return TritonBackend()
        case BackendKind.CUDA:   return CudaBackend()
```

好处：**拼写错会立刻报 AttributeError**（`BackendKind.TORCHH` 直接崩），而不是像魔法字符串 `"torchh"` 那样静默出错到运行时才炸。你的引擎将来要做"多后端可切换"（Torch/Triton/CUDA），`Enum` 是标配。

---

## 3. 深入细节：修掉你的 `_is_residual_proj` 补丁

你 `model.py` 里有这么一段（脆弱信号）：

```python
self.mlp = MLP(config.n_embd, config.dropout)
self.mlp._is_residual_proj = True   # 用动态属性"偷偷"传状态
```

然后 `_init_weights` 里靠 `hasattr(module, "_is_residual_proj")` 判断是否缩放初始化。**问题**：这是用"运行时动态属性"传递"设计意图"，靠 `hasattr` 这种脆弱约定，拼写错、漏设、别的类误设都不报错。

**工业改法**：把意图变成**显式参数**（dataclass 字段或构造参数）：

```python
@dataclass
class InitConfig:
    std: float = 0.02
    residual_scale: bool = False      # 显式字段，而非魔法属性

# 或者在 __init__ 里明确传：
self.mlp = MLP(config.n_embd, config.dropout, is_residual_proj=True)
```

然后在 `_init_weights` 里用 `isinstance` + 显式字段判断，而不是 `hasattr`。**原则：能用类型/参数表达的设计意图，绝不用运行时动态属性**（第 17 篇设计模式会展开"显式优于隐式"）。

---

## 4. 回到你的项目

1. **给 `Config` 加 `frozen=True` + `__post_init__` 校验**：把 `n_embd % n_head == 0`、`multiple_of` 相关约束从 `assert` 移到 `__post_init__`。
2. **给 `Sequence.length` 改成 `@property`**：内部 `_length`，外部只读。
3. **把 `_is_residual_proj` 补丁改成显式字段**（见第 3 节）。
4. **给 `BackendKind` 建一个 `Enum`**，为多后端切换铺路。

**今日动手任务（20 分钟）**：完成上面 4 条中的至少 2 条，跑一遍 `pytest 推理引擎/tests/` 确认没改坏行为。

---

## 5. 自测题

**Q1（概念）**：`to_torch("cuda")` 传字符串会发生什么？`Enum` 相比"用字符串常量"的核心优势是什么？

**Q2（应用）**：下面 dataclass 有什么 bug？怎么修？

```python
@dataclass
class Request:
    tokens: list = []
```

**Q3（思考）**：为什么 `frozen=True` 的 dataclass 适合当"配置对象"和"缓存 key"？结合第 01 篇的不可变对象知识回答。

<details>
<summary>点击展开答案</summary>

**A1**：`to_torch("cuda")` 里 `"cuda" is Device.CUDA` 为 False，会返回 `"cpu"`——**静默出错**（本该是 cuda 却给了 cpu）。这就是魔法字符串的危险：错误不报错，而是"默默地走错分支"。`Enum` 的优势：① 拼写错立刻 `AttributeError`；② 有限取值、IDE 能补全；③ 每个成员是单例，可用 `is` 比较；④ 可携带值/方法。

**A2**：`tokens: list = []` 是**可变默认值**，所有 `Request()` 实例共享同一个 list，改一个污染所有。修复：`tokens: list = field(default_factory=list)`。

**A3**：`frozen=True` 使实例创建后不可变（类似第 01 篇的不可变对象）：① **配置对象**：不会被误改，传递时无需防御性拷贝，天然线程安全；② **缓存 key**：不可变 → 可哈希（`__hash__` 可用），且哈希值稳定（否则对象作为 dict key 时哈希变了会导致查找失效）；③ 值相等即同一状态，便于比较和去重。
</details>

---

## 6. 延伸阅读

- [dataclasses 官方文档](https://docs.python.org/3/library/dataclasses.html) —— `field`/`frozen`/`__post_init__` 全细节。
- [enum 官方文档](https://docs.python.org/3/library/enum.html) —— `auto()`/`IntEnum`/`Flag`。
- [ABC 与抽象基类](https://docs.python.org/3/library/abc.html)。
- [PEP 557 — dataclass](https://peps.python.org/pep-0557/) 的设计动机。

---

> **下一篇 → `05_现代语法糖.md`**：`match` 结构化匹配、海象运算符、解包、f-string 调试语法——这些 3.10+ 的现代写法，让你的代码少写一半、更不易错。
