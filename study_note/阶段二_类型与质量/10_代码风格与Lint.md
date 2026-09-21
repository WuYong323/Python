# 10 · 代码风格与 Lint：让机器替你守住"整洁"

> **一句话定位**：代码风格不是审美，是"降低团队/未来的你阅读成本"。用 `ruff` + `pre-commit` 把格式化、检查自动化，你就永远不用再为空格、import 顺序、未用变量分心。
> **前置依赖**：07 类型
> **预计时长**：半天
> **对应你的项目**：`backend.py` 里 `x=x.float()`、`self.n_head=n_head` 无空格，`model.py` 的 import 顺序乱——本篇一键全修。

---

## 0. 先跑起来

先确认工具（你机器上 ruff 已装）：

```powershell
ruff --version        # 若没有：uv tool install ruff
```

```python
# 10_style_demo.py —— 故意写得"脏"，待会儿让 ruff 修
import sys,os

def  bad_function(x,y):
    result=x+y
    unused_var=42
    if result>10:
        return {"data":result , "ok":True}
    else:
        return None
```

**执行两条命令**：

```powershell
ruff format 10_style_demo.py     # ① 自动格式化（空格、换行、引号）
ruff check  10_style_demo.py     # ② 静态检查（未用变量、坏习惯）
```

`ruff format` 会把 `x+y` 变成 `x + y`、`{"data":result , "ok":True}` 变成 `{"data": result, "ok": True}`；`ruff check` 会报 `unused_var` 未使用、`import sys,os` 未使用/应分行。

**自动修复**：`ruff check --fix 10_style_demo.py` 会删掉未用的 import 和变量。

---

## 1. 一句话直觉

> **Lint = 代码的"体检医生"，Formatter = 代码的"理发师"。医生挑毛病（未用变量、危险写法、坏味道），理发师管造型（空格、换行、引号统一）。**

类比：你不需要手动记住"运算符两边要空格""import 要按字母序""多行要缩进 4 格"这些规矩——把规矩写进配置文件，**`ruff` 自动执行**。你只负责写逻辑，机器负责管整洁。这跟工业里"用机床保证公差，而不是靠师傅手感"是一个道理。

---

## 2. 核心原理

### 2.1 为什么选 ruff（而不是 black + flake8 + isort）

`ruff` 是 Astral 出的、用 **Rust** 写的极快工具，**一个工具同时干格式化 + 检查 + import 排序**：

| 工具 | 作用 | 速度 |
|------|------|------|
| black | 格式化 | 慢（Python 实现） |
| flake8 / isort | 检查 / import 排序 | 慢 |
| **ruff** | **格式化 + 检查 + import 排序 三合一** | **快 10~100 倍** |

**2025 年的共识：新项目直接用 `ruff`**，不再单独配 black/flake8/isort。你的项目也该如此。

### 2.2 `ruff format`：只管"长什么样"

它负责（不可配置项极少，这是刻意的——减少争论）：
- 运算符、逗号、冒号两侧空格：`x=x+1` → `x = x + 1`
- 每行宽度（默认 88 字符，超过自动换行）
- 引号统一（默认双引号）
- 尾部逗号、空行

```powershell
ruff format .                 # 格式化整个项目
ruff format --check .         # 只检查不修改（CI 里用）
```

### 2.3 `ruff check`：挑"坏味道"和"潜在 bug"

它按**规则集**检查，常用规则组：

| 规则组 | 抓什么 |
|--------|--------|
| `E`/`F` | 语法错误、未定义名、未用 import（flake8 基础） |
| `I` | import 顺序（isort） |
| `B` | 常见 bug（如可变默认参数、裸 except） |
| `UP` | 现代写法升级（如 `Optional[X]`→`X | None`） |
| `SIM` | 简化写法（如 `if x: return True` → `return x`） |
| `N` | 命名规范 |
| `S` | 安全（如 `torch.load` 无 `weights_only`） |

```powershell
ruff check .                 # 检查整个项目
ruff check --fix .           # 能自动修的自动修
```

> 注意：`S` 安全规则**不会默认开启**，需手动加。它正好能抓你 `torch.load` 缺 `weights_only` 的问题。

### 2.4 配置：写进 `pyproject.toml`

```toml
[tool.ruff]
line-length = 100                # 你习惯更长行可设 100

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM", "N", "S"]   # 开启的规则组
ignore = ["E501"]                # 忽略"行太长"（已由 format 处理）

[tool.ruff.format]
quote-style = "double"           # 双引号
```

> 这里注意：`[tool.ruff.lint]` 是 ruff 0.2+ 的新配置结构，旧教程写 `[tool.ruff]` 顶层。以你装的版本为准，`ruff --version` 确认后看对应文档。

### 2.5 `pre-commit`：把检查挂进 git

`pre-commit` 在**每次 `git commit` 前**自动跑检查，不合格就拦下提交：

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.0
    hooks:
      - id: ruff            # check
      - id: ruff-format     # format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.11.0
    hooks:
      - id: mypy
```

```powershell
uv tool install pre-commit
pre-commit install            # 安装成 git hook
pre-commit run --all-files    # 手动全量跑一次
```

从此，**不规范的代码根本进不了 git**，团队协作时风格天然统一。

---

## 3. 深入细节：PEP 8 核心要点速查

自动化工具负责机械部分，但有些**命名/结构**规矩工具管不了，得靠人：

| 项 | 规范 | 反例 |
|----|------|------|
| 模块/函数/变量 | `snake_case` | `getBatch`、`MyFunc` |
| 类 | `PascalCase` | `class blockAllocator` |
| 常量 | `UPPER_SNAKE` | `maxLayers = 128` |
| 私有属性 | `_single` 内部、`__double` 名称改写 | 滥用 `__` |
| 空行 | 顶层两行、类内一行 | 随意 |
| import | 标准库 → 第三方 → 本地，每组空行隔开 | `model.py` 里 `from pathlib import Path` 插在 torch 中间 |
| 每行长度 | ≤ 88（ruff 默认） | 一屏拖不到头 |

**你的 `model.py` 命名**其实已经不错（`CausalSelfAttention`、`_init_weights`、`n_embd` 都规范），主要是**空格和 import 顺序**问题，`ruff` 一键能修。

---

## 4. 回到你的项目

**今日动手任务（20 分钟，一次到位）**：

1. 在 `PythonProject1` 根目录建 `pyproject.toml`（若没有），加入上面的 `[tool.ruff]` 配置。
2. 跑 `ruff format .` 和 `ruff check --fix .`，把 `推理引擎/`、`cuda_ext` 的 Python 文件全部规范化。
3. 打开 `backend.py` 看 `x=x.float()` 是否变成 `x = x.float()`，`model.py` 的 import 是否被排序分组。
4. `git diff` 看改了什么（若有 git），确认是"纯风格"改动、没改逻辑。
5. （可选）装 pre-commit 并 `pre-commit install`。

**验收标准**：`ruff check .` 零告警（或只剩你明确 ignore 的）；`ruff format --check .` 通过。

> ⚠️ 注意：`ruff format` 会改动**所有** Python 文件（包括 `.venv` 里的吗？不会，ruff 默认跳过 `.venv`、`site-packages` 等）。格式化后**务必重跑你的 `pytest 推理引擎/tests/`**，确认行为没变——虽然 format 理论上不改语义，但这是工程纪律。

---

## 5. 自测题

**Q1（概念）**：`ruff format` 和 `ruff check` 分别管什么？为什么要把它们分开？

**Q2（应用）**：下面代码有哪些 ruff 会抓的问题？`ruff check --fix` 能自动修哪些、哪些要手动？

```python
import os,sys
def f(x):
    y=x+1
    if y==2: return True
    else: return False
```

**Q3（思考）**：为什么团队协作里"格式统一"要靠工具（ruff）而不是"大家自觉遵守规范"？工具的强制性和人工约定的本质区别是什么？

<details>
<summary>点击展开答案</summary>

**A1**：`ruff format` 管**格式**（空格、换行、引号、行宽），是"只改样式、不改语义"的确定性重排；`ruff check` 管**语义风险**（未用变量、可变默认参数、裸 except、import 顺序、安全）。分开是因为：格式问题可 100% 安全自动改，检查问题则有些需人工判断（如删未用变量要确认没副作用），分开能"先安全格式化、再谨慎检查"。

**A2**：`import os,sys`（应分行、且未使用，`ruff check --fix` 会删）；`f` 缺类型标注（严格模式下才报）；`y==2` 无空格（`ruff format` 修）；`if y==2: return True else: return False` 是坏味道（`SIM` 规则建议 `return y == 2`，`--fix` 可自动简化）；`else: return False` 冗余。自动修：删 import、加空格、简化布尔返回；手动：补类型标注、判断 `os/sys` 是否真需要。

**A3**：工具强制是**确定性、可执行、零遗漏**的——规则写成配置文件，每次提交自动校验，任何人无法"忘了"或"看漏"；人工约定靠自觉，会因疲劳、疏忽、水平差异而失效，且 code review 花大量时间争论空格。本质区别：**工具把规范从"文化"变成"机制"**，文化会衰减，机制能持久。这也是 CI/CD、pre-commit 存在的根本原因。
</details>

---

## 6. 延伸阅读

- [ruff 官方文档](https://docs.astral.sh/ruff/) —— 配置、规则、format。
- [pre-commit 官方文档](https://pre-commit.com/) —— git hook 管理。
- [PEP 8 — 风格指南](https://peps.python.org/pep-0008/) —— 权威风格规范。

---

> **阶段二完成 ✅** 你已经拥有：类型标注 + mypy 静态检查 + 异常/日志体系 + ruff/pre-commit 自动化。这是"工业代码"和"脚本"的分水岭。

> **下一篇 → `阶段三/11_GIL与并发模型.md`**：GIL 到底锁住了什么、线程/进程/协程怎么选、为什么你的推理 serving 必须用异步——进入性能和并发的核心地带。
