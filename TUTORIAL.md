# MetaE-former 项目接手教程

> 目标读者：**完全没有接触过深度学习**，但会写一点 Python、知道什么是矩阵的同学。
> 读完之后你应该能做到三件事：① 把项目跑通；② 说清楚每一行关键代码在干什么；③ 独立改出你自己的任务。
>
> 对应论文：Kuang et al., *Rapid inverse design of large-scale freeform metaoptics with the neighborhood-attention transformer*, PhotoniX 2026。
>
> 本仓库是**独立复现**，不是论文官方代码。所有与论文的差异在第 2.4 节集中列出，务必先读，否则你会把"复现精度差距"误判成"代码 bug"。

---

## 目录

- [0. 五分钟跑通（先跑，再理解）](#0-五分钟跑通先跑再理解)
- [1. 零基础概念速查](#1-零基础概念速查)
- [2. 这个项目在解决什么问题](#2-这个项目在解决什么问题)
- [3. 文件地图](#3-文件地图)
- [4. 形状账本（最重要的一张表）](#4-形状账本最重要的一张表)
- [5. 逐模块精读](#5-逐模块精读)
- [6. 二次开发手册](#6-二次开发手册)
- [7. 排错手册](#7-排错手册)
- [8. 术语表](#8-术语表)
- [9. 给你的第一个练习](#9-给你的第一个练习)
- [附：一页速查](#附一页速查)

---

## 0. 五分钟跑通（先跑，再理解）

### 0.1 用哪个 Python

**不要用系统默认的 `python`**（那是 WorkBuddy 自带的 3.13，没装 torch）。本项目固定使用：

```bash
PY="C:/Users/Fischer/AppData/Local/Microsoft/WindowsApps/python.exe"
```

实测版本：Python 3.14.2 / torch 2.11.0+cpu / numpy 2.4.1 / scipy 1.17.0 / matplotlib 3.10.8。

第一次用之前先验证：

```bash
"$PY" -c "import torch; print(torch.__version__)"
# 期望输出：2.11.0+cpu
```

如果报 `ModuleNotFoundError: No module named 'torch'`，说明你用错了解释器——换回上面那条绝对路径。

### 0.2 三条命令

在工作目录 `C:/Users/Fischer/Desktop/MetaE_former` 下：

```bash
# ① 跑一遍模型评估，确认权重能加载（实测 26 秒）
"$PY" evaluate.py --ckpt checkpoints/pretrain.pt --n 48

# ② 做一次涡旋光逆向设计（数分钟，产物在 outputs/vortex6/）
"$PY" design.py --task vortex --l 6 --focal 20e-6 --ckpt checkpoints/finetune2.pt --out outputs/vortex6

# ③ 打开 GUI（可视化操作，不用记命令）
"$PY" gui_app.py
```

**第 ① 步的实测输出**（你的结果应该长这样，差太多说明环境有问题）：

```
== Table 1 (per pattern family) ==
  family |    n |  amp MAE | angle MAE
  random |   48 |    0.123 |     0.171
  island |   48 |    0.096 |     0.077
  stripe |   48 |    0.134 |     0.087
 overall |  144 |    0.118 |     0.112
(paper Table 1, FDTD ground truth: 0.037-0.114 amp / 0.067-0.123 angle)
```

注意两个数：`overall amp 0.118 / angle 0.112`。它们和论文 Table 1 的区间（0.037–0.114 / 0.067–0.123）**基本重叠但偏上限**——这就是 0.8M 小模型 + BPM 标签的合理水平（第 2.4 节）。如果你看到 `amp > 0.3`，那是权重或数据坏了，不是"复现精度不够"。

第 ② 步成功后，`outputs/vortex6/` 里会有：

| 文件 | 内容 |
|---|---|
| `design_result.png` | 六联图：折射率分布、输出相位、输出振幅、焦面光斑、FWHM 剖面、x-z 传播 |
| `design.npz` | 全部原始数组（`ri` / `field` / `focal_int` / `xz` / `meta`） |
| `summary.json` | NA、FWHM、效率、耗时 |

### 0.3 无界面自检（改代码后必跑）

```bash
"$PY" gui_app.py --selftest
```

它会依次跑通「物理直接设计 → 训练 12 步 → 代理模型设计 → 评估」四条链路，最后打印 `[selftest] OK`。
**这是你改完代码后最快的回归测试**，比手工点 GUI 可靠。

---

## 1. 零基础概念速查

先建立直觉。以下十个词你会在代码里反复遇到。

### 1.1 张量（Tensor）

就是**多维数组**。`numpy.ndarray` 的亲戚，区别是张量能记录"我是怎么算出来的"，从而支持自动求导。

代码里最常见的三种形状：

```python
(B, 25, 25)       # 一批 B 个 25×25 的折射率图（B = batch size，一次处理几个样本）
(B, 2, 100, 100)  # 一批 B 个 100×100 的复光场，2 = [实部, 虚部]
(B, C, H, W)      # PyTorch 卷积的标准布局：批、通道、高、宽
```

**你必须养成看 `shape` 的习惯。** 本项目 90% 的报错都是形状对不上。

### 1.2 神经网络 / 参数

神经网络就是一个**带参数的函数**。你喂进去一个输入，它吐出一个输出；"参数"是函数里那些可调的数值（本项目的 `fast` 预设约 80 万个参数）。

`metae/model.py` 里 `MetaEFormer` 干的事，本质是一句话：

```
输入 折射率图 (25×25)  ──网络──▶  输出 出射复光场 (100×100)
```

### 1.3 训练（Training）

神经网络的参数一开始是随机的，输出全是噪声。训练就是**反复修正参数**，让它越来越接近正确答案：

```
猜一个输出  →  和标准答案比，算误差(loss)  →  按误差反向调整参数  →  重复几万次
```

这个"反向调整"的算法叫**反向传播 + 梯度下降**。你不需要会推公式，只需要知道：

- **loss 变小 = 模型在变好**；
- 调参的步长叫**学习率（lr）**，太大震荡、太小爬得慢。

### 1.4 损失函数（Loss）

衡量"猜得有多错"的标尺。本项目用两个（论文 Eq.1–3）：

```
L_total = L_幅度 + α · L_相位
L_幅度   = mean(|E_pred - E_true|)                 ← 复场逐元素绝对误差
L_相位   = mean((1 - cos(θ_pred - θ_true)) / 2)    ← 相位差，范围 [0,1]
```

**为什么要用 `1-cos` 而不是直接相减？** 因为相位是周期量：359° 和 1° 只差 2°，但直接相减会得到 358° 的巨大误差。`cos` 天生对 2π 周期，绕过了这个问题。这是光学项目里极其常见的坑。

### 1.5 代理模型（Surrogate Model）

这是本项目的核心思想，务必理解。

真实的物理仿真（FDTD / RCWA）非常慢：算一个 5µm 的小块要几秒到几分钟。如果逆向设计要迭代 200 次 × 16 个块，就要跑几小时。

于是我们**先训练一个神经网络去模仿物理仿真器**：

```
结构 → [真仿真器] → 光场      （慢，但准，用来造训练数据）
结构 → [神经网络]  → 光场      （快 1000 倍，但有小误差）
```

训练好之后，把神经网络当成"物理的替身（代理）"塞进优化循环，就能秒级完成设计。**这就是论文标题里 "Rapid" 的来源。**

代价是：代理模型有误差，设计结果会打折扣。这一点在第 2.4 节和 5.7 节会反复出现。

### 1.6 逆向设计（Inverse Design）

正常物理是"给结构，算光场"（正向）。工程上我们要的是反过来的：

```
我想要一个能产生涡旋光的超表面  →  请问每个位置该放多高的柱子？
```

这就是逆向设计。传统做法靠查表（先造一个相位-结构对照表，再按需要的相位去查），只能处理规则结构；**逆向设计**是直接把"目标光场"写成损失函数，用梯度下降反求结构，因此能做任意形状（freeform）。

### 1.7 可微（Differentiable）

如果一个计算过程从头到尾都能求导，它就能被塞进梯度下降里当损失函数。

- `torch.fft.fft2`、`torch.exp`、`torch.abs` → 可微 ✅
- 硬阈值 `n > 1.5`、`argmax`、`np.load` → 不可微 ❌

本项目所有物理计算都用 PyTorch 写（而不是 numpy），就是为了保证可微。**这是你改代码时最容易破坏的东西。**

### 1.8 批处理（Batch）与并行 patch

`B` 个样本同时算，GPU/CPU 的向量化能一次算完，比 for 循环快得多。

本项目的 patch 优化把 16 个 5µm 小块拼成一个 batch 一起优化（`design.py:139`），这既加快了速度，也是论文 Fig.5 的做法。

### 1.9 二值化惩罚（Binarization Penalty）

真实的超表面柱子只有两种状态：**有材料**（n=3.526）或**没材料**（n=1.0）。但梯度下降天生喜欢连续值（n=1.7、n=2.3 这种"半个柱子"没法加工）。

解决办法：在 loss 里加一项，把连续的折射率往两端推：

```python
n_norm = tanh(v)                 # 参数 v 映射到 [-1, 1]
penalty = mean(1 - n_norm**2)    # 在 ±1 处为 0，在 0 处为 1 → 越接近两端惩罚越小
loss = L_phase + β · penalty     # β 随迭代从 0 线性升到 5
```

β 从 0 慢慢升到 5，叫 **ramp**（斜坡式升温）。一开始只优化相位（结构自由），后期逐渐逼它变成黑白，避免一上来就被二值约束卡死。

### 1.10 检查点（Checkpoint）

`.pt` 文件，本质是一个字典：`{"model": 参数字典, "preset": "fast", "step": 1600, ...}`。

本仓库现有四个：

| 文件 | preset | step | 用途 |
|---|---|---|---|
| `pretrain.pt` | fast | 1600 | 预训练基线 |
| `finetune.pt` | fast | 1600 | 任务微调版（涡旋） |
| `finetune2.pt` | fast | 2800 | **当前最好用的，设计时优先用这个** |
| `selftest.pt` | tiny | 12 | 自检用的玩具权重，别拿它做设计 |

加载逻辑在 `metae/utils.py:23` 的 `load_model()`：它会读 `ck["preset"]` 来决定建多大模型，所以**权重和 preset 必须配对**，否则报 `size mismatch`。

---

## 2. 这个项目在解决什么问题

### 2.1 物理装置

想象一块 5µm × 5µm 的方形玻璃片，上面立着 25×25 根硅柱，柱子间距 200nm，高度统一 800nm：

```
        入射平面波 (λ=1064nm, x 偏振)
              ↓  ↓  ↓  ↓
        ┌─────────────────────┐
        │ █ █ █ ░ █ ░ █ █ █  │  ← 每根柱子的"粗细"独立可调
        │ █ ░ ░ █ █ ░ ░ █ ░  │     → 等效折射率 n ∈ [1.0, 3.526]
        │ ...                 │
        └─────────────────────┘  高度 800nm
              ↓
        出射复光场 (在柱顶上方 z=λ 处，100×100 采样点，50nm/px)
```

光穿过高折射率的柱子会"走得更慢"，累积相位 φ ≈ k₀·(n−1)·h。所以**调柱子的等效折射率 = 调局部相位**。把 25×25 个相位拼起来，就能塑造出射波前——这就是超表面（metasurface）。

这张 25×25 的折射率图，就是神经网络要学、要设计的**唯一自由变量**。整篇论文的输入输出都是它。

### 2.2 论文的四步流程

```
① 造数据          自由形状超原子（4 类，Fig.2）→ 真仿真器打标签得到光场
        ↓
② 预训练          MetaE-former 学 "结构 → 光场" 的映射（Fig.3）
        ↓
③ 任务微调        用预训练模型先设计一批任务结构，再用真仿真器重新打标签，
                  用差分学习率微调（Fig.4）——即论文说的 self-supervised data engine
        ↓
④ 任意任务设计    目标相位切 5µm 块 → 逐块梯度反演 → 拼接 → 验证（Eq.4-7）
```

对应到本仓库的四个脚本：

| 论文步骤 | 脚本 | 产物 |
|---|---|---|
| ① + ② | `pretrain.py` | `checkpoints/pretrain.pt` |
| ③ | `finetune.py` | `checkpoints/finetune.pt` |
| ④ | `design.py` | `outputs/<任务>/design_result.png` |
| 性能表 | `evaluate.py` | Table 1 风格指标 |

### 2.3 为什么必须切成小块

1mm 口径需要 57121 个 patch。网络只吃 25×25，所以论文规定：**每 5µm 一块，块与块之间重叠 16%（84% 切割比）**，各块独立设计后再拼接。

84% 的来历：块的场网格是 100×100 px，但只取中心 84×84 px（4.2µm）参与拼接。重叠区保证拼接处相位连续。公式在 `design.py:96`：

```python
ext   = (n_p - 1) * CROP + PITCH_PX   # 扩展画布 = (n_p-1)*84 + 100
valid = n_p * CROP                     # 有效口径 = n_p * 84
```

### 2.4 本复现与论文的关键差异（**必读**）

| 要素 | 论文 | 本仓库 | 后果 |
|---|---|---|---|
| 真值求解器 | RCWA / FDTD（全波，含反射、多重散射） | **split-step BPM 近似**（`metae/solver.py`） | 绝对数值不能与论文对标；反射被忽略 |
| 数据量 | 26 万样本，20 epoch | 流式生成，1600 step（约 3.8 万样本） | 模型欠拟合 |
| 网络规模 | 16 块 NhA，embed 256，16 heads（约 40M 参数） | `fast`：embed 48，7 块（约 0.8M 参数） | 表达能力差一档 |
| 设计口径 | 1mm，57121 patch | 4×4 或 16×1 patch（16.8µm） | 拼接累积误差更明显 |
| 设计精度 | 论文 Fig.6 光斑干净 | 涡旋环居中但环宽偏大；艾里点阵不完美 | **这是模型容量 + 求解器差异，不是 bug** |

一句诚实的话：**这套代码复现的是"方法链路"，不是"论文数值"。** 如果你要冲论文级精度，路线是换更大的 preset（`--preset paper`）+ 用真 FDTD 打标签，两者都需要大量算力。第 6.5 节给了具体做法。

---

## 3. 文件地图

### 3.1 库代码（`metae/`，共 9 个文件，抄一遍就能懂）

| 文件 | 行数 | 干什么 | 你的改动频率 |
|---|---|---|---|
| `solver.py` | 130 | **物理仿真器**：split-step BPM + 角谱传播 | 想换真 FDTD 就改这里 ⭐ |
| `data.py` | 135 | 四类自由形状超原子采样器 | 想喂自己的数据就改这里 ⭐ |
| `model.py` | 165 | MetaE-former 网络（NAT U-Net） | 想加大模型改 `PRESETS` |
| `losses.py` | 53 | Eq.1–3 的两个损失项 | 想换优化目标改这里 |
| `metrics.py` | 50 | 归一化 MAE、幅度/角度 MAE | 加指标 |
| `tasks.py` | 66 | 目标相位生成：透镜/涡旋/艾里/自定义 | **加新任务改这里** ⭐ |
| `pipeline.py` | 428 | 可复用高层函数，GUI 与 CLI 共用 | 核心逻辑都在 `design_from_phase` |
| `utils.py` | 40 | 归一化、加载 checkpoint | 少动 |
| `__init__.py` | 25 | 包导出 | 少动 |

### 3.2 入口脚本（根目录）

| 脚本 | 作用 | 什么时候用 |
|---|---|---|
| `pretrain.py` | 预训练 CLI | 重训基线模型 |
| `finetune.py` | 采集任务数据 + 差分学习率微调 | 换任务（如艾里）后重训 |
| `evaluate.py` | 三类样本指标评估 | 确认权重质量 |
| `design.py` | **主设计入口**（论文 Eq.4-7） | 日常出结果 ⭐ |
| `generate.py` | 设计 + 一键出图（GUI 后端也用） | 出交付图 |
| `gui_app.py` | tkinter 图形界面（含 `--selftest`） | 不想敲命令 |
| `direct_design.py` | **物理直接设计**（全口径可微 BPM，走代理） | 追求精度、能等几十分钟 ⭐ |
| `design_fig6.py` / `make_fig6.py` | 复现论文 Fig.6 的批量设计与出图 | 写论文/汇报 |
| `refine_display.py` | 对已存结构做高分辨前向验证（50nm 全网格 + 角谱） | 怀疑代理模型骗你时 |
| `bootstrap_finetune2.py` | 生成 `finetune2.pt` 的引导脚本 | 想复现当前最好权重 |

### 3.3 数据产物

```
checkpoints/          权重（.pt）+ 训练历史（pretrain_history.json）+ 微调数据缓存（.npz）
outputs/              所有实验产物，按任务名分目录
  fig6_*/             Fig.6 复现的中间结构（design.npz）
  fig6_fields_mt.npz  高分辨前向缓存
  fig6_replication.png 最终成图
  paper/              从 PDF 提取的原始位图（标定用）
```

### 3.4 推荐阅读顺序

不要从 `model.py` 开始（那是最抽象的）。按这个顺序读，每一步都能立刻看到输出：

```
1. metae/tasks.py        (66 行)  → 目标相位长什么样，先画出来看
2. metae/solver.py       (130 行) → 物理怎么算的，跑一个样本看输入输出
3. metae/data.py         (135 行) → 数据从哪来
4. metae/model.py        (165 行) → 网络怎么搭的
5. metae/losses.py       (53 行)  → 怎么算错
6. pretrain.py           (154 行) → 把上面拼成训练循环
7. design.py             (292 行) → 把训练好的模型当工具用
8. metae/pipeline.py     (428 行) → 最后再看，它是上面所有的封装
```

---

## 4. 形状账本（最重要的一张表）

深度学习项目 90% 的 bug 出自形状/单位。把这张表贴在显示器上。

| 阶段 | 变量 | 形状 | 数值范围 | 像素尺寸 | 代码位置 |
|---|---|---|---|---|---|
| 结构（物理量） | `n_map` | `(B, 25, 25)` | 1.0 ~ 3.526 | 200 nm/px（总 5µm） | `data.py:124` |
| 网络输入 | `normalize_input(n)` | `(B, 25, 25)` | −1 ~ +1 | 同上 | `utils.py:14` |
| 网络输出 | `pred` | `(B, 2, 100, 100)` | 通道0=实部，通道1=虚部 | 50 nm/px（总 5µm） | `model.py:136` |
| 复光场 | `complex(pred[:,0], pred[:,1])` | `(B, 100, 100)` | 幅度已按 `max|E|` 归一到 ≤1 | 同上 | `design.py:188` |
| 损失窗（84% 切割） | `pc[:, 8:92, 8:92]` | `(B, 84, 84)` | — | 4.2 µm | `design.py:161` |
| 目标相位 | `tgt` | `(n_p², 84, 84)` | rad（可任意 wrap） | 4.2 µm 窗口 | `design.py:144` |
| 拼接后全画布 | `field` | `(ext, ext)` | complex64 | ext=(n_p−1)·84+100 | `design.py:192` |
| 有效口径 | `field_c` | `(n_p·84, n_p·84)` | complex64 | n_p·4.2 µm | `design.py:198` |
| 输出结构 | `ri` | `(n_p·25, n_p·25)` | 1.0 ~ 3.526 | 200 nm/px | `design.py:193` |
| 焦面强度 | `inten` | `(n_p·84·PAD)²` | W/a.u. | 50 nm/px | `design.py:214` |

### 4.1 三个必须记住的常数

```python
PITCH_PX = 100    # 5µm patch 的场网格边长（50nm × 100 = 5µm）
CROP     = 84     # 84% 切割 → 只取中心 84 px = 4.2 µm 参与拼接
DX       = 50e-9  # 场采样步长 50 nm
```

以及一个容易忘的换算：**折射率图的 1 个像素 = 200nm，场图的 1 个像素 = 50nm**。所以 `25 px`（结构）↔ `100 px`（场），比例恒为 **4:1**。这个 4 在 `model.py` 里体现为两次上采样（25→50→100）。

### 4.2 一个必跑的验证脚本

新建 `check_shapes.py` 放到根目录：

```python
import torch
from metae.data import FreeformAtomSampler
from metae.solver import SplitStepBPM
from metae.utils import normalize_input
from metae.model import build_model

sampler = FreeformAtomSampler(generator=torch.Generator().manual_seed(0))
solver = SplitStepBPM()
model = build_model("fast")

nm = sampler.sample(2)
print("结构      ", tuple(nm.shape), nm.min().item(), nm.max().item())   # (2,25,25) 1.0 3.526
fld = solver(nm)
print("真值光场  ", tuple(fld.shape), fld.abs().max().item())             # (2,2,100,100) ≈0.95
pred = model(normalize_input(nm))
print("代理光场  ", tuple(pred.shape))                                    # (2,2,100,100)
print("参数量    ", sum(p.numel() for p in model.parameters())/1e6, "M")  # 0.82 M
```

实测输出（可直接对照）：

```
结构       (2, 25, 25) 1.0 3.526
真值光场   (2, 2, 100, 100) 0.9532
代理光场   (2, 2, 100, 100)
参数量     0.8213 M
```

> 注意 `fld.abs().max()` 打印的是 **0.95 而不是 1.0**，这不是 bug：`fld` 是 `(B,2,H,W)` 的实/虚 **双通道实数**张量，`abs()` 逐通道取绝对值。物理上 `max|E|=1` 指的是复模 `sqrt(re²+im²)`，单看实部或虚部当然 ≤1。要验证归一化，写 `torch.hypot(fld[:,0], fld[:,1]).max()`，它才等于 1.0。

跑通它，你就建立了"数据流"的直觉。

---

## 5. 逐模块精读

### 5.1 `metae/solver.py` —— 物理裁判

这是整个项目**最需要你亲手改**的文件。它不是神经网络，是物理。

#### 它在算什么

多切片 BPM（beam propagation method）：把 800nm 高的柱子沿 z 切成 8 片，每片做两件事——

1. **乘相位**：光在折射率 n 的介质里走 dz，累积相位 `exp(i·k₀·(n−1)·dz)`；
2. **衍射传播**：把这层场往后传 dz，用角谱法（FFT 到频域乘 `exp(i·kz·dz)` 再逆变换）。

8 片交替做完，再自由传播 λ=1064nm 到柱顶上方，得到 100×100 的复场。核心就 12 行（`solver.py:94-102`）：

```python
phase_per_slice = 1j * self.k0 * (n - 1.0) * self.dz
E = torch.ones_like(n, dtype=torch.complex64)      # 入射平面波，振幅 1
for _ in range(self.n_slices):                      # 8 片
    E = E * torch.exp(phase_per_slice)              # ① 相位累积
    E = self._propagate(E, self.dz)                 # ② 衍射传播
E = self._propagate(E, self.wl)                     # 柱顶 → z=λ
```

最后除以 `max|E|` 归一化（`solver.py:103`）。**这一步极其重要**：所有下游损失、指标都建立在"幅度 ≤ 1"的约定上。如果你改归一化方式，`losses.py` 的 amplitude 项含义就变了。

#### 三个值得注意的设计

- `edge_sigma_px=0.6`（`solver.py:39`）：柱子边缘是硬跳变，直接上 FFT 会有振铃。用 5×5 高斯核做亚像素平滑，物理上相当于"真实加工也有圆角"。
- `kz` 用复平方根（`solver.py:55`）：保证倏逝波（kx²+ky²>k₀²）自动衰减而不是爆炸。
- `angular_spectrum_propagate()`（`solver.py:116`）：独立的传播函数，用于把拼接后的全口径场传到焦面。**它和 BPM 内部的 `_propagate` 是同一套数学，但签名不同**（前者接受任意 dx 与 n_bg）。

#### 换成真 FDTD 时怎么改

唯一要求：**输入 `(B,25,25)` 折射率，输出 `(B,2,100,100)` 复场**，且保持可微。

如果 FDTD 不可微（通常如此），有两条路：

- **离线打标签**（推荐）：先用 FDTD 预生成 N 万组 `(结构, 场)` 存成 `.npz`，然后把 `data.py` 改成从文件读，训练时不再调用 solver。
- **伴随法**：用 FDTD 的 adjoint 梯度手工写反向传播，工作量大但可在优化循环内直接反演。

改动点只有 `data.py` 的采样器返回结构、以及训练脚本里 `fld = solver(nm)` 这一行换成 `fld = load_labeled(idx)`。

---

### 5.2 `metae/data.py` —— 数据从哪来

论文造 26 万样本要下载几百 GB。本仓库**流式生成**：每次 `sample(batch)` 现场随机画一批，永不落盘。

四个族（`data.py:63-121`）：

| 族 | 占比 | 生成方式 | 长什么样 |
|---|---|---|---|
| `random_pattern` | 72% | 高斯模糊随机噪声，逐样本拉伸到 [0,1] | 云朵状斑块 |
| `island_pattern` | 12% | 3~8 个随机矩形叠加，折射率 ∈[1.33, 3.526] | 岛状团块 |
| `stripe_pattern` | 16% | `cos(2πf(cosτ·X + sinτ·Y) + φ)`，随机频率/角度/相位 | 斜条纹 |
| `task_pattern` | 微调阶段 | 由预训练模型反演得到的中间结构（见 `finetune.py:41`） | 任务相关 |

每族内部还有 **60% 连续 / 40% 二值** 的划分（`continuous_ratio=0.6`）。这个比例很重要：网络必须同时见过"光滑的"和"只有 1.0/3.526 两种值"的结构，否则设计阶段做二值化时会崩。

#### 怎么接自己的数据

继承并重写两个方法即可：

```python
class MySampler(FreeformAtomSampler):
    def sample(self, B, mixture="pretrain", probs=None):
        # 返回 (B, 25, 25)，值域 [1.0, 3.526]，dtype float32
        return my_structures[torch.randint(len(my_structures), (B,))]
```

注意：**必须是 torch 张量、必须在 CPU 上、必须落在 [1.0, 3.526]**。传 numpy 数组会静默变慢（没有报错，但会退化成逐样本循环）。

---

### 5.3 `metae/model.py` —— 网络结构

#### 一句话版本

U 形编码器-解码器，编码器用**邻域注意力**代替卷积，解码器用转置卷积上采样，最后输出实部/虚部两个通道。

```
25×25 ─stem─▶ 25×25 ─3×NABlock─▶ 25×25 (dim=48)
                                     │ down1 (stride 2)
                                     ▼
                                  13×13 ─2×NABlock─▶ 13×13 (dim=96)
                                     │ down2
                                     ▼
                                   7×7  ─2×NABlock─▶ 7×7
                                     │ up1 + skip
                                     ▼
                                  13×13 ─fuse─▶ 13×13
                                     │ up2 + skip
                                     ▼
                                  25×25 ─fuse─▶ 25×25
                                     │ up3 (×2)
                                     ▼
                                  50×50
                                     │ up4 (×2)
                                     ▼
                                 100×100 ─head─▶ 2×100×100
```

#### 什么是"邻域注意力"（NhA）

普通自注意力让每个像素看**全图**，代价是 O(N²)：100×100 就是 1 亿次交互，CPU 跑不动。

邻域注意力（`model.py:21`）限制每个像素只看**周围 5×5 的窗口**，代价降到 O(N·k²)。对超表面这个任务天然合理——光在柱子之间的耦合本来就是短程的（邻近几根柱子才互相影响），这正是论文标题 "neighborhood-attention" 的物理动机。

实现上（`model.py:40-59`）：把 5×5 邻域用 `F.unfold` 展开成 `k²=25` 个偏移，然后对每个像素做 25 路 softmax 加权。还加了一个可学习的**相对位置偏置** `rpb`（`model.py:33`），让网络能学到"左边邻居和右边邻居权重不同"。

#### 三个 preset

```python
PRESETS = {
    "paper": dict(dim=256, heads=16, blocks=(8,4,4), mlp_ratio=4.0),  # 论文规模，~40M 参数，CPU 跑不动
    "fast":  dict(dim=48,  heads=4,  blocks=(3,2,2), mlp_ratio=2.0),  # 本仓库默认，~0.8M
    "tiny":  dict(dim=32,  heads=4,  blocks=(2,1,1), mlp_ratio=2.0),  # 冒烟测试用
}
```

改模型规模的唯一入口就是这张表（`model.py:155`）。如果你加自己的 preset，**记得同步重新训练**——`load_model` 会按 ckpt 里记的 preset 建网络，preset 不匹配就加载失败。

#### 输出为什么是 2 通道而不是复数

PyTorch 的复数支持仍有坑（部分算子无梯度、`view` 不支持）。拆成 `[实部, 虚部]` 两个实数通道最稳，需要复数时再 `torch.complex(pred[:,0], pred[:,1])` 拼回去（`losses.py:29`）。

---

### 5.4 `metae/losses.py` 与 `metrics.py`

`losses.py` 只有 53 行，但有个值得注意的选择（`losses.py:34`）：

```python
la = (pred - target).abs().mean()                     # L1，不是 L2
lt = (1 - cos(θ_pred - θ_target)).mean() / 2          # 周期相位
total = la + alpha * lt
```

- **为什么用 L1 而不是 L2**：L2 对大误差惩罚过重，容易让网络只盯着少数几个最亮的点；L1 更均衡。
- **`alpha`**：论文推荐 `4√π ≈ 7.09`（两项量级配平）。本仓库默认 1.0，原因是 `lt` 已经归一到 [0,1]，与 `la`（约 0.1）量级相当，不需要再放大。
- **`amp_weighted_phase`** 选项（`losses.py:43`）：按 `|E|` 加权，让暗区相位不那么重要。物理上更合理（暗处相位本来就是噪声），但默认关闭以对齐论文。

`metrics.py` 里的 `normalized_mae` 是论文 Table 1 的口径（`metrics.py:13`）：

```python
scale = max|E_true|                                        # 每个样本的峰值幅度
d     = mean|Re E_pred - Re E_true|,  mean|Im E_pred - Im E_true|   # 实部、虚部分别求
norm  = 0.5 * (d_real + d_imag) / scale                    # 两通道平均后归一
```

注意三点：① 是**实部虚部分别**算误差再平均，不是直接对复数取模；② 分母是该样本的 `max|E|`，不是总和；③ 结果是一个无量纲的相对误差，论文里约 0.11。

对比时统一用这个，不要自己算 `mean absolute difference`——后者会被暗区和亮区同等加权，数字好看但没意义。

---

### 5.5 训练：`pretrain.py` 与 `finetune.py`

#### 预训练循环（`pretrain.py:88-148`）

去掉日志后只剩 20 行，这是所有深度学习训练循环的模板：

```python
for step in range(1, steps + 1):
    lr = warmup_then_cosine(step)          # ① 调学习率
    nm  = sampler.sample(batch)            # ② 造数据（现场随机生成）
    fld = solver(nm)                       # ③ 真仿真器打标签（no_grad，因为标签不需要梯度）
    pred = model(normalize_input(nm))      # ④ 网络预测
    loss = metaformer_loss(pred, fld)      # ⑤ 算误差
    opt.zero_grad(); loss.backward(); opt.step()   # ⑥ 反向传播 + 更新参数
```

**这六步是所有训练脚本的骨架。** 你以后看任何 PyTorch 代码，都在找这六步。

几个值得解释的细节：

- **warmup + cosine**（`pretrain.py:90-95`）：前 200 步学习率从 0 线性爬升（避免一开始就把随机初始化的参数带飞），之后余弦退火到接近 0（后期精细收敛）。
- **`clip_grad_norm_(..., 1.0)`**（`pretrain.py:110`）：梯度裁剪。偶尔某一步梯度会异常大，裁掉防止参数一步跳飞。这是训练不稳定的常见保险。
- **只在 val 变好时保存**（`pretrain.py:139`）：`score < best` 才写盘，所以 `pretrain.pt` 是"验证集最优"，不是"最后一步"。

#### 微调为什么要"差分学习率"

论文的观察：预训练学到的**浅层**（提取局部结构特征）和**深层**（整合全局）需要不同的调整幅度。实现就是一个 5 组的 `build_param_groups`（`finetune.py:88`）：

| 系数 | 模块 | 直观解释 |
|---|---|---|
| 0.5a | `up1..up4`, `fuse1/2`, `cpe4/5`, `head` | 上采样层最敏感，几乎不动 |
| 5a | `stem`, `cpe1`, `s1` | 浅层，小幅调整 |
| 8a | `cpe2`, `s2` | 中深层 |
| 10a | `cpe3`, `s3` | 最深，调整幅度最大 |
| 15a | `down1`, `down2` | 下采样层最快（要适应新任务的输入分布） |

`a = 5e-5`（`finetune.py:116` 的 `--lr-base`）。注意 `AdamW` 接收的是 param group 列表，每组自带 `lr`，所以**不能再用单一 lr 的 scheduler**，否则会把差分覆盖掉。

#### 自监督数据引擎（`finetune.py:41` 的 `collect_task_data`）

这是论文 Fig.2d 的精髓，也是最容易被忽略的一步：

```
用预训练模型反演涡旋任务 → 保存每次迭代的中间 RI 快照（不是最终结果！）
                        → 用真求解器给这些快照打标签
                        → 得到"任务相关的训练集"
                        → 用它微调
```

**为什么要中间快照而不是最终结果**：最终结果是二值化的漂亮结构，分布太窄；中间快照包含从随机到收敛的全过程，分布更接近设计时的真实状态。论文称之为 "archive every intermediate RI profile"。

---

### 5.6 设计：`design.py` 与 `pipeline.design_from_phase`

这是你日常最常用的部分，也是论文 Eq.4-7 的落地。

#### 完整流程（`design.py:63` 的 `main`）

```
1. 加载 checkpoint              load_model()
2. 算几何                       ext = (n_p-1)*84+100,  valid = n_p*84,  NA = sin(atan(D/2F))
3. 生成全局目标相位             build_target_phase(task, n_pix=ext, ...)
4. 切成 patch 目标             tgt[i,j] = 目标相位[i*84+8 : i*84+92, ...]
5. 优化所有 patch 的 RI         batch 内并行，Adam，iters=250
6. 拼接                         field[块位置] = 预测场；ri[块位置] = 结构
7. 传到焦面                     角谱传播 z = focal − λ
8. 算指标                       FWHM、效率、x-z 扫描
9. 出图/存盘
```

#### 第 5 步的核心循环（`design.py:154-176`）

```python
v = torch.randn(n_p*n_p, 25, 25) * 0.1     # 优化变量（无约束实数值）
v.requires_grad_(True)                      # 声明"这个要被求导"
opt = torch.optim.Adam([v], lr=0.1)

for it in range(1, iters+1):
    beta = ramp(it)                                    # 0 → 5，二值化权重
    n_norm = torch.tanh(v)                             # 映射到 (-1, 1)
    n_abs  = n1 + (n2-n1) * (n_norm+1)/2               # 映射到 (1.0, 3.526)
    pred   = model(normalize_input(n_abs))             # ← 代理模型当物理用
    pc     = complex(pred[:,0], pred[:,1])[:, 8:92, 8:92]
    l_phase  = ((1 - cos(angle(pc) - tgt)) / 2).mean() # 相位误差
    penalty  = (1 - n_norm**2).mean()                  # Eq.4 二值惩罚
    loss     = l_phase + beta * penalty
    opt.zero_grad(); loss.backward(); opt.step()
```

**为什么要 `tanh` 参数化**：直接优化 `n` 需要额外加 clip 约束（否则会跑到 0 或 5）。`tanh` 天生把值域卡在 (−1,1)，物理映射后自动落在 [n1,n2] 内，不需要任何约束代码。这是逆向设计里的标准技巧。

#### 关于"初始值"的一个大坑

默认 `v = randn * 0.1`，映射后 n ≈ 2.26（即 `[1.0, 3.526]` 区间的**中点**，实测 `v=0.1 → n=2.389`）。对透镜这种平滑目标，优化器能从随机起点爬出来；但对**相位跳变剧烈**的目标（艾里、高阶涡旋），随机起点会让优化器卡在鞍点——实测相位损失停在 0.46 再也不降，且迭代第 1 步的相位损失恒为 **0.50**（`1-cos` 在随机相位下的期望值，可作为"优化器完全没学到东西"的诊断阈值）。

解法是**全息初始化**（`pipeline.py:190-200`）：

```python
# 由目标相位直接反解折射率：φ = k0·(n−1)·h  ⟹  n = 1 + wrap(φ)/(k0·h)
n0 = 1.0 + torch.remainder(t25, 2*pi) / (k0 * 800e-9)
n0 = n0.clamp(n1, n2)
v  = torch.atanh((n0-n1)/(n2-n1)*2 - 1)   # 反解回 tanh 参数空间
```

调用方式：`design_from_phase(..., init="phase")`。**遇到"损失不降"先试这个**，往往一步到位。

注意 `wrap(φ)` 必须取 `remainder(φ, 2π)` 而不是 `φ % 2π`（负数处理不同），且 `n0` 要 clamp——因为 800nm 高的硅柱最大相位约 12 rad，超出 2π 的目标相位物理上做不到，必须裁掉。

#### 4×4 patch 的拼接细节

```python
for i in range(n_p):
    for j in range(n_p):
        field[i*84 : i*84+100, j*84 : j*84+100] = pc[k]   # 场：100px 全贴（含重叠）
        ri[i*25 : (i+1)*25,  j*25 : (j+1)*25]  = n_abs[k] # 结构：25px 不重叠瓷砖
```

**注意两者的区别**：场是按 100px 整块贴上去的（重叠区后贴的覆盖先贴的），结构是按 25px 无重叠平铺。这看起来不一致，但物理上是对的——5µm 的 patch 确实无缝拼接，而场图的重叠区只是"用哪一块的预测"的选择问题，84% 切割保证了重叠区差异很小。

---

### 5.7 `direct_design.py` —— 物理直接设计（重要补充）

#### 它和 `design.py` 的根本区别

| | `design.py`（代理） | `direct_design.py`（直接） |
|---|---|---|
| 优化循环里用什么算光场 | 神经网络（快，有误差 ~1 rad） | 真 BPM（准，慢） |
| 口径 | 分 patch 后拼接 | 全口径一次反演，不切块 |
| 耗时 | 2~5 分钟 | 几十分钟 |
| 拼接误差 | 有（块边界相位不连续） | 无 |
| 论文对应 | 论文正统方法（Fig.5/6） | **论文没有对应章节** |
| 什么时候用 | 日常设计、需要快速迭代 | 验收、怀疑代理模型骗你时 |

#### 为什么它值得存在

代理模型必然有误差。实测在二值结构上，`fast` 预设的相位误差约 **0.9~1.6 rad**——这已经接近 π/2，足以让光斑散掉。用 `direct_design.py` 可以回答一个关键问题：**"如果物理是完美的，这套目标相位能出好看的光斑吗？"**

- 能 → 说明问题在代理模型（该加大模型）
- 不能 → 说明目标相位或传播参数本身有问题（该改物理设定）

这个"归因"能力比结果本身更有价值。

#### 用法

```bash
"$PY" direct_design.py outputs/direct_metalens metalens
"$PY" direct_design.py outputs/direct_vortex6 vortex 6
"$PY" direct_design.py outputs/direct_airy airy 10.0 --n-pixels 100
```

第二个参数是任务，第三个是 `extra`（涡旋=拓扑荷，艾里=立方强度 S）。

---

## 6. 二次开发手册

以下每个场景都给出**改动点 + 最小代码 + 验证方法**。

### 6.1 场景一：加一个新的目标光场（最常见的需求）

假设你要做"贝塞尔光束"。

**第 1 步**：在 `metae/tasks.py` 加生成函数（紧跟 `airy_phase` 之后）：

```python
def bessel_phase(X, Y, k_r=2*math.pi/2e-6):
    """零阶贝塞尔光束的圆锥相位：φ = -k_r · r"""
    return -k_r * torch.hypot(X, Y)
```

**第 2 步**：在 `build_target_phase` 的分支里注册（`tasks.py:41`）：

```python
elif kind == "bessel":
    ph = bessel_phase(X, Y, kw.get("k_r", 2*math.pi/2e-6))
```

**第 3 步**：把 `design.py` 的 `--task` 可选值加上它（`design.py:65`），并确保目标相位在 `main` 里被正确调用：

```python
ap.add_argument("--task", default="metalens",
                choices=["metalens", "vortex", "airy", "bessel", "custom"])
```

**第 4 步**：验证。先只画相位，不跑设计：

```python
import numpy as np, matplotlib.pyplot as plt
from metae.tasks import build_target_phase
ph = build_target_phase("bessel", n_pix=336, k_r=2*np.pi/2e-6).numpy()
plt.imshow(ph, cmap="twilight"); plt.colorbar(); plt.savefig("bessel_phase.png")
```

看相位图是否符合预期（同心环、中心对称），再跑设计。**永远先验证目标相位，再花 5 分钟等设计。**

⚠️ 注意：如果你的相位是"rad 绝对值"形式的全局量（比如 `k_r·r` 能到 100 rad），一定要检查是否超出硅柱能实现的范围（约 12 rad @ 800nm 高）。超出部分会被 clamp 掉，光斑直接塌。

### 6.2 场景二：换成真实 FDTD 数据

**最快路径**（离线打标签，不需要可微）：

1. 用 Lumerical / Meep 生成 N 组 `(结构, 复场)`，存成两个 `.npz`：
   ```python
   np.savez("fdtd_dataset.npz",
            n_maps=structures,   # (N, 25, 25) float32
            fields=fields)       # (N, 2, 100, 100) float32，[实部, 虚部]
   ```
2. 在 `metae/data.py` 加一个文件采样器：
   ```python
   class FileSampler:
       def __init__(self, path):
           d = np.load(path)
           self.n = torch.from_numpy(d["n_maps"])
           self.f = torch.from_numpy(d["fields"])
       def sample(self, B, **kw):
           idx = torch.randint(len(self.n), (B,))
           return self.n[idx], self.f[idx]     # 同时返回标签
   ```
3. 把 `pretrain.py` 的这几行改成用现成标签：
   ```python
   # 原： nm = sampler.sample(batch);  fld = solver(nm)
   nm, fld = sampler.sample(batch)
   ```

其余全部不变。这就是"求解器可替换"的设计价值。

### 6.3 场景三：加大模型（逼近论文）

改 `metae/model.py:155` 的 `PRESETS`：

```python
"mid": dict(dim=96, heads=8, blocks=(5,3,3), kernel=5, mlp_ratio=4.0),   # ~3M 参数
```

然后重训（`--preset mid`）。经验值：

| preset | 参数 | 每步耗时（24 线程 CPU） | 建议 steps |
|---|---|---|---|
| tiny | 0.2M | ~0.2 s | 50（只做冒烟） |
| fast | 0.8M | ~0.6 s | 1600~3000 |
| mid | 3M | ~2 s | 5000+ |
| paper | 40M | ~30 s | 20000+（实际需要 GPU） |

⚠️ **换 preset 后必须重训，不能拿旧权重加载**。ckpt 里记的 preset 会覆盖你的命令行参数（`utils.py:27`），这是有意的防错设计。

### 6.4 场景四：加一个新指标

在 `metae/metrics.py` 加函数，然后在 `pipeline.evaluate_model`（`pipeline.py:310`）的 `rows.append(...)` 里加一个键，CSV 导出会自动带上（因为它用 `DictWriter` + `fieldnames=rows[0].keys()`）。

### 6.5 场景五：跑论文级 Fig.6

已经是成品脚本，直接跑：

```bash
# 涡旋（单块）
"$PY" design_fig6.py vortex

# 艾里（4 条行带并行，避免内存爆）
OMP_NUM_THREADS=4 "$PY" design_fig6.py airy_band 0 1
OMP_NUM_THREADS=4 "$PY" design_fig6.py airy_band 1 2
OMP_NUM_THREADS=4 "$PY" design_fig6.py airy_band 2 3
OMP_NUM_THREADS=4 "$PY" design_fig6.py airy_band 3 4
"$PY" design_fig6.py airy_stitch      # 拼回完整结构

# 出图
"$PY" make_fig6.py
```

⚠️ **并行时务必设 `OMP_NUM_THREADS=4`**。本机 24 核，起 5 个进程若各自开 20 线程会严重超订阅，实测从 6s/迭代退化到 32s/迭代并撑爆内存。

### 6.6 场景六：GUI 加一个控件

`gui_app.py` 有 931 行，结构是这样的：

```
run_selftest()            ← 无头自检
MainWindow
  ├─ Tab 1: 训练          → 调 pipeline.train_model
  ├─ Tab 2: 逆向设计      → 调 pipeline.design_from_phase
  ├─ Tab 3: 场可视化验证  → 调 refine_display / direct_design
  └─ Tab 4: 评估          → 调 pipeline.evaluate_model
```

加参数的流程：① 在对应 Tab 的 `__init__` 里加控件；② 在点击回调里读控件值；③ 透传给 `pipeline` 的函数。**不要把业务逻辑写进 GUI**——GUI 只负责收集参数和画图。

---

## 7. 排错手册

按"症状 → 原因 → 解法"组织，都是本项目实际踩过的坑。

### 7.1 环境类

| 症状 | 原因 | 解法 |
|---|---|---|
| `ModuleNotFoundError: No module named 'torch'` | 用错解释器（系统 python 是 3.13 无 torch） | 用 `C:/Users/Fischer/AppData/Local/Microsoft/WindowsApps/python.exe` |
| `size mismatch for stem.weight` | ckpt 的 preset 与当前不一致 | 不要手改 preset，让它从 ckpt 读；或按 ckpt 的 preset 重建 |
| GUI 打不开 / 报 `No module named 'torch'` | 用错解释器（界面本身用标准库 tkinter，不需要额外 GUI 依赖） | 同上，torch 只装在 WindowsApps 那个 python |

### 7.2 训练类

| 症状 | 原因 | 解法 |
|---|---|---|
| loss 完全不动（如卡在 0.46） | 随机初始值落在鞍点，目标相位跳变太剧烈 | `init="phase"` 全息初始化 |
| loss 变 NaN | 学习率过大 / 梯度爆炸 | 降 lr 到 1e-4；确认 `clip_grad_norm_` 在；检查输入是否越界 |
| 训练快但推理差很多 | 过拟合 / 数据分布太窄 | 增大 batch、加噪声、检查 `continuous_ratio` |
| 多进程时极慢 | OpenMP 线程超订阅 | `OMP_NUM_THREADS=4` |
| 日志不刷新、看不到进度 | stdout 缓冲 | `python -u`；或看 `flush=True` 是否在每个 print 上 |

### 7.3 设计类

| 症状 | 原因 | 解法 |
|---|---|---|
| `目标相位图需为 (336,336) 像素` | patches 参数与相位图尺寸不匹配 | `ext = (patches-1)*84+100`，按此生成 |
| 光斑全白/全饱和 | `imshow` 的 vmin/vmax 没按数据定 | 用 `W / W.max()` 归一化后再画 |
| 光斑跑到窗口外 | 目标相位的一阶衍射方向在窗口外 | 加大 `pad_eval`（艾里用 4） |
| 拼接处有明显接缝 | 84% 切割不够 / 块间相位不连续 | 检查 `tgt` 是否用中心 84px；确认 patch 位置对齐 |
| 拓扑荷读出 +12.9（明显错） | 在出射面测相位，不是焦面 | 传米制单位给 `focal_field`，或先传播再测 |
| 涡旋中心有十字亮线 | 圆孔破坏了艾里的可分离性 | 用方孔（`aperture` 传方形边界） |
| 二值化后结构全 1.0 或全 3.526 | beta 上升太快 / 学习率过大 | 降 `beta_final`，或延后 `binarize_from` |

### 7.4 结果判读类（最容易误判的一类）

| 现象 | 是不是 bug |
|---|---|
| 光斑比论文模糊 | **不是**。模型 0.8M vs 论文 40M，BPM vs FDTD。属预期差距 |
| FWHM 比衍射极限小 | **是**。检查 FWHM 计算窗口里是否有旁瓣被算进主峰 |
| 效率 90%+ | **可疑**。检查是否分母用了归一化后的强度 |
| 相位 MAE 很小但光斑很烂 | **正常**。相位在暗区无意义，要按幅度加权评估 |
| 涡旋环宽明显大于理论 | **正常**。环宽 ∝ 1/NA，4 块口径 NA 很低 |

---

## 8. 术语表

| 术语 | 英文 | 一句话解释 |
|---|---|---|
| 超表面 | metasurface | 亚波长结构阵列，能任意塑造波前 |
| 超原子 | meta-atom | 单个可调结构单元（本项目 = 一根柱子） |
| 自由形状 | freeform | 结构不受规则形状约束，每根柱子独立优化 |
| 逆向设计 | inverse design | 从目标光场反求结构 |
| 代理模型 | surrogate model | 用神经网络模仿慢速物理仿真器 |
| 邻域注意力 | neighborhood attention | 每个像素只看 k×k 邻域的自注意力，O(N·k²) |
| 正向问题 | forward problem | 结构 → 光场（仿真） |
| BPM | beam propagation method | 分步傅里叶传播，忽略反射的近似求解器 |
| 角谱法 | angular spectrum | FFT 域传播，`exp(i·kz·dz)` |
| 二值化惩罚 | binarization penalty | 把连续折射率推向两端的正则项 |
| 差分学习率 | differential LR | 不同层用不同学习率 |
| 检查点 | checkpoint | 保存的模型参数文件 |
| 批 | batch | 一次同时处理的样本数 |
| 拓扑荷 | topological charge (ℓ) | 涡旋相位绕一圈累积 2πℓ |
| NA | numerical aperture | 数值孔径 = n·sin(arctan(D/2F)) |
| FWHM | full width at half maximum | 半高全宽，焦斑尺寸 |
| 84% 切割 | 84% cutting ratio | patch 只取中心 84% 参与拼接 |

---

## 9. 给你的第一个练习

按顺序做完这四个小任务，你就真正接手了：

1. **改一个数字**：把 `design.py` 的 `--iters` 从 250 改到 500，看 `design_loss_curve.csv` 尾部的损失降了多少。体会"迭代次数"的边际收益。
2. **加一个任务**：按 6.1 加"贝塞尔"，先只画相位图，确认形状对，再跑完整设计。
3. **做一次归因**：对同一个涡旋目标，分别跑 `design.py`（代理）和 `direct_design.py`（直接），对比焦面光斑。如果直接设计明显更好，说明你的代理模型还有提升空间。
4. **破坏一次可微性**：故意把 `design.py` 里的 `torch.tanh(v)` 改成 `numpy.tanh(v.numpy())`，看报什么错。理解"可微"在工程上意味着什么。

---

## 附：一页速查

```bash
PY="C:/Users/Fischer/AppData/Local/Microsoft/WindowsApps/python.exe"
cd C:/Users/Fischer/Desktop/MetaE_former

"$PY" gui_app.py                                              # GUI
"$PY" gui_app.py --selftest                                   # 回归自检

"$PY" pretrain.py --preset fast --steps 1600 --batch 24 --lr 3e-4
"$PY" finetune.py --ckpt checkpoints/pretrain.pt --collect 800 --steps 300
"$PY" evaluate.py --ckpt checkpoints/pretrain.pt --n 48

"$PY" design.py --task metalens --focal 20e-6 --out outputs/metalens
"$PY" design.py --task vortex --l 6 --ckpt checkpoints/finetune2.pt --out outputs/vortex6
"$PY" design.py --task airy --out outputs/airy
"$PY" design.py --task custom --phase my_phase.npy --out outputs/custom

"$PY" direct_design.py outputs/direct_vortex6 vortex 6 --iters 2500
"$PY" make_fig6.py
```

**核心常数**：λ=1064nm · 柱高 800nm · 柱距 200nm · n∈[1.0, 3.526] · patch 5µm · 场采样 50nm · 84% 切割 · 网络 25→13→7→13→25→50→100。

**改代码前先问自己三句**：形状对吗？可微吗？单位是米还是微米？
