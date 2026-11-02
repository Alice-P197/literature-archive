# 论文复现实验室 · Paper Repro Lab

把机器学习、计算成像与光学设计方向的论文，拆成可运行、可修改、可验证的代码档案。

每一篇固定交付四样东西：**可运行的源码**、**已训练的权重**、**和原文并排的结果图**，
以及一份**写清哪里做不到的差异说明**。

站点：<https://alice-p197.github.io/paper-repro-lab/>

---

## 收录

| # | 论文 | 方向 | 交付 |
|---|---|---|---|
| 01 | **MetaE-former**：用邻域注意力代理模型设计大尺寸超表面<br>PhotoniX 2026 | 神经网络 + 逆向设计 | Python 源码、3 个预训练权重、Fig.3 / Fig.6 复现图、桌面端 GUI、零基础教程 |
| 02 | **Optical generative models**：让光在自由空间里完成一次生成<br>Nature 644, 903–911 (2025) | 光学计算 + 生成模型 | MATLAB 源码（11 个脚本）、两套原型权重、复现报告 |

---

## 仓库结构

```
├── docs/                              站点（GitHub Pages 源目录）
│   ├── index.html                     门户：项目卡片由数据渲染
│   ├── guide.html                     复现方法与写作规范
│   ├── reproductions/
│   │   ├── _template.html             空白项目页骨架
│   │   ├── metae-former.html          项目 01 详情页
│   │   └── optical-generative-models.html   项目 02 详情页
│   ├── assets/
│   │   ├── css/site.css               唯一样式表
│   │   ├── js/site.js                 唯一脚本
│   │   ├── data/projects.js           项目清单（新增一篇改这里）
│   │   └── img/{metae,optgen}/
│   └── downloads/                     代码包与报告 PDF
│
├── metae/                             MetaE-former 的核心库
├── pretrain.py  finetune.py  design.py  evaluate.py
├── gui_app.py                         tkinter 桌面界面
├── direct_design.py                   可微 BPM 全口径直设计
├── design_fig6.py  make_fig6.py       Fig.6 复现流水线
├── checkpoints/                       预训练 / 微调权重
├── TUTORIAL.md                        零基础代码接手教程（约 970 行）
│
└── reproductions/
    └── optical-generative-models/     项目 02 的 MATLAB 源码与原型权重
```

---

## 快速上手

### 项目 01 · MetaE-former

```bash
pip install -r requirements.txt

# 评估现成权重（约 1 分钟）
python evaluate.py --ckpt checkpoints/pretrain.pt --n 48

# 做一次涡旋光逆向设计
python design.py --task vortex --l 6 --focal 20e-6 --ckpt checkpoints/finetune2.pt --out outputs/vortex6

# 打开桌面界面
python gui_app.py
```

首次接手建议先读 **[TUTORIAL.md](TUTORIAL.md)**：从「张量是什么」讲到「怎么加一个新的目标光场」，所有命令与数值都实测过。

### 项目 02 · 光学生成模型

```matlab
setenv('OGEN_DATASET','mnist');   train_prototypes     % 阶段一，约 18 分钟
setenv('OGEN_DATASET','mnist');   generate_mixture      % 阶段二，约 1 分钟
```

---

## 新增一篇复现

站点的结构就是为这件事准备的，三个动作：

1. 在 `docs/assets/data/projects.js` 加一条记录 —— 门户卡片自动出现
2. 复制 `docs/reproductions/_template.html` 为 `<slug>.html`，只填正文
3. 图片放 `docs/assets/img/<slug>/`，宽 ≤ 1300 px

详细规范（章节顺序、差异表格式、制图约定、发布检查清单）见站点的 `guide.html`。

---

## 设计原则

- **差异表优先于结论**：每篇都有「论文做法 / 本复现做法 / 影响」三列，结论只能写在差异表允许的范围内。
- **指标可复算**：页面上每个数字都对应一条能跑的命令。
- **未完成项不隐去**：用标记标出，而不是删除。
- **只讲方法差异**：硬件条件、运行速度这类环境差异不作为成果表述。

---

## 许可

代码以 MIT 许可发布。论文插图版权归原作者与期刊所有，站内仅用于结果对照。
本站为独立复现，非论文作者官方实现。
