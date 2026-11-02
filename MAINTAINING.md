# 站点维护说明

> 这份文档面向维护者，不发布到站点。访客看到的是 `docs/guide.html`（复现方法论）。

## 站点结构

纯静态，无构建步骤。样式、脚本、数据各一份，所有页面共用。

```
docs/
├── index.html                     门户：项目卡片由数据渲染
├── guide.html                     复现方法论（面向访客）
├── reproductions/
│   ├── _template.html             空白项目页骨架，复制它开始新的一篇
│   ├── metae-former.html          项目 01
│   └── optical-generative-models.html   项目 02
├── assets/
│   ├── css/site.css               唯一的样式表：设计令牌 + 全部组件
│   ├── js/site.js                 唯一的脚本：灯箱、复制、渐显、卡片渲染
│   ├── data/projects.js           ★ 项目清单，新增一篇改这里
│   └── img/
│       ├── metae/                 项目 01 的图
│       └── optgen/                项目 02 的图
└── downloads/                     代码包与 GUI 分卷（.001/.002/.003）
```

| 文件 | 要点 |
|---|---|
| `site.css` | 改配色只改顶部 `:root` 变量，全站跟着变。组件按用途分段，新增组件加到文件末尾。 |
| `site.js` | 所有功能都做了缺元素保护，任意页面加载同一份脚本不会报错。 |
| `projects.js` | 门户的唯一数据源：卡片、方向筛选、搜索索引全部由它生成。改这里，首页自动更新，不需要碰 HTML。 |

## 新增一篇复现：三个动作

1. 在 `docs/assets/data/projects.js` 的 `REPRODUCTIONS` 末尾追加一条记录 —— 门户卡片、方向按钮、搜索索引自动出现
2. 复制 `docs/reproductions/_template.html` 为 `<slug>.html`，只填正文
3. 图片放 `docs/assets/img/<slug>/`，宽 ≤ 1300 px

做完这三步，门户首页与页间导航（上下篇）会自动接上。

### 数据字段约定

| 字段 | 必填 | 说明 |
|---|---|---|
| `no` | ✓ | 卡片角标编号，按收录顺序顺延，不要跳号 |
| `venue` | ✓ | 期刊与年份，一行，例：`Nature 644 · 2025` |
| `title` | ✓ | 只写标题，不带副标题 |
| `sum` | ✓ | 一句话，≤ 40 字；细节留给复现页 |
| `cats` | ✓ | 方向 id 数组，取自 `window.CATEGORIES`，可多值 |
| `tags` | | 技术关键词，只进搜索索引，卡片上不显示 |
| `cover` | ✓ | 16:10 封面图 |
| `href` | ✓ | 相对 `docs/` 的路径 |

新增方向时，在 `window.CATEGORIES` 里加一行 `{ id, label }`。没有复现的方向不会显示按钮 —— 等第一篇归档进来，按钮自动出现，不需要改 HTML。

`_template.html` 放在 `reproductions/` 下是必须的：放 `docs/` 根会让页内所有 `../assets/` 失效。

## 页面骨架约定

章节顺序固定为：基础 / 论文 / 结果 / 方法 / 上手 / 下载 / 引用；有界面的项目在「结果」之后插一节「GUI」。
换一篇论文时阅读路径不变，这是让读者不用重新适应的前提。

## 文案与制图规范

这些规则不是审美偏好，是为了让读者能比较不同项目、能判断结论强度。

| 项 | 规则 | 理由 |
|---|---|---|
| 差异表 | 每篇必须有「论文做法 / 本复现做法 / 影响」三列 | 结论的适用范围由它界定 |
| 结果图 | 必须注明来源与口径：谁的图、什么条件、单位与比例尺 | 避免把论文原图当成自己的结果 |
| 数字 | 只写自己跑出来的数，并给出对应命令 | 读者要能复算 |
| 图片 | 宽 ≤ 1300 px；照片类用 JPEG q86，线稿用 PNG | 首屏加载时间 |
| 文案 | 只讲方法与结果的差异，不把硬件条件、运行速度当作成果 | 这些不是方法贡献，写进去会稀释结论 |
| 术语 | 首次出现给全称与一句话解释，之后用缩写 | 跨领域读者能跟上 |
| 未完成 | 用 `warn` 标记，不要隐去 | 省略会被误读为已完成 |

## 发布前检查

| 项 | 检查内容 |
|---|---|
| 结构 | 章节编号连续、锚点条链接全部有效、面包屑路径正确 |
| 资源 | 所有 `img` 路径存在；图片无空白占位；各 ZIP 与分卷链接可下载 |
| 交互 | 灯箱可开可关（Esc 生效）、复制按钮有反馈、导航当前页高亮 |
| 数据 | `projects.js` 里的 href 与文件名一致；新增卡片出现在首页 |
| 窄屏 | 900 px 以下卡片单列、截图单列、参数表可横向滚动 |
| 声明 | 许可、引用、未复现项都在 |

自动校验（在仓库根目录执行）：

```bash
python _check_site.py
```

覆盖标签配对、页内锚点、图片存在性、本地内链、样式脚本引用、CSS 类覆盖，输出「问题总数」。

## 代码包

`docs/downloads/` 下三个交付物，改动源码后都要重新打包并提交：

| 文件 | 来源 | 说明 |
|---|---|---|
| `metae-former-v1.0.0.zip` | 仓库根目录 | 排除 `docs/`、`.git`、`__pycache__`、`reproductions/`、`outputs/` 与本文档 |
| `optical-generative-models-v1.0.0.zip` | `reproductions/optical-generative-models/` | MATLAB 源码 + 原型权重 + 日志 |
| `metae-gui-windows-x64-v1.0.0.zip.001~003` | PyInstaller 产物目录 | 见下 |

### GUI 免安装包

体积约 215 MB（PyTorch 的 `torch_cpu.dll` 一个就占 254 MB，压缩后 76 MB），超过 GitHub 单文件 100 MB 限制，所以按 75 MB 切三卷。

生成流程：

```bash
pyinstaller MetaEFormerGUI.spec        # 产物在 dist/MetaEFormerGUI/
# 把 checkpoints/finetune2.pt 复制进 _internal/checkpoints/（spec 里只声明了两个权重）
./MetaEFormerGUI.exe --selftest        # 必须跑，确认打包没有漏模块
rm -rf MetaEFormerGUI/outputs MetaEFormerGUI/checkpoints   # 清掉自检产物
# 压缩该目录，再按 75 MB 切片，输出到 docs/downloads/
```

切片后必须验证：合并三卷 → `zipfile.testzip()` 返回 `None`，且 `_internal/checkpoints/` 下三个权重齐全。

Windows 合并命令（页面上也给了）：

```bat
copy /b metae-gui-windows-x64-v1.0.0.zip.001 + metae-gui-windows-x64-v1.0.0.zip.002 + metae-gui-windows-x64-v1.0.0.zip.003 metae-gui.zip
```

