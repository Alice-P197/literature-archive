/* ============================================================
   复现档案数据源
   ------------------------------------------------------------
   新增一篇复现只需要三步：
     1. 在 REPRODUCTIONS 末尾追加一条记录（no 顺延）
     2. 复制 reproductions/_template.html 为 <slug>.html 并填写内容
     3. 图片放到 assets/img/<slug>/
   门户的卡片、方向筛选、搜索索引全部由本文件自动生成，不用改 HTML。
   ============================================================ */

/* 研究方向的登记表。
   这里只负责「有哪些方向」和「显示什么名字」，顺序即筛选条的排列顺序。
   某个方向还没有复现时不会显示按钮；等第一篇归档进来，按钮自动出现。
   以后扩到成像、材料、量子等方向，在这里加一行即可。 */
window.CATEGORIES = [
  { id: "optical-computing",  label: "光学计算" },
  { id: "generative",         label: "生成模型" },
  { id: "metasurface",        label: "超表面" },
  { id: "inverse-design",     label: "逆向设计" },
  { id: "surrogate",          label: "代理模型" },
  { id: "computational-imaging", label: "计算成像" },
  { id: "ml-method",          label: "机器学习方法" },
  { id: "optimization",       label: "优化方法" },
  { id: "materials",          label: "材料与器件" },
  { id: "quantum",            label: "量子光子学" },
  { id: "biomedical",         label: "生物医学光子学" }
];

/* 档案条目。按收录时间排序，no 就是卡片上的编号。
   字段约定：
     no        序号，门户卡片角标
     venue     期刊与年份，一行
     title     标题，不写副标题
     sum       一句话摘要，不超过 40 字，细节留给复现页
     cats      方向 id，可多个，用于筛选与搜索
     tags      技术关键词，只进搜索索引，不在卡片上显示
     cover     封面图，建议 16:10
     href      复现页相对路径 */
window.REPRODUCTIONS = [
  {
    no: 1,
    slug: "optical-generative-models",
    status: "done",
    statusText: "已复现",
    venue: "Nature 644 · 2025",
    title: "光学生成模型：让光在自由空间里完成一次生成",
    sum: "相位种子经衍射解码层传播成像，十个类别全部条件生成成功。",
    cats: ["optical-computing", "generative"],
    tags: ["MATLAB", "角谱传播", "衍射神经网络", "MNIST"],
    cover: "assets/img/optgen/cover-mnist.jpg",
    href: "reproductions/optical-generative-models.html"
  },
  {
    no: 2,
    slug: "metae-former",
    status: "done",
    statusText: "已复现",
    venue: "PhotoniX 7:69 · 2026",
    title: "MetaE-former：用邻域注意力代理模型设计大尺寸超表面",
    sum: "把结构到光场的正演蒸馏进 Transformer，再分块反解相位、拼接成整片器件。",
    cats: ["metasurface", "inverse-design", "surrogate"],
    tags: ["PyTorch", "Transformer", "可微 BPM", "自由曲面"],
    cover: "assets/img/metae/fig6_repro.jpg",
    href: "reproductions/metae-former.html"
  }
];
