/* ============================================================
   复现档案数据源
   ------------------------------------------------------------
   新增一篇复现，只需要：
     1. 在 REPRODUCTIONS 数组里加一条记录
     2. 复制 _template.html 为 reproductions/<slug>.html 并填写内容
     3. 图片放到 assets/img/<slug>/
   门户首页的卡片网格会自动出现，不需要改 HTML。
   ============================================================ */

window.REPRODUCTIONS = [
  {
    slug: "metae-former",
    status: "done",
    statusText: "已复现",
    venue: "PhotoniX · 2026",
    title: "MetaE-former：用邻域注意力代理模型设计大尺寸超表面",
    sum: "把「结构 → 光场」的正演过程蒸馏进一个 Transformer，再用可微 BPM 在 5 µm 的块上反解相位、拼接成整片器件。本页给出完整链路、三个权重、复现结果与一套桌面端 GUI。",
    cover: "assets/img/metae/fig6_repro.jpg",
    kpis: [
      { v: "0.112", k: "归一化 MAE" },
      { v: "0.82 M", k: "代理参数量" },
      { v: "5 µm", k: "单块设计口径" }
    ],
    tags: ["PyTorch", "Transformer", "可微 BPM"],
    href: "reproductions/metae-former.html"
  },
  {
    slug: "optical-generative-models",
    status: "done",
    statusText: "已复现",
    venue: "Nature 644 · 2025",
    title: "光学生成模型：让光在自由空间里完成一次生成",
    sum: "相位种子经角谱传播、衍射解码层，最后落在传感器上成像。MNIST 与 Fashion-MNIST 十个类别全部条件生成成功，并附带蒸馏与插值实验。",
    cover: "assets/img/optgen/cover-mnist.jpg",
    kpis: [
      { v: "0.0082", k: "原型拟合 MSE" },
      { v: "10 类", k: "条件生成覆盖" },
      { v: "2 套", k: "数据集权重" }
    ],
    tags: ["MATLAB", "角谱传播", "生成模型"],
    href: "reproductions/optical-generative-models.html"
  }
];

/* 正在做 / 计划做的方向，只作为档案页的路线图展示，不生成卡片 */
window.ROADMAP = [
  {
    title: "衍射光学网络的端到端联合优化",
    note: "把多层衍射面与后端读出网络放进同一个梯度图，比较分层训练与联合训练的差异。"
  },
  {
    title: "超表面光谱响应的代理模型",
    note: "输入从单波长扩展到波长维度，验证邻域注意力在 3D 输入上的可扩展性。"
  },
  {
    title: "逆向设计的结果可信度评估",
    note: "用全波求解器抽样复核 BPM 代理给出的结构，给出误差随结构复杂度的分布。"
  }
];
