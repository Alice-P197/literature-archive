/* ============================================================
   文献档案数据源
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
  },
  {
    no: 3,
    slug: "doe-image-reconstruction",
    status: "done",
    statusText: "已复现",
    venue: "Optik 35 · 1972 (GS) + MRAF",
    title: "纯相位 DOE 设计：信号窗口 IFTA + MRAF 再现目标图像",
    sum: "高斯入射 1.5 mm 相位片，50 mm 自由空间再现 40 mm 图像，RMSE 0.0014、效率 42.7%。",
    cats: ["inverse-design", "optimization", "computational-imaging"],
    tags: ["Python", "IFTA", "Gerchberg-Saxton", "MRAF", "DOE"],
    cover: "assets/img/doe-image-reconstruction/doe_compare.png",
    href: "reproductions/doe-image-reconstruction.html"
  },
  {
    no: 4,
    slug: "slm-phase-optimization",
    status: "done",
    statusText: "已复现",
    venue: "伴随梯度逆设计方法",
    title: "SLM 螺旋锥透镜旁瓣抑制：径向相位参数化 + 伴随梯度优化",
    sum: "1920×1080 SLM 螺旋锥透镜，伴随梯度优化后主瓣能量占比 3.06%→87.9%，旁瓣 −4.5→−15.5 dB。",
    cats: ["optimization", "inverse-design", "computational-imaging"],
    tags: ["Python", "伴随梯度", "Adam", "SLM", "贝塞尔光束"],
    cover: "assets/img/slm-phase-optimization/preview_gauss_result.png",
    href: "reproductions/slm-phase-optimization.html"
  },
  {
    no: 5,
    slug: "fresnel-diffraction-matlab",
    status: "done",
    statusText: "已复现",
    venue: "经典 FFT 衍射算法",
    title: "菲涅尔衍射三种 FFT 算法对比：S-FFT / T-FFT / D-FFT",
    sum: "LG 涡旋光束过环孔传播，统一接口下对比三种 FFT 衍射算法的输出窗口与精度。",
    cats: ["computational-imaging", "optical-computing"],
    tags: ["MATLAB", "Fresnel衍射", "角谱法", "LG光束", "FFT"],
    cover: "assets/img/fresnel-diffraction-matlab/fresnel_compare_d30.png",
    href: "reproductions/fresnel-diffraction-matlab.html"
  },
  {
    no: 6,
    slug: "rotational-doppler-vortex-detection",
    status: "done",
    statusText: "已复现",
    venue: "红外与激光工程 50(9) · 2021",
    title: "涡旋光旋转多普勒测速：任意入射条件下的转速探测（丁友 2021 复现）",
    sum: "复现丁友 2021 公式，特征点几何平均法提取转速，误差 0.0000%，频谱展宽与理论边界一致。",
    cats: ["optical-computing", "computational-imaging"],
    tags: ["MATLAB", "轨道角动量", "旋转多普勒", "花瓣模", "拍频"],
    cover: "assets/img/rotational-doppler-vortex-detection/fig_cases_compare.png",
    href: "reproductions/rotational-doppler-vortex-detection.html"
  },
  {
    no: 7,
    slug: "double-phase-encoding-vortex-array",
    status: "done",
    statusText: "已复现",
    venue: "Optics Letters 39(7) · 2014",
    title: "双相位编码生成涡旋光阵列：4f 系统 + 纯相位 SLM",
    sum: "把 HG(5,1)+i·HG(1,5) 螺旋厄米高斯场编码进纯相位 SLM，经 4f 滤波恢复 OAM 阵列。",
    cats: ["optical-computing", "metasurface", "generative"],
    tags: ["MATLAB", "双相位编码", "4f系统", "OAM", "角谱法"],
    cover: "assets/img/double-phase-encoding-vortex-array/dpe_validation.png",
    href: "reproductions/double-phase-encoding-vortex-array.html"
  }
];

