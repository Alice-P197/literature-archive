"""MetaE-former GUI: 训练 / 相位图逆向设计 / 场可视化验证 / 模型评估 / CSV 导出.

Run:            python gui_app.py
Packaged exe:   MetaEFormerGUI.exe  (built with PyInstaller, see build_exe.bat)
Headless test:  python gui_app.py --selftest
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
import traceback

# ---------------------------------------------------------------- paths
if getattr(sys, "frozen", False):  # running as PyInstaller exe
    BASE_DIR = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    APP_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    APP_DIR = BASE_DIR
sys.path.insert(0, BASE_DIR)
os.chdir(APP_DIR)  # checkpoints / outputs land next to the exe

SELFTEST = "--selftest" in sys.argv

import matplotlib

matplotlib.use("TkAgg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei UI", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

CKPT_DIRS = [os.path.join(APP_DIR, "checkpoints"), os.path.join(BASE_DIR, "checkpoints")]
OUT_DIR = os.path.join(APP_DIR, "outputs")

PITCH_PX, CROP = 100, 84

if SELFTEST:
    _run_selftest_later = True
else:
    _run_selftest_later = False

# GUI imports (tkinter needed either way for exe build)
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import torch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from metae.pipeline import (
    builtin_phase,
    design_from_phase,
    evaluate_model,
    load_phase_map,
    save_design_csvs,
    save_rows_csv,
    save_matrix_csv,
    train_model,
)
from metae.solver import angular_spectrum_propagate
from direct_design import run_direct_design, propagation_stacks

TASKS = ["lens (metalens)", "vortex", "airy", "custom (外部相位图)"]
QUANTS = ["强度 |E|²", "振幅 |E|", "实部 Re(E)", "虚部 Im(E)", "相位 arg(E)"]
VIEWS = ["xy 目标面切片", "xz 传播剖面", "yz 传播剖面"]

# ---------------------------------------------------------------- palette
C_BG = "#EEF2F7"       # app background
C_CARD = "#FFFFFF"     # card background
C_ACC = "#2563EB"      # accent blue
C_ACC_D = "#1D4ED8"
C_TXT = "#0F172A"      # main text
C_MUT = "#64748B"      # muted text
C_HEAD = "#0F172A"     # header band
C_OK = "#059669"
C_WARN = "#D97706"

FONT = ("Microsoft YaHei UI", 10)
FONT_B = ("Microsoft YaHei UI", 10, "bold")
FONT_T = ("Microsoft YaHei UI", 13, "bold")
FONT_S = ("Microsoft YaHei UI", 9)


def _qty(E):
    """按名称映射复场 -> 显示数组。"""
    return {
        QUANTS[0]: np.abs(E) ** 2,
        QUANTS[1]: np.abs(E),
        QUANTS[2]: np.real(E),
        QUANTS[3]: np.imag(E),
        QUANTS[4]: np.angle(E),
    }


def _safe(s: str) -> str:
    """matplotlib DejaVu/YaHei 缺 µ/² 字形，图内文字用 ASCII 替换。"""
    return s.replace("µ", "u").replace("²", "^2")


# ===================================================================== GUI
class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("MetaE-former 智能超表面设计工作台")
        root.geometry("1300x900")
        root.configure(bg=C_BG)

        self._setup_style()

        self.q: "queue.Queue[tuple]" = queue.Queue()
        self.job: threading.Thread | None = None
        self.stop_flag = threading.Event()
        self.model = None
        self.ckpt_of_model = None
        self.last_res = None
        self.last_eval = None
        self.last_hist = None
        # 可视化状态
        self.viz = None   # dict(E_pad, dx, stack_xz, stack_yz, z_axis, focal_um)

        # ------------------------------------------------------------ header
        head = tk.Frame(root, bg=C_HEAD, height=64)
        head.pack(fill="x")
        tk.Label(head, text="MetaE-former", font=("Segoe UI", 17, "bold"),
                 bg=C_HEAD, fg="#7DD3FC").pack(side="left", padx=(18, 6), pady=10)
        tk.Label(head, text="智能超表面设计工作台 · 可微物理逆向设计",
                 font=("Microsoft YaHei UI", 10), bg=C_HEAD, fg="#94A3B8"
                 ).pack(side="left", pady=10)
        self.hstatus = tk.Label(head, text="就绪", font=FONT, bg=C_HEAD, fg="#E2E8F0")
        self.hstatus.pack(side="right", padx=16)

        nb = ttk.Notebook(root)
        nb.pack(fill="both", expand=True, padx=10, pady=8)
        self.tab_train = ttk.Frame(nb, style="Card.TFrame")
        self.tab_design = ttk.Frame(nb, style="Card.TFrame")
        self.tab_viz = ttk.Frame(nb, style="Card.TFrame")
        self.tab_eval = ttk.Frame(nb, style="Card.TFrame")
        nb.add(self.tab_train, text="  ① 训练  ")
        nb.add(self.tab_design, text="  ② 相位图逆向设计  ")
        nb.add(self.tab_viz, text="  ③ 场可视化验证  ")
        nb.add(self.tab_eval, text="  ④ 模型评估  ")

        self._build_train_tab()
        self._build_design_tab()
        self._build_viz_tab()
        self._build_eval_tab()

        self.status = tk.StringVar(value="就绪")
        ttk.Label(root, textvariable=self.status, anchor="w", padding=(10, 3),
                  style="Muted.TLabel").pack(fill="x")
        root.after(120, self._poll)

    # ------------------------------------------------------------- style
    def _setup_style(self):
        st = ttk.Style(self.root)
        st.theme_use("clam")
        st.configure(".", background=C_BG, foreground=C_TXT, font=FONT)
        st.configure("Card.TFrame", background=C_CARD)
        st.configure("TNotebook", background=C_BG, borderwidth=0, tabmargins=(0, 4, 0, 0))
        st.configure("TNotebook.Tab", background=C_BG, foreground=C_MUT,
                     font=FONT_B, padding=(14, 8), borderwidth=0)
        st.map("TNotebook.Tab",
               background=[("selected", C_CARD)],
               foreground=[("selected", C_ACC)])
        st.configure("Card.TLabelframe", background=C_CARD, borderwidth=0)
        st.configure("Card.TLabelframe.Label", background=C_CARD,
                     foreground=C_ACC, font=FONT_B)
        st.configure("TLabel", background=C_CARD, foreground=C_TXT, font=FONT)
        st.configure("Muted.TLabel", background=C_BG, foreground=C_MUT, font=FONT_S)
        st.configure("Head.TLabel", background=C_CARD, foreground=C_MUT, font=FONT_S)
        st.configure("TFrame", background=C_CARD)
        st.configure("TButton", background="#E2E8F0", foreground=C_TXT, font=FONT,
                     padding=(10, 6), borderwidth=0)
        st.map("TButton", background=[("active", "#CBD5E1")])
        st.configure("Accent.TButton", background=C_ACC, foreground="#FFFFFF",
                     font=FONT_B, padding=(12, 6), borderwidth=0)
        st.map("Accent.TButton", background=[("active", C_ACC_D), ("disabled", "#93C5FD")])
        st.configure("Stop.TButton", background="#FEE2E2", foreground="#B91C1C",
                     font=FONT, padding=(10, 6), borderwidth=0)
        st.map("Stop.TButton", background=[("active", "#FECACA")])
        st.configure("TEntry", fieldbackground="#FFFFFF", padding=4)
        st.configure("TCombobox", fieldbackground="#FFFFFF", padding=4)
        st.configure("TSpinbox", fieldbackground="#FFFFFF", padding=4, arrowsize=11)
        st.configure("TRadiobutton", background=C_CARD, font=FONT)
        st.configure("Accent.TRadiobutton", background=C_CARD, font=FONT_B,
                     foreground=C_TXT)
        st.map("Accent.TRadiobutton", foreground=[("selected", C_ACC)])
        st.configure("Horizontal.TProgressbar", background=C_ACC,
                     troughcolor="#E2E8F0", borderwidth=0, thickness=6)

    # ------------------------------------------------------------- helpers
    def card(self, parent, title, pad=(10, 8)):
        f = ttk.Labelframe(parent, text="  " + title + "  ", style="Card.TLabelframe",
                           padding=pad)
        f.pack(fill="x", padx=12, pady=(8, 2))
        return f

    def log(self, tab_txt, msg):
        tab_txt.configure(state="normal")
        tab_txt.insert("end", msg + "\n")
        tab_txt.see("end")
        tab_txt.configure(state="disabled")

    def start_job(self, name, fn):
        if self.job and self.job.is_alive():
            messagebox.showwarning("忙碌", "已有任务在运行，请先停止或等待完成")
            return
        self.stop_flag.clear()
        self.status.set(f"运行中: {name}")
        self.hstatus.configure(text=f"运行中: {name}…", fg="#FCA5A5")

        def wrap():
            try:
                fn()
                self.q.put(("done", name))
            except Exception:
                err = traceback.format_exc()
                self.q.put(("error", f"{name} 失败:\n{err}"))

        self.job = threading.Thread(target=wrap, daemon=True)
        self.job.start()

    def _poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    txt, msg = payload
                    self.log(txt, msg)
                elif kind == "call":
                    payload()
                elif kind == "done":
                    self.status.set(f"完成: {payload}")
                    self.hstatus.configure(text="就绪", fg="#E2E8F0")
                elif kind == "error":
                    self.status.set("出错")
                    self.hstatus.configure(text="出错", fg="#F87171")
                    messagebox.showerror("错误", payload)
        except queue.Empty:
            pass
        self.root.after(120, self._poll)

    # ============================================================= 训练 tab
    def _build_train_tab(self):
        f = self.tab_train
        top = self.card(f, "预训练设置（随机/岛状/条纹超原子数据流式生成，无需数据集文件）")
        self.tr_preset = tk.StringVar(value="fast")
        self.tr_steps = tk.IntVar(value=800)
        self.tr_batch = tk.IntVar(value=24)
        self.tr_lr = tk.DoubleVar(value=3e-4)
        opts = [("网络规模", ttk.Combobox(top, textvariable=self.tr_preset,
                 values=["tiny", "fast", "paper"], state="readonly", width=8)),
                ("训练步数", ttk.Spinbox(top, textvariable=self.tr_steps, from_=50, to=100000, width=8)),
                ("batch", ttk.Spinbox(top, textvariable=self.tr_batch, from_=4, to=256, width=8)),
                ("学习率", ttk.Spinbox(top, textvariable=self.tr_lr, from_=1e-6, to=1e-2, increment=1e-5, width=9))]
        for i, (lab, w) in enumerate(opts):
            ttk.Label(top, text=lab, style="Head.TLabel").grid(row=0, column=2 * i, padx=(12, 2), pady=6, sticky="e")
            w.grid(row=0, column=2 * i + 1, padx=2, pady=6)

        btns = ttk.Frame(top)
        btns.grid(row=0, column=100, padx=14)
        self.btn_train = ttk.Button(btns, text="▶  开始训练", style="Accent.TButton",
                                    command=self.on_train)
        self.btn_train.pack(side="left", padx=4)
        self.btn_stop_train = ttk.Button(btns, text="■ 停止", style="Stop.TButton",
                                         command=self.stop_flag.set, state="disabled")
        self.btn_stop_train.pack(side="left", padx=4)
        ttk.Button(btns, text="导出训练曲线 CSV", command=self.on_export_train).pack(side="left", padx=4)

        self.tr_prog = ttk.Progressbar(f, mode="indeterminate", style="Horizontal.TProgressbar")
        self.tr_prog.pack(fill="x", padx=14, pady=(6, 0))

        self.tr_txt = tk.Text(f, height=7, state="disabled", font=("Consolas", 9),
                              bg="#0F172A", fg="#7DD3FC", relief="flat")
        self.tr_txt.pack(fill="both", expand=False, padx=14, pady=6)

        self.fig_tr = Figure(figsize=(5.6, 3.3), dpi=100, facecolor=C_CARD)
        cv = FigureCanvasTkAgg(self.fig_tr, master=f)
        cv.get_tk_widget().pack(fill="both", expand=True, padx=14, pady=(0, 10))

    def on_train(self):
        def job():
            self.q.put(("call", lambda: (self.btn_stop_train.configure(state="normal"),
                                         self.tr_prog.start(24))))
            try:
                def progress(step, steps, loss, la, lt, va, vt, hist):
                    self.last_hist = hist
                    if step % max(1, steps // 100) == 0:
                        self.q.put(("call", self.draw_train))

                hist, ck = train_model(
                    preset=self.tr_preset.get(), steps=self.tr_steps.get(), batch=self.tr_batch.get(),
                    lr=self.tr_lr.get(), ckpt_out=os.path.join(APP_DIR, "checkpoints", "gui_trained.pt"),
                    log=lambda m: self.q.put(("log", (self.tr_txt, m))),
                    should_stop=self.stop_flag, progress=progress, threads=8,
                )
                self.last_hist = hist
                self.q.put(("call", self.draw_train))
            finally:
                self.q.put(("call", lambda: (self.btn_stop_train.configure(state="disabled"),
                                             self.tr_prog.stop())))

        self.start_job("预训练", job)

    def draw_train(self):
        h = self.last_hist
        self.fig_tr.clear()
        if not h or not h["step"]:
            return
        ax1 = self.fig_tr.add_subplot(121)
        ax1.plot(h["step"], h["loss"], "-o", ms=3, color=C_ACC)
        ax1.set_title("training loss", fontsize=10)
        ax1.set_xlabel("step"); ax1.grid(alpha=0.3)
        ax2 = self.fig_tr.add_subplot(122)
        ax2.plot(h["step"], h["val_amp"], "-o", ms=3, color=C_ACC, label="amp MAE")
        ax2.plot(h["step"], h["val_ang"], "-s", ms=3, color=C_OK, label="angle MAE")
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3); ax2.set_xlabel("step")
        ax2.set_title("validation", fontsize=10)
        for ax in (ax1, ax2):
            ax.set_facecolor("#F8FAFC")
        self.fig_tr.tight_layout()
        try:
            self.fig_tr.canvas.draw_idle()
        except Exception:
            pass

    def on_export_train(self):
        if not self.last_hist:
            messagebox.showinfo("提示", "还没有训练记录，请先训练")
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="train_history.csv",
                                         filetypes=[("CSV", "*.csv")])
        if not p:
            return
        h = self.last_hist
        save_rows_csv([{"step": s, "loss": l, "val_amp": a, "val_ang": b}
                       for s, l, a, b in zip(h["step"], h["loss"], h["val_amp"], h["val_ang"])], p)
        self.status.set(f"已保存 {p}")

    # ========================================================= 设计 tab
    def _build_design_tab(self):
        f = self.tab_design

        mode = self.card(f, "设计模式")
        row = ttk.Frame(mode)
        row.pack(fill="x")
        self.dg_mode = tk.StringVar(value="direct")
        ttk.Radiobutton(row, text="物理直接设计 — 可微 BPM 全口径优化 · 无需模型 · 光斑最干净（推荐）",
                        variable=self.dg_mode, value="direct", style="Accent.TRadiobutton",
                        command=self._on_mode_change).pack(side="left", padx=(4, 18), pady=4)
        ttk.Radiobutton(row, text="代理模型设计 — NAT U-Net · 需 checkpoint · 速度快",
                        variable=self.dg_mode, value="surrogate",
                        command=self._on_mode_change).pack(side="left", pady=4)

        # checkpoint 选择（代理模式）
        top = ttk.Labelframe(f, text="  模型 checkpoint（仅代理模式需要）  ",
                             style="Card.TLabelframe", padding=(10, 6))
        self.ckpt_widgets = []
        self.dg_ckpt = tk.StringVar()
        self.cmb_ck = ttk.Combobox(top, textvariable=self.dg_ckpt, width=70)
        self.cmb_ck.pack(side="left", padx=4, pady=4)
        self.ckpt_widgets.append(self.cmb_ck)
        for cmd, txt in ((self.refresh_ckpts, "刷新"), (self.browse_ckpt, "浏览..."),
                         (self.on_load_model, "加载模型")):
            b = ttk.Button(top, text=txt, command=cmd)
            b.pack(side="left", padx=3)
            self.ckpt_widgets.append(b)
        top.pack(fill="x", padx=12, pady=(2, 4))

        src = self.card(f, "目标相位")
        self.dg_task = tk.StringVar(value=TASKS[0])
        self.dg_focal = tk.DoubleVar(value=20.0)
        self.dg_l = tk.IntVar(value=6)
        self.dg_strength = tk.DoubleVar(value=10.0)
        self.dg_phase_file = tk.StringVar(value="")
        items = [("任务", ttk.Combobox(src, textvariable=self.dg_task, values=TASKS,
                                      state="readonly", width=18)),
                 ("焦距(µm)", ttk.Spinbox(src, textvariable=self.dg_focal, from_=1, to=1000, width=7)),
                 ("拓扑荷 l", ttk.Spinbox(src, textvariable=self.dg_l, from_=-20, to=20, width=5)),
                 ("Airy强度", ttk.Spinbox(src, textvariable=self.dg_strength, from_=1, to=200, width=6))]
        for i, (lab, w) in enumerate(items):
            ttk.Label(src, text=lab, style="Head.TLabel").grid(row=0, column=2 * i, padx=(10, 2), pady=6, sticky="e")
            w.grid(row=0, column=2 * i + 1, pady=6)
        self.btn_browse_phase = ttk.Button(src, text="加载外部相位图…", command=self.browse_phase)
        self.btn_browse_phase.grid(row=0, column=100, padx=16)
        ttk.Label(src, textvariable=self.dg_phase_file, width=30, style="Head.TLabel"
                  ).grid(row=0, column=101)

        opt = self.card(f, "设计参数")
        self.dg_patches = tk.IntVar(value=4)
        self.dg_iters = tk.IntVar(value=1500)
        self.dg_lr = tk.DoubleVar(value=0.1)
        self.dg_bin = tk.DoubleVar(value=0.75)
        self.dg_ramp = tk.DoubleVar(value=0.3)
        rows = [("patches(N×N)", ttk.Spinbox(opt, textvariable=self.dg_patches, from_=1, to=16, width=6)),
                ("迭代数", ttk.Spinbox(opt, textvariable=self.dg_iters, from_=10, to=20000, width=7)),
                ("Adam lr(代理)", ttk.Spinbox(opt, textvariable=self.dg_lr, from_=0.001, to=1.0, increment=0.01, width=6)),
                ("二值化起点(代理)", ttk.Spinbox(opt, textvariable=self.dg_bin, from_=0, to=20000, width=7)),
                ("ramp比例(直接)", ttk.Spinbox(opt, textvariable=self.dg_ramp, from_=0.05, to=0.9, increment=0.05, width=6))]
        for i, (lab, w) in enumerate(rows):
            ttk.Label(opt, text=lab, style="Head.TLabel").grid(row=0, column=2 * i, padx=(10, 2), pady=6, sticky="e")
            w.grid(row=0, column=2 * i + 1)

        btns = ttk.Frame(f, style="Card.TFrame")
        btns.pack(fill="x", padx=14, pady=(6, 2))
        self.btn_design = ttk.Button(btns, text="▶  开始逆向设计", style="Accent.TButton",
                                     command=self.on_design)
        self.btn_design.pack(side="left", padx=4)
        ttk.Button(btns, text="■ 停止", style="Stop.TButton",
                   command=self.stop_flag.set).pack(side="left", padx=4)
        self.btn_csv = ttk.Button(btns, text="保存结果 CSV（RI/相位/光斑/指标）",
                                  command=self.on_export_design, state="disabled")
        self.btn_csv.pack(side="left", padx=10)

        self.dg_metrics = tk.StringVar(value="— 尚未设计 —")
        mlab = tk.Label(f, textvariable=self.dg_metrics, font=FONT_B, bg="#EFF6FF",
                        fg=C_ACC_D, anchor="w", padx=12, pady=8)
        mlab.pack(fill="x", padx=14, pady=6)

        self.dg_txt = tk.Text(f, height=6, state="disabled", font=("Consolas", 9),
                              bg="#0F172A", fg="#7DD3FC", relief="flat")
        self.dg_txt.pack(fill="x", padx=14, pady=(0, 4))

        self.fig_dg = Figure(figsize=(9.4, 5.2), dpi=100, facecolor=C_CARD)
        cv = FigureCanvasTkAgg(self.fig_dg, master=f)
        cv.get_tk_widget().pack(fill="both", expand=True, padx=14, pady=(0, 10))
        f.configure(padding=0)

    def _on_mode_change(self):
        direct = self.dg_mode.get() == "direct"
        for w in self.ckpt_widgets:
            w.configure(state="disabled" if direct else "normal")

    def refresh_ckpts(self):
        cks = find_ckpts()
        self.cmb_ck["values"] = cks
        if cks and not self.dg_ckpt.get():
            for pref in ("finetune", "gui_trained", "pretrain"):
                for p in cks:
                    if pref in os.path.basename(p):
                        self.dg_ckpt.set(p)
                        break
            if not self.dg_ckpt.get():
                self.dg_ckpt.set(cks[0])

    def browse_ckpt(self):
        p = filedialog.askopenfilename(filetypes=[("checkpoint", "*.pt")])
        if p:
            self.dg_ckpt.set(p)
            vals = list(self.cmb_ck["values"])
            if p not in vals:
                self.cmb_ck["values"] = vals + [p]

    def browse_phase(self):
        p = filedialog.askopenfilename(filetypes=[
            ("相位图", "*.npy *.csv *.txt *.png *.jpg *.jpeg *.bmp"), ("所有文件", "*.*")])
        if p:
            self.dg_phase_file.set(p)
            self.dg_task.set(TASKS[3])

    def on_load_model(self):
        def job():
            from metae.utils import load_model

            p = self.dg_ckpt.get()
            if not p or not os.path.exists(p):
                raise FileNotFoundError(f"checkpoint 不存在: {p}")
            t0 = time.time()
            model, ck = load_model(p)
            self.model, self.ckpt_of_model = model, p
            self.q.put(("log", (self.dg_txt,
                        f"已加载 {os.path.basename(p)} (preset={ck.get('preset')}, step={ck.get('step','?')}) "
                        f"耗时 {time.time()-t0:.1f}s")))
        self.start_job("加载模型", job)

    def on_load_model_sync(self):
        from metae.utils import load_model

        self.refresh_ckpts()
        p = self.dg_ckpt.get()
        if not p:
            raise FileNotFoundError("没有可用的 checkpoint，请先训练或浏览选择 .pt 文件")
        model, ck = load_model(p)
        self.model, self.ckpt_of_model = model, p
        self.q.put(("log", (self.dg_txt, f"已加载 {os.path.basename(p)} (preset={ck.get('preset')})")))

    def _task_code(self):
        t = self.dg_task.get()
        if t.startswith("custom"):
            return "custom"
        return "metalens" if t.startswith("lens") else t.split(" ")[0]

    def on_design(self):
        def job():
            task = self._task_code()
            direct = self.dg_mode.get() == "direct"
            custom = task == "custom"
            if custom and not self.dg_phase_file.get():
                raise ValueError("任务为 custom 时请先加载外部相位图文件")
            if custom:
                task = "custom"
            elif task == "custom":
                task = "metalens"

            if direct:
                # ---- 物理直接设计 ----
                extra = None
                phase_map = None
                if custom:
                    phase_map = load_phase_map(self.dg_phase_file.get(), 100)
                    self.q.put(("log", (self.dg_txt, "已加载外部相位图 → 100x100（物理直接设计）")))
                else:
                    extra = (self.dg_l.get() if task == "vortex"
                             else self.dg_strength.get() if task == "airy" else None)
                outdir = os.path.join(OUT_DIR, f"gui_direct_{task}")
                self.q.put(("log", (self.dg_txt,
                            f"物理直接设计: 任务={task}"
                            + (f" l={extra}" if task == 'vortex' else
                               f" S={extra}" if task == 'airy' else
                               " (外部相位图)" if custom else "")
                            + f", {self.dg_iters.get()} iters (无需模型, 损失定义在真实传播焦面上)")))
                res = run_direct_design(
                    outdir, task, extra, iters=self.dg_iters.get(), lr=0.03,
                    binarize_from=self.dg_ramp.get(), n_pixels=100, threads=8,
                    log=lambda m: self.q.put(("log", (self.dg_txt, m))),
                    should_stop=self.stop_flag, phase_map=phase_map,
                    focal=self.dg_focal.get() * 1e-6,
                )
                self.last_res = res
                self._prepare_viz_from_direct(res)
                self.q.put(("call", self.draw_design))
                self.q.put(("call", lambda: self.btn_csv.configure(state="normal")))
                msg = (f"直接设计完成 | 二值化 {res['binariness']*100:.0f}% | 出口相位MAE {res['phase_mae']:.3f} rad "
                       f"| 优化 {res['opt_time']:.0f}s ({res['ms_per_patch_iter']:.0f} ms/iter)")
                if task == "metalens":
                    msg += (f" | FWHM {res['fwhm_x']*1e6:.2f}µm (衍射极限 {res['diff_limit']*1e6:.2f}µm) "
                            f"| 效率 {res['efficiency']*100:.0f}%")
                elif task == "vortex":
                    msg += f" | 拓扑荷 r=1.5..4µm: {', '.join(res['charges'])} (目标 {int(res['extra'])})"
                self.q.put(("call", lambda: self.dg_metrics.set(msg)))
                self.q.put(("log", (self.dg_txt, msg + f" | 结果已存 {outdir}")))
                return

            # ---- 代理模型设计 ----
            if self.model is None:
                self.q.put(("log", (self.dg_txt, "模型未加载，正在自动加载...")))
                self.on_load_model_sync()
            model = self.model
            n_p = self.dg_patches.get()
            ext = (n_p - 1) * CROP + PITCH_PX
            if custom:
                tgt = load_phase_map(self.dg_phase_file.get(), ext)
                self.q.put(("log", (self.dg_txt, f"已加载相位图并缩放到 {ext}x{ext}")))
            else:
                tgt = builtin_phase(task, ext, focal=self.dg_focal.get() * 1e-6,
                                    l=self.dg_l.get(), strength=self.dg_strength.get())
            focal = self.dg_focal.get() * 1e-6 if (task_is_lens(self.dg_task.get()) and not custom) else None
            self.q.put(("log", (self.dg_txt,
                        f"开始设计: {n_p}x{n_p} patches (口径 {n_p*4.2:.1f}µm), {self.dg_iters.get()} iters")))
            res = design_from_phase(
                model, tgt, patches=n_p, iters=self.dg_iters.get(), lr=self.dg_lr.get(),
                n1=1.0, n2=3.526, binarize_from=self.dg_bin.get(),
                focal=focal, log=lambda m: self.q.put(("log", (self.dg_txt, m))),
                should_stop=self.stop_flag, threads=8,
                pad_eval=4 if task == "airy" else 1,
            )
            self.last_res = res
            self._prepare_viz_from_surrogate(res)
            self.q.put(("call", self.draw_design))
            self.q.put(("call", lambda: self.btn_csv.configure(state="normal")))
            msg = (f"代理设计完成 | 相位MAE {res['phase_mae']:.3f} rad | RMSE {res['phase_rmse']:.3f} rad "
                   f"| 优化 {res['opt_time']:.0f}s ({res['ms_per_patch_iter']:.0f} ms/patch-iter) "
                   f"| 推理 {res['infer_time']:.2f}s")
            if focal:
                msg += (f" | FWHM {res['fwhm_x']*1e6:.2f}µm (衍射极限 {res['diff_limit']*1e6:.2f}µm) "
                        f"| 效率 {res['efficiency']*100:.0f}%")
            self.q.put(("call", lambda: self.dg_metrics.set(msg)))
            self.q.put(("log", (self.dg_txt, msg)))

        self.start_job("逆向设计", job)

    # ------------------------------------------------------------ viz state
    def _prepare_viz_from_direct(self, res):
        self.viz = dict(E_pad=torch.from_numpy(res["E_pad"]), dx=res["dx_field"],
                        stack_xz=res["stack_xz"], stack_yz=res["stack_yz"],
                        z_axis=res["z_axis"], focal_um=res["z_axis"][-1] * 1e6)

    def _prepare_viz_from_surrogate(self, res):
        E = res["amp"] * np.exp(1j * res["phase_pred"])
        g = E.shape[0]
        xz_c, yz_c, z_axis, E_pad = propagation_stacks(
            E.astype(np.complex64), g, 50e-9, 20e-6, n_z=121)
        self.viz = dict(E_pad=E_pad, dx=50e-9, stack_xz=xz_c, stack_yz=yz_c,
                        z_axis=z_axis, focal_um=20.0)

    # ------------------------------------------------------- design drawing
    def draw_design(self):
        res = self.last_res
        if not res:
            return
        self.fig_dg.clear()
        um = res["aperture"] * 1e6
        ax = self.fig_dg.add_subplot(2, 3, 1)
        ax.imshow(res["ri"], cmap="gray", vmin=res["n1"], vmax=res["n2"])
        ax.set_title("RI 折射率排布", fontsize=10)
        ax = self.fig_dg.add_subplot(2, 3, 2)
        ax.imshow(res["phase_target"], cmap="twilight", vmin=-np.pi, vmax=np.pi)
        ax.set_title("目标相位", fontsize=10)
        ax = self.fig_dg.add_subplot(2, 3, 3)
        ax.imshow(res["phase_pred"], cmap="twilight", vmin=-np.pi, vmax=np.pi)
        ax.set_title(f"实现相位 (MAE {res['phase_mae']:.3f} rad)", fontsize=10)
        ax = self.fig_dg.add_subplot(2, 3, 4)
        if "ideal_int" in res:
            um5 = res["focal_int"].shape[0] * 0.05
            ext = [-um5 / 2, um5 / 2, um5 / 2, -um5 / 2]
            ax.imshow(np.sqrt(res["ideal_int"]), cmap="inferno", extent=ext)
            ax.set_title("理想光斑 (物理模板)", fontsize=10)
            ax = self.fig_dg.add_subplot(2, 3, 5)
            ax.imshow(np.sqrt(res["focal_int"]), cmap="inferno", extent=ext)
            ttl = "设计焦面光斑"
            if res.get("task") == "metalens":
                ttl += f" (FWHM {res['fwhm_x']*1e6:.2f}µm)"
            ax.set_title(ttl, fontsize=10)
        else:
            ax.imshow(res["amp"], cmap="inferno", vmin=0, vmax=1)
            ax.set_title("|E| at z=λ", fontsize=10)
            ax = self.fig_dg.add_subplot(2, 3, 5)
            if "focal_int" in res:
                ax.imshow(np.sqrt(res["focal_int"]), cmap="inferno",
                          extent=[-um / 2, um / 2, um / 2, -um / 2])
                ax.set_title(f"焦面 (FWHM {res['fwhm_x']*1e6:.2f}µm)", fontsize=10)
            else:
                ax.imshow(np.sqrt(res["amp"]**2), cmap="inferno")
                ax.set_title("强度（自定义相位无焦面指标）", fontsize=10)
        ax = self.fig_dg.add_subplot(2, 3, 6)
        if res["loss_hist"]:
            ax.plot(res["loss_hist"], color=C_ACC)
            ax.set_xlabel("iter"); ax.grid(alpha=0.3)
            ax.set_title("设计损失", fontsize=10)
        for a in self.fig_dg.axes:
            a.set_xticks([]); a.set_yticks([]); a.set_facecolor("#F8FAFC")
        self.fig_dg.tight_layout()
        try:
            self.fig_dg.canvas.draw_idle()
        except Exception:
            pass

    def on_export_design(self):
        if not self.last_res:
            messagebox.showinfo("提示", "请先完成一次设计")
            return
        d = filedialog.askdirectory(initialdir=OUT_DIR if os.path.isdir(OUT_DIR) else APP_DIR,
                                    title="选择保存目录（将写入设计 CSV）")
        if not d:
            return
        save_design_csvs(self.last_res, d)
        self.status.set(f"CSV 已保存到 {d}")
        messagebox.showinfo("完成", f"已保存:\nri_map.csv\nphase_target.csv\nphase_pred.csv\n"
                                    f"focal_intensity.csv(若有)\ndesign_metrics.csv\ndesign_loss_curve.csv\n\n{d}")

    # ============================================================ 场可视化 tab
    def _build_viz_tab(self):
        f = self.tab_viz
        ctl = self.card(f, "视图控制（数据来自最近一次设计结果；先到 ② 页完成一次设计）")
        self.vz_view = tk.StringVar(value=VIEWS[0])
        self.vz_qty = tk.StringVar(value=QUANTS[0])
        self.vz_z = tk.DoubleVar(value=20.0)
        for i, v in enumerate(VIEWS):
            ttk.Radiobutton(ctl, text=v, variable=self.vz_view, value=v,
                            style="Accent.TRadiobutton", command=self.draw_viz
                            ).grid(row=0, column=i, padx=(6, 16), pady=4, sticky="w")
        ttk.Label(ctl, text="物理量", style="Head.TLabel").grid(row=0, column=10, padx=(16, 2))
        cb_qty = ttk.Combobox(ctl, textvariable=self.vz_qty, values=QUANTS, state="readonly", width=12)
        cb_qty.grid(row=0, column=11, pady=4)
        cb_qty.bind("<<ComboboxSelected>>", lambda e: self.draw_viz())
        ttk.Label(ctl, text="z (µm)", style="Head.TLabel").grid(row=0, column=12, padx=(16, 2))
        self.vz_zspin = ttk.Spinbox(ctl, textvariable=self.vz_z, from_=0.0, to=60.0,
                                    increment=0.5, width=7, command=self.draw_viz)
        self.vz_zspin.grid(row=0, column=13, pady=4)
        ttk.Button(ctl, text="↻ 刷新", style="Accent.TButton",
                   command=self.draw_viz).grid(row=0, column=14, padx=14)
        ttk.Button(ctl, text="导出当前视图 PNG", command=self.on_viz_png).grid(row=0, column=15, padx=2)
        ttk.Button(ctl, text="导出数据 CSV", command=self.on_viz_csv).grid(row=0, column=16, padx=2)
        self.vz_scale = tk.StringVar(value="linear")
        ttk.Label(ctl, text="强度标尺", style="Head.TLabel").grid(row=0, column=17, padx=(16, 2))
        cb_scale = ttk.Combobox(ctl, textvariable=self.vz_scale, values=["linear", "log"],
                                state="readonly", width=7)
        cb_scale.grid(row=0, column=18)
        cb_scale.bind("<<ComboboxSelected>>", lambda e: self.draw_viz())

        self.viz_info = tk.StringVar(value="— 暂无设计结果 —")
        tk.Label(f, textvariable=self.viz_info, font=FONT_S, bg="#F1F5F9", fg=C_MUT,
                 anchor="w", padx=12, pady=4).pack(fill="x", padx=14, pady=(4, 0))

        self.fig_vz = Figure(figsize=(7.6, 5.6), dpi=100, facecolor=C_CARD)
        cv = FigureCanvasTkAgg(self.fig_vz, master=f)
        cv.get_tk_widget().pack(fill="both", expand=True, padx=14, pady=(4, 10))

    def draw_viz(self):
        if not self.viz:
            return
        v = self.viz
        view, qty = self.vz_view.get(), self.vz_qty.get()
        dx = v["dx"]
        self.fig_vz.clear()
        ax = self.fig_vz.add_subplot(111)

        if view == VIEWS[0]:  # xy 切片
            z = float(self.vz_z.get()) * 1e-6
            with torch.no_grad():
                E = angular_spectrum_propagate(v["E_pad"], z, dx).numpy()
            n = E.shape[0]
            ext_um = [-n * dx * 1e6 / 2, n * dx * 1e6 / 2] * 2
            m = _qty(E)
            data = m[qty]
            if qty == QUANTS[0] and self.vz_scale.get() == "log":
                data = np.log10(data + 1e-12)
                ax.set_title(_safe(f"xy @ z={self.vz_z.get():.1f}um  {qty} (log10)"), fontsize=11)
            else:
                ax.set_title(_safe(f"xy @ z={self.vz_z.get():.1f}um  {qty}"), fontsize=11)
            cmap = "twilight" if qty == QUANTS[4] else ("RdBu_r" if qty in (QUANTS[2], QUANTS[3]) else "inferno")
            vmax = 1 if qty == QUANTS[1] else None
            im = ax.imshow(data, cmap=cmap, extent=ext_um, origin="upper",
                           vmin=-np.pi if qty == QUANTS[4] else None,
                           vmax=np.pi if qty == QUANTS[4] else vmax)
            ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
            self._cur_csv = data
            self.fig_vz.colorbar(im, ax=ax, shrink=0.85)
        else:  # xz / yz 剖面
            stack = v["stack_xz"] if view == VIEWS[1] else v["stack_yz"]
            m = _qty(stack)
            data = m[qty]
            if qty == QUANTS[0] and self.vz_scale.get() == "log":
                data = np.log10(data + 1e-12)
            n = stack.shape[1]
            z_um = v["z_axis"] * 1e6
            ext = [-n * dx * 1e6 / 2, n * dx * 1e6 / 2, z_um[-1], 0]
            cmap = "twilight" if qty == QUANTS[4] else ("RdBu_r" if qty in (QUANTS[2], QUANTS[3]) else "inferno")
            im = ax.imshow(data, cmap=cmap, extent=ext, aspect="auto",
                           vmin=-np.pi if qty == QUANTS[4] else None,
                           vmax=np.pi if qty == QUANTS[4] else None)
            ax.set_title(_safe(f"{view.split(' ')[0]}  {qty}") + (" (log10)" if "log" in self.vz_scale.get() and qty == QUANTS[0] else ""), fontsize=11)
            ax.set_xlabel("x (um)" if view == VIEWS[1] else "y (um)")
            ax.set_ylabel("z (µm)")
            self._cur_csv = data
            self.fig_vz.colorbar(im, ax=ax, shrink=0.85)
        ax.set_facecolor("#F8FAFC")
        self.fig_vz.tight_layout()
        try:
            self.fig_vz.canvas.draw_idle()
        except Exception:
            pass

    def on_viz_png(self):
        if not self.viz:
            messagebox.showinfo("提示", "暂无设计结果，请先到 ② 页完成一次设计")
            return
        p = filedialog.asksaveasfilename(defaultextension=".png", initialfile="field_view.png",
                                         filetypes=[("PNG", "*.png")])
        if not p:
            return
        self.fig_vz.savefig(p, dpi=200, facecolor="white")
        self.status.set(f"已保存 {p}")

    def on_viz_csv(self):
        if not getattr(self, "_cur_csv", None):
            messagebox.showinfo("提示", "暂无视图数据，请先刷新一次视图")
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="field_data.csv",
                                         filetypes=[("CSV", "*.csv")])
        if not p:
            return
        save_matrix_csv(self._cur_csv, p, f"# {self.vz_view.get()} · {self.vz_qty.get()}")
        self.status.set(f"已保存 {p}")

    # ============================================================ 验证 tab
    def _build_eval_tab(self):
        f = self.tab_eval
        top = self.card(f, "模型验证（对随机/岛状/条纹三类测试样本预测场，统计归一化 MAE）")
        self.ev_n = tk.IntVar(value=32)
        ttk.Label(top, text="每类样本数", style="Head.TLabel").pack(side="left", padx=(6, 4))
        ttk.Spinbox(top, textvariable=self.ev_n, from_=4, to=512, width=7).pack(side="left", padx=2)
        self.btn_eval = ttk.Button(top, text="▶  开始验证", style="Accent.TButton",
                                   command=self.on_eval)
        self.btn_eval.pack(side="left", padx=14)
        ttk.Button(top, text="■ 停止", style="Stop.TButton", command=self.stop_flag.set).pack(side="left")
        self.btn_evalcsv = ttk.Button(top, text="保存验证指标 CSV", command=self.on_export_eval,
                                      state="disabled")
        self.btn_evalcsv.pack(side="left", padx=14)

        self.ev_txt = tk.Text(f, height=8, state="disabled", font=("Consolas", 9),
                              bg="#0F172A", fg="#7DD3FC", relief="flat")
        self.ev_txt.pack(fill="x", padx=14, pady=6)

        self.fig_ev = Figure(figsize=(6.6, 4.2), dpi=100, facecolor=C_CARD)
        cv = FigureCanvasTkAgg(self.fig_ev, master=f)
        cv.get_tk_widget().pack(fill="both", expand=True, padx=14, pady=(0, 10))

    def on_eval(self):
        def job():
            if self.model is None:
                self.on_load_model_sync()
            rows = evaluate_model(
                self.model, n_per=self.ev_n.get(),
                log=lambda m: self.q.put(("log", (self.ev_txt, m))),
                should_stop=self.stop_flag, threads=8,
            )
            self.last_eval = rows
            self.q.put(("call", self.draw_eval))
            self.q.put(("call", lambda: self.btn_evalcsv.configure(state="normal")))

        self.start_job("模型验证", job)

    def draw_eval(self):
        rows = self.last_eval
        if not rows:
            return
        self.fig_ev.clear()
        fams = [r for r in rows if r["family"] != "overall"]
        x = np.arange(len(fams))
        w = 0.27
        ax = self.fig_ev.add_subplot(111)
        for k, (key, lab, c) in enumerate([("amp_mae", "amplitude MAE", C_ACC),
                                           ("angle_mae", "angle MAE", C_OK),
                                           ("norm_mae", "normalized MAE", C_WARN)]):
            ax.bar(x + (k - 1) * w, [r[key] for r in fams], w, label=lab, color=c)
        ax.set_xticks(x, [r["family"] for r in fams])
        ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
        ax.set_facecolor("#F8FAFC")
        ax.set_title("held-out test metrics", fontsize=10)
        self.fig_ev.tight_layout()
        try:
            self.fig_ev.canvas.draw_idle()
        except Exception:
            pass

    def on_export_eval(self):
        if not self.last_eval:
            messagebox.showinfo("提示", "请先运行验证")
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="eval_metrics.csv",
                                         filetypes=[("CSV", "*.csv")])
        if not p:
            return
        save_rows_csv(self.last_eval, p)
        self.status.set(f"已保存 {p}")


def task_is_lens(task_label: str) -> bool:
    return task_label.startswith("lens") or task_label.startswith("vortex")


def find_ckpts():
    seen, out = set(), []
    for d in CKPT_DIRS:
        if os.path.isdir(d):
            for f_ in sorted(os.listdir(d)):
                if f_.endswith(".pt"):
                    p = os.path.join(d, f_)
                    if p not in seen:
                        seen.add(p)
                        out.append(p)
    return out


# ================================================================== selftest
def run_selftest() -> int:
    """Headless end-to-end test of everything the GUI does; writes CSVs."""
    ok = True
    out = os.path.join(APP_DIR, "outputs", "selftest")

    print("[selftest] direct design (physics) ...", flush=True)
    res = run_direct_design(os.path.join(out, "direct"), "metalens", None, iters=8,
                            n_pixels=25, threads=8, log=print)
    save_design_csvs(res, os.path.join(out, "direct"))
    ok &= np.isfinite(res["phase_mae"]) and os.path.exists(
        os.path.join(out, "direct", "design_metrics.csv"))
    assert res["stack_xz"].shape[0] > 1 and res["E_pad"].shape == (200, 200)
    # custom phase map path
    rng = np.random.default_rng(0)
    pm = (rng.random((40, 40)) * 2 - 1) * np.pi
    resc = run_direct_design(os.path.join(out, "direct_custom"), "custom", None, iters=6,
                             n_pixels=25, threads=8, log=print, phase_map=pm)
    ok &= np.isfinite(resc["phase_mae"])
    print(f"[selftest] direct binariness={res['binariness']:.2f} "
          f"phaseMAE={res['phase_mae']:.3f} | custom phaseMAE={resc['phase_mae']:.3f}", flush=True)

    print("[selftest] train ...", flush=True)
    hist, ck = train_model(preset="tiny", steps=12, batch=4, ckpt_out="checkpoints/selftest.pt",
                           log=print, threads=8)
    print("[selftest] surrogate design ...", flush=True)
    from metae.utils import load_model

    model, _ = load_model(ck)
    n_p = 2
    ext = (n_p - 1) * CROP + PITCH_PX
    tgt = builtin_phase("lens", ext, focal=20e-6)
    res = design_from_phase(model, tgt, patches=n_p, iters=10, focal=20e-6, log=print, threads=8)
    save_design_csvs(res, out)
    print("[selftest] evaluate ...", flush=True)
    rows = evaluate_model(model, n_per=2, log=print, threads=8)
    save_rows_csv(rows, os.path.join(out, "eval_metrics.csv"))
    save_matrix_csv(res["ri"], os.path.join(out, "ri_check.csv"))
    ok &= (os.path.exists(os.path.join(out, "design_metrics.csv"))
           and np.isfinite(res["phase_mae"]))
    print(f"[selftest] {'OK' if ok else 'FAILED'} -> {out}", flush=True)
    return 0 if ok else 1


def main():
    if _run_selftest_later:
        sys.exit(run_selftest())
    root = tk.Tk()
    app = App(root)
    app.refresh_ckpts()
    app._on_mode_change()

    def on_close():
        app.stop_flag.set()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
