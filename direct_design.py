# -*- coding: utf-8 -*-
"""可微物理直接逆向设计（演示规模全口径，无 patch、无代理）：

1. 物理初始化：目标相位 -> 折射率 n(x) = 1 + wrap(phi)/(k0*h)（柱内相位累积公式反解）
2. Adam 直接优化整片 RI，损失定义在真实 BPM 传播后的焦面上（任务专属理想光斑）
3. 二值化 ramp（Eq.4）推向可制造双值结构
4. 产物用同一物理前向验证 -> result_fig.png

用法: python direct_design.py <outdir> <task> [extra]
  metalens: python direct_design.py outputs/direct_metalens metalens
  vortex:   python direct_design.py outputs/direct_vortex6 vortex 6
  airy:     python direct_design.py outputs/direct_airy airy 10.0
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metae.solver import SplitStepBPM, angular_spectrum_propagate, N_LO, N_HI
from metae.tasks import build_target_phase, WL

DX = 50e-9
FOCAL = 20e-6
N_BG = 1.0
LAM = WL
H = 800e-9


# ================================================================ core API
def run_direct_design(outdir, task="metalens", extra=None, iters=2500, lr=0.03,
                      binarize_from=0.35, n_pixels=100, threads=8,
                      log=None, should_stop=None, phase_map=None, focal=FOCAL,
                      field_dx=DX, airy_apo=0.0):
    """全口径可微物理直接逆向设计。返回结果 dict（与 GUI / CSV 导出兼容）。

    task: metalens / vortex / airy / custom;  extra: vortex 拓扑荷 l 或 airy 强度 S
    phase_map: task=custom 时的目标相位 (n_pixels×n_pixels, rad)；提供时忽略 extra
    focal: 焦距/观察距离（米）；field_dx: 场网格采样（默认 50nm，大口径可用 100nm 提速）
    """
    def _log(m):
        if log:
            log(m)

    torch.set_num_threads(threads)
    torch.manual_seed(0)
    os.makedirs(outdir, exist_ok=True)

    n_g = n_pixels
    grid = int(round(n_g * 200e-9 / field_dx))
    aperture = n_g * 200e-9
    l_exp = int(extra) if task == "vortex" else None
    s_str = extra if task == "airy" else None
    k0 = 2 * np.pi / LAM

    # ---------------------------------------------------------------- 目标
    if task == "custom" and phase_map is not None:
        from torch.nn.functional import interpolate
        pm = torch.from_numpy(np.asarray(phase_map, dtype=np.float32))[None, None]
        phi_ri = interpolate(pm, size=(n_g, n_g), mode="bilinear",
                             align_corners=False)[0, 0].numpy()
        phi_grid = interpolate(pm, size=(grid, grid), mode="bilinear",
                               align_corners=False)[0, 0].numpy()
    else:
        c_ri = (np.arange(n_g) - (n_g - 1) / 2) * 200e-9
        tk = "lens" if task == "metalens" else task
        task_kw = ({"l": l_exp} if task == "vortex" else
                   {"aperture_m": aperture / 2, "strength": s_str} if task == "airy" else {})
        phi_ri = build_target_phase(tk, n_pix=n_g, dx=200e-9, focal=focal, n_bg=N_BG,
                                    **task_kw).numpy()
        phi_grid = build_target_phase(tk, n_pix=grid, dx=field_dx, focal=focal, n_bg=N_BG,
                                      **task_kw).numpy()
    task_kw = task_kw if (task != "custom" and phase_map is None) else {}

    # 焦面理想光斑：物理一致数值模板 —— 理想相位元件(平面波×目标相位)经同一 AS 传播
    # 得到观察面光斑，不做夫琅禾费近似。
    big_eval = grid * 2
    c_m = (np.arange(big_eval) - big_eval / 2) * field_dx
    Xg, Yg = np.meshgrid(c_m, c_m, indexing="ij")
    E_id = np.zeros((big_eval, big_eval), dtype=np.complex64)
    o1 = (big_eval - grid) // 2
    ax1 = (np.arange(grid) - grid / 2) * field_dx
    if task == "airy":
        # 方形孔径：保持 x/y 立方相位可分离，Airy 瓣阵为分立方格
        # （圆孔会破坏可分离性，在主瓣 +x/+y 侧产生连续亮条伪影）
        ap_mask = ((np.abs(ax1[:, None]) <= aperture / 2 + 1e-12)
                   & (np.abs(ax1[None, :]) <= aperture / 2 + 1e-12))
    else:
        ap_mask = (np.hypot(ax1[:, None], ax1[None, :]) <= aperture / 2 + 1e-12)
    elem = np.exp(1j * phi_grid) * ap_mask
    if task == "airy" and airy_apo > 0:
        # 对角切趾（论文式衰减艾里串）：入射振幅沿 +x+y 指数衰减，传播后头在
        # 显示左上、沿对角衰减；rot180 与 AS 传播严格对易，用于摆正头的方位
        apo = np.exp(+airy_apo * (ax1[:, None] + ax1[None, :]) * 1e6 / 2)
        elem = elem * apo
        elem = np.rot90(elem, 2)
    E_id[o1:o1 + grid, o1:o1 + grid] = elem.astype(np.complex64)
    with torch.no_grad():
        Ef_id = angular_spectrum_propagate(torch.from_numpy(E_id), focal - LAM, field_dx, n_bg=N_BG)
        I_ideal = (Ef_id.abs() ** 2).numpy().astype(np.float64)
    I_ideal /= I_ideal.sum() / grid ** 2     # 功率定标（与结构焦面总功率一致）
    I_ideal_t = torch.from_numpy(I_ideal.astype(np.float32))

    if task == "vortex":
        rr = np.hypot(Xg, Yg)
        r_max = grid * field_dx / 2 - field_dx          # 不超出网格半径
        bins = np.arange(0, min(8e-6, r_max), field_dx)
        idx = np.digitize(rr.ravel(), bins)
        prof_r = np.array([I_ideal.ravel()[idx == k].mean() if np.any(idx == k) else 0.0
                           for k in range(1, len(bins))])
        r0 = bins[int(np.argmax(prof_r))] + field_dx / 2
        _log(f"[ideal] vortex ring peak radius = {r0*1e6:.2f} um")

    # ---------------------------------------------------------------- 求解器
    sol = SplitStepBPM(grid=grid, dx=field_dx)

    def focal_intensity(n_abs):
        E = sol(n_abs.unsqueeze(0), return_complex=True)[0]
        # L2 定标：入射平面波 L2 = grid，功率守恒，杜绝能量缩放作弊
        E = E * (grid / (E.abs().pow(2).sum().sqrt() + 1e-12))
        big = torch.zeros(grid * 2, grid * 2, dtype=torch.complex64)
        o = grid // 2
        big[o:o + grid, o:o + grid] = E
        Ef = angular_spectrum_propagate(big, focal - LAM, field_dx, n_bg=N_BG)
        return E, Ef

    # ---------------------------------------------------------------- 初始化
    n_init = 1.0 + (np.mod(phi_ri + np.pi, 2 * np.pi)) / (k0 * H)
    n_init = np.clip(n_init, N_LO + 0.02, N_HI - 0.02)
    v0 = (n_init - N_LO) / (N_HI - N_LO) * 2 - 1
    v = torch.atanh(torch.from_numpy(v0 * 0.98).float()).requires_grad_(True)
    _log(f"[init] phase-informed RI in [{n_init.min():.2f},{n_init.max():.2f}]")

    opt = torch.optim.Adam([v], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)
    n_start = int(binarize_from * iters)
    BETA = 5.0
    loss_hist = []

    t0 = time.time()
    for it in range(1, iters + 1):
        if should_stop is not None and should_stop.is_set():
            _log("[stop] 用户中断，保存当前结果")
            break
        beta = 0.0
        if it > n_start:
            beta = BETA * (it - n_start) / max(1, iters - n_start)
        n_abs = N_LO + (N_HI - N_LO) * (torch.tanh(v) + 1) / 2
        if it == n_start + 1:
            ri_cont = n_abs.detach().numpy().copy()   # ramp 起点 = 连续(未二值化)设计
        E, Ef = focal_intensity(n_abs)
        If = Ef.abs() ** 2
        l_int = ((If - I_ideal_t) ** 2).mean()   # 全窗功率守恒比较，无归一化自由度
        loss = l_int + beta * (1 - torch.tanh(v) ** 2).mean()
        if task == "vortex":
            big2 = grid * 2
            r_px = r0 / field_dx
            ang = np.linspace(0, 2 * np.pi, 720, endpoint=False)
            xs = ((big2 // 2 + r_px * np.cos(ang)) % big2).astype(int)
            ys = ((big2 // 2 + r_px * np.sin(ang)) % big2).astype(int)
            ph_ring = torch.angle(Ef[ys, xs])
            l_wind = ((1 - torch.cos(ph_ring - l_exp * torch.from_numpy(ang).float())) / 2).mean()
            loss = loss + 0.5 * l_wind
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        loss_hist.append(float(l_int.item()))
        if it % 100 == 0 or it == 1:
            _log(f"  iter {it:4d} | int {l_int.item():.4f} | beta {beta:.2f} | {time.time()-t0:.0f}s")

    n_abs = N_LO + (N_HI - N_LO) * (torch.tanh(v) + 1) / 2
    ri_final = n_abs.detach().numpy()
    binr = ((ri_final < 1.3) | (ri_final > 3.2)).mean()
    opt_time = time.time() - t0
    _log(f"[opt] done {iters} iters in {opt_time:.0f}s, binariness {binr:.2f}")

    with torch.no_grad():
        E, Ef = focal_intensity(torch.from_numpy(ri_final))
    I_f = (Ef.abs() ** 2).numpy()
    I_f /= I_f.max()
    # xz / yz 传播栈（可视化验证用）
    xz_c, yz_c, z_axis, E_pad = propagation_stacks(E.numpy(), grid, field_dx, focal)
    # 连续(未二值化)设计的物理场（对标论文 Fig.6 左列）
    ri_cont_res = None
    if binarize_from < 1.0:
        with torch.no_grad():
            E_ct, Ef_ct = focal_intensity(torch.from_numpy(ri_cont))
        I_ct = (Ef_ct.abs() ** 2).numpy()
        I_ct /= I_ct.max()
        xz_ct, yz_ct, _, E_pad_ct = propagation_stacks(E_ct.numpy(), grid, field_dx, focal)
        ri_cont_res = dict(ri=ri_cont, I_focal=I_ct, E_pad=E_pad_ct.numpy(),
                           stack_xz=xz_ct, stack_yz=yz_ct, E_exit=E_ct.numpy())
    np.savez_compressed(os.path.join(outdir, "refined.npz"),
                        ri=ri_final, task=task, extra=extra,
                        I_ideal=I_ideal, I_focal=I_f, E_exit=E.numpy(),
                        E_pad=E_pad.numpy(), stack_xz=xz_c, stack_yz=yz_c,
                        z_axis=z_axis,
                        ri_cont=ri_cont_res["ri"] if ri_cont_res else np.zeros((1, 1)),
                        I_focal_cont=ri_cont_res["I_focal"] if ri_cont_res else np.zeros((1, 1)),
                        E_pad_cont=ri_cont_res["E_pad"] if ri_cont_res else np.zeros((1, 1), np.complex64),
                        stack_xz_cont=ri_cont_res["stack_xz"] if ri_cont_res else np.zeros((1, 1), np.complex64),
                        stack_yz_cont=ri_cont_res["stack_yz"] if ri_cont_res else np.zeros((1, 1), np.complex64),
                        meta=dict(iters=iters, lr=lr, opt_time=opt_time, field_dx=field_dx,
                                  loss=float(l_int.item()), binariness=float(binr)))
    _log(f"[saved] {outdir}/refined.npz")

    # 物理验证指标：出口相位 MAE（相对目标相位）
    ph_exit = np.angle(E.numpy())
    if task == "custom" and phase_map is not None:
        ph_tgt = phi_grid
    else:
        ph_tgt = build_target_phase(tk, n_pix=grid, dx=field_dx, focal=focal, n_bg=N_BG,
                                    **task_kw).numpy()
    err = np.angle(np.exp(1j * (ph_exit - ph_tgt)))
    ph_mae = float(np.abs(err).mean())

    na = np.sin(np.arctan(aperture / 2 / focal))
    res = dict(ri=ri_final, phase_target=ph_tgt, phase_pred=ph_exit,
               phase_mae=ph_mae, phase_rmse=float(np.sqrt((err ** 2).mean())),
               final_phase_loss=float(l_int.item()), aperture=aperture, na=na,
               focal_int=I_f, ideal_int=I_ideal, loss_hist=loss_hist,
               opt_time=opt_time, infer_time=0.0, ms_per_patch_iter=opt_time / max(1, iters) * 1000,
               n1=N_LO, n2=N_HI, binariness=float(binr), task=task, extra=extra,
               iters=iters, amp=np.abs(E.numpy()), dx_field=field_dx,
               stack_xz=xz_c, stack_yz=yz_c, z_axis=z_axis, E_pad=E_pad.numpy())
    if ri_cont_res is not None:
        res["cont"] = ri_cont_res
    if task == "metalens":
        # FWHM / 效率：焦面中心剖面
        c = I_f.shape[0] // 2
        row = I_f[c]
        half = row.max() / 2
        xs_above = np.where(row >= half)[0]
        fwhm = (xs_above[-1] - xs_above[0]) * field_dx * 1e6 if len(xs_above) else float("nan")
        res["fwhm_x"] = fwhm / 1e6
        res["fwhm_y"] = fwhm / 1e6
        res["diff_limit"] = 0.51 * LAM / na
        half_w = min(150, I_f.shape[0] // 2 - 1)      # ±7.5µm 收集窗（网格小时自适应）
        win = I_f[c - half_w:c + half_w + 1, c - half_w:c + half_w + 1]
        res["efficiency"] = float(win.sum() / I_f.sum())
    if task == "vortex":
        # 拓扑荷：绕环心（光轴）测缠绕
        ph_f = np.angle(Ef.numpy())
        charges = []
        cx = cy = grid
        for r_um in (1.5, 2.0, 2.5, 3.0, 4.0):
            r_px = r_um * 1e-6 / field_dx
            ang = np.linspace(0, 2 * np.pi, 720, endpoint=False)
            xs = (cx + r_px * np.cos(ang)).astype(int) % (grid * 2)
            ys = (cy + r_px * np.sin(ang)).astype(int) % (grid * 2)
            ch = np.unwrap(ph_f[ys, xs])
            charges.append(ch[-1] / (2 * np.pi))
        res["charges"] = [f"{c:+.2f}" for c in charges]
    return res


# ================================================================ field stacks
def propagation_stacks(E_exit, grid, dx, z_max, n_z=161):
    """出口复场 -> xz / yz 中心剖面传播栈。返回 (xz, yz, z_axis, E_padded)。

    xz: (n_z, grid*2) 沿 y=中心 的 |E| 相空间剖面；yz 同理；z 从 0（出口面）到 z_max。
    """
    big = torch.zeros(grid * 2, grid * 2, dtype=torch.complex64)
    o = grid // 2
    big[o:o + grid, o:o + grid] = torch.from_numpy(np.asarray(E_exit, dtype=np.complex64))
    zs = np.linspace(0.0, z_max, n_z).astype(np.float32)
    xz = np.zeros((n_z, grid * 2), dtype=np.complex64)
    yz = np.zeros((n_z, grid * 2), dtype=np.complex64)
    step = float(zs[1] - zs[0])
    with torch.no_grad():
        cur = big.clone()
        for k in range(n_z):
            xz[k] = cur[grid, :].numpy()
            yz[k] = cur[:, grid].numpy()
            if k < n_z - 1:
                cur = angular_spectrum_propagate(cur, step, dx)
    return xz, yz, zs, big


# ================================================================ CLI
def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("task", choices=["metalens", "vortex", "airy", "custom"])
    ap.add_argument("extra", nargs="?", type=float, default=None)
    ap.add_argument("--iters", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--binarize-from", type=float, default=0.35)
    ap.add_argument("--n-pixels", type=int, default=100)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--focal", type=float, default=FOCAL)
    ap.add_argument("--field-dx", type=float, default=DX,
                    help="场网格采样 (m)；大口径建议 100e-9 提速")
    ap.add_argument("--phase", type=str, default=None,
                    help="task=custom 时的目标相位图 (.npy/.csv/.png, 弧度或灰度)")
    args = ap.parse_args()
    phase_map = None
    if args.task == "custom":
        if not args.phase:
            ap.error("task=custom 需要 --phase <相位图文件>")
        from metae.pipeline import load_phase_map
        phase_map = load_phase_map(args.phase, args.n_pixels)
    res = run_direct_design(args.outdir, args.task, args.extra, args.iters, args.lr,
                            args.binarize_from, args.n_pixels, args.threads, log=print,
                            phase_map=phase_map, focal=args.focal, field_dx=args.field_dx)
    if res["task"] == "metalens":
        print(f"FWHM {res['fwhm_x']*1e6:.2f}um (diff limit {res['diff_limit']*1e6:.2f}um) "
              f"| eff {res['efficiency']*100:.0f}% | phase MAE {res['phase_mae']:.3f} rad")
    elif res["task"] == "vortex":
        print(f"charges r=1.5..4um: {', '.join(res['charges'])} (target {int(res['extra'])})")
    else:
        print(f"phase MAE {res['phase_mae']:.3f} rad")


if __name__ == "__main__":
    _cli()
