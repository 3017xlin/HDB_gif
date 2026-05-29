#!/usr/bin/env python3
"""
plot_pressure_from_openfoam.py

直接读 OpenFOAM 算例, 用 ParaView 式真切割(vol.slice + tripcolor)渲染压力切片, 出 GIF。
两个版本:
  fixed : 固定 [-512,512]^2 方块, 所有 case 一致。
  zoom  : 迎风前 >=2 排的稳健窗口, 边界保证不穿过任何建筑(找街缝 + 栅格验证 + 安全兜底)。
每个版本各出 real 和 detrend。

渲染:
  vol.slice(normal='z') 做真几何切割 -> triangulate -> tripcolor(Gouraud) -> 楼边干脆、场平滑;
  楼 = 切割留下的空洞 -> 背景填灰。
压力 = OpenFOAM 运动学压力 p * rho (Pa)。
detrend 默认减该层均值(等价于去掉垂直背景); 若确认是无量纲 p_train 可用 --detrend_gz 改成减 c*z。

输出结构(OUT_DIR 下):
  <id>/<version>/<real|detrend>/frame_XXXX_z..png     (18 个 case 文件夹, 放每帧)
  <version>_<real|detrend>/<id>.gif                    (4 个 GIF 文件夹)
其中 <id> = case 的四位数字(如 1546), <version> = fixed/zoom。

依赖: pyvista, numpy, scipy, matplotlib, imageio
用法:
  python plot_pressure_from_openfoam.py --inspect      # 只跑第一个 case, 出它全部 GIF
  python plot_pressure_from_openfoam.py                # 全部 case
"""

import os, re, gc, time, argparse, warnings
os.environ["OMP_NUM_THREADS"] = "4"
warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path
from scipy import ndimage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import Normalize
import imageio.v2 as imageio
import pyvista as pv

try:                                   # 关掉 OpenFOAM 多面体切片的无害刷屏警告
    import vtk
    vtk.vtkObject.GlobalWarningDisplayOff()
except Exception:
    pass


# ── 配置 ──────────────────────────────────────────────────────────────────────
BASE    = Path(r"C:\card_v2")
OUT_DIR = None                  # 默认 BASE/"figures"

RHO     = 1.225
P_FIELD = "p"
G = 9.81                        # detrend = (p_kin + G*z)*rho, 抵消运动学静压 -g·z 趋势

FIXED_HALF  = 512.0             # fixed 版方块 [-512,512]^2
REGION_HALF = 560.0             # 建筑都在这之内, 检测/裁剪都在此区域

DZ            = 5.6             # 帧 z 间隔
GROUND_OFFSET = 1.0
HEIGHT_DZ     = 6.0             # 测楼高的竖向扫描步长
MAX_BUILDING_H = 200.0
DETECT_N      = 240             # 检测栅格
MIN_DET_PX    = 5

# zoom 参数
N_ROWS      = 2                 # 迎风前几排
ROW_GAP     = 28.0             # 沿风向: 质心间距 < 此值算同一排 (m)
FRONT_MARGIN = 30.0            # 迎风边往上游开阔地外扩
BACK_GAP     = 22.0            # 背风边放在下游这么远的缝里
SIDE_MARGIN  = 25.0            # 两侧外扩
EDGE_MARGIN  = 5.0            # 验证: 边离楼至少这么远 (m)
REPAIR_STEP  = 3.0
REPAIR_ITERS = 90

CLIP_REAL = 0.5
CLIP_DP   = 99.0

CMAP          = "RdBu_r"
BUILDING_GREY = "#cfcfcf"
BFACE = "#F5C242"; BEDGE = "#2D2D2D"
FRAME_MS  = 1000; DPI = 130; FIG_W = 12; FIG_H = 10

VERSIONS = ["fixed", "zoom"]
KINDS    = ["real", "detrend"]

WIND = {"N": {"blow": (0, -1), "frm": (0,  1)},
        "S": {"blow": (0,  1), "frm": (0, -1)},
        "E": {"blow": (-1, 0), "frm": (1,  0)},
        "W": {"blow": (1,  0), "frm": (-1, 0)}}


# ── OpenFOAM ──────────────────────────────────────────────────────────────────
def find_case_dir(root, depth=0):
    if (root / "system").is_dir() and (root / "constant").is_dir():
        return root
    if depth > 3:
        return None
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        r = find_case_dir(sub, depth + 1)
        if r:
            return r
    return None


def parse_wind(name):
    m = re.search(r"_([NSEW])[-_]", name)
    if not m:
        return None, None, None
    d = m.group(1).upper()
    return d, WIND[d]["blow"], WIND[d]["frm"]


def case_id(name):
    m = re.search(r"case_HDB_(\d{4})", name)
    return m.group(1) if m else re.sub(r"-\d{8}T.*$", "", name)


def read_openfoam_volume(case_dir):
    foam = case_dir / f"{case_dir.name}.foam"
    if not foam.exists():
        foam.write_text("")
    reader = pv.get_reader(str(foam))
    times = list(getattr(reader, "time_values", []) or [])
    used_t = None
    if times:
        used_t = times[-1]; reader.set_active_time_value(used_t)
    if hasattr(reader, "cell_to_point_creation"):
        reader.cell_to_point_creation = True
    multi = reader.read()
    vol = None
    try:
        if "internalMesh" in multi.keys():
            vol = multi["internalMesh"]
    except Exception:
        pass
    if vol is None:
        for blk in multi:
            if isinstance(blk, pv.UnstructuredGrid) and blk.n_cells > 0:
                vol = blk; break
    if vol is None or vol.n_cells == 0:
        raise RuntimeError("没找到 internalMesh")
    if P_FIELD not in vol.point_data:
        vol = vol.cell_data_to_point_data()
    if P_FIELD not in vol.point_data:
        raise RuntimeError(f"网格里没有 '{P_FIELD}'")
    return vol, used_t


# ── 建筑检测(栅格) ───────────────────────────────────────────────────────────
def interior_voids(valid):
    inv = ~valid
    lab, _ = ndimage.label(inv)
    border = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    border.discard(0)
    return (inv & ~np.isin(lab, list(border))) if border else inv


def probe_valid(vol, z, region, n):
    (rx0, rx1), (ry0, ry1) = region
    dx = (rx1 - rx0) / (n - 1); dy = (ry1 - ry0) / (n - 1)
    probe = pv.ImageData(dimensions=(n, n, 1), spacing=(dx, dy, 1.0), origin=(rx0, ry0, z))
    s = probe.sample(vol)
    return np.asarray(s["vtkValidPointMask"]).reshape(n, n).astype(bool)


def scan_buildings(vol, b, region):
    """地面 footprint + 竖向扫描得每栋楼顶高。返回 buildings 列表 + 地面占据栅格。"""
    (rx0, rx1), (ry0, ry1) = region
    z0 = b[4] + GROUND_OFFSET
    ground = interior_voids(probe_valid(vol, z0, region, DETECT_N))
    lab, n = ndimage.label(ground)
    dx = (rx1 - rx0) / (DETECT_N - 1); dy = (ry1 - ry0) / (DETECT_N - 1)
    blds, fps = [], []
    for L in range(1, n + 1):
        fp = (lab == L)
        if fp.sum() < MIN_DET_PX:
            continue
        ys, xs = np.where(fp)
        wx = rx0 + xs * dx; wy = ry0 + ys * dy
        blds.append({"cx": float(wx.mean()), "cy": float(wy.mean()),
                     "x0": float(wx.min()), "x1": float(wx.max()),
                     "y0": float(wy.min()), "y1": float(wy.max()),
                     "area": int(fp.sum()), "top": z0})
        fps.append(fp)
    z = z0 + HEIGHT_DZ
    zmax = min(b[5], z0 + MAX_BUILDING_H)
    while z <= zmax + 1e-6:
        occ = interior_voids(probe_valid(vol, z, region, DETECT_N))
        any_left = False
        for bd, fp in zip(blds, fps):
            if (occ & fp).sum() > 0.3 * bd["area"]:
                bd["top"] = z; any_left = True
        if not any_left:
            break
        z += HEIGHT_DZ
    return blds, ground


# ── 窗口: fixed 与 zoom ────────────────────────────────────────────────────────
def fixed_window(b):
    x0 = max(b[0], -FIXED_HALF); x1 = min(b[1], FIXED_HALF)
    y0 = max(b[2], -FIXED_HALF); y1 = min(b[3], FIXED_HALF)
    return (x0, x1, y0, y1)


def estate_window(blds, region):
    """安全兜底: 框住所有楼 + 往开阔地外扩, 边必在楼群外 -> 不穿楼。"""
    (rx0, rx1), (ry0, ry1) = region
    m = SIDE_MARGIN
    x0 = max(rx0, min(d["x0"] for d in blds) - m)
    x1 = min(rx1, max(d["x1"] for d in blds) + m)
    y0 = max(ry0, min(d["y0"] for d in blds) - m)
    y1 = min(ry1, max(d["y1"] for d in blds) + m)
    return (x0, x1, y0, y1)


def edges_clear(window, occ, region):
    """四条边带 EDGE_MARGIN 去查占据栅格, 无楼=True。"""
    (rx0, rx1), (ry0, ry1) = region
    ny, nx = occ.shape
    dx = (rx1 - rx0) / (nx - 1); dy = (ry1 - ry0) / (ny - 1)
    x0, x1, y0, y1 = window
    if x1 - x0 < 5 or y1 - y0 < 5:
        return False
    xi = lambda x: int(np.clip(round((x - rx0) / dx), 0, nx - 1))
    yi = lambda y: int(np.clip(round((y - ry0) / dy), 0, ny - 1))
    j0, j1 = sorted((xi(x0), xi(x1))); i0, i1 = sorted((yi(y0), yi(y1)))
    mx = max(1, int(round(EDGE_MARGIN / dx))); my = max(1, int(round(EDGE_MARGIN / dy)))
    if occ[max(0, i0 - my):i0 + my + 1, j0:j1 + 1].any(): return False
    if occ[max(0, i1 - my):i1 + my + 1, j0:j1 + 1].any(): return False
    if occ[i0:i1 + 1, max(0, j0 - mx):j0 + mx + 1].any(): return False
    if occ[i0:i1 + 1, max(0, j1 - mx):j1 + mx + 1].any(): return False
    return True


def compute_zoom_window(blds, occ, region, frm, b):
    """迎风前 N_ROWS 排: 沿风向把楼聚成排, 取前两排, 加前/后/侧余量(直接返回, 不做穿楼校验)。"""
    if frm is None or len(blds) < 2:
        return estate_window(blds, region), "fallback(no-wind/too-few)"

    depth_y = abs(frm[1]) >= abs(frm[0])
    up = 1.0 if (frm[1] if depth_y else frm[0]) > 0 else -1.0   # 上游方向(世界坐标)
    dval = (lambda d: d["cy"] * up) if depth_y else (lambda d: d["cx"] * up)  # 越大越上游

    s = sorted(blds, key=dval, reverse=True)        # 上游在前
    rows = [[s[0]]]
    for d in s[1:]:
        if dval(rows[-1][-1]) - dval(d) <= ROW_GAP:
            rows[-1].append(d)
        else:
            rows.append([d])
    sel = [d for r in rows[:N_ROWS] for d in r]
    if not sel:
        return estate_window(blds, region), "fallback(empty-sel)"

    sx0 = min(d["x0"] for d in sel); sx1 = max(d["x1"] for d in sel)
    sy0 = min(d["y0"] for d in sel); sy1 = max(d["y1"] for d in sel)

    if depth_y:
        if up > 0:   # 上游=大Y
            y_front, y_back = sy1 + FRONT_MARGIN, sy0 - BACK_GAP
        else:        # 上游=小Y
            y_front, y_back = sy0 - FRONT_MARGIN, sy1 + BACK_GAP
        x_lo, x_hi = sx0 - SIDE_MARGIN, sx1 + SIDE_MARGIN
        y_lo, y_hi = sorted((y_front, y_back))
    else:
        if up > 0:   # 上游=大X
            x_front, x_back = sx1 + FRONT_MARGIN, sx0 - BACK_GAP
        else:
            x_front, x_back = sx0 - FRONT_MARGIN, sx1 + BACK_GAP
        y_lo, y_hi = sy0 - SIDE_MARGIN, sy1 + SIDE_MARGIN
        x_lo, x_hi = sorted((x_front, x_back))

    (rx0, rx1), (ry0, ry1) = region
    clip = lambda v, lo, hi: max(lo, min(hi, v))
    win = (clip(x_lo, rx0, rx1), clip(x_hi, rx0, rx1),
           clip(y_lo, ry0, ry1), clip(y_hi, ry0, ry1))
    return win, f"first{len(rows[:N_ROWS])}rows/{len(sel)}blds"


# ── 真切割 + tripcolor 渲染 ────────────────────────────────────────────────────
def extract_triangles(faces):
    if faces.size == 0:
        return np.empty((0, 3), int)
    if faces.size % 4 == 0 and bool((faces[::4] == 3).all()):
        return faces.reshape(-1, 4)[:, 1:]
    tris, i = [], 0
    while i < len(faces):
        k = int(faces[i])
        if k == 3:
            tris.append(faces[i + 1:i + 4])
        i += k + 1
    return np.array(tris, int) if tris else np.empty((0, 3), int)


def slice_at_z(vol, zl):
    try:
        sl = vol.slice(normal="z", origin=(0.0, 0.0, zl))
    except Exception:
        return None
    if sl is None or sl.n_points == 0:
        return None
    sl = sl.triangulate()
    tris = extract_triangles(np.asarray(sl.faces))
    if len(tris) == 0:
        return None
    pts = np.asarray(sl.points)
    vals = np.asarray(sl.point_data[P_FIELD])
    return {"z": float(zl), "x": pts[:, 0].copy(), "y": pts[:, 1].copy(),
            "tris": tris, "vals": vals.copy()}


def visible_tris(sld, window, finder, occ, region):
    """窗口内的三角形; 再丢掉"形心既在参考建筑轮廓洞里、又在地面footprint里"的三角形
    -> 每栋楼在自己屋顶以上也统一显示成灰(屋顶以下本就是空洞, 不受影响)。"""
    x0, x1, y0, y1 = window
    inw = (sld["x"] >= x0) & (sld["x"] <= x1) & (sld["y"] >= y0) & (sld["y"] <= y1)
    keep = inw[sld["tris"]].any(axis=1)
    tris = sld["tris"][keep]
    if finder is not None and occ is not None and len(tris) > 0:
        cx = sld["x"][tris].mean(axis=1); cy = sld["y"][tris].mean(axis=1)
        ti = np.asarray(finder(cx, cy))                      # <0 = 落在参考轮廓的洞里
        (rx0, rx1), (ry0, ry1) = region; ny, nx = occ.shape
        jj = np.clip(np.round((cx - rx0) / (rx1 - rx0) * (nx - 1)).astype(int), 0, nx - 1)
        ii = np.clip(np.round((cy - ry0) / (ry1 - ry0) * (ny - 1)).astype(int), 0, ny - 1)
        drop = (ti < 0) & occ[ii, jj]                        # 两门齐: 轮廓洞 且 地面建筑
        tris = tris[~drop]
    return tris


def field_of(sld, kind):
    f = sld["vals"] * RHO
    if kind == "real":
        return f
    return (sld["vals"] + G * sld["z"]) * RHO      # detrend: 抵消静压 -g·z, 所以 +g·z


def compute_scales(slices, window, finder, occ, region):
    rvals, dvals = [], []
    for sld in slices:
        tris = visible_tris(sld, window, finder, occ, region)
        if len(tris) == 0:
            continue
        vid = np.unique(tris)
        v = sld["vals"][vid]
        rvals.append(v * RHO)
        dvals.append((v + G * sld["z"]) * RHO)
    rv = np.concatenate(rvals) if rvals else np.array([0.0, 1.0])
    dv = np.concatenate(dvals) if dvals else np.array([-1.0, 1.0])
    real = (float(np.percentile(rv, CLIP_REAL)), float(np.percentile(rv, 100 - CLIP_REAL)))
    dpmax = max(float(np.percentile(np.abs(dv), CLIP_DP)), 1e-3)
    return real, dpmax


def draw_wind(ax, wind):
    wl, blow = wind
    if blow is None:
        return
    dx, dy = blow; cx, cy, L = 0.10, 0.93, 0.05
    ax.annotate("", xy=(cx + dx * L, cy + dy * L), xytext=(cx - dx * L, cy - dy * L),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="#1a1a1a", lw=2.5, mutation_scale=20), zorder=20)
    ax.text(cx, cy + L + 0.04, f"Wind {wl} (from)", transform=ax.transAxes,
            fontsize=9, fontweight="bold", color="#1a1a1a", ha="center", va="bottom",
            zorder=20, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1a1a1a", alpha=0.85))


def draw_side(ax_s, heights, zl, ground, z_ceil):
    ax_s.set_facecolor("#f8f8f8")
    span = max(z_ceil - ground, 1.0)
    show = sorted(heights, reverse=True)[:60] or [ground + 1]
    n = max(len(show), 1); bw = min(0.7 / n, 0.06)
    for i, top in enumerate(show):
        xc = (i + 0.5) / n
        hf = (top - ground) / span
        a = 0.9 if zl < top else 0.18
        ec = BEDGE if zl < top else "#bbb"
        ax_s.add_patch(plt.Rectangle((xc - bw / 2, 0.0), bw, hf,
                       facecolor=BFACE, edgecolor=ec, linewidth=0.4, alpha=a))
    zf = (zl - ground) / span
    ax_s.axhline(y=zf, color="#e74c3c", lw=2, zorder=10)
    ax_s.text(1.0, zf, f" {zl:.0f}m", fontsize=8, color="#e74c3c", fontweight="bold",
              va="center", ha="left")
    ax_s.set_xlim(-0.05, 1.18); ax_s.set_ylim(-0.03, 1.05)
    ax_s.set_xticks([]); ax_s.set_yticks([])
    ax_s.set_title("Side view", fontsize=8)


def render_frame(sld, window, kind, vlim, wind, heights, ground, z_ceil,
                 finder, occ, region, case_label, idx, n, fp):
    tris = visible_tris(sld, window, finder, occ, region)
    field = field_of(sld, kind)
    x0, x1, y0, y1 = window
    cmap = matplotlib.colormaps[CMAP]
    norm = Normalize(vlim[0], vlim[1])          # 每帧新建, 不跨图复用
    tag = "p (Pa) absolute" if kind == "real" else "p + rho*g*z (Pa)"

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
    ax  = fig.add_axes([0.07, 0.08, 0.65, 0.84])
    ax.set_facecolor(BUILDING_GREY)                 # 空洞(楼)显灰
    if len(tris) > 0:
        triang = mtri.Triangulation(sld["x"], sld["y"], tris)
        ax.tripcolor(triang, field, shading="gouraud", cmap=cmap, norm=norm)
    draw_wind(ax, wind)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
    ax.set_xlabel("X (m)", fontsize=11); ax.set_ylabel("Y (m)", fontsize=11)
    ax.set_title(f"{case_label}    z = {sld['z']:.1f} m    |    {tag}",
                 fontsize=12, fontweight="bold", pad=10)

    cax = fig.add_axes([0.74, 0.25, 0.015, 0.50])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax); cb.set_label(tag, fontsize=10); cb.ax.tick_params(labelsize=8)

    ax_s = fig.add_axes([0.80, 0.08, 0.17, 0.22])
    draw_side(ax_s, heights, sld["z"], ground, z_ceil)
    fig.text(0.04, 0.015, case_label, fontsize=8, color="#999", style="italic")
    fig.text(0.97, 0.015, f"Frame {idx + 1}/{n}", fontsize=8, color="#aaa", ha="right")
    fig.savefig(fp, dpi=DPI, facecolor="white")
    plt.close(fig)


def assemble_gif(fpaths, gif_path):
    imgs = [imageio.imread(str(f)) for f in fpaths]
    tgt = imgs[0].shape
    out = []
    for im in imgs:
        if im.shape != tgt:
            c = np.full(tgt, 255, dtype=np.uint8)
            sl = tuple(slice(0, min(a, b)) for a, b in zip(im.shape, tgt))
            c[sl] = im[sl]; im = c
        out.append(im)
    imageio.mimsave(str(gif_path), out, duration=FRAME_MS, loop=0)


# ── 单个 case ─────────────────────────────────────────────────────────────────
def process_case(name):
    t0 = time.time()
    outer = BASE / name
    cid = case_id(name)
    print(f"\n{'=' * 64}\n  [{cid}] {name}\n{'=' * 64}")
    case_dir = find_case_dir(outer)
    if case_dir is None:
        print("  [SKIP] 没找到 case 目录"); return
    vol, used_t = read_openfoam_volume(case_dir)
    b = vol.bounds
    wl, blow, frm = parse_wind(name)
    print(f"  time={used_t} n_cells={vol.n_cells:,} wind {wl} Z[{b[4]:.0f},{b[5]:.0f}]")

    region = ((max(b[0], -REGION_HALF), min(b[1], REGION_HALF)),
              (max(b[2], -REGION_HALF), min(b[3], REGION_HALF)))
    blds, occ = scan_buildings(vol, b, region)
    if not blds:
        print("  [SKIP] 没检测到建筑"); return
    print(f"  检测到 {len(blds)} 栋楼, 最高 {max(d['top'] for d in blds):.0f}m")

    win_fixed = fixed_window(b)
    win_zoom, zinfo = compute_zoom_window(blds, occ, region, frm, b)
    print(f"  fixed {tuple(round(v) for v in win_fixed)}")
    print(f"  zoom  {tuple(round(v) for v in win_zoom)}  [{zinfo}]")
    windows = {"fixed": win_fixed, "zoom": win_zoom}

    # z 扫描: 地面 -> 整片最高楼 + 2 帧
    ground = b[4] + GROUND_OFFSET
    top_all = max(d["top"] for d in blds)
    z_levels = np.arange(ground, top_all + 2 * DZ + 1e-6, DZ)
    print(f"  切片 {len(z_levels)} 层 ({ground:.0f}->{z_levels[-1]:.0f}m) ...")

    slices = []
    for zl in z_levels:
        sld = slice_at_z(vol, float(zl))
        if sld is not None:
            slices.append(sld)
    if not slices:
        print("  [SKIP] 切不出面"); return
    del vol; gc.collect()

    # 用最底层(楼实心)建一个 crisp 建筑轮廓查找器, 给屋顶帧做规范化
    try:
        ref = slices[0]
        finder = mtri.Triangulation(ref["x"], ref["y"], ref["tris"]).get_trifinder()
    except Exception as e:
        print(f"  [WARN] 轮廓 finder 建失败, 屋顶帧不规范化: {e}"); finder = None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ver in VERSIONS:
        window = windows[ver]
        x0, x1, y0, y1 = window
        in_win = lambda d: x0 <= d["cx"] <= x1 and y0 <= d["cy"] <= y1
        heights = [d["top"] for d in blds if in_win(d)] or [d["top"] for d in blds]
        z_top = max(heights) + 2 * DZ
        sub = [s for s in slices if s["z"] <= z_top + 1e-6] or slices
        z_ceil = max(heights) + 5.0
        real_rng, dp_max = compute_scales(sub, window, finder, occ, region)
        print(f"    [{ver}] {len(sub)} 帧, real[{real_rng[0]:.1f},{real_rng[1]:.1f}] detrend±{dp_max:.2f}")

        for kind in KINDS:
            vlim = real_rng if kind == "real" else (-dp_max, dp_max)
            fdir = OUT_DIR / cid / ver / kind
            fdir.mkdir(parents=True, exist_ok=True)
            fpaths = []
            for idx, sld in enumerate(sub):
                fp = fdir / f"frame_{idx:04d}_z{sld['z']:.1f}m.png"
                try:
                    render_frame(sld, window, kind, vlim, (wl, blow), heights, ground, z_ceil,
                                 finder, occ, region, cid, idx, len(sub), fp)
                    fpaths.append(fp)
                except Exception as e:
                    print(f"      [WARN] frame {idx} z={sld['z']:.0f}: {e}")
            if not fpaths:
                continue
            gdir = OUT_DIR / f"{ver}_{kind}"
            gdir.mkdir(parents=True, exist_ok=True)
            assemble_gif(fpaths, gdir / f"{cid}.gif")
            print(f"      OK -> {ver}_{kind}/{cid}.gif")

    del slices; gc.collect()
    print(f"  done ({time.time() - t0:.1f}s)")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    global BASE, OUT_DIR, DZ, DETECT_N, N_ROWS, ROW_GAP, G, FIXED_HALF, REGION_HALF
    ap = argparse.ArgumentParser(
        description="OpenFOAM 压力切片 GIF (fixed + zoom, ParaView 式真切割渲染)")
    ap.add_argument("--inspect", action="store_true", help="只跑第一个 case (含其全部 GIF)")
    ap.add_argument("--base", default=str(BASE), help="算例根目录 (含 case_HDB_* )")
    ap.add_argument("--out_dir", default=None, help="输出目录 (默认 BASE/figures)")
    ap.add_argument("--dz", type=float, default=DZ, help="帧 z 间隔 (m)")
    ap.add_argument("--detect_n", type=int, default=DETECT_N, help="检测栅格分辨率")
    ap.add_argument("--n_rows", type=int, default=N_ROWS, help="zoom 迎风排数")
    ap.add_argument("--row_gap", type=float, default=ROW_GAP, help="同排质心间距阈值 (m)")
    ap.add_argument("--detrend_gz", type=float, default=None,
                    help="覆盖 detrend 的 g (运动学场减 g*z); 默认 9.81")
    ap.add_argument("--fixed_half", type=float, default=FIXED_HALF, help="fixed 方块半边长 (m)")
    ap.add_argument("--region_half", type=float, default=REGION_HALF, help="检测区域半边长 (m)")
    args = ap.parse_args()

    BASE = Path(args.base)
    OUT_DIR = Path(args.out_dir) if args.out_dir else (BASE / "figures")
    DZ = args.dz; DETECT_N = args.detect_n; N_ROWS = args.n_rows; ROW_GAP = args.row_gap
    if args.detrend_gz is not None: G = args.detrend_gz
    FIXED_HALF = args.fixed_half; REGION_HALF = args.region_half

    folders = sorted(d.name for d in BASE.glob("case_HDB_*") if d.is_dir())
    if not folders:
        print(f"在 {BASE} 没找到 case_HDB_* 目录"); return
    print(f"共 {len(folders)} 个 case, 全部输出到: {OUT_DIR}")
    if args.inspect:
        folders = folders[:1]

    for ci, name in enumerate(folders, 1):
        print(f"\n###### [{ci}/{len(folders)}] ######")
        try:
            process_case(name)
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            import traceback; traceback.print_exc()
    print("\nDone.")


if __name__ == "__main__":
    main()