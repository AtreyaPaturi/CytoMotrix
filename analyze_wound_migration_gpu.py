#!/usr/bin/env python3
"""
analyze_wound_migration_gpu.py

GPU-accelerated scratch-wound (collective migration) analysis for the
Celldiscoverer 7 time-lapse exported by split_czi_by_well_time.py.

Input : the export folder containing manifest.csv and the per-time-point PNGs
        (16-bit raw or --png-8bit both work).
Output: <out>/per_frame.csv, per_field.csv, per_condition.csv,
        figures (SVG + PNG), and QC overlays per field.

Method (per frame, on the GPU with PyTorch):
  1. downsample (block mean), normalise by the frame median, light denoise
  2. local standard deviation (texture) in a sliding window
     -> cell monolayer = textured, open wound = smooth
  3. threshold log(local SD) with Otsu's method, threshold fixed per field
     from the first frame (both classes are present at t0)
  4. morphological closing + opening (min/max pooling) to remove debris and
     smooth cell interiors
  (CPU) 5. keep the largest smooth connected region = wound, fill holes
Metrics: wound area, mean wound width (= area / wound length), relative
area (% of t0), closure (%), and front velocity = -d(width)/dt / 2 (um/h,
per migrating edge) from a linear fit while the wound is > 5 % open.

Install (once):
    pip install torch numpy scipy pillow matplotlib
    (for NVIDIA GPUs pick the CUDA build from https://pytorch.org)

Run:
    python3 analyze_wound_migration_gpu.py --in /path/to/export

Options:
    --in PATH          export folder with manifest.csv
    --out PATH         results folder (default <in>/migration_analysis)
    --device D         auto | cuda | mps | cpu (default auto)
    --batch N          frames per GPU batch (default 8; lower if GPU memory is short)
    --pixel-size UM    override pixel size (default: read from PNG metadata)
    --downsample N     block-mean downsampling before analysis (default 2)
    --smooth-um UM     denoising box filter before texture (default 4)
    --window-um UM     texture window (default 40)
    --close-um UM      closing radius, removes debris in the gap (default 8)
    --open-um UM       opening radius, removes smooth patches in the monolayer (default 15)
    --thr-offset F     added to the Otsu threshold in log-SD units; negative =
                       stricter (smaller wound), positive = looser (default 0)
    --min-area-frac F  wound smaller than this fraction of the field = closed (default 0.005)
    --fit-min-rel PCT  use frames with rel. area above this for the velocity fit (default 5)
"""

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

import torch
import torch.nn.functional as F

# Nature Publishing Group palette (ggsci "npg")
NPG = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", "#8491B4",
       "#91D1C2", "#DC0000", "#7E6148", "#B09C85"]


# ------------------------------------------------------------------ helpers
def pick_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    dev = torch.device(name)
    if dev.type == "cuda":
        print(f"Device: CUDA - {torch.cuda.get_device_name(0)}")
    elif dev.type == "mps":
        print("Device: Apple MPS")
    else:
        print("Device: CPU (no GPU found - this will be slower; same results)")
    return dev


def load_manifest(root):
    mf = root / "manifest.csv"
    if not mf.exists():
        sys.exit(f"manifest.csv not found in {root} - point --in at the export folder")
    with open(mf, newline="") as fh:
        rows = list(csv.DictReader(fh))
    fields = {}
    for r in rows:
        fields.setdefault((r["well"], int(r["field"])), []).append(r)
    for k in fields:
        fields[k].sort(key=lambda r: int(r["timepoint"]))
    return fields


def split_condition(cond, well):
    if cond and "_" in cond:
        cell, treat = cond.split("_", 1)
        return cell.replace("-", " "), treat
    return (cond or well), ""


def read_png(path):
    im = Image.open(path)
    arr = np.asarray(im)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(-1)
    return arr.astype(np.float32), im.info.get("dpi")


def odd(n):
    n = max(3, int(round(n)))
    return n if n % 2 else n + 1


# -------------------------------------------------------------- GPU kernels
def box_mean(x, k):
    """Separable box filter with reflect padding. x: (B,1,H,W)."""
    r = k // 2
    x = F.pad(x, (0, 0, r, r), mode="reflect")
    x = F.avg_pool2d(x, (k, 1), stride=1)
    x = F.pad(x, (r, r, 0, 0), mode="reflect")
    return F.avg_pool2d(x, (1, k), stride=1)


def local_log_sd(x, k):
    m = box_mean(x, k)
    m2 = box_mean(x * x, k)
    return torch.log((m2 - m * m).clamp_min(0).sqrt() + 1e-6)


def dilate(m, r):
    if r <= 0:
        return m
    m = F.max_pool2d(m, (2 * r + 1, 1), stride=1, padding=(r, 0))
    return F.max_pool2d(m, (1, 2 * r + 1), stride=1, padding=(0, r))


def erode(m, r):
    return -dilate(-m, r) if r > 0 else m


def otsu(v, bins=256):
    v = v.flatten()
    if v.numel() > 2_000_000:
        v = v[torch.randint(0, v.numel(), (2_000_000,), device=v.device)]
    lo = torch.quantile(v, 0.001).item()
    hi = torch.quantile(v, 0.999).item()
    h = torch.histc(v.clamp(lo, hi), bins=bins, min=lo, max=hi)
    p = h / h.sum()
    centers = lo + (torch.arange(bins, device=v.device) + 0.5) * (hi - lo) / bins
    w0 = p.cumsum(0)
    mu = (p * centers).cumsum(0)
    sb = (mu[-1] * w0 - mu) ** 2 / (w0 * (1 - w0)).clamp_min(1e-12)
    i = int(torch.argmax(sb[:-1]))
    return lo + (i + 1) * (hi - lo) / bins


# --------------------------------------------------------- per-field analysis
def wound_component(mask, min_px):
    lab, n = ndi.label(mask)
    if n == 0:
        return np.zeros_like(mask)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    i = int(sizes.argmax())
    if sizes[i] < min_px:
        return np.zeros_like(mask)
    return ndi.binary_fill_holes(lab == i)


def analyse_field(rows, root, dev, a):
    paths = [root / r["path"] for r in rows]
    with ThreadPoolExecutor(8) as ex:
        loaded = list(ex.map(read_png, paths))
    imgs = [im for im, _ in loaded]
    dpi = loaded[0][1]

    px = a.pixel_size
    if px is None and dpi and dpi[0] > 0:
        cand = 25400.0 / float(dpi[0])
        px = cand if 0.05 < cand < 20 else None
    if px is None:
        sys.exit("Pixel size not found in PNG metadata - pass --pixel-size (um)")

    ds = max(1, a.downsample)
    pxd = px * ds
    k = odd(a.window_um / pxd)
    ks = odd(a.smooth_um / pxd) if a.smooth_um > 0 else 1
    r_close = int(round(a.close_um / pxd))
    r_open = int(round(a.open_um / pxd))

    thr = None
    masks = []
    with torch.no_grad():
        for i in range(0, len(imgs), a.batch):
            x = torch.from_numpy(np.stack(imgs[i:i + a.batch]))[:, None].to(dev)
            if ds > 1:
                x = F.avg_pool2d(x, ds)
            med = x.flatten(1).median(dim=1).values.clamp_min(1e-6)
            x = x / med[:, None, None, None]
            if ks > 1:
                x = box_mean(x, ks)                       # denoise
            s = local_log_sd(x, k)
            if thr is None:
                thr = otsu(s[0]) + a.thr_offset
            m = (s < thr).float()
            m = erode(dilate(m, r_close), r_close)        # closing
            m = dilate(erode(m, r_open), r_open)          # opening
            masks.append(m[:, 0].bool().cpu().numpy())
    masks = np.concatenate(masks)

    H, W = masks.shape[1:]
    min_px = a.min_area_frac * H * W
    wounds = [wound_component(m, min_px) for m in masks]

    ys, xs = np.nonzero(wounds[0])
    horizontal = True if xs.size == 0 else (np.ptp(xs) >= np.ptp(ys))
    length_um = (W if horizontal else H) * pxd

    areas = np.array([w.sum() * pxd * pxd for w in wounds])
    A0 = areas[0]
    per_frame = []
    for r, A in zip(rows, areas):
        rel = 100 * A / A0 if A0 > 0 else np.nan
        per_frame.append({
            "time_min": int(r["time_min"]), "timepoint": int(r["timepoint"]),
            "wound_area_um2": A, "mean_width_um": A / length_um,
            "rel_area_pct": rel, "closure_pct": 100 - rel})

    info = {"pixel_um": px, "analysis_pixel_um": pxd, "window_px": k,
            "smooth_px": ks,
            "threshold_logsd": thr, "orientation":
            "horizontal" if horizontal else "vertical"}
    return per_frame, info, imgs, wounds, ds


def velocity_fit(per_frame, fit_min_rel):
    t = np.array([f["time_min"] for f in per_frame]) / 60.0
    w = np.array([f["mean_width_um"] for f in per_frame])
    rel = np.array([f["rel_area_pct"] for f in per_frame])
    keep = rel >= fit_min_rel
    if keep.sum() < 3:
        return np.nan, np.nan, int(keep.sum())
    slope, icpt = np.polyfit(t[keep], w[keep], 1)
    pred = slope * t[keep] + icpt
    ss_res = ((w[keep] - pred) ** 2).sum()
    ss_tot = ((w[keep] - w[keep].mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return -slope / 2.0, r2, int(keep.sum())


# ----------------------------------------------------------------------- QC
def get_font(size):
    for n in ("DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "arialbd.ttf"):
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def qc_montage(imgs, wounds, per_frame, ds, title, path):
    idx = sorted({0, len(imgs) // 2, len(imgs) - 1})
    H, W = wounds[0].shape
    ref = np.asarray(Image.fromarray(imgs[0]).resize((W, H), Image.BOX))
    lo, hi = np.percentile(ref, [0.5, 99.5])
    hi = max(hi, lo + 1)
    panels = []
    font = get_font(max(14, H // 22))
    for i in idx:
        g = np.asarray(Image.fromarray(imgs[i]).resize((W, H), Image.BOX))
        g8 = (np.clip((g - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
        rgb = np.stack([g8] * 3, -1)
        edge = wounds[i] & ~ndi.binary_erosion(wounds[i], iterations=2)
        rgb[edge] = (230, 75, 53)                         # NPG red outline
        im = Image.fromarray(rgb)
        d = ImageDraw.Draw(im)
        f = per_frame[i]
        txt = f"{f['time_min']} min   rel. area {f['rel_area_pct']:.0f}%"
        d.text((10, 10), txt, font=font, fill=(255, 255, 255),
               stroke_width=2, stroke_fill=(0, 0, 0))
        panels.append(im)
    gap = 12
    mont = Image.new("RGB", (len(panels) * W + (len(panels) - 1) * gap, H + 50),
                     (255, 255, 255))
    for j, p in enumerate(panels):
        mont.paste(p, (j * (W + gap), 50))
    ImageDraw.Draw(mont).text((10, 10), title, font=get_font(28), fill=(0, 0, 0))
    mont.thumbnail((2400, 2400))
    path.parent.mkdir(parents=True, exist_ok=True)
    mont.save(path)


# ------------------------------------------------------------------ figures
def make_figures(frames, fields, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.fonttype": "none", "font.size": 8,
                         "font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.8})

    conds = []
    for f in fields:
        c = (f["cell_line"], f["treatment"])
        if c not in conds:
            conds.append(c)
    treatments = []
    for _, t in conds:
        if t not in treatments:
            treatments.append(t)
    cells = []
    for c, _ in conds:
        if c not in cells:
            cells.append(c)
    tcolor = {t: [NPG[3], NPG[0], NPG[2], NPG[1], NPG[4], NPG[5]][i % 6]
              for i, t in enumerate(treatments)}
    cstyle = {c: s for c, s in zip(cells, [("-", "o", True), ("--", "s", False),
                                           (":", "^", True), ("-.", "D", False)])}

    # ---- time course: relative wound area, mean +/- SD of fields
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for cell, treat in conds:
        by_t = {}
        for r in frames:
            if r["cell_line"] == cell and r["treatment"] == treat:
                by_t.setdefault(r["time_min"], []).append(r["rel_area_pct"])
        t = np.array(sorted(by_t)) / 60.0
        m = np.array([np.nanmean(by_t[k]) for k in sorted(by_t)])
        sd = np.array([np.nanstd(by_t[k], ddof=1) if len(by_t[k]) > 1 else 0
                       for k in sorted(by_t)])
        ls, mk, filled = cstyle[cell]
        col = tcolor[treat]
        ax.fill_between(t, m - sd, m + sd, color=col, alpha=0.15, lw=0)
        ax.plot(t, m, ls=ls, color=col, lw=1.2, marker=mk, ms=3.5,
                markevery=max(1, len(t) // 8), mfc=col if filled else "white",
                mec=col, label=f"{cell} {treat}")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Wound area (% of t0)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(out / f"wound_area_timecourse.{ext}", dpi=300)
    plt.close(fig)

    # ---- front velocity + closure at end: per field points, mean +/- SD
    for key, ylab, fname in (("front_velocity_um_per_h", "Front velocity (µm/h)",
                              "front_velocity"),
                             ("closure_end_pct", "Wound closure at end (%)",
                              "closure_end")):
        fig, ax = plt.subplots(figsize=(3.2, 2.6))
        xt, xl = [], []
        for i, (cell, treat) in enumerate(conds):
            vals = np.array([f[key] for f in fields
                             if f["cell_line"] == cell and f["treatment"] == treat],
                            dtype=float)
            vals = vals[np.isfinite(vals)]
            _, mk, filled = cstyle[cell]
            col = tcolor[treat]
            jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else [0]
            ax.scatter(i + np.asarray(jitter), vals, s=14, marker=mk,
                       facecolor=col if filled else "white", edgecolor=col,
                       lw=0.8, zorder=3)
            if len(vals):
                sd = vals.std(ddof=1) if len(vals) > 1 else 0
                ax.errorbar(i, vals.mean(), yerr=sd, fmt="none", ecolor="black",
                            capsize=3, lw=0.8, zorder=4)
                ax.hlines(vals.mean(), i - 0.25, i + 0.25, color="black", lw=1.2,
                          zorder=4)
            xt.append(i)
            xl.append(f"{cell}\n{treat}")
        ax.set_xticks(xt)
        ax.set_xticklabels(xl, fontsize=6, rotation=45, ha="right")
        ax.set_ylabel(ylab)
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        for ext in ("svg", "png"):
            fig.savefig(out / f"{fname}.{ext}", dpi=300)
        plt.close(fig)


# ---------------------------------------------------------------------- main
def write_csv(path, rows, cols):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v)
                        for k, v in r.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--pixel-size", type=float, default=None)
    ap.add_argument("--downsample", type=int, default=2)
    ap.add_argument("--smooth-um", type=float, default=4.0)
    ap.add_argument("--window-um", type=float, default=40.0)
    ap.add_argument("--close-um", type=float, default=8.0)
    ap.add_argument("--open-um", type=float, default=15.0)
    ap.add_argument("--thr-offset", type=float, default=0.0)
    ap.add_argument("--min-area-frac", type=float, default=0.005)
    ap.add_argument("--fit-min-rel", type=float, default=5.0)
    a = ap.parse_args()

    root = Path(a.inp)
    if root.suffix.lower() == ".czi" or not root.is_dir():
        sys.exit(f"--in must be the PNG export folder that contains manifest.csv "
                 f"(the --out folder of split_czi_by_well_time.py), not '{root}'.\n"
                 f"e.g.  --in /path/to/export")
    fields = load_manifest(root)
    out = Path(a.out) if a.out else root / "migration_analysis"
    out.mkdir(parents=True, exist_ok=True)
    dev = pick_device(a.device)
    print(f"{len(fields)} fields found\n")

    all_frames, all_fields = [], []
    t_start = time.time()
    for (well, fnum), rows in fields.items():
        t0 = time.time()
        cond = rows[0]["condition"]
        cell, treat = split_condition(cond, well)
        per_frame, info, imgs, wounds, ds = analyse_field(rows, root, dev, a)
        vel, r2, nfit = velocity_fit(per_frame, a.fit_min_rel)

        base = {"well": well, "condition": cond, "cell_line": cell,
                "treatment": treat, "field": fnum}
        for f in per_frame:
            all_frames.append({**base, **f})
        all_fields.append({**base,
                           "wound_area_t0_um2": per_frame[0]["wound_area_um2"],
                           "width_t0_um": per_frame[0]["mean_width_um"],
                           "closure_end_pct": per_frame[-1]["closure_pct"],
                           "front_velocity_um_per_h": vel, "fit_r2": r2,
                           "fit_n_frames": nfit, **info})
        tag = f"{well}_{cond}_F{fnum}" if cond else f"{well}_F{fnum}"
        qc_montage(imgs, wounds, per_frame, ds, f"{well}  {cell} {treat}  F{fnum}",
                   out / "qc" / f"{tag}_qc.png")
        print(f"{well} F{fnum} {cell:<9} {treat:<14} width0 "
              f"{per_frame[0]['mean_width_um']:6.0f} um  closure "
              f"{per_frame[-1]['closure_pct']:5.1f}%  v {vel:5.2f} um/h  "
              f"({time.time() - t0:.1f} s)")

    # per condition summary
    summary = []
    seen = []
    for f in all_fields:
        c = (f["cell_line"], f["treatment"])
        if c in seen:
            continue
        seen.append(c)
        grp = [g for g in all_fields if (g["cell_line"], g["treatment"]) == c]
        row = {"cell_line": c[0], "treatment": c[1], "n_fields": len(grp)}
        for k in ("width_t0_um", "closure_end_pct", "front_velocity_um_per_h"):
            v = np.array([g[k] for g in grp], dtype=float)
            v = v[np.isfinite(v)]
            row[f"{k}_mean"] = float(v.mean()) if v.size else np.nan
            row[f"{k}_sd"] = float(v.std(ddof=1)) if v.size > 1 else np.nan
        summary.append(row)

    write_csv(out / "per_frame.csv", all_frames,
              ["well", "condition", "cell_line", "treatment", "field", "timepoint",
               "time_min", "wound_area_um2", "mean_width_um", "rel_area_pct",
               "closure_pct"])
    write_csv(out / "per_field.csv", all_fields,
              ["well", "condition", "cell_line", "treatment", "field",
               "wound_area_t0_um2", "width_t0_um", "closure_end_pct",
               "front_velocity_um_per_h", "fit_r2", "fit_n_frames", "pixel_um",
               "analysis_pixel_um", "smooth_px", "window_px", "threshold_logsd",
               "orientation"])
    write_csv(out / "per_condition.csv", summary, list(summary[0].keys()))
    make_figures(all_frames, all_fields, out)
    print(f"\nDone in {time.time() - t_start:.0f} s -> {out}")
    print("Check the QC overlays in qc/ before trusting the numbers.")


if __name__ == "__main__":
    main()
