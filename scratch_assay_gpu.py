#!/usr/bin/env python3
"""
scratch_assay_gpu.py

GPU scratch-wound analysis for the Celldiscoverer 7 time-lapse exported by
split_czi_by_well_time.py (16-bit or 8-bit PNGs + manifest.csv).

Built on the entropy + Otsu method of
  * S. Bhattiprolu, "Python for Microscopists" 021-scratch_assay_using_scikit_image.py
  * E. Gad, github.com/eyadgad/Image-Processing-and-Analysis-of-the-Scratch-Assay
and extended with:
  * local entropy computed on the GPU (PyTorch), radius given in um so it
    scales with pixel size (the reference disk(3) px is ~2 um here - too small
    for 0.69 um/px CD7 images, it sees camera noise as texture)
  * denoising + consistent grey-level quantisation for the whole time series
  * Otsu threshold fixed from t0 (default) or per frame as in the references
  * ALL open space is counted:
      wound area      = every open pixel inside the t0 scratch region
                        (all fragments, not just the largest piece)
      total open area = wound area + open regions outside the scratch that are
                        larger than --min-open-um2 (real gaps in the monolayer)
  * physical units (um, um^2, h), mean wound width, front velocity
  * per-field linear regression report (slope, R^2, SE, p) as in the GitHub repo
  * optional entropy / binary map export and animated GIFs (as in the GitHub repo)
  * per-frame / per-field / per-condition CSVs, publication figures (SVG + PNG),
    QC overlays

Install (once):
    pip install torch numpy scipy pillow matplotlib imageio
    (for NVIDIA GPUs use the CUDA build from https://pytorch.org)

Run:
    python3 scratch_assay_gpu.py --in /path/to/export
    python3 scratch_assay_gpu.py --in /path/to/export --gif --save-maps

Main options:
    --in PATH             export folder containing manifest.csv
    --out PATH            results folder (default <in>/scratch_analysis)
    --device D            auto | cuda | mps | cpu
    --batch N             frames per GPU batch (default 8)
    --pixel-size UM       override pixel size (default: from PNG metadata)
    --downsample N        block-mean downsampling before analysis (default 2)
    --smooth-um UM        denoising before entropy (default 4)
    --entropy-radius-um R entropy disk radius (default 20)
    --levels N            grey levels for entropy (default 16)
    --otsu t0|per-frame   threshold from first frame (default) or each frame
    --thr-offset BITS     added to the Otsu threshold; + = more open, - = less
    --min-speck-um2 A     discard open specks smaller than this (default 200)
    --fill-debris-um2 A   fill debris / single particles inside open space
                          smaller than this (default 150)
    --min-open-um2 A      open regions OUTSIDE the scratch count toward total
                          open area only above this size (default 20000; smaller
                          smooth patches in a confluent epithelial monolayer are
                          usually flat cell bodies, not gaps)
    --fit-min-rel PCT     frames with wound > PCT % of t0 used for fits (default 5)
    --save-maps           save entropy + binary images for every frame
    --gif                 animated GIF per field: input | entropy | open area
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
from scipy.stats import linregress

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
        print("Device: CPU (no GPU found - slower, same results)")
    return dev


def load_manifest(root):
    mf = root / "manifest.csv"
    if not mf.exists():
        sys.exit(f"manifest.csv not found in {root} - point --in at the PNG "
                 f"export folder (the --out of split_czi_by_well_time.py)")
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


# -------------------------------------------------------------- GPU kernels
def box_mean(x, k):
    r = k // 2
    x = F.pad(x, (0, 0, r, r), mode="reflect")
    x = F.avg_pool2d(x, (k, 1), stride=1)
    x = F.pad(x, (r, r, 0, 0), mode="reflect")
    return F.avg_pool2d(x, (1, k), stride=1)


def disk_kernel(r, dev):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    k = (x * x + y * y <= r * r).astype(np.float32)
    return torch.from_numpy(k / k.sum())[None, None].to(dev)


def local_entropy(x01, r, levels, kern):
    """Shannon entropy (bits) of grey levels in a disk of radius r.
    GPU equivalent of skimage.filters.rank.entropy. x01: (B,1,H,W) in [0,1]."""
    q = (x01 * (levels - 1)).round().clamp(0, levels - 1)
    xp = F.pad(q, (r, r, r, r), mode="reflect")
    ent = torch.zeros_like(x01)
    for lv in range(levels):
        p = F.conv2d((xp == lv).float(), kern)
        ent -= p * torch.log2(p.clamp_min(1e-12))
    return ent


def otsu(v, bins=256):
    v = v.flatten()
    if v.numel() > 2_000_000:
        v = v[torch.randint(0, v.numel(), (2_000_000,), device=v.device)]
    lo = torch.quantile(v, 0.001).item()
    hi = torch.quantile(v, 0.999).item()
    if hi <= lo:
        return hi
    h = torch.histc(v.clamp(lo, hi), bins=bins, min=lo, max=hi)
    p = h / h.sum()
    centers = lo + (torch.arange(bins, device=v.device) + 0.5) * (hi - lo) / bins
    w0 = p.cumsum(0)
    mu = (p * centers).cumsum(0)
    sb = (mu[-1] * w0 - mu) ** 2 / (w0 * (1 - w0)).clamp_min(1e-12)
    i = int(torch.argmax(sb[:-1]))
    return lo + (i + 1) * (hi - lo) / bins


# -------------------------------------------------------- CPU post-process
def drop_small(mask, min_px):
    if min_px <= 0:
        return mask
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_px
    keep[0] = False
    return keep[lab]


def fill_small_holes(mask, max_px):
    """Fill closed non-open islands (debris, single particles) < max_px."""
    if max_px <= 0:
        return mask
    lab, n = ndi.label(~mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    small = sizes < max_px
    small[0] = False
    small[border] = False
    return mask | small[lab]


def largest_filled(mask):
    lab, n = ndi.label(mask)
    if n == 0:
        return np.zeros_like(mask)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return ndi.binary_fill_holes(lab == int(sizes.argmax()))


# ------------------------------------------------------- per-field analysis
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
    a_px = pxd * pxd
    ks = odd(a.smooth_um / pxd) if a.smooth_um > 0 else 1
    r = max(1, int(round(a.entropy_radius_um / pxd)))
    kern = disk_kernel(r, dev)

    thr0, lo, hi = None, None, None
    raw_masks, ent8, img8, thrs = [], [], [], []
    keep_maps = a.save_maps or a.gif
    with torch.no_grad():
        for i in range(0, len(imgs), a.batch):
            x = torch.from_numpy(np.stack(imgs[i:i + a.batch]))[:, None].to(dev)
            if ds > 1:
                x = F.avg_pool2d(x, ds)
            med = x.flatten(1).median(dim=1).values.clamp_min(1e-6)
            x = x / med[:, None, None, None]
            if lo is None:                                  # fixed from t0
                s = x[0].flatten()
                s = s[torch.randint(0, s.numel(), (min(s.numel(), 2_000_000),),
                                    device=dev)]
                lo = torch.quantile(s, 0.005).item()
                hi = max(torch.quantile(s, 0.995).item(), lo + 1e-6)
            x01 = ((x - lo) / (hi - lo)).clamp(0, 1)
            if ks > 1:
                x01 = box_mean(x01, ks)
            ent = local_entropy(x01, r, a.levels, kern)
            for j in range(ent.shape[0]):
                if a.otsu == "per-frame":
                    t = otsu(ent[j]) + a.thr_offset
                else:
                    if thr0 is None:
                        thr0 = otsu(ent[0]) + a.thr_offset
                    t = thr0
                thrs.append(t)
                raw_masks.append((ent[j, 0] <= t).cpu().numpy())
            if keep_maps:
                emax = float(np.log2(a.levels))
                ent8 += list((ent[:, 0].clamp(0, emax) / emax * 255)
                             .byte().cpu().numpy())
                img8 += list((x01[:, 0] * 255).byte().cpu().numpy())

    H, W = raw_masks[0].shape
    speck = a.min_speck_um2 / a_px
    debris = a.fill_debris_um2 / a_px
    min_open = a.min_open_um2 / a_px
    masks = [fill_small_holes(drop_small(m, speck), debris) for m in raw_masks]

    W0 = largest_filled(masks[0])                       # scratch region at t0
    if not W0.any():
        print("   ! no open region found at t0 in this field")
    ys, xs = np.nonzero(W0)
    horizontal = True if xs.size == 0 else (np.ptp(xs) >= np.ptp(ys))
    length_um = (W if horizontal else H) * pxd

    wounds, totals = [], []
    for m in masks:
        wnd = m & W0
        other = drop_small(m & ~W0, min_open)
        wounds.append(wnd)
        totals.append(wnd | other)

    A0 = wounds[0].sum() * a_px
    per_frame = []
    for rw, wnd, tot, t in zip(rows, wounds, totals, thrs):
        A = wnd.sum() * a_px
        rel = 100 * A / A0 if A0 > 0 else np.nan
        per_frame.append({
            "timepoint": int(rw["timepoint"]), "time_min": int(rw["time_min"]),
            "wound_area_um2": A, "total_open_area_um2": tot.sum() * a_px,
            "mean_width_um": A / length_um, "rel_area_pct": rel,
            "closure_pct": 100 - rel, "threshold_bits": t})

    info = {"pixel_um": px, "analysis_pixel_um": pxd, "smooth_px": ks,
            "entropy_radius_px": r, "levels": a.levels,
            "threshold_t0_bits": thrs[0], "orientation":
            "horizontal" if horizontal else "vertical",
            "scratch_length_um": length_um}
    maps = {"img8": img8, "ent8": ent8, "W0": W0, "wounds": wounds,
            "totals": totals, "imgs": imgs}
    return per_frame, info, maps


def fits(per_frame, fit_min_rel):
    t = np.array([f["time_min"] for f in per_frame]) / 60.0
    A = np.array([f["wound_area_um2"] for f in per_frame])
    w = np.array([f["mean_width_um"] for f in per_frame])
    rel = np.array([f["rel_area_pct"] for f in per_frame])
    keep = rel >= fit_min_rel
    out = {"fit_n_frames": int(keep.sum())}
    if keep.sum() < 3:
        out.update({k: np.nan for k in ("area_slope_um2_per_h", "area_intercept_um2",
                                        "area_r2", "area_se", "area_p",
                                        "front_velocity_um_per_h", "width_r2")})
        return out
    la = linregress(t[keep], A[keep])
    lw = linregress(t[keep], w[keep])
    out.update({"area_slope_um2_per_h": la.slope, "area_intercept_um2": la.intercept,
                "area_r2": la.rvalue ** 2, "area_se": la.stderr, "area_p": la.pvalue,
                "front_velocity_um_per_h": -lw.slope / 2.0,
                "width_r2": lw.rvalue ** 2})
    return out


# ----------------------------------------------------------- QC and extras
def overlay_rgb(g8, wound, total, W0):
    rgb = np.stack([g8] * 3, -1).astype(np.float32)
    other = total & ~wound
    for m, col in ((wound, (77, 187, 213)), (other, (243, 155, 127))):
        rgb[m] = 0.55 * rgb[m] + 0.45 * np.array(col, np.float32)
    edge = W0 & ~ndi.binary_erosion(W0, iterations=2)
    rgb[edge] = (230, 75, 53)
    return rgb.astype(np.uint8)


def base8(maps, i, a_ref):
    if maps["img8"]:
        return maps["img8"][i]
    H, W = maps["W0"].shape
    g = np.asarray(Image.fromarray(maps["imgs"][i]).resize((W, H), Image.BOX))
    lo, hi = a_ref
    return (np.clip((g - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def qc_montage(maps, per_frame, title, path):
    n = len(per_frame)
    idx = sorted({0, n // 2, n - 1})
    H, W = maps["W0"].shape
    g0 = np.asarray(Image.fromarray(maps["imgs"][0]).resize((W, H), Image.BOX))
    ref = np.percentile(g0, [0.5, 99.5])
    ref = (ref[0], max(ref[1], ref[0] + 1))
    font = get_font(max(14, H // 22))
    panels = []
    for i in idx:
        im = Image.fromarray(overlay_rgb(base8(maps, i, ref), maps["wounds"][i],
                                         maps["totals"][i], maps["W0"]))
        f = per_frame[i]
        ImageDraw.Draw(im).text(
            (10, 10), f"{f['time_min']} min   wound {f['rel_area_pct']:.0f}% of t0",
            font=font, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        panels.append(im)
    gap, head = 12, 80
    mont = Image.new("RGB", (len(panels) * W + (len(panels) - 1) * gap, H + head),
                     (255, 255, 255))
    for j, p in enumerate(panels):
        mont.paste(p, (j * (W + gap), head))
    d = ImageDraw.Draw(mont)
    d.text((10, 8), title, font=get_font(28), fill=(0, 0, 0))
    d.text((10, 44), "blue = wound (open, inside t0 scratch)   orange = other "
           "open gaps   red line = t0 scratch", font=get_font(20), fill=(60, 60, 60))
    mont.thumbnail((2400, 2400))
    path.parent.mkdir(parents=True, exist_ok=True)
    mont.save(path)


def save_maps(maps, per_frame, tag, out):
    edir, bdir = out / "maps" / tag / "entropy", out / "maps" / tag / "binary"
    edir.mkdir(parents=True, exist_ok=True)
    bdir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(per_frame):
        name = f"T{f['timepoint']:03d}_{f['time_min']:04d}min.png"
        Image.fromarray(maps["ent8"][i]).save(edir / name)
        Image.fromarray((maps["totals"][i] * 255).astype(np.uint8)).save(bdir / name)


def save_gif(maps, per_frame, tag, out, fps=8, max_w=1500):
    import imageio.v2 as imageio
    frames = []
    font = None
    for i, f in enumerate(per_frame):
        g = maps["img8"][i]
        ov = overlay_rgb(g, maps["wounds"][i], maps["totals"][i], maps["W0"])
        panel = np.concatenate([np.stack([g] * 3, -1),
                                np.stack([maps["ent8"][i]] * 3, -1), ov], axis=1)
        im = Image.fromarray(panel)
        im.thumbnail((max_w, max_w))
        if font is None:
            font = get_font(max(12, im.height // 14))
        ImageDraw.Draw(im).text((8, 6), f"{f['time_min']} min", font=font,
                                fill=(255, 255, 255), stroke_width=2,
                                stroke_fill=(0, 0, 0))
        frames.append(np.asarray(im))
    gdir = out / "gifs"
    gdir.mkdir(exist_ok=True)
    imageio.mimsave(gdir / f"{tag}.gif", frames, duration=1.0 / fps, loop=0)


def field_fit_plot(per_frame, fit, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.array([f["time_min"] for f in per_frame]) / 60
    A = np.array([f["wound_area_um2"] for f in per_frame]) / 1e3
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    ax.plot(t, A, "o", ms=3, color=NPG[3], mfc="white")
    if np.isfinite(fit["area_slope_um2_per_h"]):
        ax.plot(t, (fit["area_slope_um2_per_h"] * t + fit["area_intercept_um2"]) / 1e3,
                color=NPG[0], lw=1)
        ax.text(0.97, 0.95, f"slope {fit['area_slope_um2_per_h'] / 1e3:.1f}"
                f" ×10³ µm²/h\nR² = {fit['area_r2']:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Wound area (×10³ µm²)")
    ax.set_title(title, fontsize=8)
    ax.set_ylim(bottom=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


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

    conds, treatments, cells = [], [], []
    for f in fields:
        c = (f["cell_line"], f["treatment"])
        if c not in conds:
            conds.append(c)
        if c[1] not in treatments:
            treatments.append(c[1])
        if c[0] not in cells:
            cells.append(c[0])
    tpal = [NPG[3], NPG[0], NPG[2], NPG[1], NPG[4], NPG[5]]
    tcolor = {t: tpal[i % 6] for i, t in enumerate(treatments)}
    cstyle = {c: s for c, s in zip(cells, [("-", "o", True), ("--", "s", False),
                                           (":", "^", True), ("-.", "D", False)])}

    for key, ylab, fname in (("rel_area_pct", "Wound area (% of t0)",
                              "wound_area_timecourse"),
                             ("total_open_area_um2", "Total open area (µm²)",
                              "total_open_area_timecourse")):
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        for cell, treat in conds:
            by_t = {}
            for r in frames:
                if r["cell_line"] == cell and r["treatment"] == treat:
                    by_t.setdefault(r["time_min"], []).append(r[key])
            ts = sorted(by_t)
            t = np.array(ts) / 60.0
            m = np.array([np.nanmean(by_t[k]) for k in ts])
            sd = np.array([np.nanstd(by_t[k], ddof=1) if len(by_t[k]) > 1 else 0
                           for k in ts])
            ls, mk, filled = cstyle[cell]
            col = tcolor[treat]
            ax.fill_between(t, m - sd, m + sd, color=col, alpha=0.15, lw=0)
            ax.plot(t, m, ls=ls, color=col, lw=1.2, marker=mk, ms=3.5,
                    markevery=max(1, len(t) // 8), mfc=col if filled else "white",
                    mec=col, label=f"{cell} {treat}")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel(ylab)
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=6.5, loc="lower left")
        fig.tight_layout()
        for ext in ("svg", "png"):
            fig.savefig(out / f"{fname}.{ext}", dpi=300)
        plt.close(fig)

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
            jit = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else [0]
            ax.scatter(i + np.asarray(jit), vals, s=14, marker=mk,
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
    ap.add_argument("--entropy-radius-um", type=float, default=20.0)
    ap.add_argument("--levels", type=int, default=16)
    ap.add_argument("--otsu", choices=["t0", "per-frame"], default="t0")
    ap.add_argument("--thr-offset", type=float, default=0.0)
    ap.add_argument("--min-speck-um2", type=float, default=200.0)
    ap.add_argument("--fill-debris-um2", type=float, default=150.0)
    ap.add_argument("--min-open-um2", type=float, default=20000.0)
    ap.add_argument("--fit-min-rel", type=float, default=5.0)
    ap.add_argument("--save-maps", action="store_true")
    ap.add_argument("--gif", action="store_true")
    a = ap.parse_args()

    root = Path(a.inp)
    if root.suffix.lower() == ".czi" or not root.is_dir():
        sys.exit(f"--in must be the PNG export folder that contains manifest.csv, "
                 f"not '{root}'.  e.g.  --in /path/to/export")
    fields = load_manifest(root)
    out = Path(a.out) if a.out else root / "scratch_analysis"
    out.mkdir(parents=True, exist_ok=True)
    dev = pick_device(a.device)
    print(f"{len(fields)} fields found\n")

    all_frames, all_fields, report = [], [], []
    t_start = time.time()
    for (well, fnum), rows in fields.items():
        t0 = time.time()
        cond = rows[0]["condition"]
        cell, treat = split_condition(cond, well)
        per_frame, info, maps = analyse_field(rows, root, dev, a)
        fit = fits(per_frame, a.fit_min_rel)
        tag = f"{well}_{cond}_F{fnum}" if cond else f"{well}_F{fnum}"
        title = f"{well}  {cell} {treat}  F{fnum}"

        base = {"well": well, "condition": cond, "cell_line": cell,
                "treatment": treat, "field": fnum}
        for f in per_frame:
            all_frames.append({**base, **f})
        all_fields.append({**base,
                           "wound_area_t0_um2": per_frame[0]["wound_area_um2"],
                           "width_t0_um": per_frame[0]["mean_width_um"],
                           "total_open_t0_um2": per_frame[0]["total_open_area_um2"],
                           "closure_end_pct": per_frame[-1]["closure_pct"],
                           **fit, **info})

        qc_montage(maps, per_frame, title, out / "qc" / f"{tag}_qc.png")
        field_fit_plot(per_frame, fit, title, out / "qc" / f"{tag}_area_fit.png")
        if a.save_maps:
            save_maps(maps, per_frame, tag, out)
        if a.gif:
            save_gif(maps, per_frame, tag, out)

        report += [f"{title}",
                   f"  Wound area: Y = {fit['area_slope_um2_per_h']:.1f} X + "
                   f"{fit['area_intercept_um2']:.1f}   (um^2 vs h)",
                   f"  R^2 = {fit['area_r2']:.3f}   SE = {fit['area_se']:.1f}   "
                   f"p = {fit['area_p']:.2g}   frames used = {fit['fit_n_frames']}",
                   f"  Width t0 = {per_frame[0]['mean_width_um']:.0f} um   "
                   f"closure at end = {per_frame[-1]['closure_pct']:.1f} %   "
                   f"front velocity = {fit['front_velocity_um_per_h']:.2f} um/h "
                   f"(width R^2 = {fit['width_r2']:.3f})", ""]
        print(f"{well} F{fnum} {cell:<9} {treat:<14} width0 "
              f"{per_frame[0]['mean_width_um']:5.0f} um  closure "
              f"{per_frame[-1]['closure_pct']:5.1f}%  v "
              f"{fit['front_velocity_um_per_h']:5.2f} um/h  R2 {fit['width_r2']:.2f}"
              f"  ({time.time() - t0:.1f} s)")

    summary, seen = [], []
    for f in all_fields:
        c = (f["cell_line"], f["treatment"])
        if c in seen:
            continue
        seen.append(c)
        grp = [g for g in all_fields if (g["cell_line"], g["treatment"]) == c]
        row = {"cell_line": c[0], "treatment": c[1], "n_fields": len(grp)}
        for k in ("width_t0_um", "closure_end_pct", "front_velocity_um_per_h",
                  "area_slope_um2_per_h"):
            v = np.array([g[k] for g in grp], dtype=float)
            v = v[np.isfinite(v)]
            row[f"{k}_mean"] = float(v.mean()) if v.size else np.nan
            row[f"{k}_sd"] = float(v.std(ddof=1)) if v.size > 1 else np.nan
        summary.append(row)

    write_csv(out / "per_frame.csv", all_frames,
              ["well", "condition", "cell_line", "treatment", "field", "timepoint",
               "time_min", "wound_area_um2", "total_open_area_um2", "mean_width_um",
               "rel_area_pct", "closure_pct", "threshold_bits"])
    write_csv(out / "per_field.csv", all_fields,
              ["well", "condition", "cell_line", "treatment", "field",
               "wound_area_t0_um2", "width_t0_um", "total_open_t0_um2",
               "closure_end_pct", "front_velocity_um_per_h", "width_r2",
               "area_slope_um2_per_h", "area_intercept_um2", "area_r2", "area_se",
               "area_p", "fit_n_frames", "pixel_um", "analysis_pixel_um",
               "smooth_px", "entropy_radius_px", "levels", "threshold_t0_bits",
               "orientation", "scratch_length_um"])
    write_csv(out / "per_condition.csv", summary, list(summary[0].keys()))
    params = "  ".join(f"{k}={v}" for k, v in vars(a).items())
    (out / "scratch_report.txt").write_text(
        "Scratch assay report (entropy + Otsu, GPU)\n" + params + "\n\n"
        + "\n".join(report))
    make_figures(all_frames, all_fields, out)
    print(f"\nDone in {time.time() - t_start:.0f} s -> {out}")
    print("Check qc/*_qc.png before trusting the numbers.")


if __name__ == "__main__":
    main()
