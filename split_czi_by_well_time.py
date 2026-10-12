#!/usr/bin/env python3
"""
split_czi_by_well_time.py

Split a Zeiss Celldiscoverer 7 time-lapse CZI (multi-scene = wells x fields)
into one PNG per well / field / time point (+ manifest CSV), and/or one
annotated MP4 video per field.

PNGs are 16-bit lossless raw data by default (usable for quantification;
look dark in ordinary image viewers - open in Fiji). --png-8bit writes
contrast-scaled 8-bit PNGs for viewing/slides instead.

Install (once):
    pip install pylibCZIrw tifffile numpy imageio imageio-ffmpeg pillow

Well -> condition labels come from a plate-map CSV (columns: well, cell_line,
treatment). Make one with make_plate_map.py (6, 8, 12, 24 or 96 wells) and
fill it in. Without --plate-map, wells are named by their well ID only.

Dry run (reads metadata only, writes nothing):
    python3 split_czi_by_well_time.py --in experiment.czi --plate-map plate_map.csv --dry-run

PNGs + videos:
    python3 split_czi_by_well_time.py --in experiment.czi --plate-map plate_map.csv --out /path/to/export --videos

Videos only:
    python3 split_czi_by_well_time.py --in experiment.czi --plate-map plate_map.csv --out /path/to/export --videos-only

Options:
    --in PATH            input .czi, or a folder containing exactly one .czi
    --out PATH           output folder (created if missing; not needed for --dry-run)
    --plate-map CSV      well, cell_line, treatment table (see make_plate_map.py)
    --interval MIN       frame interval in minutes (default 10)
    --layout by-time     <Well_Cond>/T###_####min/<file>.png   (default)
    --layout by-field    <Well_Cond>/F#/<file>.png
    --png-8bit           contrast-scaled 8-bit PNGs instead of raw 16-bit
    --stack              also write one T-stack TIFF per field (for tracking)
    --videos             also write one MP4 per field into <out>/videos/
    --videos-only        write only the MP4s (no TIFFs)
    --fps N              video frame rate (default 10 -> 73 frames = ~7 s)
    --video-scale F      downscale factor for videos, e.g. 0.5 (default 1.0)
    --fields-per-well N  fallback grouping if wells cannot be read from metadata (default 3)
    --overwrite          allow writing into a non-empty output folder
"""

import argparse
import csv
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import tifffile
from pylibCZIrw import czi as pyczi


# ---------------------------------------------------------------- plate map
def normalise_well(w):
    """'a01' / 'A1 ' -> 'A1'."""
    m = re.match(r"^\s*([A-Pa-p])\s*0*(\d{1,2})\s*$", w or "")
    return f"{m.group(1).upper()}{int(m.group(2))}" if m else (w or "").strip()


def load_plate_map(path):
    """Read well, cell_line, treatment CSV -> {well: (cell_line, treatment)}.
    Rows with an empty cell_line and treatment are unused wells and skipped."""
    p = Path(path)
    if not p.exists():
        sys.exit(f"Plate map not found: {p}  (make one with make_plate_map.py)")
    with open(p, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    need = {"well", "cell_line", "treatment"}
    have = {k.strip().lower() for k in (rows[0].keys() if rows else [])}
    if not need <= have:
        sys.exit(f"Plate map {p} needs the columns: well, cell_line, treatment")
    labels = {}
    for r in rows:
        r = {k.strip().lower(): (v or "").strip() for k, v in r.items() if k}
        well = normalise_well(r["well"])
        if not (r["cell_line"] or r["treatment"]):
            continue
        if well in labels:
            sys.exit(f"Plate map {p}: well {well} is listed twice")
        labels[well] = (r["cell_line"], r["treatment"])
    if not labels:
        sys.exit(f"Plate map {p} has no filled-in wells")
    return labels


def condition_name(cell, treat):
    """Folder/file condition tag 'Cell-line_Treatment'. The analysis scripts
    split on the first '_' and turn '-' back into spaces in the cell line."""
    cell = re.sub(r"[\s_]+", "-", cell.strip())
    treat = re.sub(r"\s+", "", treat.strip())
    return f"{cell}_{treat}" if treat else cell


# ----------------------------------------------------------------- metadata
def parse_scenes(xml_str):
    """Return {scene_index: {'name':..., 'well':...}} from CZI XML metadata."""
    root = ET.fromstring(xml_str)
    out = {}
    for sc in root.iter("Scene"):
        idx = sc.get("Index")
        if idx is None:
            continue
        idx = int(idx)
        name = (sc.get("Name") or "").strip()
        well = None
        arr = sc.findtext("ArrayName")
        if arr and arr.strip():
            well = arr.strip()
        if well is None:
            shp = sc.find("Shape")
            if shp is not None and shp.get("Name"):
                well = shp.get("Name").strip()
        if well is None:
            m = re.match(r"^([A-Pa-p]\d{1,2})\b", name)
            if m:
                well = m.group(1).upper()
        prev = out.get(idx)
        if prev is None or (prev["well"] is None and well is not None):
            out[idx] = {"name": name, "well": well}
    return out


def pixel_size_um(xml_str):
    root = ET.fromstring(xml_str)
    for d in root.iter("Distance"):
        if d.get("Id") == "X":
            v = d.findtext("Value")
            if v:
                return float(v) * 1e6  # stored in metres
    return None


def safe(s):
    return re.sub(r"[^A-Za-z0-9._+-]+", "-", s).strip("-")


# ---------------------------------------------------------------- image I/O
def dim_range(bb, key):
    start, end = bb.get(key, (0, 1))
    return range(start, end)


def read_plane(doc, scene, t, z, c):
    arr = doc.read(plane={"T": t, "Z": z, "C": c}, scene=scene)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr


def read_timepoint(doc, scene, t, Z, C):
    """Return (array, axes) for one time point, singleton Z/C dropped."""
    img = np.stack([np.stack([read_plane(doc, scene, t, z, c) for c in C])
                    for z in Z])                        # Z, C, Y, X
    axes = "ZCYX"
    if img.shape[1] == 1:
        img, axes = img[:, 0], "ZYX"
    if img.shape[0] == 1:
        img, axes = img[0], axes[1:]
    return img, axes


def plane2d(img, axes):
    """Middle Z, first C -> single 2D plane for video."""
    if "Z" in axes:
        img = img[img.shape[0] // 2]
        axes = axes.replace("Z", "", 1)
    while img.ndim > 2:
        img = img[0]
    return img


def write_tif(path, arr, px_um, interval_min, axes):
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.dtype in (np.uint8, np.uint16, np.float32):
        kw = {"imagej": True, "metadata": {"axes": axes, "unit": "um",
                                           "finterval": interval_min * 60}}
        if px_um:
            kw["resolution"] = (1.0 / px_um, 1.0 / px_um)
        tifffile.imwrite(path, arr, **kw)
    else:
        tifffile.imwrite(path, arr)


def write_png(path, img2d, px_um, lo=None, hi=None):
    """16-bit lossless PNG of the raw data, or 8-bit contrast-scaled if lo/hi given."""
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    if lo is not None:
        arr = to_uint8(img2d, lo, hi)
    elif img2d.dtype == np.uint8:
        arr = img2d
    else:
        arr = np.clip(img2d, 0, 65535).astype(np.uint16)
    kw = {}
    if px_um:
        dpi = 25400.0 / px_um                           # pixels per inch
        kw["dpi"] = (dpi, dpi)
    Image.fromarray(arr).save(path, **kw)


# -------------------------------------------------------------------- video
def contrast_limits(doc, scene, T, Z, C, lo_pct=0.5, hi_pct=99.5):
    """One display range per field (first/middle/last frame) -> no flicker."""
    ts = sorted({T[0], T[len(T) // 2], T[-1]})
    px = np.concatenate([plane2d(*read_timepoint(doc, scene, t, Z, C)).ravel()
                         for t in ts]).astype(np.float32)
    lo, hi = np.percentile(px, [lo_pct, hi_pct])
    return float(lo), float(max(hi, lo + 1))


def to_uint8(img, lo, hi):
    x = (img.astype(np.float32) - lo) / (hi - lo)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def get_font(size):
    from PIL import ImageFont
    for name in ("DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf",
                 "arialbd.ttf", "Arial Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)       # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def annotate(gray8, text_tr, text_br, font, stroke):
    """White text with black outline: top-right label, bottom-right time."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(gray8).convert("RGB")
    d = ImageDraw.Draw(im)
    W, H = im.size
    m = max(8, int(0.02 * min(W, H)))
    style = dict(font=font, fill=(255, 255, 255),
                 stroke_width=stroke, stroke_fill=(0, 0, 0))

    b = d.multiline_textbbox((0, 0), text_tr, font=font,
                             stroke_width=stroke, align="right")
    d.multiline_text((W - m - b[2], m - b[1]), text_tr, align="right", **style)

    b = d.textbbox((0, 0), text_br, font=font, stroke_width=stroke)
    d.text((W - m - b[2], H - m - b[3]), text_br, **style)
    return np.asarray(im)


def prep_frame(gray8, scale):
    from PIL import Image
    if scale != 1.0:
        im = Image.fromarray(gray8)
        im = im.resize((max(2, int(im.width * scale)),
                        max(2, int(im.height * scale))), Image.LANCZOS)
        gray8 = np.asarray(im)
    h, w = gray8.shape
    return gray8[: h - h % 2, : w - w % 2]              # H.264 needs even dims


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--plate-map", default=None)
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--layout", choices=["by-time", "by-field"], default="by-time")
    ap.add_argument("--stack", action="store_true")
    ap.add_argument("--videos", action="store_true")
    ap.add_argument("--videos-only", action="store_true")
    ap.add_argument("--png-8bit", action="store_true")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--video-scale", type=float, default=1.0)
    ap.add_argument("--fields-per-well", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    want_tif = not a.videos_only        # per-time-point PNG export
    want_vid = a.videos or a.videos_only
    if not a.dry_run:
        try:
            import PIL  # noqa: F401
        except ImportError:
            sys.exit("PNG export needs: pip install pillow")
    if want_vid and not a.dry_run:
        try:
            import imageio.v2 as imageio  # noqa: F401
            import imageio_ffmpeg          # noqa: F401
            import PIL                     # noqa: F401
        except ImportError:
            sys.exit("Videos need: pip install imageio imageio-ffmpeg pillow")

    if not a.dry_run and not a.out:
        sys.exit("--out is required (except with --dry-run)")
    plate = load_plate_map(a.plate_map) if a.plate_map else {}

    inp = Path(a.inp)
    if not inp.exists():
        sys.exit(f"Input not found: {inp}  (is the drive mounted / VPN up?)")
    if inp.is_dir():
        czis = sorted(inp.glob("*.czi"))
        if len(czis) == 1:
            inp = czis[0]
        elif not czis:
            sys.exit(f"No .czi file in folder: {inp}")
        else:
            sys.exit("Several .czi files in folder - pass one with --in:\n  "
                     + "\n  ".join(str(p) for p in czis))
        print(f"Using {inp}")

    if not a.dry_run and os.name != "nt" and re.match(r"^[A-Za-z]:[\\/]", a.out or ""):
        sys.exit(f"Output '{a.out}' is a Windows drive path but this is Linux.\n"
                 f"Pass a Linux path, e.g. --out /mnt/nas/my_export")

    with pyczi.open_czi(str(inp)) as doc:
        xml = doc.raw_metadata
        bb = doc.total_bounding_box
        scene_rects = doc.scenes_bounding_rectangle
        meta = parse_scenes(xml)
        px = pixel_size_um(xml)

        scene_ids = sorted(scene_rects.keys()) if scene_rects else [0]
        T, Z, C = dim_range(bb, "T"), dim_range(bb, "Z"), dim_range(bb, "C")

        # ---- scene table: well + field number
        table = []
        for s in scene_ids:
            m = meta.get(s, {"name": "", "well": None})
            table.append({"scene": s, "name": m["name"], "well": m["well"]})
        if any(r["well"] is None for r in table):
            print(f"! Well names not found in metadata for some scenes; "
                  f"grouping every {a.fields_per_well} scenes as one well.")
            for i, r in enumerate(table):
                r["well"] = f"Well{i // a.fields_per_well + 1:02d}"
        counters = {}
        for r in table:
            counters[r["well"]] = counters.get(r["well"], 0) + 1
            r["field"] = counters[r["well"]]
            lab = plate.get(normalise_well(r["well"]))
            r["label"] = lab
            r["cond"] = condition_name(*lab) if lab else ""

        print(f"File      : {inp}")
        print(f"Dims      : T={len(T)}  Z={len(Z)}  C={len(C)}  scenes={len(scene_ids)}")
        print(f"Pixel size: {px:.4f} um" if px else "Pixel size: not found")
        print(f"Duration  : {(len(T) - 1) * a.interval:.0f} min at {a.interval:g} min/frame")
        print("\nscene  well   field  condition            scene-name")
        for r in table:
            print(f"{r['scene']:>5}  {r['well']:<6} {r['field']:>5}  "
                  f"{r['cond'] or '(unmapped)':<20} {r['name']}")
        wells = sorted({r["well"] for r in table})
        print(f"\n{len(wells)} wells: {', '.join(wells)}")
        bad = [w for w, n in counters.items() if n != a.fields_per_well]
        if bad:
            print(f"! Wells without {a.fields_per_well} fields: {bad}")
        if plate:
            present = {normalise_well(w) for w in wells}
            unmapped = [w for w in wells if normalise_well(w) not in plate]
            if unmapped:
                print(f"! Wells in the CZI but not in the plate map "
                      f"(named by well ID only): {unmapped}")
            absent = sorted(w for w in plate if w not in present)
            if absent:
                print(f"! Wells in the plate map but not in the CZI: {absent}")
        else:
            print("! No --plate-map given: wells are named by well ID only "
                  "(make one with make_plate_map.py)")

        if a.dry_run:
            print("\nDry run only - nothing written.")
            return

        out = Path(a.out)
        check = out / "videos" if a.videos_only else out
        if check.exists() and any(check.iterdir()) and not a.overwrite:
            sys.exit(f"Output folder not empty: {check}  (use --overwrite)")
        out.mkdir(parents=True, exist_ok=True)

        if want_vid:
            import imageio.v2 as imageio
            (out / "videos").mkdir(exist_ok=True)

        manifest = []
        for r in table:
            s, well, f, cond = r["scene"], r["well"], r["field"], r["cond"]
            tag = safe(f"{well}_{cond}" if cond else well)
            wdir = out / tag
            stack = [] if (a.stack and want_tif) else None

            writer = font = None
            lo = hi = None
            if want_vid or (want_tif and a.png_8bit):
                lo, hi = contrast_limits(doc, s, T, Z, C)
            if want_vid:
                cell, treat = r["label"] or (well, "")
                label_tr = f"{cell}\n{treat}" if treat else cell
                vpath = out / "videos" / f"{tag}_F{f}.mp4"
                writer = imageio.get_writer(
                    str(vpath), fps=a.fps, codec="libx264", quality=8,
                    pixelformat="yuv420p", macro_block_size=1,
                    ffmpeg_log_level="error")

            for ti, t in enumerate(T):
                img, axes = read_timepoint(doc, s, t, Z, C)
                tmin = int(round(ti * a.interval))
                tlabel = f"T{ti + 1:03d}_{tmin:04d}min"

                if want_tif:
                    fname = f"{tag}_F{f}_{tlabel}.png"
                    p = (wdir / tlabel / fname if a.layout == "by-time"
                         else wdir / f"F{f}" / fname)
                    if a.png_8bit:
                        write_png(p, plane2d(img, axes), px, lo, hi)
                    else:
                        write_png(p, plane2d(img, axes), px)
                    manifest.append([well, cond, f, s, r["name"], ti + 1, tmin,
                                     str(p.relative_to(out))])
                    if stack is not None:
                        stack.append(img)

                if writer is not None:
                    g8 = prep_frame(to_uint8(plane2d(img, axes), lo, hi),
                                    a.video_scale)
                    if font is None:
                        fsize = max(14, int(0.045 * g8.shape[0]))
                        font, stroke = get_font(fsize), max(1, fsize // 12)
                    writer.append_data(annotate(g8, label_tr, f"{tmin} min",
                                                font, stroke))

                print(f"\r{tag} F{f}: {ti + 1}/{len(T)}", end="", flush=True)
            print()

            if writer is not None:
                writer.close()
            if stack is not None:
                write_tif(wdir / "stacks" / f"{tag}_F{f}_Tstack.tif",
                          np.stack(stack), px, a.interval, "T" + axes)

        if want_tif:
            with open(out / "manifest.csv", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["well", "condition", "field", "scene_index",
                            "scene_name", "timepoint", "time_min", "path"])
                w.writerows(manifest)
            print(f"\n{len(manifest)} PNGs -> {out}")
        if want_vid:
            print(f"{len(table)} videos -> {out / 'videos'}")


if __name__ == "__main__":
    main()
