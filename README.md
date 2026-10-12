# CytoMotrix

Scratch-wound (collective cell migration) analysis for time-lapse images from a
Zeiss Celldiscoverer 7. It splits a multi-scene `.czi` into per-well,
per-time-point images and then measures wound closure and migration-front
velocity on the GPU (PyTorch; falls back to CPU).

## Pipeline

```
experiment.czi ──► split_czi_by_well_time.py ──► PNGs + manifest.csv ──► analyze_wound_migration_gpu.py ─┐
                          ▲                                           └─► scratch_assay_gpu.py ──────────┴─► CSVs, figures, QC
                   plate_map.csv  ◄── make_plate_map.py
```

| Script | What it does |
|---|---|
| `make_plate_map.py` | Writes a plate-map template for **6, 8, 12, 24 or 96-well** plates, optionally pre-filled with cell lines (per row) and treatments (per column). |
| `split_czi_by_well_time.py` | Splits the CZI into 16-bit (or 8-bit) PNGs per well / field / time point, writes `manifest.csv`, and can also write annotated MP4s and T-stack TIFFs. |
| `analyze_wound_migration_gpu.py` | Wound detection by **local texture (standard deviation) + Otsu**. Uses the largest open region as the wound. |
| `scratch_assay_gpu.py` | Wound detection by **local entropy + Otsu**. Counts all open space inside the t0 scratch, plus linear-regression reports per field and optional entropy maps and GIFs. |

The two analysis scripts are alternatives. Run either or both on the same export
and compare the QC overlays.

## Install

Python ≥ 3.9.

```bash
pip install -r requirements.txt
```

For NVIDIA GPUs, install the CUDA build of PyTorch from <https://pytorch.org>
first. Apple-silicon Macs use MPS automatically.

## Usage

### 1. Define the plate layout

```bash
# blank template: open it in Excel and fill in cell_line + treatment per well
python make_plate_map.py --wells 24 --out plate_map.csv

# or pre-fill: one cell line per plate row, one treatment per plate column
python make_plate_map.py --wells 6 --out plate_map.csv \
    --cell-lines "HeLa WT,HeLa KO" --treatments "Control,DrugA,DrugA+DrugB"
```

| Wells | Rows × columns |
|---|---|
| 6 | A–B × 1–3 |
| 8 | A–B × 1–4 |
| 12 | A–C × 1–4 |
| 24 | A–D × 1–6 |
| 96 | A–H × 1–12 |

Use `--by-column` to put cell lines on columns and treatments on rows instead.
Wells left blank count as unused. The file is a plain three-column CSV
(`well,cell_line,treatment`), so you can also write it by hand. See
[`examples/plate_map_6well_example.csv`](examples/plate_map_6well_example.csv).

### 2. Split the CZI

```bash
# check wells, fields and the condition mapping first (writes nothing)
python split_czi_by_well_time.py --in experiment.czi --plate-map plate_map.csv --dry-run

# export 16-bit PNGs (+ optional videos)
python split_czi_by_well_time.py --in experiment.czi --plate-map plate_map.csv \
    --out /path/to/export --videos
```

Wells are read from the CZI's `ArrayName`/scene metadata. The dry run warns you
about any well that is in the CZI but not in the plate map, or the other way round.

### 3. Analyse

```bash
python analyze_wound_migration_gpu.py --in /path/to/export
python scratch_assay_gpu.py --in /path/to/export --gif --save-maps
```

Each script writes `per_frame.csv`, `per_field.csv`, `per_condition.csv`,
publication-style figures (SVG + PNG), and QC overlays in `qc/`. **Check the QC
overlays before trusting the numbers.** If the wound outline is too tight or
too loose, adjust `--thr-offset`. Run any script with `--help` to see all options.

## Metrics

- **Wound area** (µm²) and **relative area** (% of t0)
- **Closure** (%) = 100 − relative area
- **Mean wound width** = area / scratch length
- **Front velocity** (µm/h, per migrating edge) = −d(width)/dt / 2, from a linear
  fit over the frames where the wound is still > 5 % open (`--fit-min-rel`)

## Acknowledgements

The entropy + Otsu approach in `scratch_assay_gpu.py` builds on
S. Bhattiprolu, *Python for Microscopists* (`021-scratch_assay_using_scikit_image.py`),
and E. Gad, [Image-Processing-and-Analysis-of-the-Scratch-Assay](https://github.com/eyadgad/Image-Processing-and-Analysis-of-the-Scratch-Assay).

## License

MIT, see [LICENSE](LICENSE).
