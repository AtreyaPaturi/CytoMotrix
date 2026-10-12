#!/usr/bin/env python3
"""
make_plate_map.py

Write a plate-map template (CSV) for a 6-, 8-, 12-, 24- or 96-well plate.
Fill in cell_line and treatment for each well you imaged (Excel, LibreOffice or
any text editor), then pass the file to split_czi_by_well_time.py --plate-map.
Wells left blank are treated as unused.

Plate formats (rows x columns):
     6 wells : A-B x 1-3
     8 wells : A-B x 1-4
    12 wells : A-C x 1-4
    24 wells : A-D x 1-6
    96 wells : A-H x 1-12

Blank template:
    python3 make_plate_map.py --wells 6 --out plate_map.csv

Pre-filled: one cell line per plate ROW, one treatment per plate COLUMN
(extra rows / columns stay blank):
    python3 make_plate_map.py --wells 6 --out plate_map.csv \\
        --cell-lines "HeLa WT,HeLa KO" --treatments "Control,DrugA,DrugA+DrugB"

    -> A1 HeLa WT  Control      A2 HeLa WT  DrugA      A3 HeLa WT  DrugA+DrugB
       B1 HeLa KO  Control      B2 HeLa KO  DrugA      B3 HeLa KO  DrugA+DrugB

Options:
    --wells N          6 | 8 | 12 | 24 | 96
    --out PATH         output CSV (default plate_map.csv)
    --cell-lines LIST  comma-separated, assigned to rows A, B, C, ...
    --treatments LIST  comma-separated, assigned to columns 1, 2, 3, ...
    --by-column        swap: cell lines go to columns, treatments to rows
    --overwrite        replace an existing file
"""

import argparse
import csv
import sys
from pathlib import Path

PLATES = {6: (2, 3), 8: (2, 4), 12: (3, 4), 24: (4, 6), 96: (8, 12)}


def wells_for(n):
    rows, cols = PLATES[n]
    return [(r, c, f"{chr(ord('A') + r)}{c + 1}")
            for r in range(rows) for c in range(cols)]


def split_list(s):
    return [x.strip() for x in s.split(",") if x.strip()] if s else []


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wells", type=int, required=True, choices=sorted(PLATES))
    ap.add_argument("--out", default="plate_map.csv")
    ap.add_argument("--cell-lines", default="")
    ap.add_argument("--treatments", default="")
    ap.add_argument("--by-column", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists() and not a.overwrite:
        sys.exit(f"{out} already exists (use --overwrite)")

    nrows, ncols = PLATES[a.wells]
    cells, treats = split_list(a.cell_lines), split_list(a.treatments)
    n_cell_slots, n_treat_slots = (ncols, nrows) if a.by_column else (nrows, ncols)
    if len(cells) > n_cell_slots:
        sys.exit(f"{len(cells)} cell lines but only {n_cell_slots} "
                 f"{'columns' if a.by_column else 'rows'} on a {a.wells}-well plate")
    if len(treats) > n_treat_slots:
        sys.exit(f"{len(treats)} treatments but only {n_treat_slots} "
                 f"{'rows' if a.by_column else 'columns'} on a {a.wells}-well plate")

    rows = []
    for r, c, well in wells_for(a.wells):
        ci, ti = (c, r) if a.by_column else (r, c)
        cell = cells[ci] if ci < len(cells) else ""
        treat = treats[ti] if ti < len(treats) else ""
        if not (cell and treat):          # half-filled well = unused
            cell = treat = ""
        rows.append({"well": well, "cell_line": cell, "treatment": treat})

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["well", "cell_line", "treatment"])
        w.writeheader()
        w.writerows(rows)

    used = sum(1 for r in rows if r["cell_line"])
    print(f"{a.wells}-well template -> {out}  ({used} of {len(rows)} wells filled)")
    grid = [[""] * ncols for _ in range(nrows)]
    for (r, c, _), row in zip(wells_for(a.wells), rows):
        grid[r][c] = f"{row['cell_line']} | {row['treatment']}" if row["cell_line"] else "-"
    wcol = max(len(x) for g in grid for x in g) + 2
    print("    " + "".join(f"{c + 1:<{wcol}}" for c in range(ncols)))
    for r, g in enumerate(grid):
        print(f"{chr(ord('A') + r):<4}" + "".join(f"{x:<{wcol}}" for x in g))


if __name__ == "__main__":
    main()
