"""
extract_noab_design.py — Deterministically extract the 7-factor process design
from the NPS SEED Center NOAB workbook for the LAFusion 2026 experiment.

Provenance (NOAB_Mixed_Designs_v4.xlsx, v4):
  * Family "up to 75 factors" -> a 128-design-point NOAB mixed design (the
    "up to 300 factors" family is 512 points; we need only 7 factors, so 128
    points is the right, non-oversized choice).
  * Sheet 'CodedValues up to 75 factors' holds the generated design. Columns
    1..50 are discrete factors (2- to 11-level blocks, 5 each); columns 51..75
    are the 25 CONTINUOUS factors (cont1..cont25), each a balanced permutation
    of the integers 1..128 (coded range [1, 128]).
  * The continuous factors are generic placeholders with no meaningful order,
    so we take the first seven (cont1..cont7) and map them, in order, onto the
    experiment's seven continuous process factors. The design is near-orthogonal
    (max |pairwise correlation| among cont1..cont7 ~ 0.004), so any 7-of-25
    choice is equivalent; cont1..cont7 is chosen for reproducibility.

Mapping (documented, since NOAB labels are placeholders):
  cont1 -> sigma_b, cont2 -> sigma_r, cont3 -> tau, cont4 -> rho,
  cont5 -> p_o,     cont6 -> p_d,     cont7 -> sd

Output: nob_design_raw.csv (128 rows, columns cont1..cont7, coded ints 1..128).
Feed to farm.py with --nob-coded-lo 1 --nob-coded-hi 128 (load_nob_design then
rescales each column to its PROCESS_FACTORS range).

    python3 extract_noab_design.py [workbook.xlsx] [out.csv]
"""
import sys
import numpy as np
import pandas as pd

WORKBOOK = sys.argv[1] if len(sys.argv) > 1 else "NOAB_Mixed_Designs_v4.xlsx"
OUT = sys.argv[2] if len(sys.argv) > 2 else "nob_design_raw.csv"
SHEET = "CodedValues up to 75 factors"
N_POINTS = 128
CONT_START_COL = 51   # 1-based column of cont1 (after 50 discrete factor columns)
N_KEEP = 7
CODED_RANGE = (1, 128)


def main():
    cv = pd.ExcelFile(WORKBOOK).parse(SHEET, header=None)
    # Data block: numeric rows, factor columns 1..75 (column 0 is a blank gutter).
    block = cv.iloc[:, 1:76].apply(pd.to_numeric, errors="coerce").dropna(how="all")
    block = block.dropna(how="any").reset_index(drop=True)
    if len(block) != N_POINTS:
        raise SystemExit(f"expected {N_POINTS} design points, got {len(block)} "
                         f"in sheet {SHEET!r} of {WORKBOOK}")

    cont = block.iloc[:, CONT_START_COL - 1: CONT_START_COL - 1 + 25]  # 25 continuous
    seven = cont.iloc[:, :N_KEEP].astype(int)
    seven.columns = [f"cont{i}" for i in range(1, N_KEEP + 1)]

    # Verify NOAB properties before writing.
    lo, hi = CODED_RANGE
    for c in seven.columns:
        col = np.sort(seven[c].to_numpy())
        if not np.array_equal(col, np.arange(lo, hi + 1)):
            raise SystemExit(f"column {c} is not a balanced permutation of "
                             f"{lo}..{hi}; workbook layout may differ — re-inspect.")
    corr = np.array(seven.corr().to_numpy(), dtype=float)
    np.fill_diagonal(corr, 0.0)
    max_corr = float(np.abs(corr).max())

    seven.to_csv(OUT, index=False)
    print(f"Wrote {OUT}: {seven.shape[0]} points x {N_KEEP} continuous factors "
          f"(cont1..cont{N_KEEP}), coded range {CODED_RANGE}.")
    print(f"NOAB check: each column is a balanced 1..{hi} permutation; "
          f"max |pairwise corr| = {max_corr:.4f} (near-orthogonal).")


if __name__ == "__main__":
    main()
