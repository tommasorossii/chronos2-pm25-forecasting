import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

IQR_K = 1.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=C.RAW_NAN_CSV)
    ap.add_argument("--clean", type=Path, default=C.DATA_DIR / "osc_prepared.csv")
    ap.add_argument("--output", type=Path, default=C.DATA_DIR / "osc_prepared_nan.csv")
    args = ap.parse_args()

    clean = pd.read_csv(args.clean, parse_dates=[C.TIME_COL])
    raw = pd.read_csv(args.raw, parse_dates=[C.TIME_COL])
    raw[C.SENSOR_COL] = raw[C.SENSOR_COL].str.strip().str.lower()
    value_cols = [C.LCS_COL, C.RS_COL] + C.METEO_COLS

    parts, rows = [], []
    for sensor in C.SELECTED_SENSORS:
        g = clean[clean[C.SENSOR_COL] == sensor].sort_values(C.TIME_COL)
        r = raw[raw[C.SENSOR_COL] == sensor]
        if r.empty:
            raise SystemExit(f"Sensore {sensor} assente nel dataset grezzo")

        n_dup = int(r[C.TIME_COL].duplicated().sum())
        r = r.groupby(C.TIME_COL)[value_cols].mean()

        q1, q3 = r[C.LCS_COL].quantile([0.25, 0.75])
        lo, hi = q1 - IQR_K * (q3 - q1), q3 + IQR_K * (q3 - q1)
        out_mask = (r[C.LCS_COL] < lo) | (r[C.LCS_COL] > hi)
        r.loc[out_mask, C.LCS_COL] = np.nan

        d = r.reindex(g[C.TIME_COL])
        d = d.reset_index().rename(columns={"index": C.TIME_COL})
        d[C.SENSOR_COL] = sensor
        d[C.CITY_COL] = g[C.CITY_COL].iloc[0]
        d["split"] = g["split"].to_numpy()
        absent = d[C.RS_COL].isna().to_numpy()
        d[C.MISSING_COL] = g[C.MISSING_COL].to_numpy().astype(bool) | absent
        d = d[[C.TIME_COL, C.SENSOR_COL, C.CITY_COL] + value_cols + [C.MISSING_COL, "split"]]
        parts.append(d)

        te = d[d["split"] == "test"]
        rows.append({
            "sensor": sensor, "city": d[C.CITY_COL].iloc[0], "n": len(d),
            "dup. mediati": n_dup,
            "outlier LCS -> NaN": int(out_mask.sum()),
            "soglia IQR LCS": round(hi, 1),
            "% NaN LCS": round(100 * d[C.LCS_COL].isna().mean(), 1),
            "% NaN LCS test": round(100 * te[C.LCS_COL].isna().mean(), 1),
            "punti valutati test": int((~te[C.MISSING_COL]).sum()),
            "punti valutati (pulito)": int((~g[g["split"] == "test"][C.MISSING_COL]).sum()),
        })

    out = pd.concat(parts, ignore_index=True)
    print(pd.DataFrame(rows).to_string(index=False))
    out.to_csv(args.output, index=False)
    print(f"\nSalvato: {args.output}  ({len(out):,} righe)")
    print("Per usarlo: export PREPARED_FILE=osc_prepared_nan.csv")


if __name__ == "__main__":
    main()
