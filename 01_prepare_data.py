import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=C.RAW_CSV)
    ap.add_argument("--output", type=Path, default=C.PREPARED_CSV)
    ap.add_argument("--allow-nan", action="store_true",
                    help="accetta NaN nelle colonne di input (versione non interpolata)")
    args = ap.parse_args()

    df = pd.read_csv(args.input, parse_dates=[C.TIME_COL])
    print(f"Lette {len(df):,} righe, {df[C.SENSOR_COL].nunique()} sensori")

    missing = [s for s in C.SELECTED_SENSORS if s not in set(df[C.SENSOR_COL])]
    if missing:
        raise SystemExit(f"Sensori OSC assenti nel CSV: {missing}")
    df = df[df[C.SENSOR_COL].isin(C.SELECTED_SENSORS)].copy()

    value_cols = [C.LCS_COL, C.RS_COL] + C.METEO_COLS
    if C.MISSING_COL not in df.columns:
        df[C.MISSING_COL] = df[[C.LCS_COL, C.RS_COL]].isna().any(axis=1)
    keep = [C.TIME_COL, C.SENSOR_COL, C.CITY_COL] + value_cols + [C.MISSING_COL]
    df = df[keep]

    parts = []
    for sensor in C.SELECTED_SENSORS:
        d = df[df[C.SENSOR_COL] == sensor].sort_values(C.TIME_COL)
        if d[C.TIME_COL].duplicated().any():
            raise SystemExit(f"Timestamp duplicati per {sensor}")
        city = d[C.CITY_COL].iloc[0]
        full = pd.date_range(d[C.TIME_COL].min(), d[C.TIME_COL].max(), freq="h")
        n_gaps = len(full) - len(d)
        d = d.set_index(C.TIME_COL).reindex(full)
        d.index.name = C.TIME_COL
        d[C.SENSOR_COL] = sensor
        d[C.CITY_COL] = city
        d[C.MISSING_COL] = d[C.MISSING_COL].fillna(True).astype(bool)
        d = d.reset_index()
        if n_gaps:
            print(f"  {sensor}: aggiunte {n_gaps} ore mancanti (NaN, is_missing=True)")

        n = len(d)
        n_train = int(np.floor(n * C.TRAIN_FRAC))
        d["split"] = np.where(np.arange(n) < n_train, "train", "test")
        parts.append(d)
    out = pd.concat(parts, ignore_index=True)

    n_nan = out[value_cols].isna().sum()
    if n_nan.any() and not args.allow_nan:
        raise SystemExit("Trovati NaN nelle colonne di input:\n"
                         f"{n_nan[n_nan > 0]}\nUsa --allow-nan per la versione grezza.")

    rows = []
    for sensor, d in out.groupby(C.SENSOR_COL, sort=False):
        te = d[d["split"] == "test"]
        rows.append({
            "sensor": sensor,
            "city": d[C.CITY_COL].iloc[0],
            "n": len(d),
            "train": int((d["split"] == "train").sum()),
            "test": len(te),
            "start": d[C.TIME_COL].min().date(),
            "end": d[C.TIME_COL].max().date(),
            "% interp. tot": round(100 * d[C.MISSING_COL].mean(), 2),
            "% interp. test": round(100 * te[C.MISSING_COL].mean(), 2),
        })
    print("\n" + pd.DataFrame(rows).to_string(index=False))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"\nSalvato: {args.output}  ({len(out):,} righe)")


if __name__ == "__main__":
    main()
