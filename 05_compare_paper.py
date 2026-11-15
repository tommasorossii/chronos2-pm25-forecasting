import argparse

import numpy as np
import pandas as pd

import config as C

PAPER_TABLE7 = [
    {"model": "ARIMAX",        "I": None, "H": None, "MAE": 13.37, "RMSE": 18.10, "SMAPE": 0.55, "R2": -0.35},
    {"model": "Transformer",   "I": 24,   "H": 12,   "MAE": 10.00, "RMSE": 13.52, "SMAPE": 0.35, "R2": 0.44},
    {"model": "Granite_tuned", "I": 52,   "H": 1,    "MAE": 3.90,  "RMSE": 6.10,  "SMAPE": 0.16, "R2": 0.83},
    {"model": "Granite_tuned", "I": 52,   "H": 12,   "MAE": 8.59,  "RMSE": 11.85, "SMAPE": 0.29, "R2": 0.57},
]

PAPER_FIG13 = {
    "CNN": {
        1:  {12: 20.19, 24: 19.69, 48: 20.87, 72: 19.04, 512: 19.96},
        12: {12: 10.35, 24: 12.20, 48: 10.93, 72: 10.65, 512: 14.09},
        24: {12: 12.15, 24: 12.90, 48: 12.57, 72: 13.24, 512: 16.78},
        48: {24: 14.08, 48: 16.26, 72: 13.25, 512: None},
        72: {48: 16.25, 72: 15.47, 512: 15.42},
    },
    "LSTM": {
        1:  {12: 19.85, 24: 19.88, 48: 20.05, 72: 21.14, 512: 20.13},
        12: {12: 10.66, 24: 10.59, 48: 11.01, 72: 10.68, 512: 10.73},
        24: {12: 11.28, 24: 10.83, 48: 11.89, 72: 12.28, 512: 11.65},
        48: {24: 13.10, 48: 14.06, 72: 13.72, 512: 13.11},
        72: {48: 15.53, 72: 13.87, 512: 15.22},
    },
    "Transformer": {
        1:  {12: 19.52, 24: 19.44, 48: 20.14, 72: 19.86, 512: 20.88},
        12: {12: 10.83, 24: 10.00, 48: 11.01, 72: 10.48, 512: 11.60},
        24: {12: 11.56, 24: 11.02, 48: 12.03, 72: 11.83, 512: 13.42},
        48: {24: 12.93, 48: 12.91, 72: 12.80, 512: 13.15},
        72: {48: 13.92, 72: 15.06, 512: 16.26},
    },
    "Granite_tuned": {
        1:  {52: 3.90, 90: 3.99, 180: 3.97, 512: 4.23},
        12: {52: 8.59, 90: 8.75, 180: 8.92, 512: 8.84},
        24: {90: 10.85, 180: 10.45, 512: 10.83},
        48: {180: 11.43, 512: 12.72},
        72: {512: 13.56},
    },
}

PAPER_TABLE8 = pd.DataFrame([
    ("Aosta",        2.58,  4.87,   0.48,   1.55,  2.29, 0.57),
    ("Badajoz",      8.26, 11.39,   0.11,   2.23,  3.49, 0.46),
    ("Bangalore",   12.63, 16.98,  -0.43,   4.53,  6.23, 0.56),
    ("Calgary",      3.88,  5.79,  -0.08,   0.96,  1.74, 0.92),
    ("Delhi",       69.04, 93.23,  -0.39,  33.51, 48.16, 0.77),
    ("Hamirpur",    23.01, 30.73,   0.36,  23.39, 30.53, 0.63),
    ("Lima",         7.15,  9.91,  -0.34,   4.19,  6.31, 0.20),
    ("Reggio",      32.59, 47.48, -10.20,   6.09,  7.85, 0.55),
    ("Southampton",  4.02,  9.44,   0.09,   2.82,  3.98, 0.55),
    ("Trento",       6.91, 10.57,  -0.08,   3.90,  5.20, 0.62),
], columns=["city", "Raw_MAE", "Raw_RMSE", "Raw_R2", "Granite_MAE", "Granite_RMSE", "Granite_R2"])


def to_markdown(df, floatfmt="{:.2f}"):
    def fmt(v):
        if isinstance(v, float):
            return "" if np.isnan(v) else floatfmt.format(v)
        return "" if v is None else str(v)
    lines = ["| " + " | ".join(df.columns) + " |",
             "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def best_from_fig13(model):
    rows = []
    for H, by_i in PAPER_FIG13[model].items():
        vals = {i: v for i, v in by_i.items() if v is not None}
        i_best = min(vals, key=vals.get)
        rows.append({"model": model, "H": H, "I": i_best, "MAE": vals[i_best]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    args = ap.parse_args()
    out_dir = C.RESULTS_DIR / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    chronos = {}
    for tag in args.tags:
        osc = pd.read_csv(C.RESULTS_DIR / tag / "metrics_osc.csv")
        chronos[tag] = osc[osc["n_sensors"] == len(C.SELECTED_SENSORS)]

    t7 = [dict(r) for r in PAPER_TABLE7]
    for tag, osc in chronos.items():
        for H in (1, 12):
            sub = osc[osc["H"] == H]
            if sub.empty:
                continue
            b = sub.sort_values("MAE").iloc[0]
            t7.append({"model": f"Chronos-2 [{tag}]", "I": int(b["I"]), "H": H,
                       "MAE": b["MAE"], "RMSE": b["RMSE"], "SMAPE": b["SMAPE"], "R2": b["R2"]})
    t7 = pd.DataFrame(t7)
    t7.to_csv(out_dir / "table7_comparison.csv", index=False)
    (out_dir / "table7_comparison.md").write_text(to_markdown(t7))
    print("Confronto stile Tabella 7 (media OSC):")
    print(t7.to_string(index=False), "\n")

    rows = []
    for m in PAPER_FIG13:
        rows += best_from_fig13(m)
    for tag, osc in chronos.items():
        for H, sub in osc.groupby("H"):
            b = sub.sort_values("MAE").iloc[0]
            rows.append({"model": f"Chronos-2 [{tag}]", "H": int(H), "I": int(b["I"]), "MAE": b["MAE"]})
    by_h = pd.DataFrame(rows)
    by_h.to_csv(out_dir / "mae_by_horizon.csv", index=False)
    piv = by_h.pivot(index="H", columns="model", values="MAE")
    print("MAE OSC per orizzonte (migliore input window per ciascun modello):")
    print(piv.round(2).to_string(), "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for model in piv.columns:
        ax.plot(piv.index, piv[model], marker="o", label=model,
                linewidth=2.5 if model.startswith("Chronos") else 1.5)
    ax.set_xticks(C.HORIZONS)
    ax.set_xlabel("Orizzonte di previsione H (h)")
    ax.set_ylabel("MAE medio OSC")
    ax.set_title("Scenario Y: Chronos-2 vs modelli del paper")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "mae_by_horizon.png", dpi=200)
    plt.close(fig)

    rows = []
    for tag, osc in chronos.items():
        for r in osc.itertuples():
            row = {"tag": tag, "I": r.I, "H": r.H, "Chronos-2": r.MAE}
            for m in ("CNN", "LSTM", "Transformer"):
                row[m] = PAPER_FIG13[m].get(r.H, {}).get(r.I)
            rows.append(row)
    same = pd.DataFrame(rows).sort_values(["tag", "H", "I"])
    same.to_csv(out_dir / "same_config.csv", index=False)

    for tag in args.tags:
        ps = pd.read_csv(C.RESULTS_DIR / tag / "metrics_per_sensor.csv")
        ps = ps[ps["H"] > 1]
        best_city = (ps.sort_values("MAE").groupby("city", as_index=False).first()
                     [["city", "I", "H", "MAE", "RMSE", "R2"]]
                     .rename(columns={"I": "Chronos_I", "H": "Chronos_H", "MAE": "Chronos_MAE",
                                      "RMSE": "Chronos_RMSE", "R2": "Chronos_R2"}))
        city = PAPER_TABLE8.merge(best_city, on="city", how="left")
        city.to_csv(out_dir / f"per_city_{tag}.csv", index=False)
        (out_dir / f"per_city_{tag}.md").write_text(to_markdown(city))
        print(f"Città per città [{tag}] (H > 1):")
        print(city.round(2).to_string(index=False), "\n")

    print(f"Tutti i file in {out_dir}")


if __name__ == "__main__":
    main()
