import argparse
import re

import numpy as np
import pandas as pd

import config as C
import common as U

NAME_RE = re.compile(r"^(?P<sensor>.+)_I(?P<I>\d+)_H(?P<H>\d+)\.csv$")


def plot_heatmap(osc, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    piv = osc.pivot(index="H", columns="I", values="MAE").reindex(
        index=C.HORIZONS, columns=C.INPUT_WINDOWS)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    im = ax.imshow(piv.values, cmap="coolwarm", vmin=0, vmax=20, aspect="auto")
    ax.set_xticks(range(len(piv.columns)), piv.columns)
    ax.set_yticks(range(len(piv.index)), piv.index)
    ax.set_xlabel("Input Window (h)")
    ax.set_ylabel("Prediction Window (h)")
    ax.set_title(title)
    for r in range(piv.shape[0]):
        for c in range(piv.shape[1]):
            v = piv.values[r, c]
            if np.isfinite(v):
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="MAE")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    run_dir = C.RESULTS_DIR / args.tag
    pred_dir = run_dir / "preds"
    files = sorted(pred_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"Nessuna previsione in {pred_dir}")

    sensor_by_file = {s.replace(" ", "_"): s for s in C.SELECTED_SENSORS}
    df_meta = U.load_prepared()[[C.SENSOR_COL, C.CITY_COL]].drop_duplicates()
    city_of = dict(zip(df_meta[C.SENSOR_COL], df_meta[C.CITY_COL]))

    rows = []
    for f in files:
        mt = NAME_RE.match(f.name)
        if not mt or mt["sensor"] not in sensor_by_file:
            continue
        sensor = sensor_by_file[mt["sensor"]]
        pred_df = pd.read_csv(f)
        m = U.metrics_on_observed(pred_df)
        rows.append({"sensor": sensor, "city": city_of.get(sensor),
                     "I": int(mt["I"]), "H": int(mt["H"]), **m,
                     "pred_NaN": int(pred_df["y_pred"].isna().sum())})
    per_sensor = pd.DataFrame(rows).sort_values(["city", "H", "I"])
    per_sensor.to_csv(run_dir / "metrics_per_sensor.csv", index=False)

    osc = (per_sensor.groupby(["I", "H"])
           .agg(MAE=("MAE", "mean"), RMSE=("RMSE", "mean"), SMAPE=("SMAPE", "mean"),
                R2=("R2", "mean"), n_sensors=("sensor", "nunique"))
           .reset_index().sort_values(["H", "I"]))
    osc.to_csv(run_dir / "metrics_osc.csv", index=False)

    incomplete = osc[osc["n_sensors"] < len(C.SELECTED_SENSORS)]
    if len(incomplete):
        print("[avviso] config con meno di 10 sensori (media OSC non confrontabile):")
        print(incomplete[["I", "H", "n_sensors"]].to_string(index=False), "\n")

    pd.set_option("display.float_format", "{:.3f}".format)
    print("Media OSC per configurazione:")
    print(osc.to_string(index=False))

    full = osc[osc["n_sensors"] == len(C.SELECTED_SENSORS)]
    png = run_dir / "heatmap_mae_osc.png"
    plot_heatmap(full, png, f"Chronos-2 Scenario Y — MAE medio OSC ({args.tag})")
    print(f"\nHeatmap: {png}")

    best = (full.sort_values("MAE").groupby("H", as_index=False).first()
            .sort_values("H"))
    best.to_csv(run_dir / "best_per_horizon.csv", index=False)
    print("\nMigliore input window per ogni orizzonte (MAE minimo):")
    print(best.to_string(index=False))

    print(f"\nProssimo passo: python 05_compare_paper.py --tags {args.tag}")


if __name__ == "__main__":
    main()
