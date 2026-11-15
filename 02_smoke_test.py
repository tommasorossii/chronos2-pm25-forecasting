import argparse
import inspect
import traceback

import config as C
import common as U

PAPER_T8 = {
    "Aosta": 1.55, "Badajoz": 2.23, "Bangalore": 4.53, "Calgary": 0.96, "Delhi": 33.51,
    "Hamirpur": 23.39, "Lima": 4.19, "Reggio": 6.09, "Southampton": 2.82, "Trento": 3.90,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor", default="ari-2433")
    ap.add_argument("--I", type=int, default=24)
    ap.add_argument("--H", type=int, default=12)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--modes", nargs="+", default=["lcs", "cal"])
    args = ap.parse_args()

    import chronos
    from chronos import Chronos2Pipeline
    print(f"chronos-forecasting {getattr(chronos, '__version__', '?')}")
    fit_params = list(inspect.signature(Chronos2Pipeline.fit).parameters)
    print(f"fit() accetta: {fit_params}")
    print(f"min_past supportato: {'min_past' in fit_params}\n")

    df = U.load_prepared()
    d = U.sensor_frame(df, args.sensor)
    city = d[C.CITY_COL].iloc[0]
    test = d[d["split"] == "test"]
    obs = test[~test[C.MISSING_COL]]

    raw = U.compute_metrics(obs[C.RS_COL], obs[C.LCS_COL])
    print(f"Sensore {args.sensor} ({city}) — I={args.I}, H={args.H}, {args.steps} step")
    print(f"Baseline LCS grezzo vs RS sul test:  MAE={raw['MAE']:.2f}  R2={raw['R2']:.2f}")
    if city in PAPER_T8:
        print(f"Paper, Granite_tuned (Tab. 8, media città):  MAE={PAPER_T8[city]:.2f}")
    print(f"Statistiche RS sul test: media={obs[C.RS_COL].mean():.2f}, "
          f"min={obs[C.RS_COL].min():.2f}, max={obs[C.RS_COL].max():.2f}\n")

    out_dir = C.RESULTS_DIR / "smoke"
    for mode in args.modes:
        print(f"=== target_mode = '{mode}' ===")
        try:
            pred_df, info = U.run_config(df, args.sensor, args.I, args.H,
                                         target_mode=mode, num_steps=args.steps)
        except Exception:
            print("  ERRORE:")
            traceback.print_exc()
            continue
        m = U.metrics_on_observed(pred_df)
        p = pred_df["y_pred"]
        print(f"  tempi: fine-tuning {info.get('ft_seconds')} s, previsione {info['pred_seconds']} s")
        print(f"  previsioni: media={p.mean():.2f}, min={p.min():.2f}, max={p.max():.2f}, "
              f"NaN={p.isna().sum()}")
        print(f"  MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  SMAPE={m['SMAPE']:.3f}  "
              f"R2={m['R2']:.2f}  (N={m['N']})\n")
        name = f"{args.sensor}_I{args.I}_H{args.H}_{mode}"
        out_dir.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(out_dir / f"{name}.csv", index=False)
        U.save_json({**info, **m}, out_dir / f"{name}.json")

    print(f"File salvati in {out_dir}")


if __name__ == "__main__":
    main()
