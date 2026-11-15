import argparse
import time

import pandas as pd

import config as C
import common as U


def parse_configs(items):
    if not items:
        return C.valid_configs()
    out = []
    for s in items:
        i, h = (int(x) for x in s.split(":"))
        out.append((i, h))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensors", nargs="+", default=C.SELECTED_SENSORS)
    ap.add_argument("--configs", nargs="+", help="coppie I:H, es. 24:12 (default: tutte)")
    ap.add_argument("--target-mode", choices=["nan", "lcs", "cal"], default="cal")
    ap.add_argument("--ft-mode", choices=["full", "lora"], default=C.FT_MODE)
    ap.add_argument("--lr", type=float, default=C.FT_LR)
    ap.add_argument("--steps", type=int, default=C.FT_NUM_STEPS)
    ap.add_argument("--batch-size", type=int, default=C.FT_BATCH_SIZE)
    ap.add_argument("--zero-shot", action="store_true",
                    help="nessun fine-tuning (solo come riferimento, non è lo Scenario Y del paper)")
    ap.add_argument("--tag", help="nome della cartella dei risultati")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    tag = args.tag or (f"Y_{args.target_mode}_zeroshot" if args.zero_shot
                       else f"Y_{args.target_mode}_{args.ft_mode}")
    run_dir = C.RESULTS_DIR / tag
    pred_dir = run_dir / "preds"
    pred_dir.mkdir(parents=True, exist_ok=True)
    U.save_json(vars(args) | {"tag": tag, "model": C.MODEL_ID}, run_dir / "run_args.json")

    df = U.load_prepared()
    configs = parse_configs(args.configs)
    jobs = [(s, i, h) for s in args.sensors for (i, h) in configs]
    print(f"Run '{tag}': {len(jobs)} job ({len(args.sensors)} sensori x {len(configs)} config)\n")

    log_rows, t_start = [], time.time()
    for k, (sensor, I, H) in enumerate(jobs, 1):
        name = f"{sensor.replace(' ', '_')}_I{I}_H{H}"
        out_csv = pred_dir / f"{name}.csv"
        if out_csv.exists() and not args.overwrite:
            print(f"[{k}/{len(jobs)}] {name}: già presente, salto")
            continue
        print(f"[{k}/{len(jobs)}] {name}")
        try:
            pred_df, info = U.run_config(
                df, sensor, I, H, target_mode=args.target_mode, zero_shot=args.zero_shot,
                ft_mode=args.ft_mode, lr=args.lr, num_steps=args.steps,
                batch_size=args.batch_size)
        except Exception as e:
            print(f"  ERRORE: {e!r}")
            log_rows.append({"sensor": sensor, "I": I, "H": H, "error": repr(e)})
            continue

        pred_df.to_csv(out_csv, index=False)
        m = U.metrics_on_observed(pred_df)
        coef = info.pop("calibration_coef", None)
        if coef:
            U.save_json(coef, run_dir / "calibration" / f"{name}.json")
        log_rows.append({**info, **m})
        elapsed = (time.time() - t_start) / 60
        print(f"  MAE={m['MAE']:.2f} RMSE={m['RMSE']:.2f} SMAPE={m['SMAPE']:.3f} "
              f"R2={m['R2']:.2f} | {elapsed:.1f} min totali")

        log_path = run_dir / "run_log.csv"
        new = pd.DataFrame(log_rows[-1:])
        new.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)

    print(f"\nFatto. Previsioni in {pred_dir}")
    print(f"Prossimo passo: python 04_evaluate.py --tag {tag}")


if __name__ == "__main__":
    main()
