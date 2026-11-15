import inspect
import json
import shutil
import tempfile
import time

import numpy as np
import pandas as pd

import config as C


def load_prepared():
    df = pd.read_csv(C.PREPARED_CSV, parse_dates=[C.TIME_COL])
    return df.sort_values([C.SENSOR_COL, C.TIME_COL]).reset_index(drop=True)


def sensor_frame(df, sensor):
    d = df[df[C.SENSOR_COL] == sensor].sort_values(C.TIME_COL).reset_index(drop=True)
    if d.empty:
        raise ValueError(f"Sensore '{sensor}' non trovato nel dataset preparato")
    return d


def to_arrays(d):
    cols = [C.LCS_COL, C.RS_COL] + C.METEO_COLS
    return {c: d[c].to_numpy(dtype=np.float32) for c in cols}


def fit_linear_calibration(d, arr):
    cols = [C.LCS_COL] + C.METEO_COLS
    X_all = np.column_stack([arr[c] for c in cols] + [np.ones(len(d), dtype=np.float32)])
    mask = (d["split"] == "train").to_numpy() & ~d[C.MISSING_COL].astype(bool).to_numpy()
    mask &= np.isfinite(X_all).all(axis=1) & np.isfinite(arr[C.RS_COL])
    coef, *_ = np.linalg.lstsq(X_all[mask].astype(np.float64),
                               arr[C.RS_COL][mask].astype(np.float64), rcond=None)
    arr["cal"] = (X_all.astype(np.float64) @ coef).astype(np.float32)
    return dict(zip(cols + ["intercept"], np.round(coef, 4).tolist()))


def covariate_names(target_mode):
    if target_mode == "nan":
        return [C.LCS_COL] + C.METEO_COLS
    if target_mode in ("lcs", "cal"):
        return list(C.METEO_COLS)
    raise ValueError(f"target_mode sconosciuto: {target_mode}")


def make_item(arr, t, I, H, target_mode, with_future):
    s = t - I + 1
    if target_mode == "nan":
        ctx = np.full(I, np.nan, dtype=np.float32)
    elif target_mode == "lcs":
        ctx = arr[C.LCS_COL][s:t + 1].copy()
    else:
        ctx = arr["cal"][s:t + 1].copy()

    if with_future:
        target = np.concatenate([ctx, arr[C.RS_COL][t + 1:t + 1 + H]])
        end = t + 1 + H
    else:
        target = ctx
        end = t + 1

    covs = {c: arr[c][s:end].copy() for c in covariate_names(target_mode)}
    return {"target": target.astype(np.float32), "past_covariates": covs}


def train_origins(n_train, I, H):
    return list(range(I - 1, n_train - H))


def test_origins(n, n_train, I, H):
    return list(range(max(I - 1, n_train - H), n - 1))


def set_seed(seed=C.SEED):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        from transformers import set_seed as hf_set_seed
        hf_set_seed(seed)
    except Exception:
        pass


def load_pipeline():
    from chronos import Chronos2Pipeline
    return Chronos2Pipeline.from_pretrained(C.MODEL_ID, device_map=C.DEVICE)


def _filter_kwargs(fn, kwargs, warn=True):
    params = inspect.signature(fn).parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return kwargs
    kept = {k: v for k, v in kwargs.items() if k in params}
    dropped = set(kwargs) - set(kept)
    if dropped and warn:
        print(f"  [avviso] {fn.__name__} non accetta: {sorted(dropped)}")
    return kept


def finetune(pipeline, train_items, I, H, ft_mode, lr, num_steps, batch_size, workdir):
    if "min_past" not in inspect.signature(pipeline.fit).parameters:
        print("  [ATTENZIONE] fit() non accetta min_past: il taglio contesto/futuro "
              "potrebbe non cadere al passo I. Aggiorna chronos-forecasting.")
    kwargs = dict(
        inputs=train_items,
        prediction_length=H,
        context_length=I,
        min_past=I,
        finetune_mode=ft_mode,
        learning_rate=lr,
        num_steps=num_steps,
        batch_size=batch_size,
        output_dir=str(workdir),
    )
    return pipeline.fit(**_filter_kwargs(pipeline.fit, kwargs))


def _to_1d(x, H):
    try:
        x = x.detach().float().cpu().numpy()
    except AttributeError:
        x = np.asarray(x, dtype=np.float32)
    x = np.squeeze(x)
    if x.size != H:
        raise ValueError(f"Forma inattesa dell'output: {x.shape}, attesi {H} valori")
    return x.reshape(H)


def predict_median(pipeline, items, H, batch_size=C.PRED_BATCH_SIZE, chunk=4096):
    out = []
    for i in range(0, len(items), chunk):
        kwargs = dict(
            inputs=items[i:i + chunk],
            prediction_length=H,
            quantile_levels=[0.5],
            batch_size=batch_size,
            cross_learning=False,
        )
        quantiles, _mean = pipeline.predict_quantiles(
            **_filter_kwargs(pipeline.predict_quantiles, kwargs, warn=(i == 0)))
        out.extend(_to_1d(q, H) for q in quantiles)
    return np.stack(out)


def aggregate(origins, preds, n, n_train, H):
    s = np.zeros(n, dtype=np.float64)
    c = np.zeros(n, dtype=np.int64)
    for t, p in zip(origins, preds):
        for k in range(H):
            j = t + 1 + k
            if n_train <= j < n and np.isfinite(p[k]):
                s[j] += p[k]
                c[j] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(c > 0, s / c, np.nan)
    return mean[n_train:], c[n_train:]


def run_config(df, sensor, I, H, target_mode="nan", zero_shot=False,
               ft_mode=C.FT_MODE, lr=C.FT_LR, num_steps=C.FT_NUM_STEPS,
               batch_size=C.FT_BATCH_SIZE, verbose=True):
    import torch

    set_seed()
    d = sensor_frame(df, sensor)
    arr = to_arrays(d)
    n = len(d)
    n_train = int((d["split"] == "train").sum())
    info = {"sensor": sensor, "I": I, "H": H, "target_mode": target_mode,
            "zero_shot": zero_shot, "n": n, "n_train": n_train}
    if target_mode == "cal":
        info["calibration_coef"] = fit_linear_calibration(d, arr)

    t0 = time.time()
    pipeline = load_pipeline()

    if not zero_shot:
        tr_orig = train_origins(n_train, I, H)
        train_items = [make_item(arr, t, I, H, target_mode, True) for t in tr_orig]
        train_items = [it for it in train_items
                       if np.isfinite(it["target"][:I]).any() and np.isfinite(it["target"][I:]).any()]
        info.update(n_train_windows=len(train_items), ft_mode=ft_mode, lr=lr,
                    num_steps=num_steps, batch_size=batch_size)
        if verbose:
            print(f"  fine-tuning su {len(train_items)} finestre ...")
        workdir = tempfile.mkdtemp(prefix="chronos2_ft_")
        try:
            pipeline = finetune(pipeline, train_items, I, H, ft_mode, lr,
                                num_steps, batch_size, workdir)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        info["ft_seconds"] = round(time.time() - t0, 1)

    t1 = time.time()
    te_orig = test_origins(n, n_train, I, H)
    test_items = [make_item(arr, t, I, H, target_mode, False) for t in te_orig]
    preds = predict_median(pipeline, test_items, H)
    y_pred, n_preds = aggregate(te_orig, preds, n, n_train, H)
    y_pred = np.maximum(y_pred, 0.0)
    info["pred_seconds"] = round(time.time() - t1, 1)

    test = d.iloc[n_train:].reset_index(drop=True)
    out = pd.DataFrame({
        C.TIME_COL: test[C.TIME_COL],
        "y_true": test[C.RS_COL],
        "y_pred": y_pred,
        "lcs": test[C.LCS_COL],
        C.MISSING_COL: test[C.MISSING_COL].astype(bool),
        "n_preds": n_preds,
    })

    del pipeline
    torch.cuda.empty_cache()
    return out, info


def compute_metrics(y_true, y_pred):
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if len(y) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "SMAPE": np.nan, "R2": np.nan, "N": 0}
    err = y - p
    denom = (np.abs(y) + np.abs(p)) / 2
    smape_terms = np.divide(np.abs(err), denom, out=np.zeros_like(err), where=denom > 0)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "SMAPE": float(np.mean(smape_terms)),
        "R2": float(1 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else np.nan,
        "N": int(len(y)),
    }


def metrics_on_observed(pred_df):
    obs = pred_df[~pred_df[C.MISSING_COL].astype(bool)]
    return compute_metrics(obs["y_true"], obs["y_pred"])


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
