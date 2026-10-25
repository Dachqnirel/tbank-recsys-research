"""Общий код обучения: один и тот же CatBoost, одни и те же данные для всех подходов."""
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from config import (PROC_DIR, RES_DIR, PRED_DIR, CAT_COLS, CATBOOST_PARAMS, EARLY_STOPPING)
from metrics import evaluate

RUNS_FILE = RES_DIR / "runs.csv"


def load_sample():
    df = pd.read_csv(PROC_DIR / "sample_df.csv", parse_dates=["last_watch_dt"])
    for c in CAT_COLS:                      # CatBoost: категории = строки без пропусков
        df[c] = df[c].fillna("unknown").astype(str)
    return df


def split_masks(df):
    return df["split"] == "train", df["split"] == "val", df["split"] == "test"


def fit_model(df, feats, cols, seed):
    """Обучение на train, early stopping по val. Test здесь НЕ оценивается."""
    tr, va, _ = split_masks(df)
    X = pd.concat([df, feats], axis=1)[cols]
    model = CatBoostClassifier(**CATBOOST_PARAMS, random_seed=seed, verbose=0)
    print(f"   ...обучаю: {len(cols)} признаков, seed={seed}", flush=True)
    t0 = time.perf_counter()
    model.fit(X[tr], df.loc[tr, "target"], cat_features=[c for c in cols if c in CAT_COLS],
              eval_set=(X[va], df.loc[va, "target"]), early_stopping_rounds=EARLY_STOPPING, verbose=0)
    return model, X, time.perf_counter() - t0


def val_auc(model, X, df):
    _, va, _ = split_masks(df)
    return roc_auc_score(df.loc[va, "target"], model.predict_proba(X[va])[:, 1])


def save_run(df, approach, seed, model, X, cols, n_hist, fe_time, train_time, extra_time=0.0):
    """Сохраняет прогнозы val и test в файлы и пишет строку в results/runs.csv.
    Печатаются только метрики val: решения принимаем по val, test смотрим в 05_report.py."""
    _, va, te = split_masks(df)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    p_val = model.predict_proba(X[va])[:, 1]
    p_test = model.predict_proba(X[te])[:, 1]
    np.save(PRED_DIR / f"{approach}_seed{seed}_val.npy", p_val)
    np.save(PRED_DIR / f"{approach}_seed{seed}_test.npy", p_test)
    m = evaluate(df[va], p_val)
    row = {"approach": approach, "seed": seed, "n_features": len(cols), "n_hist_features": n_hist,
           "fe_time_s": round(fe_time, 2), "selection_time_s": round(extra_time, 1),
           "train_time_s": round(train_time, 1), "best_iteration": model.get_best_iteration(),
           "val_roc_auc": round(m["roc_auc"], 4), "val_logloss": round(m["logloss"], 4),
           "val_ndcg@10": round(m["ndcg@10"], 4)}
    append_run(row)
    print(f"   {approach:16s} seed={seed} | признаков {len(cols):3d} | обучение {train_time:6.1f} с"
          f" | деревьев {row['best_iteration']} | VAL AUC {m['roc_auc']:.4f} LogLoss {m['logloss']:.4f}"
          f" NDCG@10 {m['ndcg@10']:.4f}")
    return row


def save_reference(df, approach, score_val, score_test):
    """Опорные точки без обучения: прогноз = готовое число (случайное, популярность и т.д.)."""
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PRED_DIR / f"{approach}_seed0_val.npy", score_val)
    np.save(PRED_DIR / f"{approach}_seed0_test.npy", score_test)
    _, va, _ = split_masks(df)
    m = evaluate(df[va], score_val)
    append_run({"approach": approach, "seed": 0, "n_features": 1, "n_hist_features": 0,
                "val_roc_auc": round(m["roc_auc"], 4), "val_logloss": round(m["logloss"], 4),
                "val_ndcg@10": round(m["ndcg@10"], 4)})
    print(f"   {approach:16s} | VAL AUC {m['roc_auc']:.4f} LogLoss {m['logloss']:.4f} NDCG@10 {m['ndcg@10']:.4f}")


def append_run(row):
    RES_DIR.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(RUNS_FILE) if RUNS_FILE.exists() else pd.DataFrame()
    runs = pd.concat([runs, pd.DataFrame([row])], ignore_index=True)
    runs = runs.drop_duplicates(subset=["approach", "seed"], keep="last")   # перезапуск заменяет строку
    runs.to_csv(RUNS_FILE, index=False)
