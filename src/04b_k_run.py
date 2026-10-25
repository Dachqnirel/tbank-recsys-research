"""Шаг 4b. Один запуск = одно K (число отобранных признаков).

Вынесено из 04_auto_features.py: серия обучений CatBoost в одном процессе
падала с segfault в _catboost.so. Здесь каждое обучение идёт отдельным процессом,
результат дописывается в results/auto_k_sweep_runs.csv.
"""
import argparse
import csv

import pandas as pd

from config import CAT_COLS, STATIC_NUM_COLS, TRAIN_SEEDS, RES_DIR
from features import build_grid
from train_utils import load_sample, fit_model, val_auc

p = argparse.ArgumentParser()
p.add_argument("--k", type=int, required=True)
args = p.parse_args()

df = load_sample()
grid = build_grid(df)
imp = (pd.read_csv(RES_DIR / "auto_permutation_importance.csv", index_col="feature")["val_auc_drop"]
       .sort_values(ascending=False))
static_cols = CAT_COLS + STATIC_NUM_COLS
top = list(imp.index[:args.k])

model, X_k, t_train = fit_model(df, grid, static_cols + top, TRAIN_SEEDS[0])
auc = val_auc(model, X_k, df)
print(f"K={args.k:3d}: val AUC {auc:.4f} (обучение {t_train:.1f} с)")

out = RES_DIR / "auto_k_sweep_runs.csv"
is_new = not out.exists()
with open(out, "a", newline="") as f:
    w = csv.writer(f)
    if is_new:
        w.writerow(["k", "val_roc_auc", "train_time_s"])
    w.writerow([args.k, round(auc, 4), round(t_train, 1)])