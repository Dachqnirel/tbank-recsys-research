"""Шаг 4c. Выбор K по val и обучение auto_selected (один seed на запуск)."""
import argparse
import time

import pandas as pd

from config import CAT_COLS, STATIC_NUM_COLS, RES_DIR
from features import build_grid
from train_utils import load_sample, fit_model, save_run

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, required=True)
args = p.parse_args()

PERM_TIME = 18.9   # секунд, из results/04_auto_output.txt (шаг 3)

sweep = pd.read_csv(RES_DIR / "auto_k_sweep_runs.csv").drop_duplicates("k", keep="last").sort_values("k")
sweep.to_csv(RES_DIR / "auto_k_sweep.csv", index=False)
best_k = int(sweep.loc[sweep["val_roc_auc"].idxmax(), "k"])
selection_time = float(sweep["train_time_s"].sum()) + PERM_TIME
print(f"Выбрано K={best_k}, полное время отбора {selection_time:.1f} с")

df = load_sample()
t0 = time.perf_counter()
grid = build_grid(df)
fe_time = time.perf_counter() - t0

imp = (pd.read_csv(RES_DIR / "auto_permutation_importance.csv", index_col="feature")["val_auc_drop"]
       .sort_values(ascending=False))
selected = list(imp.index[:best_k])
pd.Series(selected, name="feature").to_csv(RES_DIR / "auto_selected_features.csv", index=False)

sel_cols = CAT_COLS + STATIC_NUM_COLS + selected
model, X_sel, t_train = fit_model(df, grid, sel_cols, args.seed)
save_run(df, "auto_selected", args.seed, model, X_sel, sel_cols, best_k, fe_time, t_train, selection_time)