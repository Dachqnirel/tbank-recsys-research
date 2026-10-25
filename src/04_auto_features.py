"""Шаг 4. Автогенерация признаков по сетке и отбор по валидации.

1) Сетка: 5 сущностей x 6 окон x 3 агрегата = 90 признаков, без участия человека.
2) auto_all: CatBoost на всей сетке (3 seed'а).
3) Отбор по val: перестановочная важность на val (насколько падает val AUC,
   если перемешать один признак), затем выбор числа признаков K по val AUC.
4) auto_selected: CatBoost на лучших K признаках (3 seed'а).
Test в этом скрипте не оценивается.
"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import CAT_COLS, STATIC_NUM_COLS, TRAIN_SEEDS, RES_DIR, WINDOWS
from features import build_grid
from train_utils import load_sample, split_masks, fit_model, val_auc, save_run

K_GRID = [10, 20, 40]          # 20 = столько же исторических признаков, сколько в ручном наборе
N_REPEATS = 2                  # сколько раз перемешиваем каждый признак

df = load_sample()
_, va, _ = split_masks(df)
static_cols = CAT_COLS + STATIC_NUM_COLS

# ---------- 1. Генерация ----------
t0 = time.perf_counter()
grid = build_grid(df)
fe_time = time.perf_counter() - t0
grid_cols = list(grid.columns)
print(f"1. Сетка: {len(grid_cols)} признаков за {fe_time:.2f} с")

# ---------- 2. Вся сетка ----------
print("2. auto_all (вся сетка):")
all_cols = static_cols + grid_cols
models = {}
for seed in TRAIN_SEEDS:
    model, X_all, t_train = fit_model(df, grid, all_cols, seed)
    save_run(df, "auto_all", seed, model, X_all, all_cols, len(grid_cols), fe_time, t_train)
    models[seed] = (model, t_train)

# ---------- 3. Отбор по val: перестановочная важность ----------
print(f"3. Перестановочная важность на val ({len(grid_cols)} признаков x {N_REPEATS} повтора)...")
model0, t_train0 = models[TRAIN_SEEDS[0]]
t0 = time.perf_counter()
X_val = X_all[va].copy()
y_val = df.loc[va, "target"]
base = roc_auc_score(y_val, model0.predict_proba(X_val)[:, 1])
rng = np.random.default_rng(0)
drops = {}
for col in grid_cols:
    original = X_val[col].to_numpy().copy()
    d = []
    for _ in range(N_REPEATS):
        X_val[col] = rng.permutation(original)
        d.append(base - roc_auc_score(y_val, model0.predict_proba(X_val)[:, 1]))
    X_val[col] = original
    drops[col] = float(np.mean(d))
perm_time = time.perf_counter() - t0
imp = pd.Series(drops).sort_values(ascending=False)
imp.rename("val_auc_drop").to_csv(RES_DIR / "auto_permutation_importance.csv", index_label="feature")
print(f"   готово за {perm_time:.1f} с; признаков с положительным вкладом: {(imp > 0).sum()} из {len(imp)}")
print("   Топ-15:")
print(imp.head(15).round(5).to_string())

# ---------- 4. Выбор K по val ----------
print("4. Выбор числа признаков K по val AUC (seed 0):")
sweep, sweep_models, t_sweep = [], {}, 0.0
for k in K_GRID:
    top = list(imp.index[:k])
    model, X_k, t_train = fit_model(df, grid, static_cols + top, TRAIN_SEEDS[0])
    t_sweep += t_train
    auc = val_auc(model, X_k, df)
    sweep.append({"k": k, "val_roc_auc": round(auc, 4), "train_time_s": round(t_train, 1)})
    sweep_models[k] = (model, X_k, t_train)
    print(f"   K={k:3d}: val AUC {auc:.4f} (обучение {t_train:.1f} с)")
# Кандидат "оставить всё": это модель из шага 2, переобучать не нужно
k_all = len(grid_cols)
auc_all = val_auc(model0, X_all, df)
sweep.append({"k": k_all, "val_roc_auc": round(auc_all, 4), "train_time_s": round(t_train0, 1)})
sweep_models[k_all] = (model0, X_all, t_train0)
print(f"   K={k_all:3d}: val AUC {auc_all:.4f} (вся сетка, модель из шага 2)")
sweep = pd.DataFrame(sweep)
sweep.to_csv(RES_DIR / "auto_k_sweep.csv", index=False)
best_k = int(sweep.loc[sweep["val_roc_auc"].idxmax(), "k"])
selected = list(imp.index[:best_k])
pd.Series(selected, name="feature").to_csv(RES_DIR / "auto_selected_features.csv", index=False)

# Полное время отбора: обучение модели на всей сетке + перестановки + перебор K
selection_time = t_train0 + perm_time + t_sweep
print(f"   Выбрано K={best_k}. Полное время отбора: {selection_time:.1f} с")

# ---------- 5. Финальная модель на отобранных признаках ----------
print("5. auto_selected:")
sel_cols = static_cols + selected
for seed in TRAIN_SEEDS:
    if seed == TRAIN_SEEDS[0]:
        model, X_sel, t_train = sweep_models[best_k]       # уже обучена в шаге 4
    else:
        model, X_sel, t_train = fit_model(df, grid, sel_cols, seed)
    save_run(df, "auto_selected", seed, model, X_sel, sel_cols, best_k, fe_time, t_train, selection_time)

# ---------- 6. Какие окна и сущности выбрал отбор ----------
def parse(col):
    entity = next(e for e in sorted(["user_genre", "user_ctype", "user", "item", "genre"], key=len, reverse=True)
                  if col.startswith(e + "_"))
    window = col.rsplit("_", 1)[1]
    agg = col[len(entity) + 1:].rsplit("_", 1)[0]
    return entity, agg, window

sel = pd.DataFrame([parse(c) for c in selected], columns=["entity", "agg", "window"])
print("6. Отобранные признаки по сущностям и окнам:")
table = pd.crosstab(sel["entity"], sel["window"]).reindex(columns=[f"{w}d" for w in WINDOWS], fill_value=0)
print(table.to_string())
sel.to_csv(RES_DIR / "auto_selected_breakdown.csv", index=False)
