"""Шаг 3. Опорные точки + CatBoost без истории + CatBoost на ручных признаках (3 seed'а)."""
import time

import numpy as np

from config import CAT_COLS, STATIC_NUM_COLS, TRAIN_SEEDS
from features import build_manual, window_features
from train_utils import load_sample, split_masks, fit_model, save_run, save_reference

df = load_sample()
tr, va, te = split_masks(df)
static_cols = CAT_COLS + STATIC_NUM_COLS
print(f"Строк: train {tr.sum():,}, val {va.sum():,}, test {te.sum():,}")

# ---------- 1. Опорные точки без обучения ----------
print("1. Опорные точки (без модели):")
rng = np.random.default_rng(0)
train_rate = df.loc[tr, "target"].mean()
save_reference(df, "ref_random", rng.random(va.sum()), rng.random(te.sum()))
save_reference(df, "ref_constant", np.full(va.sum(), train_rate), np.full(te.sum(), train_rate))
item30 = window_features(df, ["item_id"], ["target"], [30], "item")["item_target_mean_30d"].fillna(train_rate)
save_reference(df, "ref_item_pop30d", item30[va].to_numpy(), item30[te].to_numpy())
user30 = window_features(df, ["user_id"], ["target"], [30], "user")["user_target_mean_30d"].fillna(train_rate)
save_reference(df, "ref_user_hist30d", user30[va].to_numpy(), user30[te].to_numpy())

# ---------- 2. CatBoost только на статических признаках ----------
print("2. CatBoost без исторических признаков (анкета + метаданные объекта):")
for seed in TRAIN_SEEDS:
    model, X, t_train = fit_model(df, df[[]], static_cols, seed)
    save_run(df, "static", seed, model, X, static_cols, 0, 0.0, t_train)

# ---------- 3. CatBoost на ручных признаках ----------
t0 = time.perf_counter()
manual = build_manual(df)
fe_time = time.perf_counter() - t0
manual_cols = static_cols + list(manual.columns)
print(f"3. Ручной набор: {manual.shape[1]} исторических признаков за {fe_time:.2f} с")
for seed in TRAIN_SEEDS:
    model, X, t_train = fit_model(df, manual, manual_cols, seed)
    save_run(df, "manual", seed, model, X, manual_cols, manual.shape[1], fe_time, t_train)
    if seed == TRAIN_SEEDS[0]:
        imp = sorted(zip(model.get_feature_importance(), manual_cols), reverse=True)[:10]
        print("   Топ-10 по встроенной важности CatBoost (seed 0):")
        for v, c in imp:
            print(f"      {c:32s} {v:6.2f}")
