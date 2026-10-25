"""Шаг 6. Масштабируемость: время расчёта признаков в зависимости от числа строк.
Модель не обучается, считаем только признаки.
Запуск: python src/06_scaling.py          (10k, 50k, 200k пользователей)
        python src/06_scaling.py all      (плюс весь датасет, нужно ~3-4 ГБ свободной памяти)"""
import sys
import time

import numpy as np
import pandas as pd

from config import RAW_DIR, RES_DIR, SAMPLE_SEED
from features import window_features, ENTITIES, VALUE_COLS, MANUAL_SPEC, WINDOWS

inter = pd.read_csv(RAW_DIR / "interactions.csv", parse_dates=["last_watch_dt"]).dropna(subset=["watched_pct"])
inter["target"] = (inter["watched_pct"] >= 50).astype(int)
items = pd.read_csv(RAW_DIR / "items.csv", usecols=["item_id", "content_type", "genres"])
items["main_genre"] = items["genres"].fillna("unknown").str.split(",").str[0].str.strip()  # для замера времени достаточно
inter = inter.merge(items[["item_id", "content_type", "main_genre"]], on="item_id", how="left")

users = inter["user_id"].unique()
sizes = [10_000, 50_000, 200_000] + ([len(users)] if "all" in sys.argv else [])
rng = np.random.default_rng(SAMPLE_SEED)
order = rng.permutation(users)

rows = []
for n in sizes:
    df = inter[inter["user_id"].isin(order[:n])].reset_index(drop=True)
    t_grid = t_manual = 0.0
    for name, keys in ENTITIES.items():                 # по одной сущности, чтобы не держать всё в памяти
        t = time.perf_counter(); window_features(df, keys, VALUE_COLS, WINDOWS, name); t_grid += time.perf_counter() - t
    for name, wins in MANUAL_SPEC:
        t = time.perf_counter(); window_features(df, ENTITIES[name], VALUE_COLS, wins, name); t_manual += time.perf_counter() - t
    rows.append({"users": n, "rows": len(df), "grid_90_s": round(t_grid, 2), "manual_18_s": round(t_manual, 2)})
    print(rows[-1])

out = pd.DataFrame(rows)
out["grid_us_per_row"] = (out["grid_90_s"] / out["rows"] * 1e6).round(2)
out.to_csv(RES_DIR / "scaling.csv", index=False)
print(out.to_string(index=False))
