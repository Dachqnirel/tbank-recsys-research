"""Шаг 2. Независимые проверки, что в признаки не попадает будущее.

Проверка A: быстрая функция совпадает с медленным "честным" циклом на случайных строках.
Проверка B: если удалить из данных всё будущее (test), признаки прошлых строк не меняются.
Проверка C: у строк "без прошлого" счётчики пользователя равны нулю.
"""
import time

import numpy as np
import pandas as pd

from config import PROC_DIR, TEST_START, WINDOWS
from features import window_features, build_grid, build_manual, ENTITIES

df = pd.read_csv(PROC_DIR / "sample_df.csv", parse_dates=["last_watch_dt"])
print(f"Строк: {len(df):,}")

# ---------- A. Сверка с медленным циклом ----------
rng = np.random.default_rng(0)
rows = rng.choice(len(df), size=300, replace=False)
n_bad = 0
for keys, name in [(["user_id"], "user"), (["user_id", "main_genre"], "ug"), (["item_id"], "item")]:
    fast = window_features(df, keys, ["watched_pct"], [7, 60], name)
    for i in rows:
        row = df.iloc[i]
        same_group = np.ones(len(df), dtype=bool)
        for k in keys:
            same_group &= (df[k] == row[k]).to_numpy()
        for w in (7, 60):
            lo = row["last_watch_dt"] - pd.Timedelta(days=w)
            hist = df[same_group & (df["last_watch_dt"] >= lo).to_numpy()
                      & (df["last_watch_dt"] < row["last_watch_dt"]).to_numpy()]
            slow_cnt = len(hist)
            slow_mean = hist["watched_pct"].mean() if slow_cnt else np.nan
            f_cnt = fast.iloc[i][f"{name}_cnt_{w}d"]
            f_mean = fast.iloc[i][f"{name}_watched_pct_mean_{w}d"]
            ok_cnt = slow_cnt == f_cnt
            ok_mean = (np.isnan(slow_mean) and np.isnan(f_mean)) or np.isclose(slow_mean, f_mean)
            if not (ok_cnt and ok_mean):
                n_bad += 1
                print("  РАСХОЖДЕНИЕ:", name, w, i, slow_cnt, f_cnt, slow_mean, f_mean)
print(f"A. Сверка с циклом: 300 строк x 3 сущности x 2 окна, расхождений: {n_bad}")
assert n_bad == 0

# ---------- B. Инвариантность к будущему ----------
past = df[df["last_watch_dt"] < pd.Timestamp(TEST_START)]
for builder in (build_grid, build_manual):
    full = builder(df).loc[past.index]
    cut = builder(past)
    pd.testing.assert_frame_equal(full, cut, check_exact=False)
print("B. Признаки train+val не меняются, если удалить test-строки: OK (сетка и ручной набор)")

# ---------- C. Строки без прошлого ----------
u = window_features(df, ["user_id"], [], [1, 7, 60], "u")
no_past = df["no_past"] == 1
assert (u.loc[no_past] == 0).all().all()
print(f"C. У {int(no_past.sum()):,} строк без прошлого все счётчики пользователя = 0: OK")

# ---------- Время расчёта ----------
t = time.perf_counter(); g = build_grid(df); t_grid = time.perf_counter() - t
t = time.perf_counter(); m = build_manual(df); t_man = time.perf_counter() - t
print(f"Время: сетка {g.shape[1]} признаков за {t_grid:.2f} с, ручной набор {m.shape[1]} признаков за {t_man:.2f} с")
print("Доля строк без истории в окне (cnt = 0) по сущностям и окнам:")
table = pd.DataFrame({name: [(g[f"{name}_cnt_{w}d"] == 0).mean() for w in WINDOWS] for name in ENTITIES},
                     index=[f"{w}d" for w in WINDOWS])
print(table.round(3).to_string())
