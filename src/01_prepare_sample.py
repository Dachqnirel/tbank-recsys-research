"""Шаг 1. Сэмпл пользователей целиком, чистка, разбиение по времени.

Запуск: python src/01_prepare_sample.py            (N_USERS из config.py)
        python src/01_prepare_sample.py 20000      (другой размер)
"""
import sys

import numpy as np
import pandas as pd

from config import (RAW_DIR, PROC_DIR, RES_DIR, SAMPLE_SEED, N_USERS, TARGET_THRESHOLD,
                    VAL_START, TEST_START, ORIGIN_TAGS, SKIP_TAGS, GENRE_FIX)

n_users = int(sys.argv[1]) if len(sys.argv) > 1 else N_USERS
PROC_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)


def tags_of(s):
    """'драмы, зарубежные' -> ['драмы', 'зарубежные']; пустое значение -> []"""
    if pd.isna(s):
        return []
    return [t.strip() for t in str(s).split(",") if t.strip()]


def main_genre_of(s):
    """Первый тег, который является жанром (не страной и не 'сериалы')."""
    for t in tags_of(s):
        if t not in SKIP_TAGS:
            return GENRE_FIX.get(t, t)
    return "unknown"


def origin_of(s):
    """Происхождение: русские / зарубежные / советские, если такой тег есть."""
    for t in tags_of(s):
        if t in ORIGIN_TAGS:
            return t
    return "unknown"


def first_country_of(s):
    t = tags_of(s)
    return t[0] if t else "unknown"


# ---------- 1. Сэмпл пользователей ----------
inter = pd.read_csv(RAW_DIR / "interactions.csv", parse_dates=["last_watch_dt"])
all_users = inter["user_id"].unique()
n_users = min(n_users, len(all_users))
rng = np.random.default_rng(SAMPLE_SEED)
chosen = rng.choice(all_users, size=n_users, replace=False)
df = inter[inter["user_id"].isin(chosen)].copy()

# Проверка полноты истории: у каждого выбранного пользователя столько же строк, сколько в исходнике
full_cnt = inter["user_id"].value_counts()
samp_cnt = df["user_id"].value_counts()
assert df["user_id"].nunique() == n_users
assert (samp_cnt == full_cnt.loc[samp_cnt.index]).all(), "история пользователя обрезана"
print(f"1. Сэмпл: {len(df):,} строк, {n_users:,} пользователей. История каждого пользователя полная.")

# ---------- 2. Пропуски и целевая переменная ----------
n_nan = int(df["watched_pct"].isna().sum())
df = df.dropna(subset=["watched_pct"])
df["target"] = (df["watched_pct"] >= TARGET_THRESHOLD).astype(int)
print(f"2. Удалено строк с пустым watched_pct: {n_nan}. target = watched_pct >= {TARGET_THRESHOLD}")

# ---------- 3. Признаки пользователя и объекта ----------
users = pd.read_csv(RAW_DIR / "users.csv")
n0 = len(df)
df = df.merge(users, on="user_id", how="left")
assert len(df) == n0, "merge с users размножил строки"

items = pd.read_csv(RAW_DIR / "items.csv",
                    usecols=["item_id", "content_type", "release_year", "genres",
                             "countries", "for_kids", "age_rating"])
items["main_genre"] = items["genres"].map(main_genre_of)
items["origin"] = items["genres"].map(origin_of)
items["main_country"] = items["countries"].map(first_country_of)
items = items.drop(columns=["genres", "countries"])
n0 = len(df)
df = df.merge(items, on="item_id", how="left")
assert len(df) == n0, "merge с items размножил строки"

# Категориальные колонки для CatBoost должны быть строками без пропусков
for col in ["age", "income", "sex", "content_type"]:
    df[col] = df[col].fillna("unknown").astype(str)
print(f"3. Пользователей без анкеты в сэмпле: {(df.drop_duplicates('user_id')['age'] == 'unknown').mean():.3f}")

# ---------- 4. Разбиение по фиксированным датам ----------
df = df.sort_values(["last_watch_dt", "user_id", "item_id"]).reset_index(drop=True)
val_start, test_start = pd.Timestamp(VAL_START), pd.Timestamp(TEST_START)
df["split"] = "train"
df.loc[df["last_watch_dt"] >= val_start, "split"] = "val"
df.loc[df["last_watch_dt"] >= test_start, "split"] = "test"

tr, va, te = (df[df["split"] == s] for s in ("train", "val", "test"))
assert tr["last_watch_dt"].max() < val_start <= va["last_watch_dt"].min()
assert va["last_watch_dt"].max() < test_start <= te["last_watch_dt"].min()

# Строка "без прошлого": у пользователя нет ни одного события в более ранний день
first_day = df.groupby("user_id")["last_watch_dt"].transform("min")
df["no_past"] = (df["last_watch_dt"] == first_day).astype(int)

print("4. Разбиение:")
for name in ("train", "val", "test"):
    p = df[df["split"] == name]
    print(f"   {name:5s}: {len(p):8,d} строк | {p['last_watch_dt'].min().date()}..{p['last_watch_dt'].max().date()}"
          f" | target=1: {p['target'].mean():.3f} | без прошлого: {p['no_past'].mean():.3f}")

# ---------- 5. Сколько пользователей попадёт в NDCG ----------
print("5. Пользователи в NDCG (>=2 строк и есть оба класса):")
for name in ("val", "test"):
    g = df[df["split"] == name].groupby("user_id")["target"].agg(["size", "sum"])
    ok = (g["size"] >= 2) & (g["sum"] > 0) & (g["sum"] < g["size"])
    print(f"   {name}: пользователей {len(g):,}, в метрике {int(ok.sum()):,} ({ok.mean():.3f}),"
          f" строк у них {int(g.loc[ok, 'size'].sum()):,}")

print("6. main_genre после чистки, топ-12:")
print(df["main_genre"].value_counts().head(12).to_string())
print("   origin:", df["origin"].value_counts().to_dict())

df.to_csv(PROC_DIR / "sample_df.csv", index=False)
print(f"7. Сохранено: {PROC_DIR / 'sample_df.csv'} ({df.shape[0]:,} x {df.shape[1]})")
