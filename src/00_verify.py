import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
inter = pd.read_csv(RAW / "interactions.csv", parse_dates=["last_watch_dt"])
users = pd.read_csv(RAW / "users.csv")
items = pd.read_csv(RAW / "items.csv")

print("1. Строк:", len(inter))
print("   дублей пары (user,item):", inter.duplicated(["user_id", "item_id"]).sum())
print("   дублей user_id в users.csv:", users["user_id"].duplicated().sum())
print("   дублей item_id в items.csv:", items["item_id"].duplicated().sum())
print("   строк с item_id, которого нет в items.csv:", (~inter["item_id"].isin(items["item_id"])).sum())

print("\n2. NaN в watched_pct:", inter["watched_pct"].isna().sum())
print("   доля watched_pct == 0:", round((inter["watched_pct"] == 0).mean(), 3))
print(inter[["total_dur", "watched_pct"]].describe().to_string())
for thr in (10, 30, 50, 70, 90):
    print(f"   доля watched_pct >= {thr}: {(inter['watched_pct'] >= thr).mean():.3f}")

print("\n3. Даты:", inter["last_watch_dt"].min(), "..", inter["last_watch_dt"].max())

per_user = inter.groupby("user_id").size()
print("\n4. Событий на пользователя:")
print(per_user.describe().to_string())
print("   доля пользователей с 1 событием:", round((per_user == 1).mean(), 3))
print("   доля пользователей из interactions, которых нет в users.csv:",
      round(1 - per_user.index.isin(users["user_id"]).mean(), 3))

print("\n5. content_type в items:")
print(items["content_type"].value_counts().to_string())

m = inter.merge(items[["item_id", "content_type"]], on="item_id", how="left")
m["week"] = m["last_watch_dt"].dt.to_period("W")
m["pos"] = (m["watched_pct"] >= 50).astype(int)
print("\n6. Доля target=1 (порог 50) по неделям: всего и по типу контента")
print(m.groupby("week").agg(rows=("pos", "size"), pos_rate=("pos", "mean")).to_string())
print(m.groupby(["week", "content_type"])["pos"].mean().unstack().round(3).to_string())

print("\n7. Жанры. Первый тег (то, что мы брали как main_genre), топ-15:")
first_tag = items["genres"].dropna().str.split(",").str[0].str.strip()
print(first_tag.value_counts().head(15).to_string())
print("\n   Все теги жанров, топ-30:")
all_tags = items["genres"].dropna().str.split(",").explode().str.strip()
print(all_tags.value_counts().head(30).to_string())

sample_path = Path("data/processed/sample_df.csv")
if sample_path.exists():
    df = pd.read_csv(sample_path, parse_dates=["last_watch_dt"])
    first = df.groupby("user_id")["last_watch_dt"].transform("min")
    df["no_past"] = df["last_watch_dt"] == first
    print("\n8. Доля строк сэмпла без единого более раннего события пользователя:")
    print(df.groupby("split")["no_past"].mean().round(3).to_string())
