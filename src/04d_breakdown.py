"""Какие сущности и окна выбрал отбор (замена блока 6 из 04_auto_features.py)."""
import pandas as pd
from config import RES_DIR, WINDOWS

selected = pd.read_csv(RES_DIR / "auto_selected_features.csv")["feature"].tolist()

def parse(col):
    entity = next(e for e in sorted(["user_genre", "user_ctype", "user", "item", "genre"], key=len, reverse=True)
                  if col.startswith(e + "_"))
    window = col.rsplit("_", 1)[1]
    agg = col[len(entity) + 1:].rsplit("_", 1)[0]
    return entity, agg, window

sel = pd.DataFrame([parse(c) for c in selected], columns=["entity", "agg", "window"])
print("Отобранные признаки по сущностям и окнам:")
print(pd.crosstab(sel["entity"], sel["window"]).reindex(columns=[f"{w}d" for w in WINDOWS], fill_value=0).to_string())
print("\nПо агрегатам:")
print(sel["agg"].value_counts().to_string())
sel.to_csv(RES_DIR / "auto_selected_breakdown.csv", index=False)