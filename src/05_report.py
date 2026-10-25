"""Шаг 5. Единственное место, где считаются метрики на test.
Собирает таблицу результатов, интервалы бутстрэпа и график для README."""
import matplotlib
matplotlib.use("Agg")                      # рисуем в файл, без окна
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import RES_DIR, PRED_DIR, N_BOOTSTRAP, NDCG_K
from metrics import evaluate, paired_bootstrap_auc
from train_utils import load_sample, split_masks, RUNS_FILE

df = load_sample()
_, _, te = split_masks(df)
test = df[te]
runs = pd.read_csv(RUNS_FILE)

ORDER = ["ref_constant", "ref_random", "ref_item_pop30d", "ref_user_hist30d",
         "static", "manual", "auto_all", "auto_selected"]
NAMES = {"ref_constant": "Константа (доля 1 в train)", "ref_random": "Случайный порядок",
         "ref_item_pop30d": "Доля досмотров объекта за 30 дн.", "ref_user_hist30d": "Доля досмотров пользователя за 30 дн.",
         "static": "CatBoost: без истории", "manual": "CatBoost: ручные признаки",
         "auto_all": "CatBoost: вся сетка", "auto_selected": "CatBoost: сетка + отбор по val"}

# ---------- 1. Метрики на test для каждого прогона ----------
rows, mean_pred = [], {}
for approach in ORDER:
    seeds = runs.loc[runs["approach"] == approach, "seed"].tolist()
    if not seeds:
        continue
    preds = [np.load(PRED_DIR / f"{approach}_seed{s}_test.npy") for s in seeds]
    mean_pred[approach] = np.mean(preds, axis=0)
    for s, p in zip(seeds, preds):
        m = evaluate(test, p, NDCG_K)
        rows.append({"approach": approach, "seed": s, "test_roc_auc": m["roc_auc"],
                     "test_logloss": m["logloss"], f"test_ndcg@{NDCG_K}": m[f"ndcg@{NDCG_K}"],
                     "ndcg_users": m["ndcg_users"]})
per_seed = pd.DataFrame(rows).merge(runs, on=["approach", "seed"], how="left")
per_seed.to_csv(RES_DIR / "test_metrics_per_seed.csv", index=False)

# ---------- 2. Сводная таблица: среднее ± разброс по seed'ам ----------
def fmt(series, digits=4):
    if series.isna().all():
        return "—"
    if len(series) == 1:
        return f"{series.iloc[0]:.{digits}f}"
    return f"{series.mean():.{digits}f} ± {series.std(ddof=0):.{digits}f}"

summary = []
for approach, g in per_seed.groupby("approach", sort=False):
    is_model = not approach.startswith("ref_")
    summary.append({
        "Подход": NAMES[approach],
        "Ист. признаков": int(g["n_hist_features"].iloc[0]),
        "Test ROC-AUC": "—" if approach == "ref_constant" else fmt(g["test_roc_auc"]),
        "Test LogLoss": "—" if approach == "ref_random" else fmt(g["test_logloss"]),
        f"Test NDCG@{NDCG_K}": "—" if approach == "ref_constant" else fmt(g[f"test_ndcg@{NDCG_K}"]),
        "Val ROC-AUC": fmt(g["val_roc_auc"]),
        "Время признаков, с": fmt(g["fe_time_s"], 2) if is_model and approach != "static" else "—",
        "Время отбора, с": fmt(g["selection_time_s"], 1) if approach == "auto_selected" else "—",
        "Обучение, с": fmt(g["train_time_s"], 1) if is_model else "—",
    })
summary = pd.DataFrame(summary)
summary.to_csv(RES_DIR / "summary.csv", index=False)
n_users = int(per_seed["ndcg_users"].iloc[0])

# ---------- 3. Парный бутстрэп разницы AUC на test ----------
pairs = [("manual", "static"), ("auto_all", "manual"), ("auto_selected", "manual"), ("auto_selected", "auto_all")]
boot = []
for a, b in pairs:
    if a in mean_pred and b in mean_pred:
        d, lo, hi = paired_bootstrap_auc(test, mean_pred[a], mean_pred[b], N_BOOTSTRAP, seed=0)
        boot.append({"Сравнение": f"{NAMES[a]} − {NAMES[b]}", "ΔAUC": f"{d:+.4f}",
                     "95% интервал": f"[{lo:+.4f}; {hi:+.4f}]",
                     "Интервал включает 0": "да" if lo <= 0 <= hi else "нет"})
boot = pd.DataFrame(boot)
boot.to_csv(RES_DIR / "bootstrap_auc_diff.csv", index=False)

# ---------- 4. Markdown для README ----------
def to_md(table):
    """Таблица в формате Markdown без дополнительных библиотек."""
    lines = ["| " + " | ".join(table.columns) + " |", "|" + "---|" * len(table.columns)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in table.itertuples(index=False)]
    return "\n".join(lines)


md = ["## Результаты на test", "",
      f"Test: {len(test):,} строк, {test['user_id'].nunique():,} пользователей. "
      f"NDCG@{NDCG_K} считается по {n_users:,} пользователям, у которых в test есть и досмотры, и недосмотры. "
      f"Для моделей: среднее ± стандартное отклонение по {per_seed.loc[per_seed['approach'] == 'manual', 'seed'].nunique()} seed'ам.", "",
      to_md(summary), "",
      f"### Разница AUC на test, парный бутстрэп по пользователям ({N_BOOTSTRAP} повторов, прогнозы усреднены по seed'ам)", "",
      to_md(boot)]
(RES_DIR / "results_table.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))

# ---------- 5. График: перестановочная важность (на val) и что вошло в отбор ----------
imp_path = RES_DIR / "auto_permutation_importance.csv"
if imp_path.exists():
    imp = pd.read_csv(imp_path).head(20).iloc[::-1]
    selected = set(pd.read_csv(RES_DIR / "auto_selected_features.csv")["feature"])
    colors = ["#2a78d6" if f in selected else "#b9b8b2" for f in imp["feature"]]
    fig, ax = plt.subplots(figsize=(9, 7), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.barh(imp["feature"], imp["val_auc_drop"], color=colors, height=0.6)
    ax.set_xlabel("Падение val ROC-AUC при перемешивании признака", color="#52514e")
    ax.set_title("Автогенерация: топ-20 признаков сетки по перестановочной важности на val",
                 color="#0b0b0b", loc="left", fontsize=11)
    ax.tick_params(colors="#52514e", labelsize=8)
    ax.grid(axis="x", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c2")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#2a78d6", label="вошёл в отбор"), Patch(color="#b9b8b2", label="не вошёл")],
              frameon=False, loc="lower right", fontsize=9, labelcolor="#52514e")
    fig.tight_layout()
    fig.savefig(RES_DIR / "auto_feature_importance.png", dpi=150)
    print(f"\nГрафик: {RES_DIR / 'auto_feature_importance.png'}")
