"""Метрики качества. Все функции принимают DataFrame с колонками user_id, target и массив score."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss


def ndcg_at_k(df, score, k=10):
    """Средний NDCG@k по пользователям.
    В расчёт берём только пользователей, у которых >= 2 строк и есть оба класса:
    если все строки пользователя 1 (или все 0), NDCG не зависит от модели."""
    d = pd.DataFrame({"user_id": df["user_id"].to_numpy(),
                      "target": df["target"].to_numpy(), "score": np.asarray(score)})
    stats = d.groupby("user_id")["target"].agg(["size", "sum"])
    good = stats.index[(stats["size"] >= 2) & (stats["sum"] > 0) & (stats["sum"] < stats["size"])]
    d = d[d["user_id"].isin(good)]

    def dcg(sorted_d):
        pos = sorted_d.groupby("user_id").cumcount().to_numpy()        # 0, 1, 2 ... внутри пользователя
        gain = sorted_d["target"].to_numpy() / np.log2(pos + 2)
        gain[pos >= k] = 0.0
        return pd.Series(gain, index=sorted_d["user_id"].to_numpy()).groupby(level=0).sum()

    by_model = d.sort_values(["user_id", "score"], ascending=[True, False], kind="mergesort")
    ideal = d.sort_values(["user_id", "target"], ascending=[True, False], kind="mergesort")
    ndcg = dcg(by_model) / dcg(ideal)
    return float(ndcg.mean()), len(good)


def evaluate(df, score, k=10):
    y = df["target"].to_numpy()
    score = np.clip(np.asarray(score, dtype=float), 1e-6, 1 - 1e-6)
    ndcg, n_users = ndcg_at_k(df, score, k)
    return {"roc_auc": roc_auc_score(y, score), "logloss": log_loss(y, score),
            f"ndcg@{k}": ndcg, "ndcg_users": n_users}


def paired_bootstrap_auc(df, score_a, score_b, n_boot=300, seed=0):
    """Доверительный интервал для разницы AUC(a) - AUC(b) на одних и тех же данных.
    Пересэмплируем ПОЛЬЗОВАТЕЛЕЙ (их строки зависимы между собой), а не отдельные строки.
    Реализация: каждый пользователь получает вес = сколько раз он выпал в выборке с возвращением."""
    y = df["target"].to_numpy()
    codes, uniq = pd.factorize(df["user_id"])
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        counts = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)))
        w = counts[codes]
        if w.sum() == 0 or len(np.unique(y[w > 0])) < 2:
            continue
        diffs.append(roc_auc_score(y, score_a, sample_weight=w) - roc_auc_score(y, score_b, sample_weight=w))
    diffs = np.array(diffs)
    return float(diffs.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
