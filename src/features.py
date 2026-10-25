"""Расчёт признаков по истории. Главное правило: для строки с датой T
используются только строки той же группы с датой от T-w до T-1 включительно.
События того же дня и будущие события в признак не попадают."""
import numpy as np
import pandas as pd

from config import WINDOWS

BIG = 100_000  # больше любого номера дня, чтобы ключи разных групп не пересекались


def window_features(df, keys, value_cols, windows, name):
    """Для каждой строки df и каждого окна w из windows считает по прошлым строкам группы keys:
       {name}_cnt_{w}d            число событий за [T-w, T-1]
       {name}_{col}_mean_{w}d     среднее колонки col за то же окно (NaN, если событий нет)
    Возвращает новый DataFrame с тем же индексом, что у df."""
    n = len(df)
    group = df.groupby(keys, sort=False).ngroup().to_numpy().astype(np.int64)
    day = (df["last_watch_dt"] - df["last_watch_dt"].min()).dt.days.to_numpy().astype(np.int64)

    # Ключ "группа, день" -> одно число; сортировка по нему = сортировка по группе, затем по дню
    key = group * BIG + day
    order = np.argsort(key, kind="mergesort")
    key_s, day_s, group_s = key[order], day[order], group[order]

    # Правая граница: позиция первой строки ТЕКУЩЕГО дня этой группы.
    # Всё, что левее, произошло строго раньше (side="left" выкидывает сам день T)
    right = np.searchsorted(key_s, key_s, side="left")

    # Префиксные суммы: сумма значений на отрезке [a, b) = pref[b] - pref[a]
    prefs = {c: np.concatenate([[0.0], np.cumsum(df[c].to_numpy(dtype=np.float64)[order])])
             for c in value_cols}

    out = {}
    for w in windows:
        # Левая граница: первая строка группы с днём >= T - w
        low_key = group_s * BIG + np.maximum(day_s - w, 0)
        left = np.searchsorted(key_s, low_key, side="left")
        cnt = (right - left).astype(np.float64)

        res = np.empty(n)
        res[order] = cnt                         # возвращаем исходный порядок строк
        out[f"{name}_cnt_{w}d"] = res.copy()
        for c in value_cols:
            s = prefs[c][right] - prefs[c][left]
            mean = np.full(n, np.nan)
            np.divide(s, cnt, out=mean, where=cnt > 0)
            res[order] = mean
            out[f"{name}_{c}_mean_{w}d"] = res.copy()
    return pd.DataFrame(out, index=df.index)


# ---------------- Сущности (группы), по которым считаются агрегаты ----------------
ENTITIES = {
    "user":       ["user_id"],                    # привычки пользователя
    "item":       ["item_id"],                    # популярность и досматриваемость объекта
    "genre":      ["main_genre"],                 # тренд жанра у всей аудитории
    "user_genre": ["user_id", "main_genre"],      # косвенная связь: пользователь x жанр
    "user_ctype": ["user_id", "content_type"],    # косвенная связь: пользователь x фильм/сериал
}
VALUE_COLS = ["watched_pct", "target"]            # что агрегируем: % досмотра и доля досмотров


def build_grid(df, windows=WINDOWS):
    """АВТОГЕНЕРАЦИЯ: все сущности x все окна x (count, mean pct, mean target)."""
    parts = [window_features(df, keys, VALUE_COLS, windows, name) for name, keys in ENTITIES.items()]
    return pd.concat(parts, axis=1)


# Ручной набор: зафиксирован ДО запусков модели на этих данных.
# Логика человека: неделя = свежий интерес, месяц = устойчивая привычка,
# для пар user x жанр / тип берём 60 дней, потому что событий у пары мало.
MANUAL_SPEC = [
    ("user",       [7, 30]),
    ("item",       [7, 30]),
    ("user_genre", [60]),
    ("user_ctype", [60]),
]


def build_manual(df):
    """РУЧНОЙ НАБОР: выбранные человеком окна + два признака-отношения."""
    parts = [window_features(df, ENTITIES[name], VALUE_COLS, wins, name) for name, wins in MANUAL_SPEC]
    feats = pd.concat(parts, axis=1)
    # Отношения, которых нет в сетке: насколько пара лучше обычного поведения пользователя
    feats["genre_affinity_60d"] = feats["user_genre_watched_pct_mean_60d"] - window_features(
        df, ["user_id"], ["watched_pct"], [60], "tmp")["tmp_watched_pct_mean_60d"]
    feats["user_genre_share_60d"] = feats["user_genre_cnt_60d"] / window_features(
        df, ["user_id"], [], [60], "tmp")["tmp_cnt_60d"].replace(0, np.nan)
    return feats
