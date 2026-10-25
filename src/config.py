"""Все настройки проекта в одном месте. Меняешь здесь, а не в скриптах."""
from pathlib import Path

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
RES_DIR = Path("results")
PRED_DIR = RES_DIR / "preds"

# Сэмпл пользователей
SAMPLE_SEED = 42
N_USERS = 50_000            # можно переопределить: python src/01_prepare_sample.py 20000

# Целевая переменная: досмотрел ли пользователь объект хотя бы на 50%
TARGET_THRESHOLD = 50.0

# Границы разбиения (фиксированные даты, не зависят от размера сэмпла)
VAL_START = "2021-07-26"    # val:  2021-07-26 .. 2021-08-08
TEST_START = "2021-08-09"   # test: 2021-08-09 .. 2021-08-22

# Чистка жанров: эти теги описывают происхождение или тип, а не жанр
ORIGIN_TAGS = {"русские", "зарубежные", "советские"}
SKIP_TAGS = ORIGIN_TAGS | {"сериалы"}
GENRE_FIX = {"мультфильмы": "мультфильм", "историческое": "исторические"}

# Статические признаки (не зависят от истории)
CAT_COLS = ["age", "income", "sex", "content_type", "main_genre", "origin", "main_country"]
STATIC_NUM_COLS = ["kids_flg", "release_year", "for_kids", "age_rating"]

# Сетка для автогенерации: окна в днях
WINDOWS = [1, 3, 7, 14, 30, 60]

# Обучение: одинаковые параметры для всех подходов, подбирается только число деревьев
# (early stopping по val). Гиперпараметры не настраивались.
CATBOOST_PARAMS = dict(
    iterations=1000,
    learning_rate=0.08,
    depth=6,
    eval_metric="AUC",
)
EARLY_STOPPING = 50
TRAIN_SEEDS = [0, 1, 2]

# Метрики
NDCG_K = 10
N_BOOTSTRAP = 300
