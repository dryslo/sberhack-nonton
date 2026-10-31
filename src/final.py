"""Итоговый конфиг: то, что прошло проверку на трёх фолдах (журнал в results/ideas.csv, эксперименты в notebooks/05_experiments).

- признаки: FEATURES из classifier.py и regressor.py, вес свежих недель HALF_LIFE = 26;
- ансамбль: CatBoost (сид 42), CatBoost (сид 7) и LightGBM, вероятности усредняются, количества в логарифмах (05_lgbm);
- порог продажи 0.55 для недель 1-4 и 0.5 для недель 5-9 (05_scale, 05_combo);
- количество переводится в целое с наименьшим ожидаемым SMAPE, если продажи = 1 + Пуассон(q - 1) (v4).
  Соответствие прогноз -> целое, выученное по фактам (05_intmap, results/integer_map.json), на сезонных фолдах
  было лучше, но на recent завышало объём и ухудшало WAPE, Bias и RMSLE, поэтому в финал не вошло.
Обучение как в валидации: каждая 2-я неделя отсчёта (ORIGIN_STEP = 2).
"""

ORIGIN_STEP = 2

PARAMS = {
    "threshold": (0.55, 0.5),
    "integer": "smape",
    "ensemble": [{}, {"random_seed": 7}, {"library": "lightgbm"}],
}

# score итогового конфига на фолдах (05_combo)
BACKTEST = {"season2025": 0.8077, "season2024": 0.8194, "recent": 0.7832}
