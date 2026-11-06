# tests/: проверки

`make test-local` запускает тесты без Docker, `make test` в образе. Настройки pytest лежат в `tests/pytest.ini`: медленный тест на полных данных помечен `slow` и по умолчанию пропускается.

## test_api.py

22 тестовых случая API через `TestClient` FastAPI, без Docker и без сети. Данные берутся из `data/sample_*.csv`.

| Тест | Что проверяет |
|---|---|
| `test_health_ready_info` | `/health`, `/ready`, `/model/info`: дата обучения и три участника ансамбля |
| `test_predict_format` | ответ POST с `return_csv=true`: колонки, 9 понедельников после последней даты, целые неотрицательные количества, заголовки `X-Cache`, `X-Forecast-Id`, `Server-Timing` |
| `test_predict_confirmation_and_download` (2 случая) | без флага и с `return_csv=false`: JSON с `forecast_id` и `success`, скачивание по id, повторный запрос из кэша и совпадение с CSV при `return_csv=true` |
| `test_predict_invalid_return_csv` | неверное значение флага даёт 422 |
| `test_predict_repeat_is_cached_and_identical` | повторный POST с теми же файлами берётся из хранилища и совпадает байт в байт |
| `test_get_filters` | GET по товару, складу, паре и по `forecast_id` |
| `test_get_errors` | GET без параметров (400), нет такого товара (404 `NOT_FOUND`), нет прогноза с таким id (404 `NO_FORECAST`) |
| `test_predict_all` | весь прогноз в CSV и JSON, CSV совпадает с ответом POST, неверный `format` (400) |
| `test_validation` (6 случаев) | нет колонки, плохая дата, меньше 180 дней, отрицательные продажи (422), пустой файл и файл только с заголовком (400) |
| `test_missing_file_field` | в запросе нет файла `stock` (422) |
| `test_accepted_variants` | отрицательный остаток и колонка `article` вместо `char_article` принимаются |
| `test_explain` | разбор по паре: 9 недель, по 5 главных признаков, количество совпадает с прогнозом |
| `test_baseline_model` | подмена модели на `baseline` через `config.json`, у неё нет объяснений (501) |
| `test_docs_on`, `test_docs_off` | при `DOCS=1` документация работает без интернета, при `DOCS=0` её нет |
| `test_reload` | без `ADMIN_TOKEN` эндпоинта нет, с неверным токеном 403, с верным модель перечитывается |

## test_forecaster.py

Медленный тест на полных данных: прогноз по последним 180 дням (столько приходит в контейнер) должен совпасть с `submission/submission.csv` во всех строках. Так проверяется главное обещание сервиса: с окном признаков до 24 недель контейнер считает тот же прогноз, что ушёл в сабмит. Вход готовит `make api-input`, запуск: `.venv/bin/python -m pytest -m slow tests`.

## test_api.sh

Проверка уже поднятого сервиса curl-ами по разделу 10.10 ТЗ плюс коды ошибок и повторный запрос: `make test-api` или `bash tests/test_api.sh URL`. Его же запускает CI.

## make_api_input.py

Готовит вход для API в формате организаторов из полных данных. `sample`: ~30 товаров на всех складах за 200 дней в `data/sample_*.csv`, для примеров и тестов. `full`: все товары за последние 180 дней в `outputs/api_input/`, для `make check-container` и медленного теста. Большие CSV читаются кусками, чтобы не держать их в памяти целиком.

## conftest.py и pytest.ini

`conftest.py` добавляет корень проекта в путь и даёт фикстуру `sample` с файлами примера. В `pytest.ini` маркер `slow` и `-m "not slow"` по умолчанию.
