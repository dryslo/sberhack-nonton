# docker/: образ и запуск сервиса

Все Docker-файлы лежат в `docker/`, а собирается образ из корня репозитория: в него нужны `src/`, `app/`, `model/` и примеры из `data/`. Короткие команды есть в Makefile: `make build`, `make test`, `make up`, `make down`.

## Dockerfile

Сборка в несколько этапов:

- `deps`: venv с библиотеками из `requirements.txt`. Сразу удаляются зависимости CatBoost для графиков (plotly, matplotlib и их пакеты): прогнозу они не нужны, а образ без них меньше на 165 МБ.
- `runtime`: рабочий образ на `python:3.13-slim`. Только CPU: CatBoost и LightGBM считают прогноз за секунды, CUDA не нужна, образ весит около 1.05 ГБ. `libgomp1` нужен LightGBM. Сервис работает не от root, есть `HEALTHCHECK` на `/health`, прогнозы лежат в томе `/cache`, uvicorn запускается с `WORKERS` процессами (по умолчанию 2).
- `test`: тот же образ плюс pytest и httpx, запускает `tests/` (`make test`). Тесты идут ровно в том окружении, в котором работает сервис.
- без `--target` собирается `runtime`.

Python 3.13 и версии библиотек модели такие же, как при финальном обучении на Kaggle (`libraries` в `model/config.json`). С другими версиями прогноз из сохранённых моделей мог бы разойтись с сабмитом.

## Dockerfile.dockerignore

Белый список того, что попадает в контекст сборки. Без него в контекст ушли бы данные организаторов (528 МБ), `.venv` и `outputs/`. Файл с таким именем Docker берёт для Dockerfile из той же папки, если сборка идёт через BuildKit, а это поведение по умолчанию начиная с Docker 23 и в Docker Desktop.

## docker-compose.yml

Один сервис `api` и том `forecast-cache`, поэтому прогнозы переживают рестарт. Проект назван `sberhack`, так что имя тома не зависит от папки, из которой запускают compose. Запуск из корня: `make up` или `docker compose -f docker/docker-compose.yml up -d --build`.

| Переменная | По умолчанию | Что делает |
|---|---|---|
| `DOCS` | `1` | при `0` отключаются `/docs`, `/redoc`, `/solution`, `/openapi.json`, а `/` отдаёт короткий JSON |
| `WORKERS` | `2` | число процессов uvicorn, прогнозы у них общие через хранилище |
| `MAX_CONCURRENT_PREDICTIONS` | `1` | одновременных расчётов на процесс (полный POST берёт до ~1 ГБ памяти) |
| `MAX_FORECASTS` | `20` | сколько последних прогнозов хранить |
| `CACHE_DIR` | `/cache` | хранилище прогнозов (том) |
| `MODEL_DIR` | `/app/model` | папка модели, для подмены модели укажите другую |
| `ADMIN_TOKEN` | не задан | если задан, появляется `POST /model/reload` с заголовком `X-Admin-Token` |

## requirements.txt

Зависимости образа с точными версиями: библиотеки модели (как при обучении), FastAPI, uvicorn, python-multipart и pydantic. Для разработки `make venv` ставит этот же файл и инструменты для ноутбуков, Kaggle и тестов.

## CI: .github/workflows/ci.yml

На каждый push и pull request: тесты в образе, сборка рабочего образа, запуск контейнера и проверка поднятого сервиса скриптом `tests/test_api.sh`. Это то же, что локально `make test`, `make build` и `make test-api`.
