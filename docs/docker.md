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
| `STORAGE_URL` | `file:///cache` в Compose; иначе не задан | `file:///путь` — диск; `redis://host:6379/0` — Redis; `rediss://host:6379/0` — Redis с TLS. Если не задан, используется `CACHE_DIR` |
| `MODEL_DIR` | `/app/model` | папка модели, для подмены модели укажите другую |
| `ADMIN_TOKEN` | не задан | если задан, появляется `POST /model/reload` с заголовком `X-Admin-Token` |

### Выбор хранилища

Оба хранилища реализуют абстрактный адаптер `ForecastStore`: `LocalForecastStore` и `RedisForecastStore`.
В них сохраняются CSV, метаданные, данные для `/explain` и указатель на последний прогноз.
`MAX_FORECASTS` ограничивает число сохранённых прогнозов в обоих режимах. API и ключ кэша не меняются.

Локально: путь в ссылке относится к файловой системе контейнера. Проброс каталога сохраняет прогнозы на диске хоста:

```sh
docker run -p 8000:8000 --mount type=bind,src=/absolute/path/forecasts,dst=/cache -e STORAGE_URL=file:///cache sberhack-forecast
```

Каталог хоста должен быть доступен для записи пользователю контейнера с UID 10001.
Без `STORAGE_URL` продолжает работать прежняя настройка `CACHE_DIR` и существующий формат файлов.
Compose по умолчанию использует `file:///cache` и именованный том `forecast-cache`.

Удалённо: передайте адрес уже запущенного Redis, доступного из контейнера:

```sh
docker run -p 8000:8000 -e STORAGE_URL=redis://redis-host:6379/0 sberhack-forecast
```

Для Compose задайте `STORAGE_URL` в окружении или в файле, переданном через `--env-file`:

```sh
docker compose --env-file storage.env -f docker/docker-compose.yml up -d --build
```

Содержимое `storage.env`: `STORAGE_URL=redis://redis-host:6379/0`.
Ссылка с авторизацией: `redis://username:password@redis-host:6379/0`; для TLS используйте `rediss://`.
Специальные символы в имени и пароле должны быть URL-кодированы.
Все экземпляры API с одной базой Redis используют общие прогнозы (префикс ключей `forecast-store:`).
Запись прогноза и удаление старых записей выполняются одной транзакцией.
В режиме Redis прогнозы не записываются в `/cache`; сохранность после перезапуска самого Redis зависит от его настроек AOF/RDB.
Переключение `STORAGE_URL` не переносит существующие прогнозы между хранилищами.

## requirements.txt

Зависимости образа с точными версиями: библиотеки модели (как при обучении), FastAPI, uvicorn, python-multipart и pydantic. Для разработки `make venv` ставит этот же файл и инструменты для ноутбуков, Kaggle и тестов.

## CI: .github/workflows/ci.yml

На каждый push и pull request: тесты в образе, сборка рабочего образа, запуск контейнера и проверка поднятого сервиса скриптом `tests/test_api.sh`. Это то же, что локально `make test`, `make build` и `make test-api`.
