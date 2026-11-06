# make help - список команд
# NB для Kaggle: 01, 02, 03, 04, 05, 06_final или имя ноутбука из notebooks/, например NB=05_intmap
PY ?= .venv/bin/python
NB ?=
IMAGE ?= sberhack-forecast
URL ?= http://localhost:8000
COMPOSE = docker compose -f docker/docker-compose.yml
DEV = jupyter nbconvert matplotlib seaborn kaggle pytest httpx fakeredis==2.40.0

.PHONY: help venv test-local check data push run status pull _nb build test up down test-api api-input check-container solution-html

help:  ## список команд
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | sed -E 's/:.*## /\t/' | expand -t 18

# окружение и тесты без Docker

venv:  ## .venv для ноутбуков, Kaggle и тестов: библиотеки модели как в образе плюс инструменты
	python3 -m venv .venv
	.venv/bin/pip install -r docker/requirements.txt $(DEV)

test-local:  ## тесты API без Docker
	$(PY) -m pytest -q tests

# Kaggle

check:  ## проверить токен Kaggle и данные на нём
	$(PY) kaggle/run.py check

data:  ## загрузить CSV из data/ на Kaggle приватным датасетом
	$(PY) kaggle/run.py data

run: _nb  ## выполнить ноутбук на Kaggle и дождаться выводов: make run NB=06_final
	$(PY) kaggle/run.py push $(NB) --wait

push: _nb  ## запустить на Kaggle без ожидания: make push NB=03
	$(PY) kaggle/run.py push $(NB)

status: _nb  ## статус запуска: make status NB=03
	$(PY) kaggle/run.py status $(NB)

pull: _nb  ## забрать выводы в notebooks/: make pull NB=03
	$(PY) kaggle/run.py pull $(NB)

_nb:
	@test -n "$(NB)" || { echo "Укажи ноутбук, например make run NB=03"; exit 1; }

# контейнер с API (файлы в docker/)

build:  ## собрать образ с API
	docker build -f docker/Dockerfile -t $(IMAGE) .

test:  ## тесты API в образе
	docker build -f docker/Dockerfile --target test -t $(IMAGE)-test . && docker run --rm $(IMAGE)-test

up:  ## поднять сервис на http://localhost:8000 (docker compose)
	$(COMPOSE) up -d --build

down:  ## остановить сервис
	$(COMPOSE) down

test-api:  ## проверить поднятый сервис по 10.10 ТЗ: make test-api URL=http://localhost:8000
	bash tests/test_api.sh $(URL)

api-input:  ## последние 180 дней всех товаров в outputs/api_input/ (нужны data/*.csv)
	$(PY) tests/make_api_input.py full

check-container: api-input  ## сравнить прогноз сервиса по 180 дням с submission/submission.csv
	curl -sf -X POST $(URL)/predict -F "sales=@outputs/api_input/sales_last180.csv" \
	    -F "stock=@outputs/api_input/stock_last180.csv" -o outputs/api_input/forecast_container.csv
	cmp outputs/api_input/forecast_container.csv submission/submission.csv && echo "прогноз контейнера совпадает с submission/submission.csv"

solution-html:  ## notebooks/solution.ipynb в app/static/solution.html для /solution
	$(PY) -m jupyter nbconvert --to html --no-input --output-dir app/static --output solution notebooks/solution.ipynb
