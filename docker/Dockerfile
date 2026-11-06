# Прогноз продаж, HTTP API по разделу 10 ТЗ. Собирается из корня репозитория:
#   docker build -f docker/Dockerfile -t sberhack-forecast .
#   docker run -p 8000:8000 sberhack-forecast    # http://localhost:8000, документация на /docs
#   docker build -f docker/Dockerfile --target test -t sberhack-forecast-test . && docker run --rm sberhack-forecast-test
# Python и версии библиотек как при финальном обучении на Kaggle, иначе прогноз может разойтись с сабмитом.

FROM python:3.13-slim AS deps
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
# зависимости CatBoost для графиков прогнозу не нужны, минус 165 МБ
RUN pip uninstall -y -q plotly matplotlib fonttools pillow contourpy kiwisolver cycler pyparsing narwhals

FROM python:3.13-slim AS runtime
# libgomp1 нужен LightGBM. Прогноз считается на CPU
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app && mkdir -p /cache && chown app:app /cache
COPY --from=deps /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    MODEL_DIR=/app/model CACHE_DIR=/cache DOCS=1 WORKERS=2 MAX_CONCURRENT_PREDICTIONS=1 MAX_FORECASTS=20
WORKDIR /app
COPY src/ src/
COPY app/ app/
COPY model/ model/
# LightGBM хранит размеры деревьев в байтах: CRLF после checkout на Windows нарушает смещения.
RUN python -c "from pathlib import Path; [p.write_bytes(p.read_bytes().replace(b'\r\n', b'\n')) for p in Path('model').glob('*.txt')]"
COPY data/sample_sales.csv data/sample_stock.csv data/
USER app
EXPOSE 8000
VOLUME ["/cache"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${WORKERS}"]

FROM runtime AS test
USER root
RUN pip install pytest==9.2.0 httpx==0.28.1 fakeredis==2.40.0
COPY tests/ tests/
USER app
CMD ["python", "-m", "pytest", "-q", "tests"]

# без --target собирается рабочий образ
FROM runtime
