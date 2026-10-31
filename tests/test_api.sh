#!/usr/bin/env bash
# Проверка API по разделу 10.10 ТЗ плюс коды ошибок и повторный запрос. Запускать при поднятом контейнере.
# bash tests/test_api.sh [BASE_URL], по умолчанию http://localhost:8000
set -euo pipefail
BASE_URL="${1:-http://localhost:8000}"
cd "$(dirname "$0")/.."
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

fail() { echo "ОШИБКА: $*"; exit 1; }
expect() {  # expect <код> <описание> <аргументы curl>, ответ в $TMP/body
    local code=$1 what=$2
    shift 2
    local got
    got=$(curl -s -o "$TMP/body" -w "%{http_code}" "$@")
    [ "$got" = "$code" ] || fail "$what: ждали $code, получили $got: $(head -c 300 "$TMP/body")"
    echo "  ок: $what -> $code"
}

echo "=== 1. Health check ==="
expect 200 "GET /health" "$BASE_URL/health"
expect 200 "GET /ready" "$BASE_URL/ready"
python3 -m json.tool "$TMP/body"

echo "=== 2. POST /predict ==="
expect 200 "POST /predict" -X POST "$BASE_URL/predict" -F "sales=@data/sample_sales.csv" -F "stock=@data/sample_stock.csv"
cp "$TMP/body" "$TMP/forecast.csv"
echo "  получен файл: $(wc -l < "$TMP/forecast.csv") строк (с заголовком)"
head -3 "$TMP/forecast.csv"
[ "$(head -1 "$TMP/forecast.csv")" = "week,warehouse_name,article,quantity" ] || fail "заголовок CSV"
WAREHOUSE=$(sed -n 2p "$TMP/forecast.csv" | cut -d, -f2)
ARTICLE=$(sed -n 2p "$TMP/forecast.csv" | cut -d, -f3)

echo "=== 3. GET /predict?article= ==="
expect 200 "по товару $ARTICLE" "$BASE_URL/predict?article=$ARTICLE"
head -3 "$TMP/body"

echo "=== 4. GET /predict?warehouse= ==="
expect 200 "по складу $WAREHOUSE" "$BASE_URL/predict?warehouse=$WAREHOUSE"
head -3 "$TMP/body"

echo "=== 5. GET /predict?article=&warehouse= ==="
expect 200 "по паре" "$BASE_URL/predict?article=$ARTICLE&warehouse=$WAREHOUSE"
[ "$(($(wc -l < "$TMP/body") - 1))" = 9 ] || fail "по паре должно быть 9 недель"
cat "$TMP/body"

echo "=== 6. GET /predict/all?format=csv ==="
expect 200 "весь прогноз, CSV" "$BASE_URL/predict/all?format=csv"
echo "  получен файл: $(wc -l < "$TMP/body") строк"
cmp -s "$TMP/body" "$TMP/forecast.csv" || fail "весь прогноз отличается от ответа POST"

echo "=== 7. GET /predict/all?format=json ==="
expect 200 "весь прогноз, JSON" "$BASE_URL/predict/all?format=json"
python3 -c "import json; d = json.load(open('$TMP/body')); print('  записей:', len(d)); print('  первые:', d[:2])"

echo "=== 8. Ошибки ==="
expect 400 "GET /predict без параметров (ожидается 400)" "$BASE_URL/predict"
expect 400 "неверный format" "$BASE_URL/predict/all?format=xml"
expect 404 "нет такого товара" "$BASE_URL/predict?article=char_article_no_such"
head -1 data/sample_sales.csv > "$TMP/header_only.csv"
expect 400 "файл без строк" -X POST "$BASE_URL/predict" -F "sales=@$TMP/header_only.csv" -F "stock=@data/sample_stock.csv"
(head -1 data/sample_sales.csv; tail -n 1500 data/sample_sales.csv) > "$TMP/short.csv"
expect 422 "история меньше 180 дней" -X POST "$BASE_URL/predict" -F "sales=@$TMP/short.csv" -F "stock=@data/sample_stock.csv"
expect 422 "нет файла stock" -X POST "$BASE_URL/predict" -F "sales=@data/sample_sales.csv"
python3 -m json.tool "$TMP/body"

echo "=== 9. Повторный POST отдаётся из хранилища ==="
curl -s -D "$TMP/headers" -o "$TMP/again.csv" -X POST "$BASE_URL/predict" -F "sales=@data/sample_sales.csv" -F "stock=@data/sample_stock.csv"
grep -qi "^x-cache: hit" "$TMP/headers" || fail "повторный POST посчитан заново, а не взят из хранилища"
cmp -s "$TMP/again.csv" "$TMP/forecast.csv" || fail "повторный POST дал другой ответ"
echo "  ок: X-Cache: hit, ответ тот же байт в байт"

echo "=== Все проверки пройдены ==="
