"""Запуск ноутбуков из notebooks/ на Kaggle. Обычно через make:

    make check         # есть ли токен Kaggle и загруженные данные
    make data          # загрузить CSV из data/ приватным датасетом (повторный вызов создаёт новую версию)
    make run NB=03     # выполнить notebooks/03_classifier.ipynb на GPU, дождаться и забрать выводы
    make push NB=03    # то же без ожидания, потом make status NB=03 и make pull NB=03

На Kaggle уходит небольшой скрипт, внутри которого модули src/ и сам ноутбук без выводов. Скрипт раскладывает их
как в проекте, выполняет ноутбук и после каждой ячейки сохраняет его. На Kaggle настройки моделей полные
(ON_KAGGLE в src/). pull переносит выводы в локальный ноутбук, но только в ячейки, которые с тех пор не менялись.
"""
import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
KAGGLE = str(Path(sys.executable).with_name("kaggle"))
NOTEBOOKS = {"01": "01_start", "02": "02_eda", "03": "03_classifier", "04": "04_regressor", "05": "05_experiment"}
DATA_FILES = ["sales_dataset_hakaton_train.csv", "stock_dataset_hakaton_train.csv"]
GPU = "NvidiaTeslaT4"  # P100 Kaggle больше не выдаёт
OUT = PROJECT / "outputs" / "kaggle"
TOKEN = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"

# __NAME__, __MODULES__ и __NOTEBOOK__ подставляются при запуске
RUNNER = '''"""Выполняет ноутбук __NAME__ на Kaggle так же, как в проекте (src/ рядом с notebooks/), и сохраняет его с выводами."""
import base64, glob, json, os, subprocess, sys

WORK = "/kaggle/working"
NAME = "__NAME__"
MODULES = json.loads(base64.b64decode("__MODULES__"))
NOTEBOOK = base64.b64decode("__NOTEBOOK__").decode()

for pkg in ("holidays", "nbclient", "nbformat"):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
import nbformat
from nbclient import NotebookClient
import platform
print("python", platform.python_version(), "| " + ", ".join(f"{m} {__import__(m).__version__}" for m in
      ("pandas", "numpy", "catboost", "IPython", "nbclient")), flush=True)

found = glob.glob("/kaggle/input/**/sales_dataset_hakaton_train.csv", recursive=True)
if not found:
    sys.exit("Нет данных: к запуску не подключён датасет с CSV (make data)")
os.environ["HACK_DATA_DIR"] = os.path.dirname(found[0])
os.environ["HACK_OUT_DIR"] = WORK

os.makedirs(f"{WORK}/src", exist_ok=True)
os.makedirs(f"{WORK}/notebooks", exist_ok=True)
for name, text in MODULES.items():
    with open(f"{WORK}/src/{name}", "w") as f:
        f.write(text)

nb = nbformat.reads(NOTEBOOK, as_version=4)
total = sum(c.cell_type == "code" for c in nb.cells)


class SavingClient(NotebookClient):
    # сохраняем после каждой ячейки, чтобы не потерять сделанное, если сессия оборвётся.
    # on_cell_executed не подходит: старый nbclient на Kaggle его не вызывает
    done = 0

    async def async_execute_cell(self, cell, cell_index, *args, **kwargs):
        result = await super().async_execute_cell(cell, cell_index, *args, **kwargs)
        if cell.cell_type == "code":
            self.done += 1
            print(f"ячейка {self.done}/{total} готова", flush=True)
            nbformat.write(self.nb, f"{WORK}/{NAME}.ipynb")
        return result


SavingClient(nb, timeout=None, kernel_name="python3", allow_errors=True,
             resources={"metadata": {"path": f"{WORK}/notebooks"}}).execute()
nbformat.write(nb, f"{WORK}/{NAME}.ipynb")
errors = sum(any(o.get("output_type") == "error" for o in c.get("outputs", [])) for c in nb.cells if c.cell_type == "code")
print("готово; ячеек с ошибкой:", errors)
'''


def user():
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    if not TOKEN.exists():
        sys.exit(f"Нет токена Kaggle ({TOKEN}): kaggle.com -> Settings -> API -> Create New Token, затем "
                 f"положи скачанный kaggle.json в {TOKEN.parent}/ и сделай chmod 600 {TOKEN}")
    return json.loads(TOKEN.read_text())["username"]


def notebook_path(name):
    found = sorted((PROJECT / "notebooks").glob(f"{name}.ipynb")) + sorted((PROJECT / "notebooks").glob(f"*/{name}.ipynb"))
    return found[0] if found else None


def notebook_name(nb):
    # короткий номер из NOTEBOOKS или имя ноутбука без .ipynb
    if nb in NOTEBOOKS:
        return NOTEBOOKS[nb]
    if notebook_path(nb):
        return nb
    sys.exit(f"Нет ноутбука {nb}: ни среди {', '.join(sorted(NOTEBOOKS))}, ни в notebooks/ и его подпапках")


def dataset_id():
    return f"{user()}/sberhack-data"


def kernel_id(name):
    return f"{user()}/sberhack-{name.replace('_', '-')}"


def kaggle(*args):
    if subprocess.run([KAGGLE, *args]).returncode:
        sys.exit(f"Не прошло: kaggle {' '.join(args[:2])}. Сообщение Kaggle выше.")


def dataset_status():
    # "ready", "pending" и т.п. или None, если датасета нет
    try:
        r = subprocess.run([KAGGLE, "datasets", "status", dataset_id()], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        sys.exit("Kaggle не ответил за минуту, проверь интернет и попробуй ещё раз")
    return r.stdout.strip().lower() if r.returncode == 0 else None


def check():
    print(f"токен: {TOKEN}, пользователь {user()}")
    status = dataset_status()
    print(f"данные на Kaggle: {dataset_id()}, {status}" if status else "данных на Kaggle нет, загрузить: make data")


def data():
    exists = dataset_status() is not None
    with tempfile.TemporaryDirectory(dir=PROJECT / "outputs") as tmp:  # тот же диск, чтобы не копировать 550 МБ
        tmp = Path(tmp)
        for f in DATA_FILES:
            try:
                os.link(PROJECT / "data" / f, tmp / f)
            except OSError:
                shutil.copy(PROJECT / "data" / f, tmp / f)
        meta = {"title": "sberhack data", "id": dataset_id(), "licenses": [{"name": "other"}]}
        (tmp / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
        if exists:
            kaggle("datasets", "version", "-p", str(tmp), "-m", "обновление данных", "-r", "skip")
        else:
            kaggle("datasets", "create", "-p", str(tmp), "-r", "skip")  # по умолчанию приватный
    print("Kaggle обрабатывает данные...", flush=True)
    for _ in range(60):
        if dataset_status() == "ready":
            break
        time.sleep(10)
    print(f"данные: https://www.kaggle.com/datasets/{dataset_id()}, {dataset_status()}")


def runner_script(name, modules=None):
    if modules is None:
        modules = {p.name: p.read_text() for p in sorted((PROJECT / "src").glob("*.py"))}
    nb = json.loads(notebook_path(name).read_text())
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    b64 = lambda s: base64.b64encode(s.encode()).decode()
    return (RUNNER.replace("__NAME__", name).replace("__MODULES__", b64(json.dumps(modules)))
            .replace("__NOTEBOOK__", b64(json.dumps(nb))))


def push(nb, wait=False):
    name = notebook_name(nb)
    if dataset_status() is None:
        sys.exit("На Kaggle нет данных, сначала make data")
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "runner.py").write_text(runner_script(name))
        meta = {"id": kernel_id(name), "title": f"sberhack {name.replace('_', ' ')}", "code_file": "runner.py",
                "language": "python", "kernel_type": "script", "is_private": "true", "enable_gpu": "true",
                "enable_tpu": "false", "enable_internet": "true", "machine_shape": GPU,
                "dataset_sources": [dataset_id()], "competition_sources": [], "kernel_sources": [], "model_sources": []}
        (Path(tmp) / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        kaggle("kernels", "push", "-p", tmp)
    print(f"запущен {name}: https://www.kaggle.com/code/{kernel_id(name)}")
    if wait:
        wait_done(nb)
        pull(nb)


def status(nb):
    # (0, статус) или (код ошибки, сообщение)
    try:
        r = subprocess.run([KAGGLE, "kernels", "status", kernel_id(notebook_name(nb))], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return 1, "Kaggle не ответил за минуту"
    return r.returncode, (r.stdout + r.stderr).strip()


def wait_done(nb, every=60):
    last, fails = None, 0
    while True:
        code, text = status(nb)
        fails = fails + 1 if code else 0
        if fails >= 5:
            sys.exit(f"Статус не получить: {text}")
        if text != last:
            print(time.strftime("%H:%M"), text, flush=True)
            last = text
        if code == 0 and any(s in text.lower() for s in ("complete", "error", "cancel")):
            return
        time.sleep(every)


def merge(executed, local):
    # выводы переносим только в ячейки, текст которых не менялся с момента запуска
    ran = {c.get("id"): c for c in json.loads(executed.read_text())["cells"]}
    nb = json.loads(local.read_text())
    filled = changed = 0
    for c in nb["cells"]:
        r = ran.get(c.get("id"))
        if c["cell_type"] != "code" or r is None:
            continue
        if "".join(r["source"]) != "".join(c["source"]):
            changed += 1
            continue
        c["outputs"], c["execution_count"] = r.get("outputs", []), r.get("execution_count")
        filled += 1
    local.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    return filled, changed


def pull(nb):
    name = notebook_name(nb)
    dest = OUT / name
    dest.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", kernel_id(name), "-p", str(dest), "-o", "-q", "--file-pattern", r"\.(ipynb|csv|json|cbm|txt)$")
    executed = dest / f"{name}.ipynb"
    if not executed.exists():
        sys.exit(f"Выполненного ноутбука нет: запуск ещё идёт (make status NB={nb}) или упал сразу, лог в {dest}/")
    local = notebook_path(name)
    filled, changed = merge(executed, local)
    print(f"выводы записаны в {local.relative_to(PROJECT)}, ячеек: {filled}" + (f", изменённых ячеек пропущено: {changed}" if changed else ""))
    for i, c in enumerate(json.loads(executed.read_text())["cells"]):
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                print(f"  ошибка в ячейке {i}: {o.get('ename')}: {o.get('evalue', '')[:200]}")
    print(f"выполненный ноутбук, CSV и лог: {dest.relative_to(PROJECT)}/")


def main():
    parser = argparse.ArgumentParser(description="Запуск ноутбуков на Kaggle")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    sub.add_parser("data")
    for cmd in ("push", "status", "pull"):
        p = sub.add_parser(cmd)
        p.add_argument("nb", help="01, 02, 03, 04, 05 или имя ноутбука из notebooks/ без .ipynb")
        if cmd == "push":
            p.add_argument("--wait", action="store_true", help="дождаться конца и забрать результаты")
    args = parser.parse_args()
    if args.cmd == "check":
        check()
    elif args.cmd == "data":
        data()
    elif args.cmd == "push":
        push(args.nb, args.wait)
    elif args.cmd == "status":
        code, text = status(args.nb)
        print(text if code == 0 else f"статуса нет: запуска ещё не было (make push NB={args.nb}) или Kaggle недоступен")
    else:
        pull(args.nb)


if __name__ == "__main__":
    main()
