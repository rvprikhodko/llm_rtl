# Инструкция для стенда

Все команды, кроме раздела 1, выполняются в Linux внутри WSL2 из корня репозитория `~/llm_rtl`.

Порядок работ:

1. Отдельный дистрибутив WSL.
2. Прокси.
3. Системные пакеты.
4. Репозитории и данные.
5. Python-окружение.
6. Проверка GPU.
7. Фильтрация датасета с iverilog.
8. Подбор `micro_batch_size`.
9. Подбор LR.
10. Основная серия.
11. Перенос результатов.

## 1. Отдельный дистрибутив WSL (PowerShell в Windows)

Создать новый дистрибутив (можно пропустить):

```powershell
wsl --install Ubuntu-24.04 --name llm-rtl
wsl -d llm-rtl
```

Важно:

- **`wsl --shutdown` или перезапуск Docker Desktop убьёт обучение.** Запуски умеют продолжаться с последнего чекпойнта (раздел 10).
- **Код, данные и кэш моделей нужно держать в `~/` (ext4), а не на `/mnt/c` или `/mnt/e`.**

## 2. Прокси

Нужно прокинуть прокси

## 3. Системные пакеты

```bash
sudo apt update
sudo apt install -y build-essential git curl tmux python3-venv python3-dev iverilog yosys
iverilog -V | head -1; yosys -V
```

## 4. Репозитории и данные

```bash
cd ~
git clone https://github.com/rvprikhodko/llm_rtl.git
cd llm_rtl
git clone https://hub.mos.ru/nltoverilog/vai-experiments-rtllm.git
mkdir -p dataset
curl -L -o dataset/Resyn27k.json https://raw.githubusercontent.com/hkust-zhiyao/RTL-Coder/main/dataset/Resyn27k.json
sha256sum dataset/Resyn27k.json
```

Ожидаемый SHA-256 датасета: `6f50e411a17e667520a6e6b6fefea59684b6cd469639483bf3440d557ace82ae`.

## 5. Python-окружение

```bash
python3 -m venv ~/venvs/llm-rtl
source ~/venvs/llm-rtl/bin/activate
pip install --upgrade pip
pip install torch
pip install -r requirements-train.txt
pip freeze > requirements-train.lock.txt
```

`pip install torch` с PyPI под Linux ставит сборку с CUDA. Если оказалась CPU-сборка, нужно поставить явно:
`pip install torch --index-url https://download.pytorch.org/whl/cu128`.
Файл `requirements-train.lock.txt` где-то сохранить: это зафиксированные версии для раздела о воспроизводимости.

Для удобства можно добавить в `~/.bashrc` строку `source ~/venvs/llm-rtl/bin/activate`.

## 6. Проверка GPU

```bash
bash scripts/env_check.sh > env_report_wsl.txt 2>&1
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.cuda.is_bf16_supported())"
```

Ожидается: `True`, `NVIDIA GeForce RTX 4070 Ti SUPER`, `(8, 9)`, `True`.

Можно скачать модель заранее, чтобы первый запуск не ждал загрузки около 3 ГБ:

```bash
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer as T; n='EleutherAI/pythia-1.4b'; T.from_pretrained(n, revision='step143000'); AutoModelForCausalLM.from_pretrained(n, revision='step143000')"
```

## 7. Фильтрация датасета с iverilog

Набор `dataset/filtered` из локального прогона собран **без** проверки синтаксиса. На стенде его нужно пересобрать с iverilog и обучать только на этом результате:

```bash
python scripts/analyze_dataset.py dataset/Resyn27k.json --out dataset/analysis --rtllm vai-experiments-rtllm/datasets/trusted/rtllm-882030d9e61b7787/cases --iverilog --workers 32
python scripts/filter_dataset.py dataset/Resyn27k.json --out dataset/filtered --rtllm vai-experiments-rtllm/datasets/trusted/rtllm-882030d9e61b7787/cases --iverilog --workers 32
```

Без iverilog осталось 23 366 записей. С ним станет меньше на число некомпилирующихся эталонов.
Нужно сохранить `dataset/analysis/report.json` и `dataset/filtered/filter_report.json`: их цифры пойдут в сравнительный анализ.

## 8. Подбор micro_batch_size

Скрипт обучает по 3 шага на самых длинных примерах, поэтому измеренная пиковая память — оценка сверху:

```bash
python scripts/probe_microbatch.py --config configs/train/pythia-1.4b.json --output runs/probe/pythia-1.4b
```

Рекомендованное значение нужно вписать в `configs/train/pythia-1.4b.json` (`micro_batch_size`).

## 9. Подбор LR

Нужно запускать в `tmux`, чтобы обучение не прервалось при закрытии терминала:

```bash
tmux new -s lr
python scripts/lr_search.py --config configs/train/pythia-1.4b.json --output runs/lr_search/pythia-1.4b
```

Отключиться от сессии, не останавливая её: `Ctrl+B`, затем `D`. Вернуться: `tmux attach -t lr`.
Ожидаемое время — около 1.5 часа (5 значений LR × 20% данных × 1 эпоха).
Если скрипт предупредит, что лучший LR на краю сетки, нужно добавить значения через `--lrs` и запустить ту же команду ещё раз:
готовые варианты будут пропущены.

Лучший LR нужно вписать в конфиг (`lr`) и сохранить конфиг вместе с `runs/lr_search/pythia-1.4b/lr_search.json`.

## 10. Основная серия: 5 сидов

```bash
tmux new -s main
for s in 0 1 2 3 4; do
  python scripts/train_lora.py --config configs/train/pythia-1.4b.json --seed $s --output runs/main/pythia-1.4b/s$s || break
done
```

Если запуск прервался (`wsl --shutdown`, перезагрузка), нужно продолжить его с последнего чекпойнта:

```bash
python scripts/train_lora.py --config configs/train/pythia-1.4b.json --seed 3 --output runs/main/pythia-1.4b/s3 --resume
```

После этого нужно перезапустить цикл с оставшимися сидами.
Оценка: около 1.5–2 часов на эпоху, то есть 3–4 часа на сид и 15–20 часов на серию.


## 11. Перенос результатов

Чекпойнты-адаптеры небольшие (десятки мегабайт), поэтому архивировать можно всё:

```bash
tar czf ~/results_$(date +%Y%m%d).tar.gz runs dataset/analysis dataset/filtered/filter_report.json env_report_wsl.txt requirements-train.lock.txt
cp ~/results_*.tar.gz /mnt/e/
```
