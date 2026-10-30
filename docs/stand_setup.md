# Инструкция для стенда

Стенд: Windows 11, WSL2, RTX 4070 Ti SUPER 16 ГБ, i9-13900K, 64 ГБ ОЗУ. Машина общая (CI лаборатории).
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

## 0. Перед поездкой на стенд

На своём компьютере закоммить и запушь текущее состояние, чтобы на стенде оказался тот же код:

```bash
git push origin main
```

Запиши хеш коммита (`git rev-parse HEAD`). Каждый запуск сохраняет его в `run_manifest.json`.

## 1. Отдельный дистрибутив WSL (PowerShell в Windows)

На стенде уже есть 14 дистрибутивов других задач. Чтобы не трогать их окружения, создай свой:

```powershell
wsl --install Ubuntu-24.04 --name llm-rtl
wsl -d llm-rtl
```

При первом входе задай имя пользователя и пароль (пароль нужен для `sudo`).
Если `--name` не поддерживается или установка из интернета закрыта, попроси у администраторов образ Ubuntu 24.04
и импортируй его через `wsl --import`.

Важно:

- **Не меняй `.wslconfig`.** Он действует на все дистрибутивы стенда. По умолчанию WSL получает 31 ГБ ОЗУ, этого достаточно.
- **`wsl --shutdown` или перезапуск Docker Desktop кем-то из коллег убьёт обучение.** Запуски умеют продолжаться с последнего чекпойнта (раздел 10).
- **Код, данные и кэш моделей держи в `~/` (ext4), а не на `/mnt/c` или `/mnt/e`.** Чтение с дисков Windows через WSL в разы медленнее.

## 2. Прокси

Подставь адрес прокси лаборатории вместо `<PROXY>` (например, `http://host:port`) и добавь строки в `~/.bashrc`:

```bash
export http_proxy=<PROXY> https_proxy=<PROXY> HTTP_PROXY=<PROXY> HTTPS_PROXY=<PROXY>
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1
export HF_HOME=$HOME/.cache/huggingface
```

Для `apt` создай файл `/etc/apt/apt.conf.d/95proxy` со строкой `Acquire::http::Proxy "<PROXY>"; Acquire::https::Proxy "<PROXY>";`.
Если пакеты ставятся через Nexus, вместо прокси укажи в `/etc/apt/sources.list.d/` и в `pip config set global.index-url` адреса зеркал Nexus.

Если прокси подменяет TLS-сертификаты, установи корпоративный CA:

```bash
sudo cp lab-ca.crt /usr/local/share/ca-certificates/ && sudo update-ca-certificates
```

В этом случае добавь в `~/.bashrc` строку `export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt`.

Проверка (должны вернуться HTTP-коды, а не `000`):

```bash
for u in https://huggingface.co https://pypi.org/simple/ https://files.pythonhosted.org https://github.com https://download.pytorch.org/whl/; do printf "%s " $u; curl -sS -o /dev/null -w "%{http_code}\n" -m 15 $u; done
```

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
Если GitHub недоступен, скопируй файл с флешки или из Windows (`cp /mnt/e/.../Resyn27k.json dataset/`) и проверь хеш.

## 5. Python-окружение

```bash
python3 -m venv ~/venvs/llm-rtl
source ~/venvs/llm-rtl/bin/activate
pip install --upgrade pip
pip install torch
pip install -r requirements-train.txt
pip freeze > requirements-train.lock.txt
```

`pip install torch` с PyPI под Linux ставит сборку с CUDA. Если оказалась CPU-сборка, поставь явно:
`pip install torch --index-url https://download.pytorch.org/whl/cu128`.
Файл `requirements-train.lock.txt` закоммить: это зафиксированные версии для раздела о воспроизводимости.

Для удобства добавь в `~/.bashrc` строку `source ~/venvs/llm-rtl/bin/activate`.

## 6. Проверка GPU

```bash
bash scripts/env_check.sh > env_report_wsl.txt 2>&1
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.cuda.is_bf16_supported())"
```

Ожидается: `True`, `NVIDIA GeForce RTX 4070 Ti SUPER`, `(8, 9)`, `True`.

Скачай модель заранее, чтобы первый запуск не ждал загрузки около 3 ГБ:

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
Сохрани `dataset/analysis/report.json` и `dataset/filtered/filter_report.json`: их цифры пойдут в ВКР.

## 8. Подбор micro_batch_size

Скрипт обучает по 3 шага на самых длинных примерах, поэтому измеренная пиковая память — оценка сверху:

```bash
python scripts/probe_microbatch.py --config configs/train/pythia-1.4b.json --output runs/probe/pythia-1.4b
```

Рекомендованное значение впиши в `configs/train/pythia-1.4b.json` (`micro_batch_size`).
`global_batch_size` не меняй: от него зависит число шагов, и он одинаков для всех архитектур.
Порог `--memory-limit-gb 13.5` оставляет запас под рабочий стол Windows и чужие процессы. Перед запуском посмотри `nvidia-smi`:
если кто-то уже занял несколько гигабайт, подожди.

## 9. Подбор LR

Запускай в `tmux`, чтобы обучение не прервалось при закрытии терминала:

```bash
tmux new -s lr
python scripts/lr_search.py --config configs/train/pythia-1.4b.json --output runs/lr_search/pythia-1.4b
```

Отключиться от сессии, не останавливая её: `Ctrl+B`, затем `D`. Вернуться: `tmux attach -t lr`.
Ожидаемое время — около 1.5 часа (5 значений LR × 20% данных × 1 эпоха).
Если скрипт предупредит, что лучший LR на краю сетки, добавь значения через `--lrs` и запусти ту же команду ещё раз:
готовые варианты будут пропущены.

Лучший LR впиши в конфиг (`lr`) и закоммить конфиг вместе с `runs/lr_search/pythia-1.4b/lr_search.json`.

## 10. Основная серия: 5 сидов

```bash
tmux new -s main
for s in 0 1 2 3 4; do
  python scripts/train_lora.py --config configs/train/pythia-1.4b.json --seed $s --output runs/main/pythia-1.4b/s$s || break
done
```

Если запуск прервался (`wsl --shutdown`, перезагрузка), продолжи его с последнего чекпойнта:

```bash
python scripts/train_lora.py --config configs/train/pythia-1.4b.json --seed 3 --output runs/main/pythia-1.4b/s3 --resume
```

После этого перезапусти цикл с оставшимися сидами.
Оценка: около 1.5–2 часов на эпоху, то есть 3–4 часа на сид и 15–20 часов на серию. Уточнишь по первому сиду.

Во время обучения:

- **`tail -f runs/main/pythia-1.4b/s0/metrics.jsonl`** — loss, токены в секунду, пиковая память.
- **`gpu_monitor.csv`** в папке запуска. Если `tokens_per_second` заметно падает, а загрузка GPU при этом 100%, на карте чужая нагрузка. Отметь время, позже эти участки можно будет исключить из замеров скорости.
- **Один запуск за раз.** Два обучения на одной карте испортят и память, и замеры GPU-часов.

## 11. Перенос результатов

Чекпойнты-адаптеры небольшие (десятки мегабайт), поэтому архивировать можно всё:

```bash
tar czf ~/results_$(date +%Y%m%d).tar.gz runs dataset/analysis dataset/filtered/filter_report.json env_report_wsl.txt requirements-train.lock.txt
cp ~/results_*.tar.gz /mnt/e/
```

Если места мало, исключи `--exclude='trainer_state.pt'`: это состояние оптимизатора, нужное только для `--resume`.

## Что делать при проблемах

| Симптом | Действие |
|---|---|
| `CUDA out of memory` | Уменьши `micro_batch_size` вдвое; проверь `nvidia-smi` на чужие процессы |
| `torch.cuda.is_available()` = False | `nvidia-smi` в WSL работает? Если нет — проверь, что дистрибутив WSL2 (`wsl -l -v` в Windows) |
| Ошибки SSL при загрузке | Корпоративный CA, раздел 2 |
| `000` в проверке сети | Не заданы переменные прокси в текущей оболочке: `source ~/.bashrc` |
| `diverged.json` в папке запуска | LR слишком велик; для основной серии такого быть не должно |
| Запуск пишет `is not empty` | Папка уже использовалась: продолжи с `--resume` или укажи новую `--output` |
