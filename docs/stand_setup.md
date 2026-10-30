# Инструкция для стенда

Порядок работ:

1. Проверка Docker и GPU.
2. Прокси, Harbor и Nexus.
3. Репозитории и данные.
4. Сборка образа.
5. Проверка GPU в контейнере.
6. Фильтрация датасета с iverilog.
7. Подбор `micro_batch_size`.
8. Подбор LR.
9. Основная серия.
10. Результаты.

## 1. Проверка Docker и GPU

Убедитесь, что Docker Desktop запущен и контейнеры видят видеокарту:

```powershell
cd E:\llm_rtl
powershell -ExecutionPolicy Bypass -File scripts\env_check.ps1 -DockerGpuTest > env_report.txt
```

В разделе `Docker GPU test` отчёта должна быть таблица `nvidia-smi` с RTX 4070 Ti SUPER.

Запуски умеют продолжаться с последнего чекпойнта (раздел 9).

## 2. Прокси, Harbor и Nexus

Нужно настроить прокси

## 3. Репозитории и данные

```powershell
cd E:\
git clone https://github.com/rvprikhodko/llm_rtl.git
cd llm_rtl
git clone https://hub.mos.ru/nltoverilog/vai-experiments-rtllm.git
mkdir dataset
curl.exe -L -o dataset\Resyn27k.json https://raw.githubusercontent.com/hkust-zhiyao/RTL-Coder/main/dataset/Resyn27k.json
(Get-FileHash dataset\Resyn27k.json -Algorithm SHA256).Hash
```

Ожидаемый SHA-256 датасета: `6F50E411A17E667520A6E6B6FEFEA59684B6CD469639483BF3440D557ACE82AE`.

## 4. Сборка образа

```powershell
docker compose build
docker compose run --rm llm cat /opt/requirements-train.lock.txt > requirements-train.lock.txt
```

Первая сборка занимает 10–20 минут: базовый образ CUDA и PyTorch весят несколько гигабайт.
Файл `requirements-train.lock.txt` содержит точные версии пакетов. Сохраните его вместе с результатами:
он нужен для раздела о воспроизводимости.

## 5. Проверка GPU в контейнере

```powershell
docker compose run --rm llm python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.cuda.is_bf16_supported())"
```

Ожидается: `True`, `NVIDIA GeForce RTX 4070 Ti SUPER`, `(8, 9)`, `True`.

Можете скачать модель заранее (около 3 ГБ), чтобы первый запуск не ждал загрузки. Модели хранятся в томе Docker `hf-cache`
и не скачиваются повторно:

```powershell
docker compose run --rm llm python -c "from transformers import AutoModelForCausalLM, AutoTokenizer as T; n='EleutherAI/pythia-1.4b'; T.from_pretrained(n, revision='step143000'); AutoModelForCausalLM.from_pretrained(n, revision='step143000')"
```

## 6. Фильтрация датасета с iverilog

На стенде датасет необходимо
пересобрать с iverilog, и обучать модели следует только на этом результате:

```powershell
docker compose run --rm llm python scripts/analyze_dataset.py dataset/Resyn27k.json --out dataset/analysis --rtllm vai-experiments-rtllm/datasets/trusted/rtllm-882030d9e61b7787/cases --iverilog --workers 32
docker compose run --rm llm python scripts/filter_dataset.py dataset/Resyn27k.json --out dataset/filtered --rtllm vai-experiments-rtllm/datasets/trusted/rtllm-882030d9e61b7787/cases --iverilog --workers 32
```

Без iverilog осталось 23 366 записей. С проверкой их станет меньше на число некомпилирующихся эталонов.
Сохраните `dataset/analysis/report.json` и `dataset/filtered/filter_report.json`: эти цифры войдут в сравнительный анализ.

## 7. Подбор micro_batch_size

Скрипт обучает по 3 шага на самых длинных примерах, поэтому измеренная пиковая память — оценка сверху:

```powershell
docker compose run --rm llm python scripts/probe_microbatch.py --config configs/train/pythia-1.4b.json --output runs/probe/pythia-1.4b
```

Впишите рекомендованное значение в `configs/train/pythia-1.4b.json` (`micro_batch_size`).
Значение `global_batch_size` менять не нужно: от него зависит число шагов, и оно одинаково для всех архитектур.
Перед запуском проверьте `nvidia-smi`: если видеокарта уже занята другими задачами, лучше подождать.

## 8. Подбор LR

```powershell
docker compose run -d --name lr-pythia llm python scripts/lr_search.py --config configs/train/pythia-1.4b.json --output runs/lr_search/pythia-1.4b
docker logs -f lr-pythia
```

Поиск занимает около 1.5 часа: 5 значений LR × 20% данных × 1 эпоха.

Впишите лучший LR в конфиг (`lr`) и сохраните конфиг вместе с `runs/lr_search/pythia-1.4b/lr_search.json`.

## 9. Основная серия: 5 сидов

Каждая модель обучается 3 эпохи. Чекпойнты сохраняются через каждую четверть эпохи (`save_every_epoch_fraction: 0.25`,
12 срезов за запуск), а также каждые 30 минут чистого GPU-времени (`save_every_gpu_seconds: 1800`).

- **Срезы по эпохам** задаются шагами оптимизатора, поэтому у всех архитектур они соответствуют одним и тем же примерам.
  По ним сравниваются архитектуры при равном объёме данных и строятся кривые обучения.
- **Срезы по GPU-времени** используются для сравнения при равных GPU-часах.

Для каждого чекпойнта в `checkpoints/index.jsonl` записываются эпоха, шаг, токены, GPU-секунды и loss на валидации.

```powershell
docker compose run -d --name main-pythia llm bash -c "for s in 0 1 2 3 4; do python scripts/train_lora.py --config configs/train/pythia-1.4b.json --seed `$s --output runs/main/pythia-1.4b/s`$s || exit 1; done"
docker logs -f main-pythia
```

Если запуск прервался, продолжите его с последнего чекпойнта (пример для сида 3):

```powershell
docker rm main-pythia
docker compose run -d --name resume-s3 llm python scripts/train_lora.py --config configs/train/pythia-1.4b.json --seed 3 --output runs/main/pythia-1.4b/s3 --resume
```

После этого запустите оставшиеся сиды той же командой, что и серию, перечислив в цикле только их.
Оценка времени: около 1.5–2 часов на эпоху, то есть 5–6 часов на сид и около суток на серию. Уточните её по первому сиду.

## 10. Результаты

Все результаты сразу сохраняются в `E:\llm_rtl\runs`
