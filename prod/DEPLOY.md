# Руководство по развёртыванию — Fraud Detection Demo

## Архитектура

```
┌─────────────────────────────────────────────────────┐
│                   Docker Network                     │
│                                                      │
│  ┌──────────────┐     HTTP      ┌────────────────┐  │
│  │ model_server │──────/log────▶│    logger      │  │
│  │  :8000       │               │    :8001       │  │
│  └──────┬───────┘               └────────────────┘  │
│         │ /predict                                   │
└─────────┼───────────────────────────────────────────┘
          │
    ┌─────┴──────┐
    │   client   │  (запускается вручную, интерактивный)
    └────────────┘
```

| Сервис         | Порт | Назначение                                      |
|----------------|------|-------------------------------------------------|
| `model_server` | 8000 | Загружает XGBoost, принимает запросы `/predict` |
| `logger`       | 8001 | Принимает и хранит логи запросов                |
| `client`       | —    | Интерактивный CLI для тестирования              |

---

## Требования

- [Docker](https://docs.docker.com/get-docker/) 24+
- [Docker Compose](https://docs.docker.com/compose/) v2.x
- Файлы моделей в корне проекта:
  - `best_xgb.pkl` — обученная модель XGBoost
  - `Prod_testing.pkl` — тестовые данные для демонстрации

---

## Быстрый старт

### 1. Перейдите в папку prod

```bash
cd prod
```

### 2. Запустите сервисы (model_server + logger)

```bash
docker compose up --build -d
```

Это запустит:
- Сборку образов (первый раз ~3–5 мин)
- Запуск `logger` (мгновенно)
- Запуск `model_server` (загрузка модели — **до 60 сек**)

### 3. Проверьте, что всё запущено

```bash
docker compose ps
```

Оба сервиса должны иметь статус `healthy`.

Также можно проверить через браузер:
- http://localhost:8000/health  — статус model_server
- http://localhost:8001/health  — статус logger

### 4. Запустите интерактивного клиента

```bash
docker compose --profile client run --rm client
```

Вы увидите интерфейс:

```
=================================================================
    СИСТЕМА ОБНАРУЖЕНИЯ МОШЕННИЧЕСТВА — ДЕМОНСТРАЦИОННЫЙ РЕЖИМ
=================================================================

⏳ Загрузка тестовых данных...
✅ Загружено 1 183 596 транзакций, 54 признака

✅ Сервер модели доступен: http://model_server:8000
-----------------------------------------------------------------
  Доступные строки: 0 — 1 183 595  (всего 1 183 596)

  Введите номер строки (или 'q' для выхода):
```

Введите номер строки → просмотрите данные → подтвердите → получите результат.

---

## API документация

После запуска swagger-документация доступна по адресам:
- http://localhost:8000/docs — model_server
- http://localhost:8001/docs — logger

### Эндпоинты model_server

| Метод | Путь      | Описание                        |
|-------|-----------|---------------------------------|
| GET   | /health   | Проверка работоспособности      |
| POST  | /predict  | Предсказание мошенничества      |

Пример запроса к `/predict`:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "Amount": 120.5,
      "Use_Chip": 0,
      "Is_Online": 1,
      "Merchant_State": "NY",
      "MCC": 5499,
      ...
    },
    "row_index": 42
  }'
```

Пример ответа:

```json
{
  "row_index": 42,
  "fraud_probability": 0.0312,
  "prediction": 0,
  "verdict": "ЛЕГИТИМНАЯ ✅",
  "threshold_used": 0.5
}
```

### Эндпоинты logger

| Метод  | Путь          | Описание                                  |
|--------|---------------|-------------------------------------------|
| GET    | /health       | Проверка и число записей                  |
| POST   | /log          | Записать событие                          |
| GET    | /logs         | Последние N записей (`?last_n=20`)        |
| GET    | /stats        | Статистика (всего, фрод, легит, доля)     |
| DELETE | /logs/clear   | Очистить лог-файл                         |

Просмотр последних 5 запросов:

```bash
curl http://localhost:8001/logs?last_n=5
```

Статистика по сессии:

```bash
curl http://localhost:8001/stats
```

---

## Управление сервисами

```bash
# Просмотр логов сервисов
docker compose logs -f model_server
docker compose logs -f logger

# Перезапуск конкретного сервиса
docker compose restart model_server

# Остановка всех сервисов
docker compose down

# Полная очистка (включая образы)
docker compose down --rmi all --volumes
```

---

## Настройка через переменные окружения

Переменные можно изменить в `docker-compose.yml` или передать через командную строку:

| Переменная         | Сервис       | По умолчанию                  | Описание                  |
|--------------------|--------------|-------------------------------|---------------------------|
| `MODEL_PATH`       | model_server | `/app/data/best_xgb.pkl`      | Путь к pkl-файлу модели   |
| `LOGGER_URL`       | model_server | `http://logger:8001/log`      | Адрес сервиса логирования |
| `FRAUD_THRESHOLD`  | model_server | `0.5`                         | Порог классификации       |
| `LOG_FILE`         | logger       | `/app/logs/requests.log`      | Путь к лог-файлу          |
| `DATA_PATH`        | client       | `/app/data/Prod_testing.pkl`  | Путь к тестовым данным    |
| `MODEL_SERVER_URL` | client       | `http://model_server:8000`    | URL сервера модели        |

Пример: изменить порог до 0.3:

```bash
FRAUD_THRESHOLD=0.3 docker compose up -d model_server
```

---

## Структура файлов

```
prod/
├── model_server/
│   ├── app.py              # FastAPI-приложение (сервер модели)
│   ├── Dockerfile
│   └── requirements.txt
├── logger/
│   ├── app.py              # FastAPI-приложение (сервис логирования)
│   ├── Dockerfile
│   └── requirements.txt
├── client/
│   ├── client.py           # Интерактивный CLI-клиент
│   ├── Dockerfile
│   └── requirements.txt
├── logs/                   # Директория логов (создаётся автоматически)
│   └── requests.log
└── docker-compose.yml
```

---

## Примечания

- **GPU:** Модель обучалась на CUDA GPU, но при инференсе автоматически переключается на CPU (`model.set_params(device="cpu")`). GPU в контейнере не требуется.
- **Первый запуск:** при `docker compose up --build` Docker скачивает базовый образ Python (~200 МБ) и устанавливает зависимости. Последующие запуски значительно быстрее (используется кэш слоёв).
- **Логи** хранятся в `prod/logs/requests.log` и доступны с хоста без входа в контейнер.
