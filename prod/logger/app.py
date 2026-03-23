"""
logger_service.py — FastAPI-сервис для логирования запросов и предсказаний
Принимает JSON-записи, сохраняет в файл, предоставляет просмотр логов.
"""

import json
import logging
import os
from datetime import datetime, timezone

from fastapi import FastAPI, Query
from pydantic import BaseModel

LOG_FILE = os.environ.get("LOG_FILE", "/app/logs/requests.log")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fraud Detection Logger",
    description="Сервис логирования запросов к модели обнаружения мошенничества",
    version="1.0.0",
)


class LogEntry(BaseModel):
    timestamp: str | None = None
    row_index: int | None = None
    fraud_probability: float | None = None
    prediction: int | None = None
    verdict: str | None = None
    features_summary: dict | None = None


def _ensure_log_dir():
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)


# ── Эндпоинты ─────────────────────────────────────────────────────────────────

@app.get("/health", summary="Проверка работоспособности")
def health():
    _ensure_log_dir()
    total = 0
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            total = sum(1 for _ in f)
    return {"status": "ok", "log_file": LOG_FILE, "total_records": total}


@app.post("/log", summary="Записать событие в лог")
async def log_request(entry: LogEntry):
    _ensure_log_dir()

    # Если timestamp не передан — ставим текущее время
    record = entry.model_dump()
    if not record.get("timestamp"):
        record["timestamp"] = datetime.now(timezone.utc).isoformat()

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Записан лог: row={record.get('row_index')}, verdict={record.get('verdict')}")
    return {"status": "logged", "timestamp": record["timestamp"]}


@app.get("/logs", summary="Просмотр последних N записей")
def view_logs(last_n: int = Query(default=20, ge=1, le=1000)):
    if not os.path.exists(LOG_FILE):
        return {"total": 0, "showing": 0, "logs": []}

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total = len(lines)
    recent_lines = lines[-last_n:]
    entries = []
    for line in recent_lines:
        try:
            entries.append(json.loads(line.strip()))
        except json.JSONDecodeError:
            continue

    return {"total": total, "showing": len(entries), "logs": entries}


@app.get("/stats", summary="Статистика по всем запросам")
def stats():
    if not os.path.exists(LOG_FILE):
        return {"total": 0, "fraud_count": 0, "legit_count": 0, "fraud_rate": 0.0}

    total = 0
    fraud_count = 0
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                total += 1
                if entry.get("prediction") == 1:
                    fraud_count += 1
            except json.JSONDecodeError:
                continue

    legit_count = total - fraud_count
    fraud_rate = round(fraud_count / total, 4) if total > 0 else 0.0

    return {
        "total": total,
        "fraud_count": fraud_count,
        "legit_count": legit_count,
        "fraud_rate": fraud_rate,
    }


@app.delete("/logs/clear", summary="Очистить лог-файл")
def clear_logs():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.truncate(0)
    return {"status": "cleared", "timestamp": datetime.now(timezone.utc).isoformat()}
