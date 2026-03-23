"""
model_server.py — FastAPI-сервис для инференса модели XGBoost
Принимает JSON с признаками, возвращает предсказание и вероятность мошенничества.
"""

import os
import json
import pickle
import logging

import httpx
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── Настройки ─────────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/data/best_xgb.pkl")
CATEGORY_MAPS_PATH = os.environ.get("CATEGORY_MAPS_PATH", "/app/category_maps.json")
LOGGER_URL = os.environ.get("LOGGER_URL", "http://logger:8001/log")
FRAUD_THRESHOLD = float(os.environ.get("FRAUD_THRESHOLD", "0.5"))

# Загружаем маппинги категорий — восстанавливают те же CategoricalDtype, что были при обучении
with open(CATEGORY_MAPS_PATH, encoding="utf-8") as _f:
    _raw_maps = json.load(_f)

CAT_DTYPES: dict[str, pd.CategoricalDtype] = {}
for _col, _info in _raw_maps.items():
    _cats = pd.array(_info["categories"], dtype=_info["dtype"])
    CAT_DTYPES[_col] = pd.CategoricalDtype(categories=_cats, ordered=False)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Глобальный контейнер модели ────────────────────────────────────────────────
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загружаем модель один раз при старте сервиса."""
    logger.info(f"Загрузка модели из {MODEL_PATH} ...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Файл модели не найден: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    # XGBoost автоматически переключается на CPU, если GPU недоступен.
    # set_params не вызываем — это вызывает ошибку при несовпадении версий.
    state["model"] = model
    logger.info("Модель успешно загружена.")
    yield
    state.clear()


app = FastAPI(
    title="Fraud Detection Model Server",
    description="Инференс XGBoost для обнаружения мошенничества",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Схема запроса ──────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    features: dict
    row_index: int | None = None  # для логирования


class PredictResponse(BaseModel):
    row_index: int | None
    fraud_probability: float
    prediction: int
    verdict: str
    threshold_used: float


# ── Вспомогательная функция: dict → DataFrame ──────────────────────────────────
def dict_to_dataframe(features: dict) -> pd.DataFrame:
    df = pd.DataFrame([features])

    # Восстанавливаем категориальные типы с теми же маппингами, что использовались при обучении.
    # Это гарантирует совпадение category codes между обучением и инференсом.
    for col, dtype in CAT_DTYPES.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)

    return df


# ── Эндпоинты ─────────────────────────────────────────────────────────────────
@app.get("/health", summary="Проверка работоспособности сервиса")
def health():
    return {
        "status": "ok",
        "model_loaded": "model" in state,
        "threshold": FRAUD_THRESHOLD,
    }


@app.post("/predict", response_model=PredictResponse, summary="Предсказание мошенничества")
async def predict(request: PredictRequest):
    model = state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    try:
        df = dict_to_dataframe(request.features)
        proba = float(model.predict_proba(df)[0][1])
        prediction = int(proba >= FRAUD_THRESHOLD)
        verdict = "МОШЕННИЧЕСТВО 🚨" if prediction == 1 else "ЛЕГИТИМНАЯ ✅"
    except Exception as e:
        logger.error(f"Ошибка предсказания: {e}")
        raise HTTPException(status_code=422, detail=str(e))

    result = PredictResponse(
        row_index=request.row_index,
        fraud_probability=round(proba, 6),
        prediction=prediction,
        verdict=verdict,
        threshold_used=FRAUD_THRESHOLD,
    )

    # Асинхронно отправляем лог в logger_service
    await _send_log(request, result)

    return result


async def _send_log(request: PredictRequest, result: PredictResponse):
    """Отправляем запись о предсказании в сервис логирования."""
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "row_index": request.row_index,
        "fraud_probability": result.fraud_probability,
        "prediction": result.prediction,
        "verdict": result.verdict,
        "features_summary": {
            k: v
            for k, v in list(request.features.items())[:10]  # первые 10 признаков
        },
    }
    try:
        async with httpx.AsyncClient() as client:
            await client.post(LOGGER_URL, json=log_entry, timeout=2.0)
    except Exception as e:
        logger.warning(f"Не удалось отправить лог: {e}")
