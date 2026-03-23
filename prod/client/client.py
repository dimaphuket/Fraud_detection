"""
client.py — Интерактивный CLI-клиент для демонстрации модели обнаружения мошенничества
Загружает Prod_testing.pkl, позволяет выбрать строку, отправляет запрос на model_server.
"""

import os
import pickle
import sys
import json
import math

import requests
import pandas as pd

# ── Настройки ─────────────────────────────────────────────────────────────────
DATA_PATH = os.environ.get("DATA_PATH", "/app/data/Prod_testing.pkl")
MODEL_SERVER_URL = os.environ.get("MODEL_SERVER_URL", "http://model_server:8000")
PREDICT_URL = f"{MODEL_SERVER_URL}/predict"

# Категориальные колонки (их значения выводим как есть, сериализуем в строку/int)
CAT_COLS = {"Merchant_State", "MCC", "Card_Brand", "Card_Type"}

# Понятные названия признаков для отображения пользователю
DISPLAY_NAMES = {
    "Amount": "Сумма транзакции ($)",
    "Use_Chip": "Использован чип (0=нет, 1=да)",
    "Is_Online": "Онлайн транзакция",
    "Merchant_State": "Штат торговца",
    "MCC": "Код категории торговца (MCC)",
    "Has_Error": "Есть ошибка",
    "Gender": "Пол (0=жен., 1=муж.)",
    "Is_Apartment": "Адрес — квартира",
    "Total_Debt": "Общий долг ($)",
    "FICO": "Кредитный рейтинг FICO",
    "Num_Credit_Cards": "Количество кредитных карт",
    "Card_Brand": "Бренд карты",
    "Card_Type": "Тип карты",
    "Has_Chip": "Карта с чипом",
    "Cards_Issued": "Выпущено карт",
    "Credit_Limit": "Кредитный лимит ($)",
    "txn_hour": "Час транзакции",
    "txn_dayofweek": "День недели (0=пн)",
    "txn_day": "День месяца",
    "is_weekend": "Выходной день",
    "account_age_months": "Возраст аккаунта (мес.)",
    "Amount_to_Income": "Сумма / Доход",
    "user_age": "Возраст клиента",
    "is_night": "Ночная транзакция",
    "is_business_hours": "Рабочие часы",
    "amount_log": "log(Сумма)",
    "amount_round_10": "Сумма кратна $10",
    "time_since_prev_txn_card_min": "Время с пред. транз. по карте (мин.)",
    "time_since_prev_txn_user_min": "Время с пред. транз. по клиенту (мин.)",
    "txn_count_5m_card": "Транзакций за 5 мин (карта)",
    "txn_count_1h_card": "Транзакций за 1 час (карта)",
    "txn_count_5m_user": "Транзакций за 5 мин (клиент)",
    "txn_count_1h_user": "Транзакций за 1 час (клиент)",
    "errors_prev_1h": "Ошибок за прошлый час",
    "merchant_fraud_rate": "Доля мошенничества у торговца",
    "state_fraud_rate": "Доля мошенничества в штате",
    "card_burst_5m": "Всплеск активности карты (5 мин)",
    "is_foreign_offline": "Заграничная офлайн-транзакция",
}


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_header():
    print("=" * 65)
    print("    СИСТЕМА ОБНАРУЖЕНИЯ МОШЕННИЧЕСТВА — ДЕМОНСТРАЦИОННЫЙ РЕЖИМ")
    print("=" * 65)


def print_separator():
    print("-" * 65)


def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        print(f"❌ Файл данных не найден: {DATA_PATH}")
        sys.exit(1)
    print(f"⏳ Загрузка тестовых данных из {DATA_PATH} ...")
    with open(DATA_PATH, "rb") as f:
        df = pickle.load(f)
    print(f"✅ Загружено {len(df):,} транзакций, {len(df.columns)} признаков\n")
    return df


def display_row(row: pd.Series, pos_index: int):
    """Выводит строку данных в читаемом формате."""
    print_separator()
    print(f"  ТРАНЗАКЦИЯ № {pos_index}  (внутренний ID: {row.name})")
    print_separator()
    # Выводим в две колонки: ключевые признаки сначала
    key_cols = [
        "Amount", "Use_Chip", "Is_Online", "Merchant_State", "MCC",
        "Card_Brand", "Card_Type", "Has_Error", "Gender",
        "Total_Debt", "FICO", "Credit_Limit",
        "txn_hour", "is_weekend", "is_night",
        "txn_count_5m_card", "txn_count_1h_card",
        "errors_prev_1h", "merchant_fraud_rate", "state_fraud_rate",
    ]
    print(f"\n  {'Признак':<42} {'Значение':>15}")
    print(f"  {'─' * 42} {'─' * 15}")
    for col in key_cols:
        if col not in row.index:
            continue
        val = row[col]
        name = DISPLAY_NAMES.get(col, col)
        if isinstance(val, float):
            val_str = f"{val:.4f}" if abs(val) < 1000 else f"{val:,.2f}"
        else:
            val_str = str(val)
        print(f"  {name:<42} {val_str:>15}")
    print()


def serialize_features(row: pd.Series) -> dict:
    """Сериализует строку DataFrame в JSON-совместимый словарь."""
    result = {}
    for col, val in row.items():
        if pd.isna(val) if not isinstance(val, str) else False:
            result[col] = None
        elif hasattr(val, "item"):  # numpy scalar
            result[col] = val.item()
        elif isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            result[col] = None
        else:
            result[col] = val
    return result


def check_server_health():
    """Проверяем доступность model_server."""
    try:
        resp = requests.get(f"{MODEL_SERVER_URL}/health", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("model_loaded"):
            print("⚠️  Сервер запущен, но модель ещё не загружена. Подождите...")
            return False
        return True
    except requests.exceptions.ConnectionError:
        print(f"❌ Не удаётся подключиться к серверу модели: {MODEL_SERVER_URL}")
        print("   Убедитесь, что model_server запущен.")
        return False
    except Exception as e:
        print(f"❌ Ошибка при проверке сервера: {e}")
        return False


def send_prediction(features: dict, pos_index: int) -> dict | None:
    """Отправляет запрос к model_server и возвращает результат."""
    payload = {"features": features, "row_index": pos_index}
    try:
        resp = requests.post(PREDICT_URL, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        print("❌ Потеряно соединение с сервером модели.")
        return None
    except requests.exceptions.Timeout:
        print("❌ Сервер не ответил вовремя (timeout 30 сек).")
        return None
    except Exception as e:
        print(f"❌ Ошибка запроса: {e}")
        return None


def display_result(result: dict):
    """Отображает результат предсказания."""
    print_separator()
    prediction = result.get("prediction", -1)
    verdict = result.get("verdict", "N/A")
    prob = result.get("fraud_probability", 0.0)
    threshold = result.get("threshold_used", 0.5)

    if prediction == 1:
        print()
        print("  ██████████████████████████████████████████████████████████")
        print(f"  ██  РЕЗУЛЬТАТ: {verdict:<45}██")
        print("  ██████████████████████████████████████████████████████████")
    else:
        print()
        print("  ┌──────────────────────────────────────────────────────────┐")
        print(f"  │  РЕЗУЛЬТАТ: {verdict:<48}│")
        print("  └──────────────────────────────────────────────────────────┘")

    print()
    bar_len = 40
    filled = int(prob * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"  Вероятность мошенничества: {prob * 100:5.2f}%")
    print(f"  [{bar}]")
    print(f"  Порог классификации: {threshold}")
    print()


def main():
    clear_screen()
    print_header()
    print()

    # Загрузка данных
    df = load_data()
    total_rows = len(df)

    # Проверка сервера
    if not check_server_health():
        sys.exit(1)
    print(f"✅ Сервер модели доступен: {MODEL_SERVER_URL}\n")

    while True:
        print_separator()
        print(f"  Доступные строки: 0 — {total_rows - 1:,}  (всего {total_rows:,})")
        print()
        user_input = input("  Введите номер строки (или 'q' для выхода): ").strip()

        if user_input.lower() in ("q", "quit", "exit", "выход"):
            print("\n  До свидания!\n")
            break

        # Валидация ввода
        try:
            pos_index = int(user_input)
        except ValueError:
            print(f"\n  ❌ Введите целое число от 0 до {total_rows - 1:,}\n")
            continue

        if not (0 <= pos_index < total_rows):
            print(f"\n  ❌ Номер строки должен быть от 0 до {total_rows - 1:,}\n")
            continue

        # Получаем строку по позиционному индексу
        row = df.iloc[pos_index]

        # Показываем данные
        print()
        display_row(row, pos_index)

        # Запрос подтверждения
        confirm = input("  Отправить транзакцию на анализ? [y/N]: ").strip().lower()
        if confirm not in ("y", "yes", "да", "д"):
            print("\n  Отменено.\n")
            continue

        # Сериализация и запрос
        print("\n  ⏳ Отправка запроса к модели...\n")
        features = serialize_features(row)
        result = send_prediction(features, pos_index)

        if result:
            display_result(result)
        else:
            print("  ❌ Не удалось получить предсказание.\n")

        # Продолжить?
        again = input("  Проверить ещё одну транзакцию? [Y/n]: ").strip().lower()
        if again in ("n", "no", "нет", "н"):
            print("\n  До свидания!\n")
            break
        print()


if __name__ == "__main__":
    main()
