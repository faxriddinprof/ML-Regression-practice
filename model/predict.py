"""
Ertangi kun uchun guruch (kg) va tushum (so'm) bashorati.

Ishlatish:
    python predict.py example_tomorrow.csv
    python predict.py example_tomorrow.csv -o natija.csv
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).with_name("sales_model.joblib")

# Foydalanuvchi kiritadigan ustunlar (hammasi kun boshida ma'lum bo'ladi)
INPUT_COLUMNS = ["date", "temp_c", "humidity", "wind", "rain", "snow", "is_holiday", "ramadan",
                 "eid", "salary_week", "promotion", "local_event", "football_match"]
FLAG_COLUMNS = ["rain", "snow", "is_holiday", "ramadan", "eid", "salary_week",
                "promotion", "local_event", "football_match"]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Kiruvchi jadvaldan model kutadigan feature'larni yasaydi. Hafta kuni va oy sanadan olinadi."""
    missing = [c for c in INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Quyidagi ustunlar yetishmayapti: {missing}")

    X = df[INPUT_COLUMNS].copy()
    X["date"] = pd.to_datetime(X["date"])
    X["weekday"] = X["date"].dt.day_of_week          # 0 = dushanba, 6 = yakshanba
    X["month"] = X["date"].dt.month
    X["is_weekend"] = (X["weekday"] >= 5).astype(int)

    for c in FLAG_COLUMNS:
        X[c] = X[c].map(lambda v: str(v).strip().lower() in ("1", "true", "ha", "yes")).astype(int)
    for c in ["temp_c", "humidity", "wind"]:
        X[c] = pd.to_numeric(X[c], errors="raise")

    return X.drop(columns="date")


def predict(df: pd.DataFrame, bundle: dict | None = None) -> pd.DataFrame:
    """Har bir qator (kun) uchun bashorat va 95% ishonch oralig'ini qaytaradi."""
    bundle = bundle or joblib.load(MODEL_PATH)
    X = prepare_features(df)
    out = pd.DataFrame({"date": pd.to_datetime(df["date"]).dt.date})

    for target, info in bundle["targets"].items():
        pred = np.exp(info["model"].predict(X))
        lo, hi = info["interval"]
        digits = 1 if target == "rice_kg" else -3        # guruch 0.1 kg, tushum 1000 so'mgacha yaxlitlanadi
        out[target] = pred.round(digits)
        out[f"{target}_min"] = (pred * lo).round(digits)
        out[f"{target}_max"] = (pred * hi).round(digits)

    return out


def main():
    parser = argparse.ArgumentParser(description="Guruch va tushum bashorati")
    parser.add_argument("input", help="kunlar ro'yxati yozilgan CSV fayl")
    parser.add_argument("-o", "--output", help="natijani CSV faylga yozish (ixtiyoriy)")
    args = parser.parse_args()

    bundle = joblib.load(MODEL_PATH)
    result = predict(pd.read_csv(args.input), bundle)

    pd.set_option("display.width", 200)
    print(result.to_string(index=False, formatters={
        c: "{:,.0f}".format for c in result.columns if c.startswith("revenue")}))
    print(f"\nModel: {bundle['trained_on']}. Oraliqlar 95% ishonch bilan.")

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"Saqlandi: {args.output}")


if __name__ == "__main__":
    main()
