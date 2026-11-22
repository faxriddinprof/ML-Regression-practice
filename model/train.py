"""
Modelni o'qitish va sales_model.joblib fayliga saqlash.

    python train.py                       # ../data/sales.csv dan o'qiydi
    python train.py --data yangi_sales.csv

1) Oxirgi 180 kun test sifatida ajratiladi va model halol baholanadi.
2) Test xatolaridan 95% ishonch oralig'i hisoblanadi.
3) Yakuniy model BARCHA ma'lumotda qayta o'qitiladi va saqlanadi.
"""
import argparse
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from predict import MODEL_PATH, prepare_features

TARGETS = ["rice_kg", "revenue_uzs"]
TEST_DAYS = 180


def build_model():
    """log(target) uchun Ridge regressiya. Faqat sklearn qismlari, shuning uchun fayl istalgan joyda ochiladi."""
    prep = ColumnTransformer([
        ("calendar", OneHotEncoder(handle_unknown="ignore"), ["weekday", "month"]),
        ("temp", make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler()), ["temp_c"]),
    ], remainder=StandardScaler())
    return make_pipeline(prep, Ridge(alpha=1.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=Path(__file__).parent.parent / "data" / "sales.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.data, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    X = prepare_features(df)
    train_idx, test_idx = df.index[:-TEST_DAYS], df.index[-TEST_DAYS:]

    bundle = {"targets": {}, "sklearn_version": sklearn.__version__,
              "trained_on": f"{df.date.min().date()} – {df.date.max().date()} ({len(df)} kun), "
                            f"o'qitilgan sana {date.today()}"}

    for target in TARGETS:
        y = df[target]

        # 1) Halol baholash: model test davrini ko'rmaydi
        model = build_model().fit(X.loc[train_idx], np.log(y[train_idx]))
        pred = np.exp(model.predict(X.loc[test_idx]))
        y_test = y[test_idx]
        mape = mean_absolute_percentage_error(y_test, pred)
        metrics = {"MAE": mean_absolute_error(y_test, pred), "MAPE": mape,
                   "aniqlik": 1 - mape, "R2": r2_score(y_test, pred)}

        # 2) 95% oraliq: test'dagi nisbiy xatolarning 2.5% va 97.5% kvantillari
        ratio = y_test / pred
        interval = tuple(np.quantile(ratio, [0.025, 0.975]))

        # 3) Yakuniy model: barcha ma'lumotda
        final = build_model().fit(X, np.log(y))
        bundle["targets"][target] = {"model": final, "interval": interval, "test_metrics": metrics}

        print(f"{target:<12} aniqlik={metrics['aniqlik']:.1%}  R²={metrics['R2']:.3f}  "
              f"MAE={metrics['MAE']:,.1f}  95% oraliq: ×[{interval[0]:.2f} … {interval[1]:.2f}]")

    joblib.dump(bundle, MODEL_PATH)
    print(f"\nSaqlandi: {MODEL_PATH}")


if __name__ == "__main__":
    main()
