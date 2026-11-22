# Guruch va tushum bashorati modeli

Ertangi kun (yoki bir necha kun) uchun quyidagilarni bashorat qiladi:
- **ishlatiladigan guruch miqdori (kg)**;
- **kunlik tushum (so'm)**.

Har bir bashorat **95% ishonch oralig'i** bilan beriladi: haqiqiy qiymat 100 kundan taxminan 95 tasida `_min` va `_max` oralig'iga tushadi.

## O'rnatish

Python 3.10+ kerak.

```bash
pip install -r requirements.txt
```

`sales_model.joblib` scikit-learn **1.9.1** bilan saqlangan. Boshqa versiyada ogohlantirish chiqishi yoki fayl ochilmasligi mumkin, shuning uchun `requirements.txt` dagi versiyani o'rnating.

## Ishlatish

1. Bashorat kerak bo'lgan kunlarni CSV faylga yozing. Namuna: `example_tomorrow.csv`.
2. Ishga tushiring:

```bash
python predict.py example_tomorrow.csv
python predict.py example_tomorrow.csv -o natija.csv    # natijani faylga saqlash
```

Natija namunasi:

```
      date  rice_kg  rice_kg_min  rice_kg_max revenue_uzs revenue_uzs_min revenue_uzs_max
2025-06-19     49.8         39.9         60.1  12,436,000      10,325,000      15,011,000
2025-06-20     98.8         79.1        119.1  25,277,000      20,986,000      30,511,000
```

Python ichidan ham chaqirish mumkin:

```python
import pandas as pd
from predict import predict
predict(pd.read_csv("example_tomorrow.csv"))
```

## Kiruvchi CSV ustunlari

| Ustun | Ma'nosi | Qiymat |
|---|---|---|
| `date` | Bashorat qilinadigan kun | `2025-06-19` |
| `temp_c` | Havo harorati (prognoz), °C | son |
| `humidity` | Namlik, % | son |
| `wind` | Shamol tezligi, m/s | son |
| `rain`, `snow` | Yomg'ir yoki qor kutilyaptimi | `True`/`False` yoki `1`/`0` |
| `is_holiday` | Bayram kunimi | `True`/`False` |
| `ramadan` | Ramazon oyimi | `True`/`False` |
| `eid` | Hayit kunimi | `True`/`False` |
| `salary_week` | Oylik beriladigan haftami | `True`/`False` |
| `promotion` | Aksiya rejalashtirilganmi | `True`/`False` |
| `local_event` | Yaqin atrofda tadbir bormi | `True`/`False` |
| `football_match` | Futbol o'yini bormi | `True`/`False` |

Hafta kuni, oy va dam olish kuni **sanadan avtomatik** aniqlanadi.

## Model aniqligi

Model o'qitishda ko'rmagan oxirgi 180 kunda (2024-12-21 – 2025-06-18) sinovdan o'tkazilgan:

| | Aniqlik (100% − o'rtacha % xato) | R² | O'rtacha xato |
|---|---|---|---|
| Guruch | **91.5%** | 0.887 | ±3.8 kg |
| Tushum | **92.5%** | 0.908 | ±0.87 mln so'm |

Ma'lumotda ~8% kunlik tasodifiy tebranish bor. Uni ob-havo, kalendar va hodisalar bilan tushuntirib bo'lmaydi, shuning uchun 95% dan yuqori aniqlikka faqat yangi ma'lumot bilan chiqish mumkin: bronlar, oldindan buyurtmalar va shu kabilar.

**Amaliy tavsiya:** guruchni bashorat va `rice_kg_max` oralig'ida tayyorlang. Bu yetishmay qolish xavfini kamaytiradi.

## Modelni yangi ma'lumot bilan qayta o'qitish

```bash
python train.py --data yangi_sales.csv
```

Fayl `data/sales.csv` bilan bir xil tuzilishda bo'lishi kerak. Skript avval oxirgi 180 kunda modelni baholaydi, keyin barcha ma'lumotda qayta o'qitib, `sales_model.joblib` ni yangilaydi. Oyiga bir marta qayta o'qitish tavsiya etiladi.

## Texnik tafsilotlar

- **Model:** Ridge regressiya, `log(target)` uchun, har bir target alohida. Ta'sirlar foizda ishlaydi: masalan, aksiya talabni ~+30% oshiradi.
- **Feature'lar:** hafta kuni va oy (one-hot), harorat va uning kvadrati, namlik, shamol hamda 9 ta hodisa belgisi.
- **Solishtirilgan modellar:** Gradient Boosting, Keras/TensorFlow (linear va MLP). Ular Ridge'dan yaxshiroq natija bermadi, tafsilotlar `projects/sales_forecast.ipynb` da.
