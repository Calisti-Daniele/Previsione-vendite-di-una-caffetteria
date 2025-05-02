import pandas as pd
import joblib
import os
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

# Carico modello, scaler e feature columns
model = joblib.load("models/best_sales_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# Carico il dataset da prevedere
df = pd.read_csv("datasets/for_prediction.csv")
df['Date'] = pd.to_datetime(df['Date'])

# 🔧 Feature Engineering
df['Month'] = df['Date'].dt.month
df['IsWeekend'] = df['DayOfWeek'].isin(['Saturday', 'Sunday'])
df['DayOfYear'] = df['Date'].dt.dayofyear
holidays = calendar().holidays(start=df['Date'].min(), end=df['Date'].max())
df['IsHoliday'] = df['Date'].isin(holidays)

# One-hot encoding sulle categoriche
categorical_cols = ['DayOfWeek', 'Weather', 'Season']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Assicuro che tutte le colonne del training siano presenti
for col in feature_columns:
    if col not in df.columns:
        df[col] = 0
X = df[feature_columns]

# Normalizzazione
X_scaled = scaler.transform(X)

# Previsione
df["Predicted_Sales"] = model.predict(X_scaled).round().astype(int)

# Salvataggio
os.makedirs("predictions", exist_ok=True)
df.to_csv("predictions/sales_predictions.csv", index=False)

print("✅ Previsioni generate correttamente in: predictions/sales_predictions.csv")
