import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar  # alternativa per festivi, se non si usano librerie esterne
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("datasets/coffe_shop_sales_dataset.csv")

df['Date'] = pd.to_datetime(df['Date'])

# Aggiungo il mese
df['Month'] = df['Date'].dt.month

# Aggiungo flag per weekend
df['IsWeekend'] = df['DayOfWeek'].isin(['Saturday', 'Sunday'])

# Aggiungo il giorno dell’anno (utile per trend temporali)
df['DayOfYear'] = df['Date'].dt.dayofyear

# Aggiungo un flag per festività (USA come proxy se non hai una libreria italiana)
holidays = calendar().holidays(start=df['Date'].min(), end=df['Date'].max())
df['IsHoliday'] = df['Date'].isin(holidays)

# Variabili categoriche da codificare
categorical_cols = ['DayOfWeek', 'Weather', 'Season']

# Applico one-hot encoding e unisco al dataframe
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Visualizzo le prime righe del dataframe aggiornato
print("✅ Dataset dopo il feature engineering:")
print(df_encoded.head())

# Salvo il nuovo dataset se necessario
df_encoded.to_csv("datasets/coffee_shop_sales_features.csv", index=False)
