import streamlit as st
import pandas as pd
import joblib
import os
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

# ------------------------
# CONFIG
# ------------------------
st.set_page_config(page_title="📈 Previsione Vendite Caffetteria", layout="centered")

st.title("☕ Coffee Sales Predictor")
st.markdown("""
Benvenuto nella dashboard di previsione vendite per una caffetteria! 📊

Carica un file `.csv` con i dati dei giorni futuri (giorno della settimana, meteo, stagione, ecc.) e scopri quante vendite aspettarti.
""")

# ------------------------
# LOAD MODELS
# ------------------------
model = joblib.load("models/best_sales_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# ------------------------
# UPLOAD
# ------------------------
uploaded_file = st.file_uploader("📤 Carica il file CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Data'] = pd.to_datetime(df['Date'])

    # ------------------------
    # FEATURE ENGINEERING
    # ------------------------
    df['Mese'] = df['Data'].dt.month
    df['Weekend'] = df['DayOfWeek'].isin(['Saturday', 'Sunday'])
    df['GiornoDellAnno'] = df['Data'].dt.dayofyear
    holidays = calendar().holidays(start=df['Data'].min(), end=df['Data'].max())
    df['Festivo'] = df['Data'].isin(holidays)

    # One-hot encoding
    categorical_cols = ['DayOfWeek', 'Weather', 'Season']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Allineamento colonne
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    X = df[feature_columns]

    # ------------------------
    # PREDICTION
    # ------------------------
    X_scaled = scaler.transform(X)
    df['Vendite Previste'] = model.predict(X_scaled).round().astype(int)

    # ------------------------
    # OUTPUT
    # ------------------------
    st.success("✅ Previsioni completate!")

    st.markdown("""
    Le vendite previste sono state calcolate considerando i fattori stagionali ☀️❄️, il meteo 🌦️, i giorni festivi 🎉 e l'effetto delle promozioni 💸.
    Usa questi dati per pianificare meglio le scorte, il personale e le promozioni nei prossimi giorni!
    """)

    st.dataframe(df[['Data', 'Vendite Previste']])

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Scarica le previsioni in CSV",
        data=csv,
        file_name="vendite_previste.csv",
        mime="text/csv"
    )
else:
    st.info("⬆️ Carica un file per iniziare.")
    with open("datasets/for_prediction.csv", "rb") as f:
        st.download_button(
            label="📁 Scarica un esempio di input",
            data=f,
            file_name="for_prediction.csv",
            mime="text/csv"
        )
