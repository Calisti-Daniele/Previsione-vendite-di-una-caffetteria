# ☕ Smart Coffee Forecast

Un progetto completo di data science per prevedere le vendite giornaliere di una caffetteria sulla base di variabili contestuali come:

- Promozioni attive
- Giorni della settimana
- Condizioni meteo (Sole, Pioggia, Neve, ecc.)
- Stagione
- Temperatura

📈 L’obiettivo è costruire un modello predittivo accurato e spiegabile che supporti le decisioni di business (es. lanci promozionali o scorte giornaliere).

## 📂 Dataset

Il dataset copre un intero anno (2023) e contiene vendite simulate ma realistiche. È formato da 365 righe, una per ogni giorno.

### Colonne incluse:

- `Date`: la data del giorno
- `DayOfWeek`: giorno della settimana
- `Promo`: 1 se era attiva una promozione, 0 altrimenti
- `Weather`: condizione meteo (Sunny, Rainy, Cloudy, Snowy)
- `Temperature`: temperatura media del giorno (°C)
- `Season`: stagione del giorno
- `Sales`: numero di vendite

## 🧠 Obiettivi

- Analisi esplorativa e visualizzazione dei dati
- Addestramento di modelli di regressione
- Previsione delle vendite future
- Interfaccia con Streamlit per l’uso interattivo

---

# ☕ Smart Coffee Forecast (EN)

This is a complete AI project to forecast daily coffee shop sales based on context variables such as:

- Active promotions
- Day of the week
- Weather conditions (Sunny, Rainy, Snowy…)
- Season
- Temperature

📈 The goal is to build an accurate and interpretable sales forecasting model to support real-world decisions (e.g., inventory planning or promo strategy).

## 📂 Dataset

The dataset covers one full year (2023) and contains 365 entries, one per day.

### Included columns:

- `Date`: date of the record
- `DayOfWeek`: weekday
- `Promo`: 1 if a promotion was active, 0 otherwise
- `Weather`: daily weather condition
- `Temperature`: average temperature in °C
- `Season`: season of the year
- `Sales`: number of sales for the day

## 🧠 Goals

- Exploratory data analysis and visualization
- Training regression models (e.g., XGBoost)
- Forecasting future sales
- Streamlit-based interactive interface
