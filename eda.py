# Importo le librerie essenziali
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Imposto un tema visivo per i grafici
sns.set(style="whitegrid")

# Carico il dataset
df = pd.read_csv("datasets/coffe_shop_sales_dataset.csv")

# Visualizzo le prime righe del dataset per capire la struttura
print("✅ Prime righe del dataset:")
print(df.head())

# Verifico il numero di righe e colonne
print(f"\n📏 Dimensioni del dataset: {df.shape[0]} righe, {df.shape[1]} colonne")

# Controllo i tipi di dato di ogni colonna
print("\n🔍 Tipi di dato:")
print(df.dtypes)

# Controllo se ci sono valori nulli
print("\n❓ Valori nulli presenti:")
print(df.isnull().sum())

# Statistiche descrittive di base
print("\n📊 Statistiche descrittive:")
print(df.describe())

# Converto la colonna Date in datetime per analisi temporali
df['Date'] = pd.to_datetime(df['Date'])

# Creo un grafico delle vendite nel tempo
plt.figure(figsize=(14, 5))
sns.lineplot(x='Date', y='Sales', data=df)
plt.title("📈 Vendite giornaliere nel tempo")
plt.xlabel("Data")
plt.ylabel("Vendite")
plt.tight_layout()
plt.show()

# Boxplot vendite per giorno della settimana
plt.figure(figsize=(10, 5))
order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
sns.boxplot(x='DayOfWeek', y='Sales', data=df, order=order)
plt.title("📦 Distribuzione vendite per giorno della settimana")
plt.xlabel("Giorno")
plt.ylabel("Vendite")
plt.tight_layout()
plt.show()

# Boxplot vendite per meteo
plt.figure(figsize=(10, 5))
sns.boxplot(x='Weather', y='Sales', data=df)
plt.title("🌦️ Vendite in base al meteo")
plt.xlabel("Meteo")
plt.ylabel("Vendite")
plt.tight_layout()
plt.show()

# Boxplot vendite per stagione
plt.figure(figsize=(10, 5))
sns.boxplot(x='Season', y='Sales', data=df, order=['Winter', 'Spring', 'Summer', 'Autumn'])
plt.title("🍂 Vendite per stagione")
plt.xlabel("Stagione")
plt.ylabel("Vendite")
plt.tight_layout()
plt.show()
