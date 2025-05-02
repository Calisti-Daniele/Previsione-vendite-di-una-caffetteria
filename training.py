import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor

# Carico il dataset con le feature ingegnerizzate
df = pd.read_csv("datasets/coffee_shop_sales_features.csv")

# Separazione feature/target
X = df.drop(columns=["Sales", "Date"])
y = df["Sales"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Salvo le colonne per il deploy
os.makedirs("models", exist_ok=True)
joblib.dump(X_train.columns.tolist(), "models/feature_columns.pkl")

# Inizializzo lo scaler e salvo
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "models/scaler.pkl")

# Definizione dei modelli e dei parametri
models = {
    "LinearRegression": {
        "model": LinearRegression(),
        "params": {}
    },
    "RandomForest": {
        "model": RandomForestRegressor(random_state=42, verbose=True),
        "params": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [5, 10, None]
        }
    },
    "XGBoost": {
        "model": XGBRegressor(random_state=42, verbosity=1),
        "params": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 6],
            "model__learning_rate": [0.05, 0.1]
        }
    },
    "KNN": {
        "model": KNeighborsRegressor(),
        "params": {
            "model__n_neighbors": [3, 5, 7]
        }
    }
}

# Grid search su tutti i modelli
results = []
best_model = None
best_score = -np.inf

for name, cfg in models.items():
    pipe = Pipeline([
        ("model", cfg["model"])
    ])
    grid = GridSearchCV(pipe, cfg["params"], cv=5, scoring="r2", n_jobs=-1)
    grid.fit(X_train_scaled, y_train)

    preds = grid.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    results.append({
        "Model": name,
        "Best Params": grid.best_params_,
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "R2": round(r2, 4)
    })

    if r2 > best_score:
        best_score = r2
        best_model = grid.best_estimator_
        joblib.dump(best_model, "models/best_sales_model.pkl")

# Salvo i risultati
results_df = pd.DataFrame(results)
results_df.to_csv("models/gridsearch_results.csv", index=False)

