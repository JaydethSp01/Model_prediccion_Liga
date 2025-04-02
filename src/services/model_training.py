import os
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, PoissonRegressor
from xgboost import XGBRegressor
from src.repositories.jugador_repository import JugadorRepository
import math

def prepare_data(df: pd.DataFrame):
    """Prepara los datos del jugador para entrenamiento."""
    feature_columns = [
        "matches", "minutesPlayed", "goals", "expectedGoals", "assists", "expectedAssists", 
        "shots", "shotsOnTarget", "passAccuracy", "dribbles", "fouls", 
        "yellowCards", "redCards", "rating"
    ]

    X = df[feature_columns].values  # Solo características del jugador
    y = np.maximum(df["goals"].values, 1e-5)  # ✅ Evita valores 0 o negativos en y

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler, feature_columns

def train_models(df: pd.DataFrame):
    """Entrena modelos de regresión y los guarda."""
    X_train, X_test, y_train, y_test, scaler, feature_columns = prepare_data(df)

    models = {
        "linear": LinearRegression().fit(X_train, y_train),
        "xgboost": XGBRegressor(eval_metric="rmse").fit(X_train, y_train),
        "poisson": PoissonRegressor().fit(X_train, y_train)  # ✅ Ahora y_train no tiene ceros
    }

    # Guardar modelos
    models_path = "models"
    os.makedirs(models_path, exist_ok=True)
    
    joblib.dump(models["linear"], os.path.join(models_path, "linear_model.pkl"))
    joblib.dump(models["xgboost"], os.path.join(models_path, "xgboost_model.pkl"))
    joblib.dump(models["poisson"], os.path.join(models_path, "poisson_model.pkl"))
    joblib.dump(scaler, os.path.join(models_path, "scaler.pkl"))
    joblib.dump(feature_columns, os.path.join(models_path, "feature_columns.pkl"))

    print("\n✅ Modelos entrenados y guardados en 'models/'.")
    print("📊 Resumen:")
    for model_name, model in models.items():
        print(f" - {model_name} ✅")

if __name__ == "__main__":
    print("\n🚀 Iniciando entrenamiento...")

    # Obtener jugadores de la BD
    jugador_repo = JugadorRepository()
    jugadores = jugador_repo.get_all()

    if not jugadores:
        print("❌ No se encontraron jugadores en la BD.")
    else:
        df = pd.DataFrame([jug.model_dump() for jug in jugadores])  # ✅ Usar model_dump() en Pydantic V2
        train_models(df)
