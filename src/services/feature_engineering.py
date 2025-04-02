# src/services/feature_engineering.py
import pandas as pd
from typing import List
from src.models.jugador import Jugador

def create_features(jugadores: List[Jugador]) -> pd.DataFrame:
    # Convertir la lista de jugadores a DataFrame
    data = [jugador.dict() for jugador in jugadores]
    df = pd.DataFrame(data)
    
    # Crear algunas features relevantes:
    # 1. Remates al arco por partido: shotsOnTarget / matches
    df["shotsOnTargetPerGame"] = df["shotsOnTarget"] / df["matches"].replace(0, 1)
    
    # 2. Faltas por partido: fouls / matches
    df["foulsPerGame"] = df["fouls"] / df["matches"].replace(0, 1)
    
    # 3. Promedio de goles por partido: goals / matches
    df["goalsPerGame"] = df["goals"] / df["matches"].replace(0, 1)
    
    # 4. Relación expectedGoals vs. goles (xG_ratio)
    df["xgRatio"] = df["expectedGoals"] / df["goals"].replace(0, 1)
    
    # Seleccionar columnas para entrenamiento
    features = [
        "matches", "minutesPlayed", "goals", "expectedGoals", "assists",
        "expectedAssists", "shots", "shotsOnTarget", "passAccuracy", "dribbles",
        "fouls", "yellowCards", "redCards", "rating", 
        "shotsOnTargetPerGame", "foulsPerGame", "goalsPerGame", "xgRatio"
    ]
    df = df[features].dropna()
    return df
