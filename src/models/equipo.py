# src/models/equipo.py
from pydantic import BaseModel, Field
from typing import List

class Equipo(BaseModel):
    id: str = Field(..., alias="_id")
    name: str
    country: str
    league: str
    players: List[str] = []  # Lista de IDs de jugadores
    goles_por_partido: float = 0.0  # Promedio de goles por partido
    tarjetas_por_partido: float = 0.0  # Promedio de tarjetas por partido

    class Config:
        allow_population_by_field_name = True