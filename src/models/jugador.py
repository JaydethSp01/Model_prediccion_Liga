# src/models/jugador.py
from pydantic import BaseModel, Field

class Jugador(BaseModel):
    id: str = Field(..., alias="_id")
    name: str
    team: str  # ID del equipo, almacenado como string
    position: str
    matches: int
    minutesPlayed: int
    goals: int
    expectedGoals: float
    assists: int
    expectedAssists: float
    shots: float
    shotsOnTarget: float
    passAccuracy: float
    dribbles: float
    fouls: float
    yellowCards: int
    redCards: int
    rating: float

    class Config:
        allow_population_by_field_name = True
