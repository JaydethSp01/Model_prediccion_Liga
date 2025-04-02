from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.services.prediction_service import PredictionService

app = FastAPI(title="Predicción de Apuestas Deportivas")

# Instancia del servicio
prediction_service = PredictionService()

class JugadorID(BaseModel):
    jugador_id: str



@app.get("/analyze-teams")
def get_team_analysis():
    """ Retorna el análisis de equipos con métricas y consejos de apuestas """
    result = prediction_service.analyze_teams()
    if not result:
        raise HTTPException(status_code=404, detail="No se encontraron datos de equipos.")
    return result


@app.get("/analyze/upcoming-matches")
def analyze_upcoming_matches():
    """ Analiza partidos próximos de una liga específica """
    result = prediction_service.analizar_partidos_proximos(
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result
