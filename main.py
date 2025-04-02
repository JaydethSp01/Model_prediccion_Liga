import uvicorn
import requests
import time
from src.controllers.prediction_controller import app
from fastapi.middleware.cors import CORSMiddleware

def test_api():
    """Función para probar automáticamente la API al iniciar el servidor."""
    time.sleep(2)  # Esperar 2 segundos para que el servidor arranque

    base_url = "http://localhost:8000"

    # Probar que la API está corriendo accediendo a la documentación Swagger
    try:
        response = requests.get(f"{base_url}/docs")
        if response.status_code == 200:
            print("\n✅ API está corriendo correctamente en:", base_url)
        else:
            print("\n⚠️ No se pudo acceder a la API en:", base_url)
    except requests.ConnectionError:
        print("\n❌ No se pudo conectar al servidor. Asegúrate de que está en ejecución.")

    # Probar predicción de jugador (con datos ficticios)
    test_player = {
        "matches": 10,
        "minutesPlayed": 900,
        "goals": 4,
        "expectedGoals": 3.5,
        "assists": 2,
        "expectedAssists": 2.8,
        "shots": 20,
        "shotsOnTarget": 12,
        "passAccuracy": 0.85,
        "dribbles": 15,
        "fouls": 5,
        "yellowCards": 1,
        "redCards": 0,
        "rating": 7.2
    }

    try:
        response = requests.post(f"{base_url}/predict/player", json=test_player)
        if response.status_code == 200:
            print("\n✅ Prueba de predicción exitosa:", response.json())
        else:
            print("\n⚠️ Error en la predicción. Código de estado:", response.status_code)
    except requests.ConnectionError:
        print("\n❌ No se pudo conectar a la API para hacer la predicción.")

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todas las fuentes (puedes especificar ['http://localhost:5173'])
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permitir todos los encabezados
)

if __name__ == "__main__":
    print("\n🚀 Iniciando servidor...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

