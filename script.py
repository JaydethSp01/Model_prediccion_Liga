from pymongo import MongoClient

# Conexión a la base de datos
client = MongoClient("mongodb://localhost:27017/")
db = client["futnexus"]  # Cambia esto por el nombre de tu base de datos
teams_collection = db["teams"]  # Cambia esto si tu colección tiene otro nombre

# Datos estimados de goles por partido y tarjetas por partido
equipos_estadisticas = {
    "Barcelona": {"goles_por_partido": 2.3, "tarjetas_por_partido": 2.1},
    "Real Madrid": {"goles_por_partido": 2.2, "tarjetas_por_partido": 2.3},
    "Atlético de Madrid": {"goles_por_partido": 1.9, "tarjetas_por_partido": 2.8},
    "Atlético de Bilbao": {"goles_por_partido": 1.7, "tarjetas_por_partido": 2.5},
    "Villarreal": {"goles_por_partido": 1.6, "tarjetas_por_partido": 2.4},
    "Real Betis": {"goles_por_partido": 1.5, "tarjetas_por_partido": 2.7},
    "Rayo Vallecano": {"goles_por_partido": 1.4, "tarjetas_por_partido": 3.0},
    "Mallorca": {"goles_por_partido": 1.3, "tarjetas_por_partido": 2.6},
    "Celta de Vigo": {"goles_por_partido": 1.4, "tarjetas_por_partido": 2.9},
    "Real Sociedad": {"goles_por_partido": 1.3, "tarjetas_por_partido": 2.4},
    "Sevilla": {"goles_por_partido": 1.2, "tarjetas_por_partido": 2.8},
    "Getafe": {"goles_por_partido": 1.1, "tarjetas_por_partido": 3.2},
    "Girona": {"goles_por_partido": 1.2, "tarjetas_por_partido": 2.3},
    "Osasuna": {"goles_por_partido": 1.0, "tarjetas_por_partido": 2.7},
    "Valencia CF": {"goles_por_partido": 1.1, "tarjetas_por_partido": 2.5},
    "Espanyol": {"goles_por_partido": 1.0, "tarjetas_por_partido": 2.6},
    "Alavés": {"goles_por_partido": 0.9, "tarjetas_por_partido": 3.1},
    "Leganés": {"goles_por_partido": 0.8, "tarjetas_por_partido": 3.0},
    "Las Palmas": {"goles_por_partido": 0.8, "tarjetas_por_partido": 2.8},
    "Real Valladolid": {"goles_por_partido": 0.7, "tarjetas_por_partido": 2.9},
}

# Actualizar cada equipo en la base de datos
for equipo, stats in equipos_estadisticas.items():
    # Buscar el equipo por nombre y actualizar sus estadísticas
    result = teams_collection.update_one(
        {"name": equipo},  # Asume que hay un campo "nombre" en tus documentos
        {"$set": {
            "goles_por_partido": stats["goles_por_partido"],
            "tarjetas_por_partido": stats["tarjetas_por_partido"]
        }},
        upsert=True  # Si el equipo no existe, lo crea
    )
    
    # Opcional: Mostrar confirmación
    print(f"Equipo {equipo} actualizado. Modificados: {result.modified_count} documentos")

# Cerrar la conexión
client.close()