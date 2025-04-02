from src.config.database import Database
from src.models.jugador import Jugador
from bson import ObjectId

class JugadorRepository:
    def __init__(self):
        db = Database.get_database()
        self.collection = db["players"]

    def get_all(self):
        jugadores_cursor = self.collection.find()
        jugadores = []
        for jugador in jugadores_cursor:
            jugador["_id"] = str(jugador["_id"])
            if "team" in jugador:
                jugador["team"] = str(jugador["team"])
            
            # Si rating es None, ponerlo en 0.0
            jugador["rating"] = jugador.get("rating", 0.0) if jugador.get("rating") is not None else 0.0

            jugadores.append(Jugador(**jugador))
        return jugadores

    def get_by_id(self, jugador_id: str):
        jugador = self.collection.find_one({"_id": ObjectId(jugador_id)})
        if jugador:
            jugador["_id"] = str(jugador["_id"])
            if "team" in jugador:
                jugador["team"] = str(jugador["team"])
            
            # Si rating es None, ponerlo en 0.0
            jugador["rating"] = jugador.get("rating", 0.0) if jugador.get("rating") is not None else 0.0

            return Jugador(**jugador)
        return None

    def get_by_team(self, team_id: str):
        """
        Obtiene todos los jugadores asociados a un equipo dado su ID.
        :param team_id: ID del equipo.
        :return: Lista de jugadores.
        """
        try:
            jugadores_cursor = self.collection.find({"team": ObjectId(team_id)})
            jugadores = []
            for jugador in jugadores_cursor:
                jugador["_id"] = str(jugador["_id"])
                jugador["team"] = str(jugador["team"])
                jugador["rating"] = jugador.get("rating", 0.0) if jugador.get("rating") is not None else 0.0
                jugadores.append(Jugador(**jugador))
            return jugadores
        except Exception as e:
            print(f"Error al obtener jugadores para el equipo con ID '{team_id}': {e}")
            return []