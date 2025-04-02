# src/repositories/equipo_repository.py
from src.config.database import Database
from src.models.equipo import Equipo
from bson import ObjectId
from src.repositories.jugador_repository import JugadorRepository

class EquipoRepository:
    def __init__(self):
        db = Database.get_database()
        self.collection = db["teams"]

    def get_all(self):
        equipos_cursor = self.collection.find()
        equipos = []
        for equipo in equipos_cursor:
            equipo["_id"] = str(equipo["_id"])
            if "players" in equipo:
                equipo["players"] = [str(pid) for pid in equipo["players"]]
            equipos.append(Equipo(**equipo))
        return equipos

    def get_by_id(self, equipo_id: str):
        equipo = self.collection.find_one({"_id": ObjectId(equipo_id)})
        if equipo:
            equipo["_id"] = str(equipo["_id"])
            if "players" in equipo:
                equipo["players"] = [str(pid) for pid in equipo["players"]]
            return Equipo(**equipo)
        return None

    def get_by_name(self, nombre_equipo: str):
        """
        Busca un equipo por su nombre en la base de datos.
        """
        try:
            equipo = self.collection.find_one({"name": {"$regex": f"^{nombre_equipo}$", "$options": "i"}})
            if equipo:
                equipo["_id"] = str(equipo["_id"])
                if "players" in equipo:
                    equipo["players"] = [str(pid) for pid in equipo["players"]]
                return Equipo(**equipo)
            return None
        except Exception as e:
            print(f"Error al buscar equipo por nombre '{nombre_equipo}': {e}")
            return None

    def get_all_teams_with_full_player_details(self):
        equipos_cursor = self.collection.find()
        equipos = []
        jugador_repo = JugadorRepository()
        
        for equipo in equipos_cursor:
            # Convertir ObjectId a string
            equipo["_id"] = str(equipo["_id"])
            team_id = equipo["_id"]
            
            # Obtener jugadores completos por team_id
            jugadores = jugador_repo.get_by_team(team_id)
            
            # Crear estructura del equipo con jugadores completos
            equipo_data = {
                "_id": equipo["_id"],
                "name": equipo.get("name", ""),
                "logo": equipo.get("logo", ""),
                "players": [{
                    "_id": str(jugador._id),
                    "name": jugador.name,
                    "position": jugador.position,
                    "goals": jugador.goals,
                    "assists": jugador.assists,
                } for jugador in jugadores]
            }
            
            equipos.append(equipo_data)
        
        return equipos

    def get_mejores_jugadores(self, team_id: str, limite: int = 3):
        """
        Obtiene los mejores jugadores de un equipo basado en goles y asistencias.
        :param team_id: ID del equipo.
        :param limite: Número máximo de jugadores a devolver.
        :return: Lista de jugadores ordenados por rendimiento.
        """
        try:
            # Obtener detalles de los jugadores asociados al equipo
            jugador_repo = JugadorRepository()
            jugadores = jugador_repo.get_by_team(team_id)

            if not jugadores:
                print(f"⚠️ No se encontraron jugadores para el equipo con ID: {team_id}")
                return []

            # Ordenar jugadores por goles y asistencias
            jugadores_ordenados = sorted(
                jugadores,
                key=lambda j: (j.goals, j.assists),
                reverse=True
            )

            # Devolver los mejores jugadores según el límite
            return jugadores_ordenados[:limite]
        except Exception as e:
            print(f"Error al obtener los mejores jugadores para el equipo con ID '{team_id}': {e}")
            return []

