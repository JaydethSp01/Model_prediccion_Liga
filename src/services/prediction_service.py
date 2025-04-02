import random
import joblib
import numpy as np
import pandas as pd
import requests
import statistics
import re
from datetime import datetime, timedelta
from collections import defaultdict
from bson import ObjectId
from serpapi import GoogleSearch
from src.repositories.jugador_repository import JugadorRepository
from src.repositories.equipo_repository import EquipoRepository
import math

SERPAPI_API_KEY = "af93ffc6b296f828732106a04ffe3ea2180780ffe203638f081583ec16406f8c"

class PredictionService:
    def __init__(self):
        self.jugador_repo = JugadorRepository()
        self.equipo_repo = EquipoRepository()
        self.team_positions_cache = {
            "Barcelona": 1,
            "Real Madrid": 2,
            "Atlético Madrid": 3,
            "Athletic": 4,
            "Villarreal": 5,
            "Betis": 6,
            "Rayo Vallecano": 7,
            "R.C.D. Mallorca": 8,
            "Celta de Vigo": 9,
            "Real Sociedad": 10,
            "Sevilla": 11,
            "Getafe": 12,
            "Girona": 13,
            "Osasuna": 14,
            "Valencia C. F.": 15,
            "RCD Espanyol": 16,
            "Alavés": 17,
            "Leganés": 18,
            "U. D. Las Palmas": 19,
            "Valladolid": 20
        }
        
        try:
            print("\n🔄 Cargando modelos...")
            self.linear_model = joblib.load("models/linear_model.pkl")
            self.xgboost_model = joblib.load("models/xgboost_model.pkl")
            self.poisson_model = joblib.load("models/poisson_model.pkl")
            self.scaler = joblib.load("models/scaler.pkl")
            self.feature_columns = joblib.load("models/feature_columns.pkl")
            print("✅ Modelos cargados correctamente.")
        except FileNotFoundError as e:
            print(f"❌ Error al cargar modelos: {e}")
            self.linear_model = None
            self.xgboost_model = None
            self.poisson_model = None
            self.scaler = None
            self.feature_columns = None

    def predict_player_probabilities(self, jugador_id):
        """Predicción avanzada de probabilidades de jugador para apuestas"""
        jugador = self.jugador_repo.get_by_id(jugador_id)

        if not jugador:
            return {"error": "Jugador no encontrado."}

        jugador_data = {
            'matches': jugador.matches or 0,
            'minutesPlayed': jugador.minutesPlayed or 0,
            'goals': jugador.goals or 0,
            'expectedGoals': jugador.expectedGoals or 0,
            'assists': jugador.assists or 0,
            'expectedAssists': jugador.expectedAssists or 0,
            'shots': jugador.shots or 0,
            'shotsOnTarget': jugador.shotsOnTarget or 0,
            'passAccuracy': jugador.passAccuracy or 0,
            'dribbles': jugador.dribbles or 0,
            'fouls': jugador.fouls or 0,
            'yellowCards': jugador.yellowCards or 0,
            'redCards': jugador.redCards or 0,
            'rating': jugador.rating or 0
        }
        input_df = pd.DataFrame([jugador_data])
    
        if not self.scaler or input_df.empty:
            return {"error": "Error en modelos"}

        X_input = self.scaler.transform(input_df)
    
        model_predictions = {
            'gol': self._ajustar_prediccion_gol(
                self.normalize_prob(self.linear_model.predict(X_input)[0]),
                jugador
            ),
            'asistencia': self._ajustar_prediccion_asistencia(
                self.normalize_prob(self.xgboost_model.predict(X_input)[0]),
                jugador
            ),
            'tarjeta_amarilla': self._ajustar_prediccion_tarjeta(
                self.normalize_prob(self.poisson_model.predict(X_input)[0]),
                jugador
            )
        }

        probabilidades_apostables = {
            'gol_exacto_0': self._calcular_prob_gol_exacto(model_predictions['gol'], 0),
            'gol_exacto_1': self._calcular_prob_gol_exacto(model_predictions['gol'], 1),
            'gol_exacto_2': self._calcular_prob_gol_exacto(model_predictions['gol'], 2),
            'mas_de_1.5_goles': self._calcular_prob_rango_goles(model_predictions['gol'], 1.5),
            'primera_asistencia': model_predictions['asistencia'],
            'tarjeta_amarilla': model_predictions['tarjeta_amarilla'],
            'tiempo_primer_gol': self._estimar_tiempo_primer_gol(jugador)
        }
    
        analisis = {
            "jugador": jugador.name,
            "equipo": self.equipo_repo.get_by_id(str(jugador.team)).name if jugador.team else "Sin equipo",
            "posicion": jugador.position,
            "probabilidades_apostables": probabilidades_apostables,
            "referencias_mercado": self._obtener_referencias_mercado(jugador.name),
            "consejos_apuesta": self._generar_consejos_apuesta_jugador(model_predictions, jugador)
        }

        return analisis

    def _generar_consejos_apuesta_jugador(self, predictions, jugador):
        """Genera consejos específicos para apostar en un jugador"""
        consejos = []
        
        # Consejo para goles
        if predictions['gol'] > 0.4:
            consejos.append({
                "tipo": "Anotador",
                "recomendacion": f"Apuesta fuerte a que {jugador.name} marcará gol",
                "confianza": "Alta" if predictions['gol'] > 0.5 else "Media",
                "cuotas_recomendadas": {
                    "Bet365": f"https://www.bet365.com/#/AC/B1/C1/D8/E{jugador.name.replace(' ', '%20')}",
                    "Betplay": f"https://www.betplay.com.co/apuestas#filter/football/{jugador.name.replace(' ', '%20')}"
                }
            })
        elif predictions['gol'] > 0.2:
            consejos.append({
                "tipo": "Anotador",
                "recomendacion": f"Considera apuesta a que {jugador.name} marcará gol",
                "confianza": "Moderada",
                "cuotas_recomendadas": {
                    "Bet365": f"https://www.bet365.com/#/AC/B1/C1/D8/E{jugador.name.replace(' ', '%20')}",
                    "Stake": f"https://stake.com/sports/soccer/player/{jugador.name.replace(' ', '-').lower()}"
                }
            })
        
        # Consejo para asistencias
        if predictions['asistencia'] > 0.35:
            consejos.append({
                "tipo": "Asistencia",
                "recomendacion": f"Buena probabilidad de que {jugador.name} de asistencia",
                "confianza": "Alta" if predictions['asistencia'] > 0.45 else "Media",
                "cuotas_recomendadas": {
                    "Betplay": f"https://www.betplay.com.co/apuestas#filter/football/asistencias"
                }
            })
        
        # Consejo para tarjetas
        if predictions['tarjeta_amarilla'] > 0.4:
            consejos.append({
                "tipo": "Tarjeta amarilla",
                "recomendacion": f"Alto riesgo de tarjeta amarilla para {jugador.name}",
                "confianza": "Alta",
                "cuotas_recomendadas": {
                    "Bet365": f"https://www.bet365.com/#/AC/B1/C1/D8/E{jugador.name.replace(' ', '%20')}%20tarjeta",
                    "WilliamHill": f"https://sports.williamhill.com/betting/es-es/football/cards"
                }
            })
        
        return consejos

    def analizar_partidos_proximos(self):
        """Analiza partidos próximos con mayor precisión y consejos de apuesta específicos"""
        try:
            partidos_proximos = self._obtener_partidos_proximos_mejorado()
            print(f"Partidos próximos obtenidos: {len(partidos_proximos)}")
            
            if not partidos_proximos:
                return {"error": "No se encontraron partidos próximos para esta liga"}
            
            analisis_partidos = []
            
            for partido in partidos_proximos:
                equipo_local = partido.get("equipo_local", "")
                equipo_visitante = partido.get("equipo_visitante", "")
                
                # Obtener análisis detallado de equipos
                metricas_local = self._obtener_metricas_equipo(equipo_local)
                metricas_visitante = self._obtener_metricas_equipo(equipo_visitante)
                
                if not metricas_local or not metricas_visitante:
                    print(f"⚠️ Métricas no disponibles para {equipo_local} o {equipo_visitante}. Saltando partido.")
                    continue
                
                # Obtener posición en liga
                pos_local = self._obtener_posicion_liga(equipo_local)
                pos_visitante = self._obtener_posicion_liga(equipo_visitante)
                
                # Generar recomendaciones mejoradas
                recomendaciones = self._generar_recomendaciones_partido_mejorado(
                    equipo_local, 
                    equipo_visitante,
                    metricas_local,
                    metricas_visitante,
                    pos_local,
                    pos_visitante
                )
                
                # Crear análisis completo del partido
                analisis_partido = {
                    "partido": f"{equipo_local} vs {equipo_visitante}",
                    "fecha": partido.get("fecha", "Fecha no disponible"),
                    "estadio": partido.get("estadio", "Estadio no disponible"),
                    "posiciones": f"{equipo_local}: {pos_local}º - {equipo_visitante}: {pos_visitante}º",
                    "recomendaciones_apuesta": recomendaciones,
                    "probabilidades": self._calcular_probabilidades_partido_mejorado(
                        metricas_local, 
                        metricas_visitante,
                        pos_local,
                        pos_visitante
                    ),
                    "jugadores_clave": self._obtener_jugadores_clave(equipo_local, equipo_visitante),
                    "links_apuestas": self._generar_links_apuestas_mejorado(equipo_local, equipo_visitante)
                }
                
                analisis_partidos.append(analisis_partido)
            
            return {
                "total_partidos": len(analisis_partidos),
                "analisis_partidos": analisis_partidos
            }
            
        except Exception as e:
            print(f"Error al analizar partidos próximos: {e}")
            return {"error": f"Error al analizar partidos próximos: {str(e)}"}

    def _obtener_partidos_proximos_mejorado(self):
        """Obtiene partidos próximos desde SerpAPI con manejo robusto de errores"""
        try:
            params = {
                "engine": "google",
                "q": "próximos partidos La Liga",
                "api_key": SERPAPI_API_KEY,
                "hl": "es",
                "gl": "es"
            }
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            partidos = []
            
            if 'sports_results' in results:
                for match in results['sports_results'].get('games', []):
                    teams = match.get('teams', [])
                    if len(teams) == 2:
                        partidos.append({
                            "equipo_local": teams[0].get('name', ''),
                            "equipo_visitante": teams[1].get('name', ''),
                            "fecha": f"{match.get('date', '')} {match.get('time', '')}",
                            "estadio": match.get('stadium', 'Estadio no disponible')
                        })
            
            # Filtramos solo equipos de La Liga conocidos
            equipos_liga = set(self.team_positions_cache.keys())
            partidos_filtrados = [
                p for p in partidos 
                if p['equipo_local'] in equipos_liga and p['equipo_visitante'] in equipos_liga
            ]
            
            return partidos_filtrados
            
        except Exception as e:
            print(f"Error al obtener partidos desde SerpAPI: {e}")
            # Datos de ejemplo como fallback
            return [
                {
                    "equipo_local": "Real Madrid",
                    "equipo_visitante": "Barcelona",
                    "fecha": "2023-10-28 21:00",
                    "estadio": "Santiago Bernabéu"
                },
                {
                    "equipo_local": "Atlético Madrid",
                    "equipo_visitante": "Valencia C. F.",
                    "fecha": "2023-10-29 18:30",
                    "estadio": "Wanda Metropolitano"
                }
            ]

    def _obtener_metricas_equipo(self, nombre_equipo):
        """
        Obtiene las métricas de un equipo por su nombre.
        """
        try:
            equipo = self.equipo_repo.get_by_name(nombre_equipo)
            if not equipo:
                print(f"⚠️ Equipo no encontrado: {nombre_equipo}")
                return None

            # Aquí puedes calcular o devolver las métricas del equipo
            metricas = {
                "goles_por_partido": equipo.goles_por_partido,
                "tarjetas_por_partido": equipo.tarjetas_por_partido,
            }
            return metricas
        except Exception as e:
            print(f"Error obteniendo métricas para {nombre_equipo}: {e}")
            return None

    def _obtener_posicion_liga(self, nombre_equipo):
        """
        Obtiene la posición actual del equipo en la liga.
        Si no está en el caché, intenta buscarlo en la base de datos o devuelve un valor predeterminado.
        """
        try:
            # Verificar si el equipo está en el caché
            if nombre_equipo in self.team_positions_cache:
                return self.team_positions_cache[nombre_equipo]

            # Si no está en el caché, buscar en la base de datos
            equipo = self.equipo_repo.get_by_name(nombre_equipo)
            if equipo and hasattr(equipo, "posicion_liga"):
                posicion = equipo.posicion_liga
                # Actualizar el caché
                self.team_positions_cache[nombre_equipo] = posicion
                return posicion

            # Si no se encuentra, devolver un valor predeterminado
            print(f"⚠️ Posición no encontrada para el equipo: {nombre_equipo}. Usando valor predeterminado.")
            return 20  # Valor predeterminado para equipos desconocidos

        except Exception as e:
            print(f"Error al obtener posición de liga para {nombre_equipo}: {e}")
            return 20  # Valor predeterminado en caso de error

    def _generar_recomendaciones_partido_mejorado(self, equipo_local, equipo_visitante,
                                                metricas_local, metricas_visitante,
                                                pos_local, pos_visitante):
        """
        Genera recomendaciones de apuestas mejoradas para un partido específico
        considerando posición en liga, rendimiento reciente y otros factores clave
        """
        diferencia_posicion = pos_local - pos_visitante if pos_local and pos_visitante else 0
        
        # Métricas ofensivas y defensivas
        goles_local = metricas_local.get("goles_por_partido", 0)
        goles_visitante = metricas_visitante.get("goles_por_partido", 0)
        tarjetas_local = metricas_local.get("tarjetas_por_partido", 0)
        tarjetas_visitante = metricas_visitante.get("tarjetas_por_partido", 0)
        
        # Ajustar por posición en liga y localía
        factor_localia = 1.15  # Equipo local tiene ventaja
        factor_posicion = 1 + (abs(diferencia_posicion) * 0.04)  # 4% por puesto de diferencia
        
        if pos_local < pos_visitante:  # Local está mejor en la tabla
            goles_local_ajustado = goles_local * factor_localia * factor_posicion
            goles_visitante_ajustado = goles_visitante / factor_posicion
        else:  # Visitante está mejor o igual
            goles_local_ajustado = goles_local * factor_localia / factor_posicion
            goles_visitante_ajustado = goles_visitante * factor_posicion
        
        # Generar recomendaciones con datos ajustados
        recomendaciones = {
            "resultado_final": self._recomendar_resultado_final_mejorado(
                goles_local_ajustado, goles_visitante_ajustado, diferencia_posicion),
            "total_goles": self._recomendar_total_goles_mejorado(
                goles_local_ajustado, goles_visitante_ajustado),
            "ambos_marcan": self._recomendar_ambos_marcan_mejorado(
                goles_local_ajustado, goles_visitante_ajustado),
            "handicap": self._recomendar_handicap(pos_local, pos_visitante),
            "tarjetas": self._recomendar_tarjetas_mejorado(
                tarjetas_local, tarjetas_visitante, diferencia_posicion),
            "apuestas_valiosas": self._identificar_apuestas_valiosas(
                goles_local_ajustado, goles_visitante_ajustado, diferencia_posicion)
        }
        
        return recomendaciones

    def _recomendar_resultado_final_mejorado(self, goles_local, goles_visitante, diferencia_posicion):
        """Recomendación mejorada del resultado final con lógica más sofisticada"""
        prob_victoria_local, prob_empate, prob_victoria_visitante = self._calcular_probabilidades_poisson(
            goles_local, goles_visitante)
        
        # Ajustar por diferencia de posición
        if abs(diferencia_posicion) > 8:  # Gran diferencia de posición
            if diferencia_posicion < 0:  # Visitante está mejor
                ajuste = min(0.2, abs(diferencia_posicion) * 0.03)
                prob_victoria_visitante += ajuste
                prob_victoria_local -= ajuste * 0.7
                prob_empate -= ajuste * 0.3
            else:  # Local está mejor
                ajuste = min(0.2, diferencia_posicion * 0.03)
                prob_victoria_local += ajuste
                prob_victoria_visitante -= ajuste * 0.7
                prob_empate -= ajuste * 0.3
        
        # Normalizar probabilidades
        total = prob_victoria_local + prob_empate + prob_victoria_visitante
        prob_victoria_local /= total
        prob_empate /= total
        prob_victoria_visitante /= total
        
        # Determinar recomendación basada en probabilidades ajustadas
        if prob_victoria_local > 0.65:
            return {
                "recomendacion": "Victoria local",
                "confianza": "Alta",
                "probabilidad": round(prob_victoria_local, 2),
                "explicacion": f"Clara ventaja local ({diferencia_posicion} puestos de diferencia)"
            }
        elif prob_victoria_visitante > 0.6:
            return {
                "recomendacion": "Victoria visitante",
                "confianza": "Alta",
                "probabilidad": round(prob_victoria_visitante, 2),
                "explicacion": f"El visitante es claramente superior ({abs(diferencia_posicion)} puestos arriba)"
            }
        elif prob_empate > 0.4:
            return {
                "recomendacion": "Empate",
                "confianza": "Media-Alta" if prob_empate > 0.45 else "Media",
                "probabilidad": round(prob_empate, 2),
                "explicacion": "Partido muy equilibrado según estadísticas"
            }
        else:
            return {
                "recomendacion": "Doble oportunidad local/empate",
                "confianza": "Media",
                "probabilidad": round(prob_victoria_local + prob_empate, 2),
                "explicacion": "Ventaja local moderada pero no decisiva"
            }

    def _recomendar_handicap(self, pos_local, pos_visitante):
        """
        Recomienda handicap basado en diferencia real de posiciones
        """
        diferencia = pos_visitante - pos_local  # Posiciones más bajas son mejores
        
        if diferencia >= 10:  # Visitante mucho mejor
            return {
                "recomendacion": "Handicap +1.5 para visitante",
                "confianza": "Alta",
                "explicacion": f"El visitante ({pos_visitante}º) supera por {diferencia} puestos al local ({pos_local}º)"
            }
        elif diferencia >= 5:
            return {
                "recomendacion": "Handicap +0.5 para visitante",
                "confianza": "Media-Alta",
                "explicacion": f"Ventaja significativa del visitante ({diferencia} puestos)"
            }
        elif diferencia <= -10:  # Local mucho mejor
            return {
                "recomendacion": "Handicap -1.5 para local",
                "confianza": "Alta",
                "explicacion": f"El local ({pos_local}º) supera por {abs(diferencia)} puestos al visitante ({pos_visitante}º)"
            }
        elif diferencia <= -5:
            return {
                "recomendacion": "Handicap -0.5 para local",
                "confianza": "Media-Alta",
                "explicacion": f"Ventaja significativa del local ({abs(diferencia)} puestos)"
            }
        else:
            return {
                "recomendacion": "Sin handicap recomendado",
                "confianza": "Baja",
                "explicacion": "Partido equilibrado en la tabla"
            }

    def _calcular_probabilidades_partido_mejorado(self, metricas_local, metricas_visitante, pos_local, pos_visitante):
        """
        Calcula probabilidades realistas usando datos históricos reales
        """
        # Obtener datos reales de goles
        goles_local = metricas_local.get("goles_por_partido", 1.2)
        goles_visitante = metricas_visitante.get("goles_por_partido", 1.0)
        
        # Ajustar por posición y localía
        diferencia_pos = pos_visitante - pos_local  # Posición más baja = mejor equipo
        factor_posicion = 1 + (abs(diferencia_pos) * 0.03)
        
        if diferencia_pos > 0:  # Visitante mejor colocado
            goles_visitante *= factor_posicion
            goles_local /= factor_posicion
        else:  # Local mejor colocado
            goles_local *= factor_posicion
            goles_visitante /= factor_posicion
        
        # Aplicar ventaja de localía
        goles_local *= 1.15
        
        # Calcular probabilidades con Poisson
        prob_victoria_local, prob_empate, prob_victoria_visitante = self._calcular_probabilidades_poisson(
            goles_local, goles_visitante)
        
        return {
            "victoria_local": round(prob_victoria_local, 3),
            "empate": round(prob_empate, 3),
            "victoria_visitante": round(prob_victoria_visitante, 3),
            "goles_esperados": {
                "local": round(goles_local, 2),
                "visitante": round(goles_visitante, 2)
            },
            "analisis_riesgo": self._analizar_riesgo_apuesta(
                prob_victoria_local, 
                prob_empate, 
                prob_victoria_visitante,
                diferencia_pos
            )
        }

    def _calcular_probabilidades_poisson(self, lambda_local, lambda_visitante):
        """Implementación mejorada del modelo Poisson"""
        max_goles = 5  # Máximo número de goles a considerar
        prob_victoria_local = 0.0
        prob_empate = 0.0
        prob_victoria_visitante = 0.0

        for i in range(max_goles):
            for j in range(max_goles):
                # Calcular probabilidad de este resultado
                prob = (math.exp(-lambda_local) * lambda_local**i / math.factorial(i)) * \
                       (math.exp(-lambda_visitante) * lambda_visitante**j / math.factorial(j))
                
                if i > j:
                    prob_victoria_local += prob
                elif i == j:
                    prob_empate += prob
                else:
                    prob_victoria_visitante += prob

        # Normalizar las probabilidades
        total = prob_victoria_local + prob_empate + prob_victoria_visitante
        return (
            prob_victoria_local / total,
            prob_empate / total,
            prob_victoria_visitante / total
        )

    def _analizar_riesgo_apuesta(self, prob_local, prob_empate, prob_visitante, diferencia_pos):
        """Análisis de riesgo mejorado con lógica más precisa"""
        analisis = []
        total_prob = prob_local + prob_empate + prob_visitante
        
        # 1. Apuesta directa
        mejor_prob = max(prob_local, prob_visitante)
        if mejor_prob > 0.7:
            analisis.append({
                "tipo": "Victoria directa",
                "riesgo": "Bajo",
                "valor": "Excelente" if mejor_prob > 0.75 else "Bueno",
                "explicacion": f"Probabilidad clara ({mejor_prob*100:.1f}%)"
            })
        elif mejor_prob > 0.55:
            analisis.append({
                "tipo": "Victoria directa",
                "riesgo": "Moderado",
                "valor": "Aceptable",
                "explicacion": f"Probabilidad favorable ({mejor_prob*100:.1f}%)"
            })
        else:
            analisis.append({
                "tipo": "Victoria directa",
                "riesgo": "Alto",
                "valor": "Pobre",
                "explicacion": "Partido muy equilibrado"
            })

        # 2. Apuestas de handicap
        if abs(diferencia_pos) >= 8:
            analisis.append({
                "tipo": "Handicap de goles",
                "riesgo": "Moderado",
                "valor": "Bueno",
                "explicacion": f"Gran diferencia de posiciones ({abs(diferencia_pos)} puestos)"
            })
        elif abs(diferencia_pos) >= 5:
            analisis.append({
                "tipo": "Handicap de goles",
                "riesgo": "Moderado-Alto",
                "valor": "Regular",
                "explicacion": f"Diferencia significativa ({abs(diferencia_pos)} puestos)"
            })

        # 3. Ambas marcan
        prob_ambos_marcan = 1 - ((1 - (prob_local / total_prob)) * (1 - (prob_visitante / total_prob)))
        if prob_ambos_marcan > 0.65:
            analisis.append({
                "tipo": "Ambos marcan - Sí",
                "riesgo": "Bajo",
                "valor": "Excelente",
                "explicacion": f"Alta probabilidad ({prob_ambos_marcan*100:.1f}%)"
            })
        elif prob_ambos_marcan > 0.5:
            analisis.append({
                "tipo": "Ambos marcan - Sí",
                "riesgo": "Moderado",
                "valor": "Bueno",
                "explicacion": f"Probabilidad favorable ({prob_ambos_marcan*100:.1f}%)"
            })

        return analisis

    # ... (Mantener los métodos restantes igual que en la versión anterior pero con posiciones actualizadas)

    def _obtener_jugadores_clave(self, equipo_local_nombre, equipo_visitante_nombre):
        """
        Obtiene los jugadores clave de los equipos local y visitante basándose en su rendimiento.
        :param equipo_local_nombre: Nombre del equipo local.
        :param equipo_visitante_nombre: Nombre del equipo visitante.
        :return: Diccionario con los jugadores clave de ambos equipos.
        """
        try:
            # Buscar el equipo local por nombre y obtener su ID
            equipo_local = self.equipo_repo.get_by_name(equipo_local_nombre)
            if not equipo_local:
                print(f"⚠️ Equipo local no encontrado: {equipo_local_nombre}")
                jugadores_local_ordenados = []
            else:
                jugadores_local = self.jugador_repo.get_by_team(equipo_local.id)
                jugadores_local_ordenados = sorted(
                    jugadores_local,
                    key=lambda j: (j.goals, j.assists),
                    reverse=True
                )[:3]  # Tomar los 3 mejores jugadores

            # Buscar el equipo visitante por nombre y obtener su ID
            equipo_visitante = self.equipo_repo.get_by_name(equipo_visitante_nombre)
            if not equipo_visitante:
                print(f"⚠️ Equipo visitante no encontrado: {equipo_visitante_nombre}")
                jugadores_visitante_ordenados = []
            else:
                jugadores_visitante = self.jugador_repo.get_by_team(equipo_visitante.id)
                jugadores_visitante_ordenados = sorted(
                    jugadores_visitante,
                    key=lambda j: (j.goals, j.assists),
                    reverse=True
                )[:3]  # Tomar los 3 mejores jugadores

            return {
                "local": [{"nombre": j.name, "goles": j.goals, "asistencias": j.assists} for j in jugadores_local_ordenados],
                "visitante": [{"nombre": j.name, "goles": j.goals, "asistencias": j.assists} for j in jugadores_visitante_ordenados]
            }
        except Exception as e:
            print(f"Error obteniendo jugadores clave: {e}")
            return {"local": [], "visitante": []}

    def _generar_links_apuestas_mejorado(self, equipo_local, equipo_visitante):
        """Genera links con mercados relevantes según el tipo de partido"""
        query = f"{equipo_local} vs {equipo_visitante}".replace(' ', '%20')
        return [
            {
                "casa": "Bet365",
                "url": f"https://www.bet365.com/#/AC/B1/C1/D8/E{query}",
                "mercados_recomendados": ["Ganador del Partido", "Over/Under 2.5", "Ambos Marcan"]
            },
            {
                "casa": "Betplay",
                "url": f"https://www.betplay.com.co/apuestas#filter/football/{query}",
                "mercados_recomendados": ["Handicap Asiático", "Resultado Exacto", "Tarjetas"]
            }
        ]
    def _ajustar_prediccion_gol(self, probabilidad, jugador):
        """Ajusta la probabilidad de gol basado en factores adicionales"""
        # Ajuste por posición
        if jugador.position in ['FW', 'ST']:
            probabilidad = 1.0  # Initialize probabilidad if not already defined
            probabilidad *= 1.2
        elif jugador.position in ['MF', 'AM']:
            probabilidad *= 1.1
        else:
            probabilidad *= 0.7
            
        # Ajuste por minutos jugados
        if jugador.minutesPlayed and jugador.matches:
            avg_minutes = jugador.minutesPlayed / jugador.matches
            probabilidad *= min(1.0, avg_minutes / 80)  # Jugadores que juegan más minutos tienen más chances
            
        return min(0.99, max(0.01, probabilidad))

    def _ajustar_prediccion_asistencia(self, probabilidad, jugador):
        """Ajusta la probabilidad de asistencia"""
        # Ajuste por posición
        if jugador.position in ['MF', 'AM', 'WF']:
            probabilidad *= 1.3
        elif jugador.position == 'FW':
            probabilidad *= 1.1
        else:
            probabilidad *= 0.6
            
        # Ajuste por precisión de pases
        if jugador.passAccuracy:
            probabilidad *= (jugador.passAccuracy / 100) * 1.2
            
        return min(0.99, max(0.01, probabilidad))

    def _ajustar_prediccion_tarjeta(self, probabilidad, jugador):
        """Ajusta la probabilidad de tarjeta amarilla"""
        # Ajuste por posición
        if jugador.position in ['DF', 'DM']:
            probabilidad *= 1.4
        elif jugador.position == 'MF':
            probabilidad *= 1.2
        else:
            probabilidad *= 0.8
            
        # Ajuste por faltas cometidas
        if jugador.fouls and jugador.matches:
            avg_fouls = jugador.fouls / jugador.matches
            probabilidad *= min(2.0, max(0.5, avg_fouls))
            
        return min(0.99, max(0.01, probabilidad))

    def normalize_prob(self, prob):
        """Normaliza la probabilidad entre 0 y 1"""
        return max(0.0, min(1.0, prob))

    def _calcular_prob_gol_exacto(self, prob_gol, n_goles):
        """Calcula la probabilidad de que un jugador marque exactamente n goles"""
        # Usando distribución de Poisson
        return (math.exp(-prob_gol) * (prob_gol ** n_goles)) / math.factorial(n_goles)

    def _calcular_prob_rango_goles(self, prob_gol, umbral):
        """Calcula la probabilidad de que un jugador marque más de X goles"""
        if umbral == 1.5:
            # P(>1.5) = 1 - P(0) - P(1)
            return 1 - self._calcular_prob_gol_exacto(prob_gol, 0) - self._calcular_prob_gol_exacto(prob_gol, 1)
        elif umbral == 0.5:
            # P(>0.5) = 1 - P(0)
            return 1 - self._calcular_prob_gol_exacto(prob_gol, 0)
        else:
            return 0.0

    def _estimar_tiempo_primer_gol(self, jugador):
        """Estima el tiempo probable del primer gol basado en estadísticas"""
        if not jugador.goals or not jugador.minutesPlayed:
            return "Sin datos suficientes"
            
        avg_goal_time = jugador.minutesPlayed / jugador.goals
        if avg_goal_time < 120:
            return f"Primera mitad ({random.randint(15, 45)}')"
        else:
            return f"Segunda mitad ({random.randint(60, 85)}')"

    def _obtener_referencias_mercado(self, nombre_jugador):
        """Obtiene referencias de mercado para un jugador"""
        try:
            params = {
                "engine": "google",
                "q": f"{nombre_jugador} apuestas",
                "api_key": SERPAPI_API_KEY,
                "hl": "es"
            }
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            referencias = []
            if "organic_results" in results:
                for result in results["organic_results"][:3]:  # Top 3 resultados
                    referencias.append({
                        "titulo": result.get("title", ""),
                        "enlace": result.get("link", ""),
                        "fuente": result.get("source", "")
                    })
            
            return referencias
        except Exception as e:
            print(f"Error al obtener referencias de mercado: {e}")
            return []

    def _recomendar_total_goles_mejorado(self, goles_local, goles_visitante):
        """
        Genera recomendaciones de apuestas basadas en el total de goles esperados
        en el partido.
        """
        total_goles = goles_local + goles_visitante

        if total_goles > 3.5:
            return {
                "recomendacion": "Más de 3.5 goles",
                "confianza": "Alta",
                "explicacion": f"Se esperan muchos goles en el partido (total esperado: {round(total_goles, 2)})"
            }
        elif total_goles > 2.5:
            return {
                "recomendacion": "Más de 2.5 goles",
                "confianza": "Media-Alta",
                "explicacion": f"Se espera un partido con goles (total esperado: {round(total_goles, 2)})"
            }
        elif total_goles > 1.5:
            return {
                "recomendacion": "Más de 1.5 goles",
                "confianza": "Media",
                "explicacion": f"Partido con probabilidad moderada de goles (total esperado: {round(total_goles, 2)})"
            }
        else:
            return {
                "recomendacion": "Menos de 1.5 goles",
                "confianza": "Media-Baja",
                "explicacion": f"Se espera un partido defensivo con pocos goles (total esperado: {round(total_goles, 2)})"
            }

    def _recomendar_ambos_marcan_mejorado(self, goles_local, goles_visitante):
        """
        Genera recomendaciones de apuestas basadas en si ambos equipos marcarán.
        """
        prob_ambos_marcan = 1 - ((1 - goles_local) * (1 - goles_visitante))

        if prob_ambos_marcan > 0.6:
            return {
                "recomendacion": "Ambos equipos marcan - Sí",
                "confianza": "Alta",
                "explicacion": f"Alta probabilidad de que ambos equipos marquen (probabilidad: {round(prob_ambos_marcan, 2)})"
            }
        elif prob_ambos_marcan > 0.4:
            return {
                "recomendacion": "Ambos equipos marcan - Sí",
                "confianza": "Media",
                "explicacion": f"Probabilidad moderada de que ambos equipos marquen (probabilidad: {round(prob_ambos_marcan, 2)})"
            }
        else:
            return {
                "recomendacion": "Ambos equipos marcan - No",
                "confianza": "Alta",
                "explicacion": f"Baja probabilidad de que ambos equipos marquen (probabilidad: {round(prob_ambos_marcan, 2)})"
            }

    def _recomendar_tarjetas_mejorado(self, tarjetas_local, tarjetas_visitante, diferencia_posicion):
        """
        Genera recomendaciones de apuestas basadas en el número de tarjetas esperadas.
        """
        total_tarjetas = tarjetas_local + tarjetas_visitante

        if total_tarjetas > 5.5:
            return {
                "recomendacion": "Más de 5.5 tarjetas",
                "confianza": "Alta",
                "explicacion": f"Se espera un partido muy físico con muchas tarjetas (total esperado: {round(total_tarjetas, 2)})"
            }
        elif total_tarjetas > 4.5:
            return {
                "recomendacion": "Más de 4.5 tarjetas",
                "confianza": "Media-Alta",
                "explicacion": f"Partido físico con probabilidad de varias tarjetas (total esperado: {round(total_tarjetas, 2)})"
            }
        elif total_tarjetas > 3.5:
            return {
                "recomendacion": "Más de 3.5 tarjetas",
                "confianza": "Media",
                "explicacion": f"Partido con probabilidad moderada de tarjetas (total esperado: {round(total_tarjetas, 2)})"
            }
        else:
            return {
                "recomendacion": "Menos de 3.5 tarjetas",
                "confianza": "Media-Baja",
                "explicacion": f"Se espera un partido disciplinado con pocas tarjetas (total esperado: {round(total_tarjetas, 2)})"
            }

    def _identificar_apuestas_valiosas(self, goles_local, goles_visitante, diferencia_posicion):
        """
        Identifica apuestas con valor basado en el análisis de goles y posiciones.
        """
        apuestas = []
        total_goles = goles_local + goles_visitante

        # Apuesta 1: Resultado exacto común
        if diferencia_posicion > 8:  # Gran diferencia
            apuestas.append({
                "tipo": "Resultado exacto 2-0 o 3-0",
                "valor": "Alto",
                "explicacion": f"El equipo local supera por {diferencia_posicion} posiciones al visitante",
                "cuota_esperada": "4.50 - 5.50"
            })

        # Apuesta 2: Total de goles
        if total_goles > 3.5 and diferencia_posicion < 5:
            apuestas.append({
                "tipo": "Más de 3.5 goles",
                "valor": "Alto",
                "explicacion": "Ambos equipos ofensivos con alta media de goles",
                "cuota_esperada": "2.50 - 3.20"
            })

        # Apuesta 3: Ambos marcan
        prob_ambos_marcan = 1 - ((1 - goles_local) * (1 - goles_visitante))
        if 0.4 < prob_ambos_marcan < 0.7 and abs(diferencia_posicion) < 8:
            apuestas.append({
                "tipo": "Ambos equipos marcan - Sí",
                "valor": "Buen valor",
                "explicacion": "Equipos con capacidad ofensiva pero no excesiva diferencia",
                "cuota_esperada": "1.80 - 2.20"
            })

        return apuestas