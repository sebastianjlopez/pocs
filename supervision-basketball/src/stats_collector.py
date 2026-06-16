"""
StatsCollector — acumula estadísticas de juego a partir de GameEvents.

Estadísticas por jugador:
  - Posesiones
  - Tiempo de posesión (segundos)
  - Pases realizados / recibidos
  - Tiros
  - Canastas
  - Pérdidas de balón (loose ball mientras poseía)

Estadísticas por equipo:
  - Todo lo anterior agregado

Al finalizar, exporta a JSON y muestra tabla en consola.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from event_engine import EventType, GameEvent


@dataclass
class PlayerStats:
    player_id:        int
    team:             int | None = None
    possessions:      int = 0
    possession_frames: int = 0         # frames totales con posesión
    passes_made:      int = 0
    passes_received:  int = 0
    shots:            int = 0
    baskets:          int = 0
    loose_balls:      int = 0          # perdidas de balón

    def possession_seconds(self, fps: float) -> float:
        return round(self.possession_frames / fps, 1)

    def shooting_pct(self) -> float:
        return round(self.baskets / self.shots * 100, 1) if self.shots > 0 else 0.0


class StatsCollector:
    """
    Escucha GameEvents y construye estadísticas.

    Uso:
        stats = StatsCollector(fps=30)
        engine = EventEngine(fps=30, on_event=stats.record)
    """

    def __init__(self, fps: float) -> None:
        self.fps = fps
        self.players: dict[int, PlayerStats] = {}
        self.events_log: list[GameEvent]     = []

        # Seguimiento de posesión activa para medir duración
        self._current_possessor: int | None   = None
        self._possession_start:  int          = 0   # frame

        # Contadores de equipo (se calculan al exportar)
        self._team_cache: dict[int, int] = {}

    # ------------------------------------------------------------------
    def record(self, event: GameEvent) -> None:
        """Llamado por EventEngine cada vez que hay un evento nuevo."""
        self.events_log.append(event)

        # Actualizar caché de equipos
        if event.player_id is not None and event.team is not None:
            self._team_cache[event.player_id] = event.team
        if event.receiver_id is not None and event.team is not None:
            # No sabemos el equipo del receptor aún — se actualiza cuando tenga evento propio
            pass

        if event.type == EventType.POSSESSION:
            self._handle_possession(event)

        elif event.type == EventType.PASS:
            self._handle_pass(event)

        elif event.type == EventType.SHOT:
            self._get_or_create(event.player_id, event.team).shots += 1

        elif event.type == EventType.BASKET:
            self._get_or_create(event.player_id, event.team).baskets += 1

        elif event.type == EventType.LOOSE_BALL:
            if self._current_possessor is not None:
                self._get_or_create(self._current_possessor).loose_balls += 1
            self._close_possession(event.frame_idx)

        elif event.type == EventType.DEAD_BALL:
            self._close_possession(event.frame_idx)

    # ------------------------------------------------------------------
    def tick(self, frame_idx: int) -> None:
        """
        Llamar una vez por frame para acumular tiempo de posesión.
        Si no se llama, possession_seconds será 0 siempre.
        """
        if self._current_possessor is not None:
            ps = self._get_or_create(self._current_possessor)
            ps.possession_frames += 1

    # ------------------------------------------------------------------
    def print_summary(self) -> None:
        """Imprime tabla de estadísticas en consola."""
        print(f"\n{'='*72}")
        print("  ESTADÍSTICAS DEL PARTIDO")
        print(f"{'='*72}")

        for team_id in (0, 1):
            team_players = [p for p in self.players.values() if p.team == team_id]
            if not team_players:
                continue

            team_label = "Equipo A" if team_id == 0 else "Equipo B"
            print(f"\n  {team_label}")
            print(f"  {'ID':>4} | {'Poses.':>6} | {'T.Pos(s)':>8} | {'Pases':>6} | {'Tiros':>6} | {'Canast':>6} | {'%':>5} | {'Pérd.':>6}")
            print(f"  " + "-" * 60)

            for p in sorted(team_players, key=lambda x: x.player_id):
                print(
                    f"  #{p.player_id:>3} | "
                    f"{p.possessions:>6} | "
                    f"{p.possession_seconds(self.fps):>8.1f} | "
                    f"{p.passes_made:>6} | "
                    f"{p.shots:>6} | "
                    f"{p.baskets:>6} | "
                    f"{p.shooting_pct():>4.0f}% | "
                    f"{p.loose_balls:>6}"
                )

            # Totales de equipo
            tot_pos  = sum(p.possessions        for p in team_players)
            tot_time = sum(p.possession_frames   for p in team_players) / self.fps
            tot_pass = sum(p.passes_made         for p in team_players)
            tot_shot = sum(p.shots               for p in team_players)
            tot_bask = sum(p.baskets             for p in team_players)
            tot_pct  = round(tot_bask / tot_shot * 100, 1) if tot_shot > 0 else 0.0
            print(f"  {'TOT':>4} | {tot_pos:>6} | {tot_time:>8.1f} | {tot_pass:>6} | {tot_shot:>6} | {tot_bask:>6} | {tot_pct:>4.0f}% |")

        # Jugadores sin equipo asignado
        unknown = [p for p in self.players.values() if p.team not in (0, 1)]
        if unknown:
            print(f"\n  Sin equipo asignado")
            for p in unknown:
                print(f"  #{p.player_id}: posesiones={p.possessions}, pases={p.passes_made}")

        print(f"\n  Total de eventos registrados: {len(self.events_log)}")
        self._print_event_breakdown()

    def _print_event_breakdown(self) -> None:
        counts: dict[str, int] = defaultdict(int)
        for ev in self.events_log:
            counts[ev.type.name] += 1
        print("  Eventos:")
        for name, count in sorted(counts.items()):
            print(f"    {name:<12}: {count}")

    # ------------------------------------------------------------------
    def export_json(self, path: str) -> None:
        data = {
            "players": {
                str(pid): {
                    "team": ("A" if p.team == 0 else "B") if p.team in (0,1) else "?",
                    "possessions":         p.possessions,
                    "possession_seconds":  p.possession_seconds(self.fps),
                    "passes_made":         p.passes_made,
                    "passes_received":     p.passes_received,
                    "shots":               p.shots,
                    "baskets":             p.baskets,
                    "shooting_pct":        p.shooting_pct(),
                    "loose_balls":         p.loose_balls,
                }
                for pid, p in self.players.items()
            },
            "teams": self._team_totals(),
            "events_total": len(self.events_log),
            "events_by_type": {
                ev_type.name: sum(1 for e in self.events_log if e.type == ev_type)
                for ev_type in EventType
            },
            "events_log": [
                {
                    "frame":  e.frame_idx,
                    "type":   e.type.name,
                    "player": e.player_id,
                    "receiver": e.receiver_id,
                    "team":   ("A" if e.team == 0 else "B") if e.team in (0,1) else None,
                    "speed_ms": round(e.speed_ms, 2),
                    "ball_pos": e.ball_pos.tolist() if e.ball_pos is not None else None,
                }
                for e in self.events_log
            ],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Estadísticas exportadas a: {path}")

    # ------------------------------------------------------------------
    def _get_or_create(self, player_id: int | None, team: int | None = None) -> PlayerStats:
        if player_id is None:
            player_id = -1
        if player_id not in self.players:
            self.players[player_id] = PlayerStats(player_id=player_id)
        ps = self.players[player_id]
        if team is not None:
            ps.team = team
        elif player_id in self._team_cache:
            ps.team = self._team_cache[player_id]
        return ps

    def _handle_possession(self, event: GameEvent) -> None:
        self._close_possession(event.frame_idx)
        ps = self._get_or_create(event.player_id, event.team)
        ps.possessions += 1
        self._current_possessor = event.player_id
        self._possession_start  = event.frame_idx

    def _handle_pass(self, event: GameEvent) -> None:
        # Quien pasó
        self._get_or_create(event.player_id, event.team).passes_made += 1
        # Quien recibió
        if event.receiver_id is not None:
            recv_team = self._team_cache.get(event.receiver_id)
            self._get_or_create(event.receiver_id, recv_team).passes_received += 1

    def _close_possession(self, frame_idx: int) -> None:
        if self._current_possessor is not None:
            frames = frame_idx - self._possession_start
            ps = self._get_or_create(self._current_possessor)
            ps.possession_frames += frames
        self._current_possessor = None

    def _team_totals(self) -> dict:
        totals = {}
        for team_id, label in ((0, "A"), (1, "B")):
            team_ps = [p for p in self.players.values() if p.team == team_id]
            if not team_ps:
                continue
            shots = sum(p.shots for p in team_ps)
            basks = sum(p.baskets for p in team_ps)
            totals[label] = {
                "possessions":        sum(p.possessions for p in team_ps),
                "possession_seconds": round(sum(p.possession_frames for p in team_ps) / self.fps, 1),
                "passes":             sum(p.passes_made for p in team_ps),
                "shots":              shots,
                "baskets":            basks,
                "shooting_pct":       round(basks / shots * 100, 1) if shots > 0 else 0.0,
            }
        return totals
