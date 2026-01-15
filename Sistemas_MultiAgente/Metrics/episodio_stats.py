from __future__ import annotations
from dataclasses import dataclass


@dataclass
class EpisodioStats:
    """
    Estatísticas de um episódio de uma experiência.

    Campos:
      - experiencia: nome da experiência (ex.: "Farol_Treino")
      - episodio: número do episódio
      - passos: número de passos até terminar ou max_passos
      - recompensa_total: soma das recompensas
      - recompensa_descontada: soma das recompensas com desconto
      - sucesso: 1 se atingiu o objetivo, 0 caso contrário
    """
    experiencia: str
    episodio: int
    passos: int
    recompensa_total: float
    recompensa_descontada: float
    sucesso: int
