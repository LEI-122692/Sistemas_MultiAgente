from __future__ import annotations
from typing import List, Iterable
import csv

from .episodio_stats import EpisodioStats

class MetricsLogger:
    """
    Gestor de persistência de métricas.
    Acumula estatísticas de execução (EpisodioStats) e exporta para formato CSV.
    """

    def __init__(self) -> None:
        self._episodios: List[EpisodioStats] = []

    def registar(self, stats: EpisodioStats) -> None:
        """Adiciona um registo de episódio ao buffer."""
        self._episodios.append(stats)

    def registar_varios(self, stats_list: Iterable[EpisodioStats]) -> None:
        """Adiciona múltiplos registos ao buffer."""
        for e in stats_list:
            self.registar(e)

    def limpar(self) -> None:
        """Limpa o buffer de episódios."""
        self._episodios.clear()

    def guardar_csv(self, caminho: str) -> None:
        """
        Escreve os dados acumulados num ficheiro CSV.
        Formata valores flutuantes para garantir legibilidade (2 e 6 casas decimais).
        """
        with open(caminho, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Escreve o cabeçalho
            writer.writerow([
                "experiencia",
                "episodio",
                "passos",
                "recompensa_total",
                "recompensa_descontada",
                "sucesso",
            ])

            # Escreve as linhas de dados
            for e in self._episodios:
                writer.writerow([
                    e.experiencia,
                    e.episodio,
                    e.passos,
                    f"{e.recompensa_total:.2f}",
                    f"{e.recompensa_descontada:.6f}",
                    e.sucesso,
                ])

        print(f"[CSV] Métricas guardadas em: {caminho}")