from __future__ import annotations
from typing import List, Tuple
import random

class MapaLabirinto:
    """
    Representação estática da grelha do Labirinto.
    Gere a lógica geométrica (paredes vs livres) e a colocação da saída.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rnd = random.Random(seed)

        # Definição da Grelha (0=Livre, 1=Parede) - Matriz 13x13
        self.grelha: List[List[int]] = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # 0
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1], # 1 (Início em 1,1)
            [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1], # 2
            [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], # 3
            [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1], # 4
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # 5
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], # 6
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # 7
            [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1], # 8
            [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # 9
            [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1], # 10
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # 11
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 12
        ]

        self.altura: int = len(self.grelha)
        self.largura: int = len(self.grelha[0])
        self._saida_x: int | None = None
        self._saida_y: int | None = None

    def dentro_limites(self, x: int, y: int) -> bool:
        """Verifica se (x,y) está dentro da matriz."""
        return 0 <= y < self.altura and 0 <= x < self.largura

    def is_parede(self, x: int, y: int) -> bool:
        """Retorna True se for parede ou estiver fora do mapa."""
        if not self.dentro_limites(x, y):
            return True
        return self.grelha[y][x] == 1

    def celulas_livres(self) -> List[Tuple[int, int]]:
        """Lista todas as coordenadas caminháveis (0)."""
        livres = []
        for j in range(self.altura):
            for i in range(self.largura):
                if self.grelha[j][i] == 0:
                    livres.append((i, j))
        return livres

    def saida_aleatoria(self) -> Tuple[int, int]:
        """
        Define uma nova posição de saída numa célula livre aleatória.
        Garante que a saída não coincide com a posição inicial (1,1).
        """
        livres = self.celulas_livres()
        if (1, 1) in livres:
            livres.remove((1, 1))

        if not livres:
            raise ValueError("Labirinto sem espaços livres!")

        self._saida_x, self._saida_y = self._rnd.choice(livres)
        return self._saida_x, self._saida_y

    def definir_saida_fixa(self, x: int, y: int) -> None:
        """Define manualmente a saída (útil para debug ou testes específicos)."""
        if not self.dentro_limites(x, y) or self.is_parede(x, y):
            raise ValueError("Célula inválida para saída.")
        self._saida_x, self._saida_y = x, y

    def saida_actual(self) -> Tuple[int, int]:
        """Devolve as coordenadas (x, y) da saída definida no momento."""
        if self._saida_x is None or self._saida_y is None:
            raise ValueError("Saída ainda não foi definida.")
        return self._saida_x, self._saida_y

    def is_saida(self, x: int, y: int) -> bool:
        """Verifica se a coordenada corresponde ao objetivo atual."""
        return self._saida_x == x and self._saida_y == y