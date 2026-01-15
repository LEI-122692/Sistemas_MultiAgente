from __future__ import annotations
from typing import Dict, Tuple

from Core import Accao, Environment, Agent

class FarolEnvironment(Environment):
    """
    Ambiente Simples 2D (Grelha) para o problema do Farol.
    O objetivo é minimizar a distância ao ponto alvo (Farol) fixo.
    """

    def __init__(self, tamanho: int = 10) -> None:
        self.N = tamanho
        self.reset()

    def reset(self) -> None:
        """Reinicia o agente e o farol nas posições padrão."""
        self.x = 1
        self.y = 1
        self.farol_x = self.N - 2
        self.farol_y = self.N - 2

    def observacaoPara(self, agente: Agent) -> Dict:
        """
        Retorna o vetor de distância relativa (dx, dy) como perceção.
        """
        dx = self.farol_x - self.x
        dy = self.farol_y - self.y
        return {"dx": dx, "dy": dy}

    def agir(self, accao: Accao, agente: Agent) -> Tuple[float, bool, Dict]:
        """
        Executa o movimento, garante limites da grelha e retorna recompensa.
        """
        # Movimento
        if accao.direcao == Accao.CIMA:
            self.y -= 1
        elif accao.direcao == Accao.BAIXO:
            self.y += 1
        elif accao.direcao == Accao.ESQUERDA:
            self.x -= 1
        elif accao.direcao == Accao.DIREITA:
            self.x += 1

        # Limites da Grelha
        self.x = max(0, min(self.N - 1, self.x))
        self.y = max(0, min(self.N - 1, self.y))

        # Recompensas
        if self.x == self.farol_x and self.y == self.farol_y:
            return 100.0, True, {}  # Sucesso
        
        return -1.0, False, {}  # Custo de passo

    def atualizacao(self) -> None:
        pass