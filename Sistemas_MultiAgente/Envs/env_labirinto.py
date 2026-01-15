from __future__ import annotations
from typing import Tuple, Dict

from Core import Environment, Accao
from .mapa_labirinto import MapaLabirinto

class LabirintoEnvironment(Environment):
    """
    Ambiente complexo de Labirinto com obstáculos.
    """

    def __init__(self):
        super().__init__(nome="Labirinto")
        self.map = MapaLabirinto()
        self.reset()

    def reset(self) -> None:
        """Coloca o agente no início e escolhe uma saída aleatória."""
        self.agent_x = 1
        self.agent_y = 1
        self.saida_x, self.saida_y = self.map.saida_aleatoria()

    def observacaoPara(self, agente) -> Tuple[int, int, int, int]:
        """
        Retorna o estado global: (pos_agente, pos_saida).
        """
        return (self.agent_x, self.agent_y, self.saida_x, self.saida_y)

    def agir(self, accao: Accao, agente) -> Tuple[float, bool, Dict]:
        """
        Processa movimento, colisões com paredes e verifica vitória.
        """
        mapeamento = {
            Accao.CIMA: (0, -1),
            Accao.BAIXO: (0, 1),
            Accao.ESQUERDA: (-1, 0),
            Accao.DIREITA: (1, 0),
        }

        if accao.direcao not in mapeamento:
            return -10.0, False, {}

        dx, dy = mapeamento[accao.direcao]
        nx = self.agent_x + dx
        ny = self.agent_y + dy

        # 1. Colisão com Parede
        if self.map.is_parede(nx, ny):
            return -5.0, False, {}

        # 2. Movimento Válido
        self.agent_x = nx
        self.agent_y = ny

        # 3. Sucesso
        if self.map.is_saida(nx, ny):
            return 100.0, True, {}

        # 4. Passo Normal
        return -1.0, False, {}

    def atualizacao(self) -> None:
        pass