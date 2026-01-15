from typing import Any, Hashable, Tuple
from .qlearning_base import QLearningAgentBase

class QLearningLabirintoAgent(QLearningAgentBase):
    """
    Agente Q-Learning especializado para o ambiente Labirinto.
    O estado é definido pelas coordenadas absolutas do agente e do objetivo.
    """

    def processar_estado(self, obs: Any) -> Hashable:
        """
        Retorna a observação completa (tuplo de coordenadas) como estado.
        """
        return obs