from typing import Dict, Any, Hashable
from .qlearning_base import QLearningAgentBase

class QLearningFarolAgent(QLearningAgentBase):
    """
    Agente Q-Learning especializado para o ambiente Farol.
    O estado é definido pelo vetor de distância relativa (dx, dy).
    """

    def processar_estado(self, obs: Dict[str, Any]) -> Hashable:
        """
        Extrai (dx, dy) da observação para usar como chave na Q-Table.
        """
        return (obs["dx"], obs["dy"])