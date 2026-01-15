from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

class Environment(ABC):
    """
    Classe base abstrata para ambientes de simulação.
    Define o contrato de interação entre o Simulador e o mundo físico/lógico.
    """

    def __init__(self, nome: str = "Ambiente"):
        self.nome = nome

    @abstractmethod
    def observacaoPara(self, agente: Any) -> Any:
        """
        Gera a observação do estado do mundo específica para um determinado agente.
        Simula os sensores do agente (ex: visão local, coordenadas).
        """
        ...

    @abstractmethod
    def agir(self, accao: Any, agente: Any) -> tuple[float, bool, dict]:
        """
        Executa a ação de um agente no ambiente e calcula as consequências.
        
        Retorna:
            - recompensa (float): Valor numérico do feedback.
            - terminou (bool): Se o episódio chegou ao fim.
            - info (dict): Dados auxiliares para debug ou logs.
        """
        ...

    def atualizacao(self) -> None:
        """
        Executa a lógica interna do ambiente (física, movimento de obstáculos, etc.).
        Chamado a cada ciclo de simulação.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reinicia o ambiente para o seu estado inicial.
        Deve ser chamado no início de cada novo episódio.
        """
        ...