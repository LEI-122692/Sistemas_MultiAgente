from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional

# Alias para flexibilidade na definição do estado (pode ser tupla, imagem, vetor, etc.)
Observation = Any

class Agent(ABC):
    """
    Classe base abstrata que define a interface fundamental de um Agente Autónomo.
    Gere o ciclo de vida de perceção, deliberação e atuação.
    """

    def __init__(self, agent_id: int):
        self.id = agent_id
        self._ultima_observacao: Optional[Observation] = None

    @classmethod
    def cria(cls, nome_ficheiro_parametros: str) -> "Agent":
        """
        Factory method para instanciar agentes a partir de ficheiros de configuração.
        Deve ser sobreposto pelas subclasses concretas.
        """
        raise NotImplementedError("O método 'cria' deve ser implementado na subclasse.")

    def observacao(self, obs: Observation) -> None:
        """
        [Perceção] Recebe e armazena a observação atual do estado do ambiente.
        """
        self._ultima_observacao = obs

    @abstractmethod
    def age(self):
        """
        [Deliberação] Implementa a política do agente.
        Deve retornar a ação a ser executada no ambiente.
        """
        ...

    def avaliacao_estado_atual(self, recompensa: float) -> None:
        """
        [Aprendizagem] Recebe o feedback (recompensa) da última ação.
        Utilizado por agentes que implementam Aprendizagem por Reforço.
        """
        pass

    def instala(self, sensor: Any) -> None:
        """
        [Configuração] Interface para instalação de sensores ou atuadores no agente.
        """
        pass

    def comunica(self, mensagem: str, de_agente: "Agent") -> None:
        """
        [Comunicação] Canal para receção de mensagens de outros agentes (Sistemas Multi-Agente).
        """
        pass

    def fim_de_episodio(self) -> None:
        """
        Hook executado no final de cada episódio.
        Útil para limpeza de memória, logs ou decaimento de parâmetros (ex: epsilon).
        """
        pass