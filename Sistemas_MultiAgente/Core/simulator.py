from __future__ import annotations
from typing import List, Optional

from .agent import Agent
from .environment import Environment
from Metrics import EpisodioStats, MetricsLogger


class Simulator:
    """
    Simulador genérico de Sistemas Multi-Agente (SMA).
    Orquestra o ciclo de vida dos agentes e ambientes em múltiplos episódios,
    gerindo o ciclo Perceção -> Deliberação -> Ação -> Aprendizagem.
    """

    def __init__(
        self,
        ambiente: Environment,
        agentes: List[Agent],
        nome_experiencia: str = "Experiencia",
        num_episodios: int = 1,
        max_passos: int = 100,
        modo_aprendizagem: bool = True,
        gamma_default: float = 0.99,
        logger: Optional[MetricsLogger] = None,
    ) -> None:
        self.ambiente = ambiente
        self.agentes = agentes
        self.nome_experiencia = nome_experiencia
        self.num_episodios = num_episodios
        self.max_passos = max_passos
        self.modo_aprendizagem = modo_aprendizagem
        self.gamma_default = gamma_default
        self.logger = logger

    @classmethod
    def cria(cls, nome_ficheiro_parametros: str) -> "Simulator":
        """Factory method para criação via ficheiro (não implementado)."""
        raise NotImplementedError(
            "Neste projeto o simulador é criado diretamente em código."
        )

    def lista_agentes(self) -> List[Agent]:
        return self.agentes

    def executa(self) -> List[EpisodioStats]:
        """Executa todos os episódios configurados e gere os logs."""
        resultados: List[EpisodioStats] = []

        print(f"### Iniciando Simulação: {self.nome_experiencia} ###")
        
        for ep in range(1, self.num_episodios + 1):
            
            stats = self._executa_episodio(ep)
            resultados.append(stats)

            print(
                f"  Episódio {stats.episodio} terminou em {stats.passos} passos "
                f"(recompensa total: {stats.recompensa_total:.2f})"
            )

            if self.logger is not None:
                self.logger.registar(stats)

        print(f"### {self.nome_experiencia} concluído. Total de {len(resultados)} episódios. ###")
        
        return resultados

    def _executa_episodio(self, numero_episodio: int) -> EpisodioStats:
        self.ambiente.reset()

        passos = 0
        recompensa_total = 0.0
        recompensa_descontada = 0.0
        fator = 1.0

        if self.agentes:
            gamma = getattr(self.agentes[0], "gamma", self.gamma_default)
        else:
            gamma = self.gamma_default

        sucesso = 0
        terminou = False

        for agente in self.agentes:
            obs = self.ambiente.observacaoPara(agente)
            agente.observacao(obs)

        for passo in range(1, self.max_passos + 1):
            passos = passo

            for agente in self.agentes:
                accao = agente.age()
                recompensa, done, _info = self.ambiente.agir(accao, agente)

                recompensa_total += recompensa
                recompensa_descontada += fator * recompensa
                fator *= gamma

                obs = self.ambiente.observacaoPara(agente)
                agente.observacao(obs)

                if self.modo_aprendizagem:
                    if hasattr(agente, "avaliacaoEstadoAtual"):
                        agente.avaliacaoEstadoAtual(recompensa)
                    elif hasattr(agente, "avaliacao_estado_atual"):
                        agente.avaliacao_estado_atual(recompensa)

                if done:
                    terminou = True
                    sucesso = 1
                    break

            if terminou:
                break

            if hasattr(self.ambiente, "atualizacao"):
                self.ambiente.atualizacao()

        for agente in self.agentes:
            if hasattr(agente, "fim_de_episodio"):
                agente.fim_de_episodio()

        return EpisodioStats(
            experiencia=self.nome_experiencia,
            episodio=numero_episodio,
            passos=passos,
            recompensa_total=recompensa_total,
            recompensa_descontada=recompensa_descontada,
            sucesso=sucesso,
        )