from __future__ import annotations
from typing import Tuple

from Agents import QLearningFarolAgent, GreedyFarolAgent
from Envs import FarolEnvironment
from Metrics import EpisodioStats, MetricsLogger


def _correr_episodio_farol(
    env: FarolEnvironment,
    agent: QLearningFarolAgent | GreedyFarolAgent,  # Suporta ambos os tipos de agente
    experiencia_nome: str,
    numero_episodio: int,
    modo_aprendizagem: bool,
    max_passos: int,
) -> EpisodioStats:

    
    env.reset()
    passos = 0
    recompensa_total = 0.0
    recompensa_descontada = 0.0
    fator = 1.0     # para desconto
    
    gamma = getattr(agent, "gamma", 0.99)

    sucesso = 0
    terminou = False 

    # primeira observação
    obs = env.observacaoPara(agent)
    agent.observacao(obs)

    for passo in range(1, max_passos + 1):
        passos = passo

        accao = agent.age()
        recompensa, done, _info = env.agir(accao, agent)

        recompensa_total += recompensa
        recompensa_descontada += fator * recompensa
        fator *= gamma

        # nova observação
        obs = env.observacaoPara(agent)
        agent.observacao(obs)

        # Aprendizagem
        if modo_aprendizagem and isinstance(agent, QLearningFarolAgent):
            agent.avaliacaoEstadoAtual(recompensa)
        
        if done:
            terminou = True
            sucesso = 1
            break

        env.atualizacao()

    # Hook de fim de episódio
    if hasattr(agent, "fim_de_episodio"):
        agent.fim_de_episodio()

    print(
        f"  Episódio {numero_episodio} terminou em {passos} passos "
        f"(recompensa total: {recompensa_total:.2f})"
    )

    return EpisodioStats(
        experiencia=experiencia_nome,
        episodio=numero_episodio,
        passos=passos,
        recompensa_total=recompensa_total,
        recompensa_descontada=recompensa_descontada,
        sucesso=sucesso,
    )


def correr_treino_farol(
    num_episodios_treino: int = 200,
    num_episodios_teste_ql: int = 40,
    num_episodios_teste_greedy: int = 40,
    max_passos: int = 100,
    caminho_csv: str = "resultados_qlearning_farol.csv",
) -> None:
    """
    Corre as 3 fases de simulação (Treino QL, Teste QL, Teste Greedy)
    e garante que todas as métricas são registadas no Logger.
    """

    env = FarolEnvironment()
    ql_agent = QLearningFarolAgent(agent_id=1, alpha=0.1, gamma=0.99, epsilon=0.2)
    greedy_agent = GreedyFarolAgent(agent_id=2)
    logger = MetricsLogger()

    # ========================== 1. TREINO (QL) ========================== #
    experiencia_treino = "Farol_QL_Treino"
    print(f"### {experiencia_treino} – MODO APRENDIZAGEM ###")

    for ep in range(1, num_episodios_treino + 1):
        stats = _correr_episodio_farol(
            env,
            ql_agent, # Agente QL
            experiencia_nome=experiencia_treino,
            numero_episodio=ep,
            modo_aprendizagem=True,
            max_passos=max_passos,
        )
        logger.registar(stats) 

    # ========================== 2. TESTE (QL pré-treinado) ======================= #
    experiencia_teste_ql = "Farol_QL_Teste"
    print(f"\n### {experiencia_teste_ql} – MODO TESTE (QL pré-treinado) ###")

    ql_agent.epsilon = 0.0 # Desativa exploração para teste

    for ep in range(1, num_episodios_teste_ql + 1):
        stats = _correr_episodio_farol(
            env,
            ql_agent, # Agente QL
            experiencia_nome=experiencia_teste_ql,
            numero_episodio=ep,
            modo_aprendizagem=False, # Modo Teste
            max_passos=max_passos,
        )
        logger.registar(stats) 

    # ========================== 3. TESTE (Greedy) ======================= #
    experiencia_teste_greedy = "Farol_Greedy_Teste"
    print(f"\n### {experiencia_teste_greedy} – MODO TESTE (Política Greedy Fixa) ###")
    
    for ep in range(1, num_episodios_teste_greedy + 1):
        stats = _correr_episodio_farol(
            env,
            greedy_agent, # Agente Greedy
            experiencia_nome=experiencia_teste_greedy,
            numero_episodio=ep,
            modo_aprendizagem=False, # Não aprende
            max_passos=max_passos,
        )
        logger.registar(stats) 

    logger.guardar_csv(caminho_csv)
    print(f"[CSV] Métricas guardadas em: {caminho_csv}")