from __future__ import annotations

from Core import Simulator
from Envs import LabirintoEnvironment
from Agents import QLearningLabirintoAgent
from Metrics import MetricsLogger


def correr_treino_labirinto(
    num_episodios_treino: int = 50000,
    num_episodios_teste: int = 50,
    max_passos: int = 1000,
    caminho_csv: str = "resultados_qlearning_labirinto.csv",
) -> None:
    """
    Executa o ciclo completo de Treino e Validação do Q-Learning no Labirinto.
    Inclui fases de treino, teste com o agente treinado e teste com agente 'baseline'.
    """

    # --- Inicialização ---
    env = LabirintoEnvironment()
    agent = QLearningLabirintoAgent(
        agent_id=1,
        alpha=0.1,
        gamma=0.9,
        epsilon=0.3,
    )
    logger = MetricsLogger()

    
    # --- FASE 1: TREINO (QL) ---
    
    experiencia_treino = "Labirinto_Treino"
    print(f"### {experiencia_treino} – MODO APRENDIZAGEM ###")

    sim_treino = Simulator(
        ambiente=env,
        agentes=[agent],
        nome_experiencia=experiencia_treino,
        num_episodios=num_episodios_treino,
        max_passos=max_passos,
        modo_aprendizagem=True,
        logger=logger,
    )
    sim_treino.executa()

    
    # --- FASE 2: TESTE (QL TREINADO) ---
    
    experiencia_teste_ql = "Labirinto_Teste_QL"
    print(f"\n### {experiencia_teste_ql} – MODO TESTE (QL Treinado) ###")

    # Política fixa: Aproveitamento Puro (sem exploração)
    agent.epsilon = 0.0

    sim_teste_ql = Simulator(
        ambiente=env,
        agentes=[agent],
        nome_experiencia=experiencia_teste_ql,
        num_episodios=num_episodios_teste,
        max_passos=max_passos,
        modo_aprendizagem=False,
        logger=logger,
    )
    sim_teste_ql.executa()


    # --- FASE 3: TESTE DE CONTROLO (BASELINE / NÃO TREINADO) ---
    
    # Cria uma nova instância do agente (Q-Table vazia) para comparação.
    agent_nao_treinado = QLearningLabirintoAgent(agent_id=2, epsilon=0.0) 

    experiencia_teste_nao_treinado = "Labirinto_Teste_Nao_Treinado"
    print(f"\n### {experiencia_teste_nao_treinado} – PROVA DE VALOR (Baseline) ###")

    sim_teste_nao_treinado = Simulator(
        ambiente=env,
        agentes=[agent_nao_treinado], 
        nome_experiencia=experiencia_teste_nao_treinado,
        num_episodios=num_episodios_teste,
        max_passos=max_passos,
        modo_aprendizagem=False,
        logger=logger,
    )
    sim_teste_nao_treinado.executa() 

    
    # --- Gravação de Resultados ---
    
    logger.guardar_csv(caminho_csv)
    print(f"[CSV] Métricas guardadas em: {caminho_csv}")

    agent.save_qtable("qtable_labirinto.pkl")
    print("[Q] Q-table do labirinto guardada em qtable_labirinto.pkl")


if __name__ == "__main__":
    correr_treino_labirinto()