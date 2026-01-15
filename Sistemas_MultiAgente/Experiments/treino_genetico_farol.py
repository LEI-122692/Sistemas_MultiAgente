from __future__ import annotations
import random
from typing import List, Tuple

# Imports do Projeto
from Core import Agent
from Envs import FarolEnvironment
from Agents.genetic_farol_agent import GeneticFarolAgent
from Metrics import MetricsLogger, EpisodioStats

# --- Hiperparâmetros do Algoritmo Genético ---
POPULACAO = 50       # Número de indivíduos por geração
GERACOES = 20        # Total de gerações a evoluir
MAX_PASSOS = 30      # Timeout por episódio (impede loops infinitos)
ELITISMO = 5         # Número de melhores agentes preservados sem mutação
TAXA_MUTACAO = 0.1   # Probabilidade de mutação por gene
FORCA_MUTACAO = 0.5  # Desvio padrão da mutação gaussiana

def correr_treino_genetico_farol():
    """
    Executa o ciclo de evolução (Algoritmo Genético) para o ambiente Farol.
    
    Processo:
    1. Inicializa uma população aleatória.
    2. Avalia a fitness de cada agente (simulação).
    3. Regista as métricas (Melhor Fitness, Sucesso).
    4. Aplica Seleção (Torneio) e Reprodução (Cruzamento/Mutação).
    5. Repete por N gerações.
    """
    print(f"\n=== Iniciando Treino Genético: Farol ({GERACOES} Gerações) ===")
    
    logger = MetricsLogger()
    env = FarolEnvironment(tamanho=10)
    
    # Geração 0: População Aleatória
    populacao = [GeneticFarolAgent(i) for i in range(POPULACAO)]

    # Ciclo Evolutivo
    for g in range(1, GERACOES + 1):
        scores: List[Tuple[float, GeneticFarolAgent, bool]] = []
        
        # 1. Avaliação da População
        for agente in populacao:
            env.reset()
            recompensa_acumulada = 0
            chegou = False
            
            for _ in range(MAX_PASSOS):
                # Construção do vetor de inputs 
                info = {
                    "x": env.x, 
                    "y": env.y, 
                    "farol_x": env.farol_x, 
                    "farol_y": env.farol_y
                }
                
                # Ciclo Perceção-Ação
                accao = agente.age(env_info=info)
                r, done, _ = env.agir(accao, agente)
                
                recompensa_acumulada += r
                
                if done: 
                    chegou = True
                    recompensa_acumulada += 100 
                    break
            
            scores.append((recompensa_acumulada, agente, chegou))

        # 2. Estatísticas da Geração
        # Ordenar por Fitness (descendente)
        scores.sort(key=lambda x: x[0], reverse=True)
        
        melhor_fit = scores[0][0]
        n_sucessos = sum(1 for s in scores if s[2])
        
        print(f"Gen {g:02d} | Melhor Fit: {melhor_fit:.2f} | Taxa Sucesso: {n_sucessos}/{POPULACAO}")

        # 3. Registo de Métricas
        stats = EpisodioStats(
            experiencia="Farol_Genetico",
            episodio=g,          
            passos=0,           
            recompensa_total=melhor_fit,
            recompensa_descontada=0.0,
            sucesso=n_sucessos   #
        )
        logger.registar(stats)
        logger.guardar_csv("resultados_genetico_farol.csv") 

        # 4. Reprodução (Nova Geração)
        nova_pop = []
        
        # 4.1. Elitismo: Preserva os melhores
        elites = [s[1] for s in scores[:ELITISMO]]
        for e in elites:
            nova_pop.append(GeneticFarolAgent(e.id, list(e.genoma)))
            
        # 4.2. Seleção e Mutação (Preencher o resto da população)
        while len(nova_pop) < POPULACAO:
            # Seleção por Torneio (Tamanho 3)
            pool = random.sample(scores, 3)
            pai = max(pool, key=lambda x: x[0])[1]
            
            filho = GeneticFarolAgent(len(nova_pop), list(pai.genoma))
            filho.mutar(taxa=TAXA_MUTACAO, forca=FORCA_MUTACAO)
            nova_pop.append(filho)
            
        populacao = nova_pop

    print("=== Treino Genético Concluído ===")

if __name__ == "__main__":
    correr_treino_genetico_farol()