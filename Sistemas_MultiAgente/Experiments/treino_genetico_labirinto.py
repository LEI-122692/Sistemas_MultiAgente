from __future__ import annotations
import random
from typing import Set, Tuple, List

# Imports do Core e Ambiente
from Envs import LabirintoEnvironment
from Agents.genetic_agent import GeneticAgent
from Metrics import MetricsLogger, EpisodioStats

# --- Hiperparâmetros de Otimização ---
POPULACAO = 150       # Dimensão da população
GERACOES = 100        # Número de gerações
MAX_PASSOS = 150      # Timeout por indivíduo
ELITISMO = 15         # Número de indivíduos preservados (Top X)
BONUS_NOVIDADE = 40.0 # Recompensa por explorar células novas
TAXA_MUTACAO = 0.1
FORCA_MUTACAO = 0.5

def obter_sensores(env: LabirintoEnvironment) -> List[int]:
    """
    Simula sensores de proximidade (Lidar) lendo o mapa diretamente.
    Retorna: [Cima, Baixo, Esquerda, Direita] onde 1=Parede, 0=Livre.
    """
    x, y = env.agent_x, env.agent_y
    m = env.map
    
    return [
        1 if m.is_parede(x, y-1) else 0, # Cima
        1 if m.is_parede(x, y+1) else 0, # Baixo
        1 if m.is_parede(x-1, y) else 0, # Esq
        1 if m.is_parede(x+1, y) else 0  # Dir
    ]

def correr_treino_genetico_labirinto():
    """
    Executa o Algoritmo Genético com Novelty Search no ambiente Labirinto.
    
    Destaques:
    - Mapa Fixo: Garante que a evolução resolve um problema estático.
    - Novelty Search: Recompensa agentes que visitam coordenadas inéditas.
    - Fitness Híbrida: Combina Recompensa (Objetivo) + Novidade (Exploração).
    """
    print(f"\n=== Iniciando Evolução Labirinto (Novelty Search) ===")
    
    # 1. Configuração do Ambiente e Logger
    logger = MetricsLogger()
    env = LabirintoEnvironment()
    env.reset()
    
    # Fixação do Cenário (Essencial para convergência do AG)
    start_x, start_y = 1, 1
    
    # Tenta definir um objetivo desafiante (canto oposto)
    try:
        if not env.map.is_parede(11, 11):
            env.saida_x, env.saida_y = 11, 11
        elif not env.map.is_parede(11, 10):
            env.saida_x, env.saida_y = 11, 10
    except Exception:
        pass # Fallback para a saída aleatória definida no reset()
        
    goal_x, goal_y = env.saida_x, env.saida_y
    
    print(f"--> Mapa: Início({start_x},{start_y}) -> Saída({goal_x},{goal_y})")
    print(f"--> Config: Pop={POPULACAO}, Gens={GERACOES}, Elitismo={ELITISMO}")

    # Arquivo de Novidade (Memória Global de Exploração)
    arquivo_novidade: Set[Tuple[int, int]] = set()
    
    # Inicialização da População (Aleatória)
    populacao = [GeneticAgent(i) for i in range(POPULACAO)]

    # --- Ciclo Evolutivo ---
    for g in range(1, GERACOES + 1):
        scores = []
        sucessos_nesta_geracao = 0
        
        for agente in populacao:
            # 2. Configuração do Episódio
            env.agent_x = start_x
            env.agent_y = start_y
            env.saida_x = goal_x
            env.saida_y = goal_y
            
            # Reset da memória do agente
            agente._ultima_observacao = None
            
            # Perceção Inicial
            obs = env.observacaoPara(agente)
            agente.observacao(obs)
            
            caminho_percorrido = set()
            caminho_percorrido.add((start_x, start_y))
            
            recompensa_acumulada = 0
            chegou = False
            
            # 3. Simulação (Vida do Agente)
            for _ in range(MAX_PASSOS):
                # A. Leitura de Sensores
                sensores = obter_sensores(env)
                
                # B. Decisão (Forward Pass)
                accao = agente.age(sensores_parede=sensores)
                
                # C. Execução
                r, done, _ = env.agir(accao, agente)
                
                # Ajuste de Recompensa (Shaping): Suavizar penalidade de parede
                if r == -5: r = -2 
                
                # D. Atualização
                obs = env.observacaoPara(agente)
                agente.observacao(obs)
                
                recompensa_acumulada += r
                
                # Registo de Exploração Local
                pos = (env.agent_x, env.agent_y)
                caminho_percorrido.add(pos)
                
                if done and r > 0: 
                    chegou = True
                    sucessos_nesta_geracao += 1
                    recompensa_acumulada += 2000 # Grande bónus por sucesso
                    break
            
            # 4. Cálculo do Fitness (Avaliação)
            
            # 4.1. Fator Novidade
            novidade = 0
            for pos in caminho_percorrido:
                if pos not in arquivo_novidade:
                    novidade += BONUS_NOVIDADE
                    arquivo_novidade.add(pos)
            
            # 4.2. Fator Distância (Heurística de Orientação)
            dist = abs(goal_x - env.agent_x) + abs(goal_y - env.agent_y)
            
            # 4.3. Função de Fitness Composta
            # Fitness = Recompensa + Exploração - CustoDistância
            fitness = recompensa_acumulada + novidade - (dist * 5)
            
            scores.append((fitness, agente, chegou))

        # 5. Estatísticas e Logs
        scores.sort(key=lambda x: x[0], reverse=True) # Ordenar (Melhor -> Pior)
        melhor_fit = scores[0][0]
        
        print(f"Gen {g:03d} | Fit: {int(melhor_fit)} | Sucesso: {sucessos_nesta_geracao}/{POPULACAO} | Explorados: {len(arquivo_novidade)}")
        
        stats = EpisodioStats(
            experiencia="Gen_Labirinto_Novelty",
            episodio=g,
            passos=len(arquivo_novidade), # Guardamos nº células exploradas no campo 'passos'
            recompensa_total=melhor_fit,
            recompensa_descontada=0.0,
            sucesso=sucessos_nesta_geracao
        )
        logger.registar(stats)
        logger.guardar_csv("resultados_genetico_labirinto.csv")

        # Critério de Convergência Antecipada
        if sucessos_nesta_geracao > POPULACAO * 0.95:
            print(">>> População convergiu (>95% sucesso). Parando treino.")
            break

        # 6. Reprodução (Nova Geração)
        nova_pop = []
        
        # 6.1. Elitismo
        elites = [s[1] for s in scores[:ELITISMO]]
        for e in elites:
            nova_pop.append(GeneticAgent(e.id, list(e.genoma)))
            
        # 6.2. Cruzamento e Mutação
        while len(nova_pop) < POPULACAO:
            # Seleção por Torneio (5 participantes)
            pool = random.sample(scores, 5)
            pai = max(pool, key=lambda x: x[0])[1]
            
            # Clonagem
            filho = GeneticAgent(len(nova_pop), list(pai.genoma))
            
            # Mutação
            filho.mutar(taxa=TAXA_MUTACAO, forca=FORCA_MUTACAO)
            nova_pop.append(filho)
            
        populacao = nova_pop

    print("=== Evolução Concluída ===")

if __name__ == "__main__":
    correr_treino_genetico_labirinto()