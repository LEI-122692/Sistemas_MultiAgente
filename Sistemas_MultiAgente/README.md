# Simulador de Sistemas Multi-Agente

Este projeto implementa uma plataforma de simulação para agentes autónomos, desenvolvida no âmbito da Unidade Curricular de **Agentes Autónomos**. 

O objetivo é comparar o desempenho de agentes reativos (Greedy), agentes de Aprendizagem por Reforço (Q-Learning) e agentes Evolutivos (Algoritmos Genéticos com Neuroevolução) em ambientes de complexidade variável.

## 📋 Funcionalidades Principais

* **Ambientes:**
    * **Farol:** Espaço contínuo convexo (sem obstáculos complexos), ideal para validação de baselines.
    * **Labirinto:** Espaço discreto não-convexo com armadilhas (*dead-ends*), exigindo planeamento ou exploração avançada.
* **Algoritmos:**
    * **Greedy (Guloso):** Heurística de distância direta.
    * **Q-Learning:** Aprendizagem por Reforço tabular (*Model-Free*).
    * **Algoritmo Genético:** Neuroevolução de pesos de uma rede neuronal. Inclui mecanismo de **Novelty Search** para superar mínimos locais.

---

## 🛠️ Instalação e Requisitos

### Pré-requisitos
* Python 3.8 ou superior.
* Biblioteca `matplotlib` (para geração de gráficos).

### Configuração
1.  Clone este repositório:
    
    git clone <INSIRA_AQUI_O_LINK_DO_SEU_GITHUB>
    cd Sistemas_MultiAgente
    

2.  Instale as dependências (se necessário):

    pip install matplotlib numpy
   

---

## 🚀 Guia de Execução (Como Testar)

O projeto possui um ponto de entrada centralizado. Para iniciar o simulador, execute o ficheiro `main.py` na raiz do projeto.

python main.py

### Menu Principal
Ao iniciar, será apresentado o seguinte menu interativo no terminal:

#### GRUPO 1: Q-LEARNING (Aprendizagem por Reforço)
* **Opção 1 - Farol (Treino Completo):**
    * Treina um agente Q-Learning no ambiente simples (Farol).
    * Executa testes comparativos contra um agente Greedy (Baseline).
* **Opção 2 - Labirinto (Treino Completo):**
    * Treina um agente Q-Learning durante 25.000 episódios no Labirinto.
    * Gera a Q-Table (`qtable_labirinto.pkl`) e exporta métricas para CSV.
* **Opção 3 - Labirinto (Demo Visual):**
    * **Nota:** Requer que a Opção 2 tenha sido executada previamente (para existir uma Q-Table).
    * Mostra uma simulação passo-a-passo no terminal do agente treinado a resolver o labirinto.

#### GRUPO 2: ALGORITMOS GENÉTICOS (Evolutivo)
* **Opção 4 - Farol (Evolução Simples):**
    * Prova de conceito de Neuroevolução num ambiente simples.
* **Opção 5 - Labirinto (Deep + Novelty Search):**
    * Executa a evolução de uma população de 150 agentes.
    * Utiliza *Novelty Search* para incentivar a exploração e encontrar a saída sem mapas prévios.

---

## 📊 Análise de Resultados

Todas as simulações geram ficheiros de log em formato `.csv` na raiz do projeto (ex: `resultados_labirinto.csv`, `resultados_farol.csv`).

Para visualizar os gráficos de desempenho (Curvas de Aprendizagem e Taxas de Sucesso), execute o script de análise dedicado:


python analise.py

Isto irá gerar e apresentar automaticamente janelas com:

1.Comparação no Labirinto: Q-Learning vs Genético (Sucesso e Recompensas).

2.Comparação no Farol: Greedy vs Q-Learning vs Genético.

3.Curvas de Evolução: Progresso da Fitness e da Novidade ao longo das gerações.

Estrutura do Projeto
Agents/: Contém a implementação das classes dos Agentes ("Cérebros").
qlearning_labirinto.py / qlearning_farol.py: Agentes de Aprendizagem por Reforço.
genetic_agent.py: Agente base com Rede Neuronal.
greedy_farol.py: Agente de controlo (heurística simples).
Core/: Motor de simulação (Simulator.py) e gestão do ciclo de vida.
Envs/: Definição da física e regras dos Ambientes (LabirintoEnvironment, FarolEnvironment).
Experiments/: Scripts que orquestram os treinos e testes (chamados pelo menu).
main.py: Ponto de entrada da aplicação (Interface CLI).
analise.py: Módulo de processamento de dados e geração de gráficos matplotlib.

Autores:

Miguel Nunes (122707) 
Ricardo Lourenço (122692)