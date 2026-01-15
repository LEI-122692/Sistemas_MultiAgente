from __future__ import annotations
import random
import math
from typing import List, Optional
from Core import Agent, Accao

class GeneticAgent(Agent):
    """
    Implementa um agente controlado por uma Rede Neuronal Artificial (Feedforward).
    
    A arquitetura inclui uma camada oculta (Hidden Layer) que permite ao agente
    aprender fronteiras de decisão não-lineares, essenciais para navegação
    com obstáculos complexos.
    """

    def __init__(self, agent_id: int, genoma: Optional[List[float]] = None):
        """
        Inicializa o agente genético.

        Args:
            agent_id (int): Identificador único do agente.
            genoma (List[float], opcional): Pesos pré-definidos da rede. 
                                            Se None, inicia com pesos aleatórios.
        """
        super().__init__(agent_id)
        
        # --- Arquitetura da Rede Neuronal ---
        # Inputs: 4 sensores de parede + 2 GPS (agente) + 2 GPS (alvo) + 1 Bias
        self.n_inputs = 9 
        self.n_hidden = 6  # Neurónios na camada oculta
        self.n_outputs = 4 # Ações possíveis (Cima, Baixo, Esq, Dir)
        
        # Cálculo do número total de pesos (genes)
        # Estrutura: (Input -> Hidden) + (Hidden -> Output)
        self.n_genes = (self.n_inputs * self.n_hidden) + (self.n_hidden * self.n_outputs)
        
        # Inicialização do Genoma
        if genoma is None:
            # Inicialização de pesos aleatória (distribuição uniforme entre -1 e 1)
            self.genoma = [random.uniform(-1.0, 1.0) for _ in range(self.n_genes)]
        else:
            self.genoma = genoma

    def age(self, sensores_parede: Optional[List[int]] = None) -> Accao:
        """
        Executa o 'Forward Pass' da rede neuronal para decidir a próxima ação.

        Args:
            sensores_parede (List[int]): Leitura dos sensores de proximidade.

        Returns:
            Accao: A ação escolhida pela rede (maior ativação na saída).
        """
        if sensores_parede is None:
            sensores_parede = [0, 0, 0, 0]

        # Decomposição da observação (x_agente, y_agente, x_alvo, y_alvo)
        obs = self._ultima_observacao
        ax, ay, sx, sy = obs
        
        # 1. Pré-processamento e Normalização dos Inputs
        # Assumindo grelha de dimensão aprox. 15x15 para normalização
        norm_x = ax / 15.0
        norm_y = ay / 15.0
        
        # Vetor de direção relativa ao objetivo
        dx = 1 if sx > ax else (-1 if sx < ax else 0)
        dy = 1 if sy > ay else (-1 if sy < ay else 0)
        
        # Montagem do vetor de entrada (+1.0 para o Bias)
        inputs = [norm_x, norm_y, dx, dy] + sensores_parede + [1.0]
        
        # 2. Processamento da Camada Oculta (Input -> Hidden)
        hidden_vals = [0.0] * self.n_hidden
        gene_idx = 0
        
        for h in range(self.n_hidden):
            soma = 0.0
            for inp in inputs:
                soma += inp * self.genoma[gene_idx]
                gene_idx += 1
            
            # Função de Ativação: Tangente Hiperbólica (Tanh)
            hidden_vals[h] = math.tanh(soma)
            
        # 3. Processamento da Camada de Saída (Hidden -> Output)
        outputs = [0.0] * self.n_outputs
        
        for o in range(self.n_outputs):
            soma = 0.0
            for h in range(self.n_hidden):
                soma += hidden_vals[h] * self.genoma[gene_idx]
                gene_idx += 1
            
            outputs[o] = soma # Saída linear (raw scores)

        # 4. Seleção da Ação (Argmax)
        maior_valor = -float('inf') # Garante que qualquer valor real será maior
        acao_idx = 0
        
        for i in range(self.n_outputs):
            if outputs[i] > maior_valor:
                maior_valor = outputs[i]
                acao_idx = i
                
        # Mapeamento do índice para a Ação
        opcoes = [Accao.CIMA, Accao.BAIXO, Accao.ESQUERDA, Accao.DIREITA]
        return Accao(tipo="mover", direcao=opcoes[acao_idx])

    def mutar(self, taxa: float = 0.1, forca: float = 0.5) -> None:
        """
        Aplica mutação gaussiana aos pesos da rede neuronal.

        Args:
            taxa (float): Probabilidade de um gene (peso) sofrer mutação.
            forca (float): Desvio padrão da mutação gaussiana.
        """
        novo_genoma = []
        
        for gene in self.genoma:
            if random.random() < taxa:
                # Aplica ruído gaussiano
                gene += random.gauss(0, forca)
                # Clamping: Limita os pesos ao intervalo [-10, 10] para estabilidade
                gene = max(-10.0, min(10.0, gene))
            
            novo_genoma.append(gene)
            
        self.genoma = novo_genoma