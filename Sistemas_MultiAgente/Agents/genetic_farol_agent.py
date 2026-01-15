from __future__ import annotations
import random
from typing import List, Optional, Dict, Any
from Core import Agent, Accao

class GeneticFarolAgent(Agent):
    """
    Implementa um agente genético com arquitetura linear (Perceptrão Simples).
    Otimizado para o ambiente Farol, mapeando inputs espaciais diretamente em ações.
    """

    def __init__(self, agent_id: int, genoma: Optional[List[float]] = None):
        """
        Inicializa o agente.

        Args:
            agent_id (int): Identificador único.
            genoma (List[float], opcional): Pesos da rede. Se None, inicializa aleatoriamente.
        """
        super().__init__(agent_id)
        
        # --- Arquitetura da Rede ---
        # Inputs: Pos X, Pos Y, Dist X, Dist Y, Bias
        self.n_inputs = 5 
        self.n_outputs = 4 
        self.n_genes = self.n_inputs * self.n_outputs
        
        if genoma is None:
            # Inicialização uniforme entre -1.0 e 1.0
            self.genoma = [random.uniform(-1.0, 1.0) for _ in range(self.n_genes)]
        else:
            self.genoma = genoma

    def age(self, env_info: Optional[Dict[str, Any]] = None) -> Accao:
        """
        Calcula a ação baseada no estado do ambiente (Forward Pass).

        Args:
            env_info (Dict[str, Any]): Dicionário com coordenadas absolutas (x, y, farol_x, farol_y).

        Returns:
            Accao: Ação com maior ativação na saída da rede.
        """
        # Fallback de segurança para falta de informação
        if env_info is None:
            opcoes = [Accao.CIMA, Accao.BAIXO, Accao.ESQUERDA, Accao.DIREITA]
            return Accao(tipo="mover", direcao=random.choice(opcoes))

        ax = env_info["x"]
        ay = env_info["y"]
        fx = env_info["farol_x"]
        fy = env_info["farol_y"]

        # 1. Normalização dos Inputs (Assumindo grelha 20x20)
        norm_x = ax / 20.0
        norm_y = ay / 20.0
        
        # Vetores de Direção
        dx = 0
        if fx > ax: dx = 1
        elif fx < ax: dx = -1
        
        dy = 0
        if fy > ay: dy = 1
        elif fy < ay: dy = -1
        
        # Vetor de Entrada (+ Bias)
        inputs = [norm_x, norm_y, dx, dy, 1.0]
        
        # 2. Processamento Linear (Sem camada oculta)
        outputs = [0.0] * self.n_outputs
        peso_idx = 0
        
        for i in range(self.n_outputs):
            valor = 0.0
            for val_in in inputs:
                valor += val_in * self.genoma[peso_idx]
                peso_idx += 1
            outputs[i] = valor

        # 3. Seleção da Ação (Argmax)
        maior_valor = -float('inf')
        acao_idx = 0
        for i in range(self.n_outputs):
            if outputs[i] > maior_valor:
                maior_valor = outputs[i]
                acao_idx = i
                
        opcoes = [Accao.CIMA, Accao.BAIXO, Accao.ESQUERDA, Accao.DIREITA]
        return Accao(tipo="mover", direcao=opcoes[acao_idx])

    def mutar(self, taxa: float = 0.1, forca: float = 0.5) -> None:
        """
        Aplica mutação gaussiana ao genoma.

        Args:
            taxa (float): Probabilidade de mutação por gene.
            forca (float): Desvio padrão da mutação.
        """
        novo_genoma = []
        for gene in self.genoma:
            if random.random() < taxa:
                gene += random.gauss(0, forca)
                # Clamping para manter estabilidade numérica [-5.0, 5.0]
                gene = max(-5.0, min(5.0, gene))
            novo_genoma.append(gene)
        self.genoma = novo_genoma