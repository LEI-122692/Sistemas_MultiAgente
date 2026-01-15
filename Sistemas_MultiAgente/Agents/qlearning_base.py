import random
from typing import Any, Dict, Tuple, Hashable, List

from Core import Agent, Accao

# Tipo para a chave da Q-table: (Estado, Ação) -> Q_Valor
EstadoQ = Tuple[Hashable, str]


class QLearningAgentBase(Agent):
    """
    Classe Base de Q-Learning. Implementa o núcleo do Aprendizagem por Reforço (RL).
    Responsável pela decisão (Epsilon-Greedy) e pela aprendizagem (Equação de Bellman).
    """

    def __init__(self, agent_id: int, alpha=0.1, gamma=0.90, epsilon=0.2):
        super().__init__(agent_id)

        # Parâmetros de calibração do RL:
        self.alpha = alpha      # Taxa de Aprendizagem (Learning Rate): Peso do novo conhecimento.
        self.gamma = gamma      # Fator de Desconto: Importância das recompensas futuras.
        self.epsilon = epsilon  # Taxa de Exploração: Probabilidade de escolher uma ação aleatória.

        # A Tabela Q (Q-Table) é a memória da política do agente, armazena Q(estado, acao).
        self.q: Dict[EstadoQ, float] = {}

        # Registos necessários para a transição (s1, a1) -> (r, s2) para a atualização.
        self._ultimo_estado = None
        self._ultima_acao = None


    # Métodos a implementar nas Subclasses

    def processar_estado(self, obs: Any) -> Hashable:
        """
        MÉTODO ABSTRATO: Converte a observação bruta (obs) na representação de ESTADO.
        """
        raise NotImplementedError

    def acoes_possiveis(self) -> List[str]:
        """Define o espaço de ações que o agente pode tomar (A)."""
        return [Accao.CIMA, Accao.BAIXO, Accao.ESQUERDA, Accao.DIREITA]

   
    # Ciclo de Vida do Agente
    
    def age(self) -> Accao:
        """
        [DELIBERAÇÃO] Seleciona a ação (a) usando a política Epsilon-Greedy.
        """
        estado = self.processar_estado(self._ultima_observacao)

        # Epsilon-Greedy: Decide entre Exploração (aleatório) e Explotação (melhor Q-valor).
        if random.random() < self.epsilon:
            # Exploração: Escolhe uma ação aleatória.
            direcao = random.choice(self.acoes_possiveis())
        else:
            # Explotação: Escolhe a ação com o maior valor esperado na Q-Table.
            direcao = self._melhor_acao(estado)

        # Guarda o par (s, a) atual para a atualização da Q-Table no próximo passo (s', a').
        self._ultimo_estado = estado
        self._ultima_acao = direcao

        return Accao(tipo="mover", direcao=direcao)

    def _melhor_acao(self, estado: Hashable) -> str:
        """
        Função auxiliar para encontrar a ação Greedy (Q-valor máximo).
        """
        # Cria tuplas (Q_valor, direcao) para todas as ações possíveis no estado atual.
        qs = [(self.q.get((estado, a), 0.0), a) for a in self.acoes_possiveis()]
        qs.sort(reverse=True) # Ordena do maior Q-valor para o menor
        return qs[0][1] # Devolve a direção correspondente ao Q-valor máximo.

    def avaliacaoEstadoAtual(self, recompensa: float):
        """
        [APRENDIZAGEM] Atualiza o Q-valor Q(s,a) usando a regra de Bellman (TD-Learning).
        Isto só é executado no Modo Aprendizagem.
        """
        if self._ultimo_estado is None or self._ultima_acao is None:
            # Ignorar o primeiro passo de cada episódio.
            return

        s1 = self._ultimo_estado      # Estado Anterior (s)
        a1 = self._ultima_acao        # Ação Tomada (a)
        s2 = self.processar_estado(self._ultima_observacao) # Novo Estado (s')

        # 1. Calcular o Valor Futuro Esperado: max_a' Q(s', a')
        max_q2 = max(self.q.get((s2, a), 0.0) for a in self.acoes_possiveis())

        # 2. Obter o Q-valor atual Q(s, a)
        antigo = self.q.get((s1, a1), 0.0)
        
        # 3. Regra de Atualização (Q-Learning):
        # Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max Q(s') - Q(s,a) ]
        novo = antigo + self.alpha * (recompensa + self.gamma * max_q2 - antigo)

        # 4. Atualizar a Q-table
        self.q[(s1, a1)] = novo

    def fim_de_episodio(self):
        """
        Hook chamado no final do episódio. Implementa o decaimento suave de epsilon (Annealing).
        """
        # Reduz epsilon ligeiramente (0.995) para favorecer a explotação ao longo do tempo.
        self.epsilon = max(0.01, self.epsilon * 0.995)
        
    
    # Persistência da Política (Modo Teste)
    
    def save_qtable(self, path: str) -> None:
        """Guarda a Q-table treinada (a política) no disco."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.q, f)

    def load_qtable(self, path: str) -> None:
        """Carrega a Q-table pré-treinada para o Modo de Teste/Avaliação."""
        import pickle
        with open(path, "rb") as f:
            self.q = pickle.load(f)