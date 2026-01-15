from typing import Dict, Any
from Core import Agent, Accao

class GreedyFarolAgent(Agent):
    """
    Agente de política determinística (Greedy) para o ambiente Farol.
    
    Implementa uma heurística baseada na distância de Manhattan, priorizando
    o movimento no eixo com maior distância absoluta ao objetivo em cada passo.
    Utilizado como baseline para comparação de desempenho.
    """
    
    def age(self) -> Accao:
        """
        Decide a ação que maximiza a aproximação imediata ao alvo.
        """
        obs: Dict[str, Any] = self._ultima_observacao
        dx = obs["dx"]
        dy = obs["dy"]

        # Verifica se o agente já se encontra nas coordenadas do objetivo
        if dx == 0 and dy == 0:
            return Accao(tipo="nenhuma", direcao=None)
        
        # Seleção de eixo prioritário (Heurística)
        if abs(dx) >= abs(dy):
            # Movimento no Eixo X
            if dx > 0:
                return Accao(tipo="mover", direcao=Accao.DIREITA)
            elif dx < 0:
                return Accao(tipo="mover", direcao=Accao.ESQUERDA)
        
        # Movimento no Eixo Y (Executado se a distância vertical for dominante)
        if dy > 0:
            return Accao(tipo="mover", direcao=Accao.BAIXO)
        elif dy < 0:
            return Accao(tipo="mover", direcao=Accao.CIMA)

        # Fallback de segurança (não deve ser atingido em condições normais)
        return Accao(tipo="mover", direcao=Accao.CIMA)