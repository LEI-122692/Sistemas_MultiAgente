from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Accao:
    """
    Representa uma ação atómica realizada pelo agente.
    Estrutura imutável contendo o tipo de ação e direção opcional.
    """
    tipo: str
    direcao: str | None = None

    # Constantes de Direção
    CIMA = "cima"
    BAIXO = "baixo"
    ESQUERDA = "esquerda"
    DIREITA = "direita"