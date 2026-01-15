from __future__ import annotations
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from Experiments import (
    correr_treino_farol,
    correr_treino_labirinto,
    demo_labirinto,
    correr_treino_genetico_farol,
    correr_treino_genetico_labirinto
)

# --- FUNÇÕES AUXILIARES ---

def verificar_qtable(path: str) -> bool:
    """
    Verifica se o ficheiro da Q-Table existe antes de tentar correr a demo.
    """
    if not os.path.exists(path):
        print("\n[ERRO] Q-Table não encontrada!")
        print(f" -> Ficheiro esperado: {path}")
        print(" -> Ação necessária: Execute primeiro o Treino do Labirinto (Opção 2).\n")
        return False
    return True

# --- INTERFACE (MENU) ---

def menu():
    print("\n==============================================")
    print("       SIMULADOR DE SISTEMAS MULTI-AGENTE     ")
    print("==============================================")
    
    print("\n--- GRUPO 1: Q-LEARNING (Aprendizagem por Reforço) ---")
    print("1 - Farol:     Treino Completo (QL + Greedy Baseline)")
    print("2 - Labirinto: Treino Completo (QL + Baseline)")
    print("3 - Labirinto: Demo Visual (Requer treino prévio)")
    
    print("\n--- GRUPO 2: ALGORITMOS GENÉTICOS (Evolutivo) ---")
    print("4 - Farol:     Evolução Simples (Rede Neuronal)")
    print("5 - Labirinto: Evolução Deep + Novelty Search")
    
    print("\n----------------------------------------------")
    print("0 - Sair")
    
    return input(" >> Escolha uma opção: ").strip()

# --- LOOP PRINCIPAL ---

def main():
    while True:
        op = menu()

        # ==========================================
        # Q-LEARNING
        # ==========================================
        
        if op == "1":
            # Farol (Problema Simples)
            correr_treino_farol(
                num_episodios_treino=250,
                num_episodios_teste_ql=40,
                num_episodios_teste_greedy=40,
                max_passos=100,
                caminho_csv="resultados_farol.csv"
            )

        elif op == "2":
            # Labirinto (Problema Complexo)
            correr_treino_labirinto(
                num_episodios_treino=25000,
                num_episodios_teste=50,
                max_passos=500,
                caminho_csv="resultados_labirinto.csv"
            )

        elif op == "3":
            # Demo Visual Labirinto
            if verificar_qtable("qtable_labirinto.pkl"):
                # Executa apenas se a tabela existir
                demo_labirinto(num_episodios=10, max_passos=500)

        # ==========================================
        # ALGORITMOS GENÉTICOS
        # ==========================================

        elif op == "4":
            # Farol Genético
            correr_treino_genetico_farol()
            
        elif op == "5":
            # Labirinto Genético (Novelty Search)
            correr_treino_genetico_labirinto()

        # ==========================================
        # SAIR
        # ==========================================
        
        elif op == "0":
            print("\nSimulação terminada. Até à próxima!")
            break

        else:
            print("\n[!] Opção inválida. Tente novamente.")

if __name__ == "__main__":
    main()