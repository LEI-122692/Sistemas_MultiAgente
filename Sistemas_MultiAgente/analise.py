import pandas as pd
import matplotlib.pyplot as plt
import os

def desenhar_grafico_qlearning_limpo(caminho_csv, nome_experiencia_treino, titulo, output_file):
    """
    Desenha APENAS a linha de tendência (Média Móvel), sem o ruído de fundo.
    """
    if not os.path.exists(caminho_csv):
        print(f"[AVISO] Ficheiro não encontrado: {caminho_csv}")
        return

    try:
        df = pd.read_csv(caminho_csv)
        
        # 1. FILTRAR DADOS 
        df_treino = df[df['experiencia'] == nome_experiencia_treino].copy()
        
        if df_treino.empty:
            print(f"[AVISO] Não encontrei dados para '{nome_experiencia_treino}'.")
            return

        # 2. ORDENAR e CALCULAR MÉDIA
        df_treino = df_treino.sort_values(by='episodio')
        df_treino['suavizado'] = df_treino['recompensa_total'].rolling(window=50).mean()

        # 3. FILTRO VISUAL (Começa só no ep 50)
        df_suave = df_treino[df_treino['episodio'] >= 50]

        plt.figure(figsize=(10, 6))
        
        # Desenha APENAS a linha forte (Tendência)
        plt.plot(df_suave['episodio'], df_suave['suavizado'], color='darkblue', linewidth=2, label='Média Móvel (Tendência)')
        
        plt.title(titulo, fontsize=14)
        plt.xlabel('Episódio', fontsize=12)
        plt.ylabel('Recompensa Média', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        plt.savefig(output_file)
        print(f"[SUCESSO] Gráfico Limpo: {output_file}")
        plt.close()

    except Exception as e:
        print(f"[ERRO] {caminho_csv}: {e}")

def desenhar_grafico_genetico(caminho_csv, titulo, output_file):
    """ Desenha Genético focado no Sucesso (Laranja) """
    if not os.path.exists(caminho_csv): return

    try:
        df = pd.read_csv(caminho_csv)
        df = df.sort_values(by='episodio')

        plt.figure(figsize=(10, 6))
        
        # Desenha linha de Sucesso
        if 'sucesso' in df.columns and df['sucesso'].max() > 0:
            plt.plot(df['episodio'], df['sucesso'], color='darkorange', linewidth=3, marker='o', markersize=4, label='Nº Agentes com Sucesso')
            plt.ylabel('População com Sucesso', fontsize=12, color='darkorange')
            plt.tick_params(axis='y', labelcolor='darkorange')
        else:
            plt.plot(df['episodio'], df['recompensa_total'], color='green', linewidth=2, label='Melhor Fitness')
            plt.ylabel('Fitness', color='green')

        plt.title(titulo, fontsize=14)
        plt.xlabel('Geração', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"[SUCESSO] Gráfico Genético: {output_file}")
        plt.close()
    except:
        pass

if __name__ == '__main__':
    print("--- GERANDO GRÁFICOS 'CLEAN' ---")

    # 1. Q-LEARNING FAROL
    desenhar_grafico_qlearning_limpo(
        "resultados_farol.csv", 
        "Farol_QL_Treino", 
        "Curva de Aprendizagem: Farol", 
        "grafico_QL_Farol.png"
    )

    # 2. Q-LEARNING LABIRINTO
    desenhar_grafico_qlearning_limpo("resultados_labirinto.csv", "QL_Labirinto_V1", "Curva de Aprendizagem: Labirinto", "grafico_QL_Labirinto.png")
    desenhar_grafico_qlearning_limpo("resultados_labirinto.csv", "Labirinto_Treino", "Curva de Aprendizagem: Labirinto", "grafico_QL_Labirinto.png")

    # 3. GENÉTICOS
    desenhar_grafico_genetico("resultados_genetico_labirinto.csv", "Evolução: Labirinto", "grafico_Gen_Labirinto.png")
    desenhar_grafico_genetico("resultados_genetico_farol.csv", "Evolução: Farol", "grafico_Gen_Farol.png")