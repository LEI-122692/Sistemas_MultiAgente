from __future__ import annotations
import time
import tkinter as tk

from Envs import LabirintoEnvironment
from Agents import QLearningLabirintoAgent


class Viewer:
    """
    Janela gráfica simples para visualizar o labirinto.
    Integrada no demo_visual para simplificar a estrutura do projeto.
    """

    def __init__(self, env, cell_size: int = 50, delay: float = 0.15):
        self.env = env
        self.cell_size = cell_size
        self.delay = delay

        # criar janela
        self.root = tk.Tk()
        self.root.title("Labirinto - Simulação")

        # dimensões da grelha (a partir do mapa do labirinto)
        m = self.env.map
        width = m.largura * cell_size
        height = m.altura * cell_size + 30  # espaço para texto

        self.canvas = tk.Canvas(self.root, width=width, height=height)
        self.canvas.pack()

        self.info_text = self.canvas.create_text(
            10, height - 15, anchor="w", text="", font=("Consolas", 10)
        )

    # -------------------------------------------------- #
    def _desenhar_labirinto(self, passo: int, recompensa: float):
        m = self.env.map
        cs = self.cell_size

        # apagar células anteriores
        self.canvas.delete("cell")

        # desenhar grelha
        for y in range(m.altura):
            for x in range(m.largura):
                x0, y0 = x * cs, y * cs
                x1, y1 = x0 + cs, y0 + cs

                if m.is_parede(x, y):
                    cor = "#333333"   # parede
                else:
                    cor = "#EEEEEE"   # caminho

                self.canvas.create_rectangle(
                    x0, y0, x1, y1, fill=cor, outline="#AAAAAA", tags="cell"
                )

        # saída (goal)
        gx, gy = self.env.saida_x, self.env.saida_y
        x0, y0 = gx * cs, gy * cs
        x1, y1 = x0 + cs, y0 + cs
        self.canvas.create_rectangle(
            x0, y0, x1, y1, fill="#7CFC00", outline="#006400", tags="cell"
        )

        # agente
        ax, ay = self.env.agent_x, self.env.agent_y
        cx0, cy0 = ax * cs + cs * 0.2, ay * cs + cs * 0.2
        cx1, cy1 = ax * cs + cs * 0.8, ay * cs + cs * 0.8
        self.canvas.create_oval(
            cx0, cy0, cx1, cy1, fill="#00BFFF", outline="#000080", tags="cell"
        )

        # texto em baixo
        self.canvas.itemconfig(
            self.info_text,
            text=f"Passo {passo} | Recompensa: {recompensa}",
        )

        self.root.update()
        time.sleep(self.delay)

    # -------------------------------------------------- #
    def reset(self, episodio: int):
        self.root.title(f"Labirinto - Episódio {episodio + 1}")

    def render_step(self, passo: int, recompensa: float):
        self._desenhar_labirinto(passo, recompensa)

    def render_episode_end(self, episodio: int, passos: int):
        self.canvas.itemconfig(
            self.info_text,
            text=f"Episódio {episodio + 1} terminado em {passos} passos.",
        )
        self.root.update()
        time.sleep(1.0)

    def close(self):
        self.root.destroy()

# FUNÇÃO DE DEMONSTRAÇÃO PRINCIPAL

def demo_labirinto(
    num_episodios: int = 1,
    max_passos: int = 80,
    delay: float = 0.15,
) -> None:
    """
    Demonstração visual do agente no labirinto numa janela gráfica.
    Política fixa (epsilon=0.0).
    """

    env = LabirintoEnvironment()
    # Assume-se que o agente QL já tem o método load_qtable definido
    agent = QLearningLabirintoAgent(agent_id=1, alpha=0.1, gamma=0.99, epsilon=0.0)
    agent.load_qtable("qtable_labirinto.pkl")
    
    viewer = Viewer(env, cell_size=50, delay=delay)

    print("\n=== Demonstração Visual – Labirinto ===")

    for ep in range(num_episodios):
        env.reset()
        obs = env.observacaoPara(agent)
        agent.observacao(obs)

        viewer.reset(ep)

        terminou = False
        for passo in range(1, max_passos + 1):
            accao = agent.age()
            recompensa, terminou, _info = env.agir(accao, agent)

            obs = env.observacaoPara(agent)
            agent.observacao(obs)

            viewer.render_step(passo, recompensa)

            if terminou:
                viewer.render_episode_end(ep, passo)
                break

        if not terminou:
            viewer.render_episode_end(ep, max_passos)

    viewer.close()
    print("\n--- Fim da demonstração ---")


if __name__ == "__main__":
    
    demo_labirinto()