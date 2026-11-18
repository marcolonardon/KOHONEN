"""
Mapa Auto-Organizável de Kohonen (SOM) aplicado à classificação de DNA.

Base de dados: "Molecular Biology (Splice-junction Gene Sequences)", UCI
Machine Learning Repository -- 3190 janelas de 60 nucleotídeos de primatas,
cada uma rotulada como:
    EI = fronteira éxon-íntron, IE = fronteira íntron-éxon, N = nenhuma.

O algoritmo segue Kohonen (1990), "The Self-Organizing Map", Proc. IEEE 78(9):
    Eq. (2')  vencedor:      ||x - m_c|| = min_i ||x - m_i||
    Eq. (7)   atualização:   m_i(t+1) = m_i(t) + h_ci(t) [x(t) - m_i(t)]
    Eq. (8)   vizinhança:    h_ci(t) = alfa(t) exp(-||r_i - r_c||^2 / sigma(t)^2)
e as recomendações práticas da Seção II-D (fase de ordenação curta com alfa
alto e vizinhança larga, seguida de um ajuste fino longo com alfa pequeno).

A rotulagem do mapa e a avaliação seguem Mohamed et al. (2013), "Enhanced
Self-Organizing Map Neural Network for DNA Sequence Classification": cada
neurônio guarda um contador por classe, o maior contador define o rótulo do
neurônio, e o resultado é medido por acurácia e precisão (Eq. 3 e 4).

Uso:
    python kohonen_som.py
    python kohonen_som.py --linhas 20 --colunas 20 --topologia retangular
"""

import argparse
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # salva os PNGs sem precisar abrir janela
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.patches import Patch, RegularPolygon

RAIZ = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# 1. Leitura e codificação da base de dados
# --------------------------------------------------------------------------

# O SOM clássico trabalha com vetores numéricos e distância euclidiana, então
# cada nucleotídeo vira 4 números (A, C, G, T). Os poucos códigos ambíguos da
# base (D, N, S, R) repartem o peso entre as bases que podem representar.
CODIGO = {
    "A": (1, 0, 0, 0),
    "C": (0, 1, 0, 0),
    "G": (0, 0, 1, 0),
    "T": (0, 0, 0, 1),
    "D": (1 / 3, 0, 1 / 3, 1 / 3),  # A, G ou T
    "N": (0.25, 0.25, 0.25, 0.25),  # qualquer base
    "S": (0, 0.5, 0.5, 0),  # C ou G
    "R": (0.5, 0, 0.5, 0),  # A ou G
}

DESCRICAO_CLASSE = {
    "EI": "EI (éxon-íntron)",
    "IE": "IE (íntron-éxon)",
    "N": "N (nenhuma)",
}


def carregar_splice(caminho):
    """Lê o arquivo splice.data: uma linha por amostra, 'classe, nome, sequência'."""
    nomes, classes, sequencias = [], [], []
    with open(caminho, encoding="ascii") as arquivo:
        for linha in arquivo:
            partes = [p.strip() for p in linha.split(",")]
            if len(partes) != 3:
                continue
            classes.append(partes[0])
            nomes.append(partes[1])
            sequencias.append(partes[2].upper())
    return np.array(nomes), np.array(classes), sequencias


def codificar(sequencias):
    """Transforma cada sequência de 60 letras em um vetor de 240 números."""
    matriz = np.array([[CODIGO[base] for base in seq] for seq in sequencias], dtype=float)
    return matriz.reshape(len(sequencias), -1)


def dividir_estratificado(y, fracao_teste, rng):
    """Separa treino e teste mantendo a proporção de cada classe."""
    treino, teste = [], []
    for classe in np.unique(y):
        indices = rng.permutation(np.flatnonzero(y == classe))
        n_teste = int(round(len(indices) * fracao_teste))
        teste.extend(indices[:n_teste])
        treino.extend(indices[n_teste:])
    return np.sort(treino), np.sort(teste)


# --------------------------------------------------------------------------
# 2. O Mapa Auto-Organizável
# --------------------------------------------------------------------------


class SOM:
    """Mapa de Kohonen bidimensional com grade hexagonal ou retangular."""

    def __init__(self, linhas, colunas, topologia="hexagonal", vizinhanca="gaussiana", semente=0):
        self.linhas = linhas
        self.colunas = colunas
        self.topologia = topologia
        self.vizinhanca = vizinhanca
        self.rng = np.random.default_rng(semente)
        self.pesos = None  # vetores m_i, um por neurônio

        # r_i: posição de cada neurônio na grade. Na grade hexagonal as linhas
        # ímpares são deslocadas meia célula, de modo que todo neurônio tem
        # 6 vizinhos à distância 1 (Fig. 2 de Kohonen, 1990).
        linha, coluna = np.divmod(np.arange(linhas * colunas), colunas)
        if topologia == "hexagonal":
            self.posicoes = np.column_stack([coluna + 0.5 * (linha % 2), linha * np.sqrt(3) / 2])
        else:
            self.posicoes = np.column_stack([coluna, linha]).astype(float)

        diferenca = self.posicoes[:, None, :] - self.posicoes[None, :, :]
        self.dist2_grade = (diferenca**2).sum(axis=2)  # ||r_i - r_c||^2
        # Vizinhos imediatos: 6 na grade hexagonal, 8 na retangular.
        self.adjacentes = (self.dist2_grade > 0) & (self.dist2_grade < 1.5**2)

    @property
    def n_neuronios(self):
        return self.linhas * self.colunas

    def _nucleo(self, vencedor, raio):
        """Função de vizinhança em torno do vencedor (sem o fator alfa)."""
        d2 = self.dist2_grade[vencedor]
        if self.vizinhanca == "gaussiana":
            return np.exp(-d2 / raio**2)  # Eq. (8)
        return (d2 <= raio**2).astype(float)  # conjunto N_c da Eq. (6)

    def treinar(
        self,
        X,
        passos_ordenacao=1000,
        passos_ajuste=None,
        alfa_inicial=0.9,
        alfa_ajuste=0.05,
        alfa_final=0.001,
        raio_inicial=None,
    ):
        """Treinamento sequencial em duas fases (Kohonen, 1990, Seção II-D).

        Fase de ordenação: alfa cai linearmente de `alfa_inicial` até
        `alfa_ajuste` e o raio da vizinhança encolhe de `raio_inicial` até 1.
        Fase de ajuste fino: alfa pequeno caindo até `alfa_final`, com a
        vizinhança reduzida aos vizinhos imediatos (raio 1).

        Devolve o histórico [(passo, erro de quantização)] para a curva de
        treinamento.
        """
        if passos_ajuste is None:
            # "o número de passos deve ser ao menos 500 vezes o número de neurônios"
            passos_ajuste = 500 * self.n_neuronios
        if raio_inicial is None:
            # "o raio inicial pode até ser maior que metade do diâmetro da rede"
            raio_inicial = max(self.linhas, self.colunas) / 2

        # Pesos iniciais: amostras sorteadas sem repetição (a única restrição
        # de Kohonen é que os valores iniciais sejam diferentes entre si).
        escolhidas = self.rng.choice(len(X), self.n_neuronios, replace=len(X) < self.n_neuronios)
        self.pesos = X[escolhidas].copy()

        total = passos_ordenacao + passos_ajuste
        sorteio = self.rng.integers(0, len(X), total)  # x(t) sorteado ao acaso
        marcos = set(np.unique(np.geomspace(1, total, 40).astype(int)))
        historico = [(0, self.erro_quantizacao(X))]

        for t in range(total):
            if t < passos_ordenacao:
                f = t / passos_ordenacao
                alfa = alfa_inicial + (alfa_ajuste - alfa_inicial) * f
                raio = raio_inicial + (1.0 - raio_inicial) * f
            else:
                f = (t - passos_ordenacao) / passos_ajuste
                alfa = alfa_ajuste + (alfa_final - alfa_ajuste) * f
                raio = 1.0

            diferenca = X[sorteio[t]] - self.pesos  # x(t) - m_i(t)
            vencedor = np.argmin(np.einsum("ij,ij->i", diferenca, diferenca))  # Eq. (2')
            h = alfa * self._nucleo(vencedor, raio)
            self.pesos += h[:, None] * diferenca  # Eq. (7)

            if t + 1 in marcos:
                historico.append((t + 1, self.erro_quantizacao(X)))
        return historico

    def distancias(self, X):
        """Distância euclidiana de cada amostra a cada neurônio (amostras x neurônios)."""
        d2 = (X**2).sum(axis=1)[:, None] - 2 * X @ self.pesos.T + (self.pesos**2).sum(axis=1)[None, :]
        return np.sqrt(np.maximum(d2, 0))

    def vencedores(self, X):
        """Índice do neurônio vencedor (BMU) de cada amostra."""
        return self.distancias(X).argmin(axis=1)

    def erro_quantizacao(self, X):
        """Distância média entre cada amostra e o seu neurônio vencedor."""
        return float(self.distancias(X).min(axis=1).mean())

    def erro_topografico(self, X):
        """Fração das amostras cujos dois melhores neurônios não são vizinhos na grade."""
        dois_melhores = np.argsort(self.distancias(X), axis=1)[:, :2]
        vizinhos = self.adjacentes[dois_melhores[:, 0], dois_melhores[:, 1]]
        return float(1 - vizinhos.mean())

    def matriz_u(self):
        """Distância média entre o peso de cada neurônio e os dos seus vizinhos."""
        entre_pesos = self.distancias(self.pesos)
        return (entre_pesos * self.adjacentes).sum(axis=1) / self.adjacentes.sum(axis=1)


# --------------------------------------------------------------------------
# 3. Rotulagem do mapa e classificação
# --------------------------------------------------------------------------


def calibrar(som, X, y, n_classes):
    """Conta, em cada neurônio, quantas amostras de cada classe ele venceu.

    O maior contador define o rótulo do neurônio; a confiabilidade é a fração
    das amostras do neurônio que pertencem a essa classe. Neurônios que não
    venceram nenhuma amostra ficam sem rótulo (-1).
    """
    contadores = np.zeros((som.n_neuronios, n_classes), dtype=int)
    np.add.at(contadores, (som.vencedores(X), y), 1)
    total = contadores.sum(axis=1)
    rotulos = np.where(total > 0, contadores.argmax(axis=1), -1)
    confiabilidade = contadores.max(axis=1) / np.maximum(total, 1)
    return contadores, rotulos, confiabilidade


def classificar(som, X, rotulos):
    """Atribui a cada amostra o rótulo do neurônio rotulado mais próximo."""
    distancias = som.distancias(X)
    distancias[:, rotulos < 0] = np.inf
    vencedores = distancias.argmin(axis=1)
    return rotulos[vencedores], vencedores


def matriz_confusao(y_real, y_previsto, n_classes):
    matriz = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(matriz, (y_real, y_previsto), 1)
    return matriz


def tabela_metricas(matriz, classes):
    """Texto com matriz de confusão, acurácia, precisão e revocação."""
    acertos = np.diag(matriz)
    precisao = acertos / np.maximum(matriz.sum(axis=0), 1)  # TP / (TP + FP)
    revocacao = acertos / np.maximum(matriz.sum(axis=1), 1)  # TP / (TP + FN)
    linhas = [f"  Acurácia geral: {100 * acertos.sum() / matriz.sum():.2f}%  ({acertos.sum()} de {matriz.sum()})"]
    linhas.append("  Matriz de confusão (linha = classe real, coluna = classe prevista):")
    linhas.append("    " + " " * 8 + "".join(f"{c:>7}" for c in classes) + "   precisão  revocação")
    for i, classe in enumerate(classes):
        valores = "".join(f"{v:>7}" for v in matriz[i])
        linhas.append(f"    {classe:<8}{valores}   {precisao[i]:>8.3f}  {revocacao[i]:>9.3f}")
    return linhas


# --------------------------------------------------------------------------
# 4. Figuras
# --------------------------------------------------------------------------

SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
TINTA_SUAVE = "#898781"
GRADE = "#e1e0d9"
SEM_AMOSTRAS = "#e1e0d9"
CORES_CLASSE = ["#2a78d6", "#eb6834", "#1baf7a"]  # distinguíveis também por daltônicos
AZUIS = LinearSegmentedColormap.from_list(
    "azuis", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
        "figure.facecolor": SUPERFICIE,
        "axes.facecolor": SUPERFICIE,
        "savefig.facecolor": SUPERFICIE,
        "text.color": TINTA,
        "axes.labelcolor": TINTA_SECUNDARIA,
        "xtick.color": TINTA_SUAVE,
        "ytick.color": TINTA_SUAVE,
    }
)


def desenhar_mapa(ax, som, cores=None, valores=None):
    """Desenha a grade do SOM: uma célula por neurônio, colorida por `cores` ou `valores`."""
    if som.topologia == "hexagonal":
        celulas = [RegularPolygon(p, 6, radius=1 / np.sqrt(3)) for p in som.posicoes]
    else:
        celulas = [RegularPolygon(p, 4, radius=np.sqrt(0.5), orientation=np.pi / 4) for p in som.posicoes]
    colecao = PatchCollection(celulas, edgecolor=SUPERFICIE, linewidth=1.5)
    if valores is not None:
        colecao.set_array(np.asarray(valores, dtype=float))
        colecao.set_cmap(AZUIS)
    else:
        colecao.set_facecolor(cores)
    ax.add_collection(colecao)
    ax.set_aspect("equal")
    ax.margins(0.01)
    ax.autoscale_view()
    ax.invert_yaxis()
    ax.axis("off")
    return colecao


def titulo(ax, principal, subtitulo):
    ax.set_title(principal, loc="left", fontsize=13, fontweight="bold", pad=24)
    ax.text(0, 1.0, subtitulo, transform=ax.transAxes, fontsize=9.5, color=TINTA_SECUNDARIA,
            va="bottom", linespacing=2.2)


def figura_mapa_classes(som, rotulos, confiabilidade, classes, caminho):
    acaso = 1 / len(classes)
    cores = []
    for rotulo, conf in zip(rotulos, confiabilidade):
        if rotulo < 0:
            cores.append(to_rgba(SEM_AMOSTRAS))
        else:
            # Quanto menor a confiabilidade do rótulo, mais clara a célula.
            cores.append(to_rgba(CORES_CLASSE[rotulo], 0.3 + 0.7 * (conf - acaso) / (1 - acaso)))
    fig, ax = plt.subplots(figsize=(8, 7.2))
    desenhar_mapa(ax, som, cores=cores)
    titulo(ax, "Mapa de classes", "Classe majoritária de cada neurônio; cor mais clara = rótulo menos confiável")
    legenda = [Patch(color=CORES_CLASSE[i], label=DESCRICAO_CLASSE.get(c, c)) for i, c in enumerate(classes)]
    legenda.append(Patch(color=SEM_AMOSTRAS, label="sem amostras"))
    ax.legend(handles=legenda, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False, fontsize=9.5)
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figura_escalar(som, valores, principal, subtitulo, rotulo_barra, caminho):
    fig, ax = plt.subplots(figsize=(8, 7.2))
    colecao = desenhar_mapa(ax, som, valores=valores)
    titulo(ax, principal, subtitulo)
    barra = fig.colorbar(colecao, ax=ax, shrink=0.7, pad=0.02)
    barra.set_label(rotulo_barra)
    barra.outline.set_visible(False)
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figura_curva(historico, passos_ordenacao, caminho):
    passos, erros = zip(*historico[1:])
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(passos, erros, color=CORES_CLASSE[0], linewidth=2)
    ax.axvline(passos_ordenacao, color=TINTA_SUAVE, linewidth=1)
    ax.text(passos_ordenacao, 0.97, " fim da ordenação", transform=ax.get_xaxis_transform(), fontsize=9,
            color=TINTA_SECUNDARIA, va="top")
    ax.set_xscale("log")
    ax.set_xlabel("Passos de treinamento (escala logarítmica)")
    ax.set_ylabel("Erro de quantização")
    ax.grid(True, axis="y", color=GRADE, linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(which="both", length=0)
    titulo(ax, "Curva de treinamento", "Distância média entre as amostras de treino e o neurônio vencedor")
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# 5. Programa principal
# --------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="SOM de Kohonen na base de DNA Splice-junction (UCI).")
    parser.add_argument("--dados", type=Path, default=RAIZ / "data" / "splice.data", help="arquivo splice.data")
    parser.add_argument("--saida", type=Path, default=RAIZ / "resultados", help="pasta dos resultados")
    parser.add_argument("--linhas", type=int, default=15, help="linhas do mapa")
    parser.add_argument("--colunas", type=int, default=15, help="colunas do mapa")
    parser.add_argument("--topologia", choices=["hexagonal", "retangular"], default="hexagonal")
    parser.add_argument("--vizinhanca", choices=["gaussiana", "bolha"], default="gaussiana")
    parser.add_argument("--passos-ordenacao", type=int, default=1000)
    parser.add_argument("--passos-ajuste", type=int, default=None, help="padrão: 500 x número de neurônios")
    parser.add_argument("--teste", type=float, default=0.3, help="fração das amostras reservada para teste")
    parser.add_argument("--semente", type=int, default=42)
    args = parser.parse_args()

    # Leitura da base
    nomes, classes_texto, sequencias = carregar_splice(args.dados)
    X = codificar(sequencias)
    classes, y = np.unique(classes_texto, return_inverse=True)
    treino, teste = dividir_estratificado(y, args.teste, np.random.default_rng(args.semente))

    relatorio = ["=" * 68, "SOM de Kohonen - sequências de DNA (Splice-junction, UCI)", "=" * 68]
    relatorio.append(f"Amostras: {len(X)}  |  atributos: {X.shape[1]} (60 nucleotídeos x 4)")
    relatorio.append("Classes: " + ", ".join(f"{c} = {(y == i).sum()}" for i, c in enumerate(classes)))
    relatorio.append(f"Treino: {len(treino)}  |  teste: {len(teste)}  (divisão estratificada, semente {args.semente})")
    print("\n".join(relatorio))

    # Treinamento
    som = SOM(args.linhas, args.colunas, args.topologia, args.vizinhanca, args.semente)
    passos_ajuste = args.passos_ajuste if args.passos_ajuste is not None else 500 * som.n_neuronios
    print(f"\nTreinando mapa {args.linhas}x{args.colunas} {args.topologia} "
          f"({args.passos_ordenacao} + {passos_ajuste} passos)...")
    inicio = time.perf_counter()
    historico = som.treinar(X[treino], args.passos_ordenacao, passos_ajuste)
    duracao = time.perf_counter() - inicio

    # Rotulagem com o conjunto de treino e classificação
    contadores, rotulos, confiabilidade = calibrar(som, X[treino], y[treino], len(classes))
    previsto_treino, _ = classificar(som, X[treino], rotulos)
    previsto_teste, vencedores_teste = classificar(som, X[teste], rotulos)
    rotulados = rotulos >= 0

    resultado = ["", f"Mapa: {args.linhas}x{args.colunas} {args.topologia}, vizinhança {args.vizinhanca}"]
    resultado.append(f"Passos: {args.passos_ordenacao} de ordenação + {passos_ajuste} de ajuste fino "
                     f"({duracao:.1f} s)")
    resultado.append(f"Erro de quantização (treino): {historico[0][1]:.4f} inicial, {historico[-1][1]:.4f} final")
    resultado.append(f"Erro de quantização (teste):  {som.erro_quantizacao(X[teste]):.4f}")
    resultado.append(f"Erro topográfico (treino):    {som.erro_topografico(X[treino]):.4f}")
    resultado.append(f"Neurônios rotulados: {rotulados.sum()} de {som.n_neuronios}  "
                     f"(confiabilidade média dos rótulos: {confiabilidade[rotulados].mean():.3f})")
    resultado.append("  por classe: " + ", ".join(f"{c} = {(rotulos == i).sum()}" for i, c in enumerate(classes)))
    resultado += ["", "CLASSIFICAÇÃO - TREINO"]
    resultado += tabela_metricas(matriz_confusao(y[treino], previsto_treino, len(classes)), classes)
    resultado += ["", "CLASSIFICAÇÃO - TESTE"]
    resultado += tabela_metricas(matriz_confusao(y[teste], previsto_teste, len(classes)), classes)
    print("\n".join(resultado))

    # Arquivos de saída
    args.saida.mkdir(parents=True, exist_ok=True)
    (args.saida / "relatorio.txt").write_text("\n".join(relatorio + resultado) + "\n", encoding="utf-8")
    with open(args.saida / "predicoes_teste.csv", "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(["nome", "classe_real", "classe_prevista", "linha_bmu", "coluna_bmu"])
        for i, previsto, vencedor in zip(teste, previsto_teste, vencedores_teste):
            escritor.writerow([nomes[i], classes[y[i]], classes[previsto], *divmod(int(vencedor), args.colunas)])

    figura_mapa_classes(som, rotulos, confiabilidade, classes, args.saida / "mapa_classes.png")
    figura_escalar(som, som.matriz_u(), "Matriz U",
                   "Distância média entre cada neurônio e seus vizinhos; escuro = fronteira entre grupos",
                   "Distância média aos vizinhos", args.saida / "matriz_u.png")
    figura_escalar(som, contadores.sum(axis=1), "Ocupação do mapa",
                   "Número de amostras de treino que cada neurônio venceu",
                   "Amostras de treino", args.saida / "ocupacao.png")
    figura_curva(historico, args.passos_ordenacao, args.saida / "curva_treinamento.png")

    print(f"\nArquivos gravados em {args.saida}:")
    for arquivo in sorted(args.saida.iterdir()):
        print(f"  {arquivo.name}")


if __name__ == "__main__":
    main()
