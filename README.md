# IBL Color Palette Generation

Repositório desenvolvido como parte do Trabalho de Conclusão de Curso (TCC), com o objetivo de investigar a aplicação de **Instance-Based Learning (IBL)** na geração de paletas de cores a partir de regras de harmonia cromática.

O projeto reúne scripts para geração das instâncias de aprendizado, amostragem de cores, geração de paletas por diferentes abordagens, extração de resultados de uma ferramenta comercial e avaliação comparativa dos resultados.

## Estrutura do projeto

| Arquivo | Descrição |
|---|---|
| `training_data_generator.py` | Gera as instâncias de aprendizado a partir de combinações de valores HSL e regras de harmonia cromática, convertendo as cores para CIELAB e registrando os deslocamentos entre as cores. |
| `sobol_sample_generator.py` | Gera uma amostra de cores de entrada utilizando sequências de Sobol, considerando os limites válidos do espaço CIELAB. |
| `ibl_palette_generator.py` | Implementa a geração de paletas por IBL, utilizando busca de vizinhos próximos no espaço CIELAB e diferentes configurações de quantidade de vizinhos e ponderação por distância. |
| `theoretical_palette_generator.py` | Gera paletas de referência aplicando diretamente as regras teóricas de harmonia cromática. |
| `commercial_palette_extractor.py` | Automatiza a consulta à ferramenta comercial e a extração das paletas geradas para as cores de entrada e esquemas definidos. |
| `pairwise_comparison_generator.py` | Gera os arquivos de comparação entre duas abordagens, realizando o pareamento das cores e calculando métricas de diferença cromática e perceptual. |
| `combined_model_analysis.py` | Consolida e compara os resultados dos seis modelos IBL em relação a uma abordagem de referência, produzindo estatísticas e gráficos. |
| `individual_analysis.py` | Realiza a análise individual de uma comparação entre duas abordagens, gerando estatísticas, percentis, gráficos e relatórios. |

## Fluxo geral

O processamento pode ser organizado nas seguintes etapas:

1. **Preparação dos dados:** geração das instâncias de aprendizado e da amostra de cores de entrada.
2. **Geração das paletas:** aplicação dos modelos IBL e das regras teóricas, além da extração das paletas da ferramenta comercial.
3. **Comparação:** pareamento das cores geradas pelas diferentes abordagens e cálculo das métricas de comparação.
4. **Análise:** avaliação individual das comparações e análise consolidada do desempenho dos modelos IBL.

## Métricas de avaliação

A avaliação considera diferenças entre as cores nos espaços CIELAB, CIELCh e HSL, incluindo a diferença perceptual **CIEDE2000 (ΔE00)**. São utilizadas estatísticas descritivas, distribuições e percentis para examinar a proximidade entre as paletas geradas pelas diferentes abordagens.

## Tecnologias e bibliotecas

O projeto foi desenvolvido em Python e utiliza as seguintes bibliotecas:

| Biblioteca | Finalidade |
|---|---|
| `NumPy` | Operações numéricas e manipulação de arrays. |
| `Pandas` | Manipulação, organização e exportação dos dados em tabelas. |
| `SciPy` | Geração de amostras utilizando sequências de Sobol. |
| `scikit-image` | Conversão e manipulação de espaços de cores, além do cálculo da diferença perceptual CIEDE2000. |
| `FAISS` | Busca eficiente de vizinhos próximos no espaço CIELAB. |
| `Playwright` | Automação do navegador para consulta à ferramenta comercial e extração de paletas. |
| `Matplotlib` | Geração de gráficos para análise e visualização dos resultados. |

## Configuração do ambiente

### Requisitos

- Python 3.10 ou superior.
- `pip` para instalação das dependências.

### Instalação

1. Clone o repositório:

```bash
git clone <URL_DO_REPOSITORIO>
cd <NOME_DO_REPOSITORIO>
```

2. Crie um ambiente virtual:

```bash
python -m venv .venv
```

3. Ative o ambiente virtual:

**Linux/macOS:**
```bash
source .venv/bin/activate
```

**Windows:**
```bash
.venv\Scripts\activate
```

4. Instale as bibliotecas necessárias:

```bash
pip install numpy pandas scipy scikit-image faiss-cpu playwright matplotlib
```

5. Instale o navegador utilizado pelo Playwright:

```bash
playwright install chromium
```

Após a instalação, o ambiente estará preparado para executar os scripts do projeto.

### Execução

Os scripts podem ser executados individualmente, conforme a etapa do processamento desejada:

```bash
python training_data_generator.py
```

Os demais scripts seguem o mesmo padrão de execução, respeitando a ordem do fluxo de processamento descrito anteriormente.
