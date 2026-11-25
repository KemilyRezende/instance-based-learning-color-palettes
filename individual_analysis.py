"""Análise estatística individual de comparações entre duas fontes de paletas.

Uso:
1. Ajuste INPUT_CSV, OUTPUT_DIR, MODEL_1 e MODEL_2.
2. Execute: python analisar_comparacao_individual.py

Organização:
- Quatro gráficos individuais de ΔE00.
- Grupo 1: uma figura 2x2 (um painel por esquema), com as médias das
  quatro métricas absolutas CIELAB.
- Grupo 2: uma figura 2x2 (um painel por esquema), com as distribuições
  das mesmas quatro métricas absolutas CIELAB.
- Grupo 3: uma figura 2x2 (um painel por esquema), com as distribuições
  das quatro métricas assinadas CIELAB.
- As duas métricas de matiz (LCh e HSL), ambas em graus, ficam juntas em
  duas figuras próprias: médias e distribuições, cada uma com quatro
  painéis (um por esquema).

As métricas são agregadas primeiro por input_hex + scheme, para que cada
cor de entrada tenha o mesmo peso dentro de cada esquema.
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURAÇÃO — ALTERE ESTES VALORES A CADA COMPARAÇÃO
# ============================================================

INPUT_CSV = "comparacao_comercial_teoria.csv"
OUTPUT_DIR = Path("Analise_Comercial_Teoria")

MODEL_1 = "Comercial"
MODEL_2 = "Teoria"

PERCENTILES = [25, 50, 75, 90, 95, 99]

SCHEMES = [
    "analogous",
    "triadic",
    "complementary",
    "split_complementary",
]

SCHEME_LABELS = {
    "analogous": "Análoga",
    "triadic": "Triádica",
    "complementary": "Complementar",
    "split_complementary": "Complementar partida",
}

EXPECTED_OUTPUTS = {
    "analogous": 2,
    "triadic": 2,
    "complementary": 1,
    "split_complementary": 2,
}

PERCEPTUAL_METRIC = "delta_E00"

# Métricas assinadas: a direção da diferença é preservada.
SIGNED_METRICS = ["delta_L", "delta_a", "delta_b", "delta_C"]

# Métricas de matiz: ambas representam diferenças angulares em graus.
# Devem estar previamente calculadas como a menor distância angular [0, 180].
ANGULAR_METRICS = ["delta_h_LCh", "delta_H_HSL"]

# Magnitudes absolutas CIELAB: compartilham a mesma unidade/escala.
MAGNITUDE_METRICS = [
    "delta_C_abs",
    "delta_L_abs",
    "delta_a_abs",
    "delta_b_abs",
]

METRIC_LABELS = {
    "delta_E00": "ΔE00",
    "delta_L_abs": "|ΔL*|",
    "delta_a_abs": "|Δa*|",
    "delta_b_abs": "|Δb*|",
    "delta_C_abs": "|ΔC*|",
    "delta_h_LCh": "Δh (LCh)",
    "delta_H_HSL": "ΔH (HSL)",
    "delta_L": "ΔL*",
    "delta_a": "Δa*",
    "delta_b": "Δb*",
    "delta_C": "ΔC*",
}

METRIC_UNITS = {
    "delta_E00": "Unidades ΔE00",
    "delta_L_abs": "Unidades CIELAB",
    "delta_a_abs": "Unidades CIELAB",
    "delta_b_abs": "Unidades CIELAB",
    "delta_C_abs": "Unidades CIELAB",
    "delta_h_LCh": "Graus",
    "delta_H_HSL": "Graus",
    "delta_L": "Unidades CIELAB",
    "delta_a": "Unidades CIELAB",
    "delta_b": "Unidades CIELAB",
    "delta_C": "Unidades CIELAB",
}

REQUIRED_COLUMNS = [
    "input_hex",
    "scheme",
    "output_id",
    "delta_E00",
    "delta_L",
    "delta_a",
    "delta_b",
    "delta_C",
    "delta_h_LCh",
    "delta_H_HSL",
]


# ============================================================
# LEITURA E VALIDAÇÃO
# ============================================================

def load_data(csv_path: str | Path) -> pd.DataFrame:
    """Carrega e valida o CSV de comparação."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {path}")

    df = pd.read_csv(path)

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "O CSV não contém todas as colunas necessárias. "
            f"Colunas ausentes: {missing}"
        )

    df = df.copy()

    for column in ["input_hex", "scheme", "output_id"]:
        df[column] = df[column].astype("string").str.strip()

    df["scheme"] = df["scheme"].str.lower()

    numeric_columns = [
        "delta_E00", "delta_L", "delta_a", "delta_b", "delta_C",
        "delta_h_LCh", "delta_H_HSL",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=[
        "input_hex", "scheme", "output_id", *numeric_columns
    ])

    invalid_schemes = sorted(set(df["scheme"]) - set(SCHEMES))
    if invalid_schemes:
        raise ValueError(f"Esquemas não reconhecidos no CSV: {invalid_schemes}")

    if df.empty:
        raise ValueError("Nenhuma linha válida foi encontrada no CSV.")

    # A chave identifica uma saída específica de uma entrada/esquema.
    duplicated = df.duplicated(
        subset=["input_hex", "scheme", "output_id"], keep=False
    )
    if duplicated.any():
        examples = df.loc[
            duplicated, ["input_hex", "scheme", "output_id"]
        ].head(10)
        raise ValueError(
            "Há saídas duplicadas para a mesma entrada/esquema/output_id. "
            f"Exemplos:\n{examples.to_string(index=False)}"
        )

    # Confere a quantidade esperada de saídas em cada entrada/esquema.
    counts = (
        df.groupby(["input_hex", "scheme"])["output_id"]
        .nunique()
        .reset_index(name="n_outputs")
    )
    counts["expected"] = counts["scheme"].map(EXPECTED_OUTPUTS)
    wrong = counts[counts["n_outputs"] != counts["expected"]]

    if not wrong.empty:
        raise ValueError(
            "Quantidade inesperada de saídas em algumas combinações "
            "entrada/esquema. Esperado: 2 para análoga, triádica e "
            "complementar partida; 1 para complementar.\n"
            f"{wrong.head(20).to_string(index=False)}"
        )

    # ΔE00 e diferenças angulares são magnitudes não negativas.
    for column in ["delta_E00", *ANGULAR_METRICS]:
        if (df[column] < 0).any():
            raise ValueError(
                f"A coluna {column} contém valores negativos. "
                "Verifique o cálculo da distância/magnitude."
            )

    return df


# ============================================================
# AGREGAÇÃO POR COR DE ENTRADA
# ============================================================

def aggregate_per_color(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera uma observação por input_hex + scheme.

    Para métricas assinadas, guarda:
      - média assinada entre as saídas;
      - média dos valores absolutos entre as saídas.

    A segunda medida evita que diferenças positivas e negativas se
    cancelem ao calcular a magnitude média.

    Para métricas angulares e ΔE00, calcula a média das saídas.
    """
    records = []

    for (input_hex, scheme), group in df.groupby(
        ["input_hex", "scheme"], sort=False
    ):
        record = {
            "input_hex": input_hex,
            "scheme": scheme,
            "n_outputs": len(group),
        }

        record[PERCEPTUAL_METRIC] = group[PERCEPTUAL_METRIC].mean()

        for metric in SIGNED_METRICS:
            record[f"{metric}_signed"] = group[metric].mean()
            record[f"{metric}_abs"] = group[metric].abs().mean()

        for metric in ANGULAR_METRICS:
            record[metric] = group[metric].mean()

        records.append(record)

    result = pd.DataFrame(records)

    # Mantém uma ordem previsível para facilitar inspeção e exportação.
    result["scheme"] = pd.Categorical(
        result["scheme"], categories=SCHEMES, ordered=True
    )
    return result.sort_values(["scheme", "input_hex"]).reset_index(drop=True)


# ============================================================
# ESTATÍSTICAS DESCRITIVAS
# ============================================================

def describe_series(values: pd.Series) -> dict:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return {
            "n": 0, "mean": np.nan, "std": np.nan, "variance": np.nan,
            "median": np.nan, "min": np.nan, "max": np.nan,
        }

    return {
        "n": int(values.count()),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
        "variance": float(values.var(ddof=1)) if len(values) > 1 else np.nan,
        "median": float(values.median()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def calculate_statistics(per_color: pd.DataFrame) -> pd.DataFrame:
    variables = [
        PERCEPTUAL_METRIC,
        *MAGNITUDE_METRICS,
        *ANGULAR_METRICS,
        *[f"{metric}_signed" for metric in SIGNED_METRICS],
    ]

    rows = []
    for scheme in SCHEMES:
        subset = per_color[per_color["scheme"] == scheme]
        for variable in variables:
            stats = describe_series(subset[variable])
            rows.append({
                "scheme": scheme,
                "scheme_label": SCHEME_LABELS[scheme],
                "metric": variable,
                **stats,
            })

    return pd.DataFrame(rows)


def calculate_percentiles(per_color: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scheme in SCHEMES:
        subset = per_color.loc[
            per_color["scheme"] == scheme, PERCEPTUAL_METRIC
        ].dropna()

        row = {
            "scheme": scheme,
            "scheme_label": SCHEME_LABELS[scheme],
            "n": int(subset.count()),
        }
        for percentile in PERCENTILES:
            row[f"p{percentile}"] = (
                float(np.percentile(subset, percentile))
                if not subset.empty else np.nan
            )
        rows.append(row)

    return pd.DataFrame(rows)


def calculate_extremes(per_color: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scheme in SCHEMES:
        values = per_color.loc[
            per_color["scheme"] == scheme, PERCEPTUAL_METRIC
        ].dropna()
        rows.append({
            "scheme": scheme,
            "scheme_label": SCHEME_LABELS[scheme],
            "n": int(values.count()),
            "min_delta_E00": float(values.min()) if not values.empty else np.nan,
            "max_delta_E00": float(values.max()) if not values.empty else np.nan,
        })
    return pd.DataFrame(rows)


# ============================================================
# FUNÇÕES AUXILIARES PARA GRÁFICOS
# ============================================================

def finish_figure(fig, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def scheme_values(per_color: pd.DataFrame, scheme: str, metric: str) -> pd.Series:
    return (
        per_color.loc[per_color["scheme"] == scheme, metric]
        .dropna()
        .astype(float)
    )


def create_four_panel_figure(title: str, ylabel: str):
    """Cria uma figura 2x2, um painel independente para cada esquema."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    fig.suptitle(title, fontsize=15, fontweight="bold")
    for ax, scheme in zip(axes, SCHEMES):
        ax.set_title(SCHEME_LABELS[scheme], fontsize=12)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    return fig, axes


# ============================================================
# GRÁFICOS DE ΔE00 — QUATRO ARQUIVOS INDIVIDUAIS
# ============================================================

def plot_mean_delta_e(stats: pd.DataFrame) -> None:
    data = stats[stats["metric"] == PERCEPTUAL_METRIC].set_index("scheme")
    values = [data.loc[s, "mean"] for s in SCHEMES]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        [SCHEME_LABELS[s] for s in SCHEMES],
        values,
    )
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_title(f"ΔE00 médio — {MODEL_1} × {MODEL_2}")
    ax.set_ylabel("ΔE00 médio")
    ax.set_xlabel("Esquema de cores")
    ax.grid(axis="y", alpha=0.25)
    finish_figure(fig, "01_delta_E00_medio_por_esquema.png")


def plot_mean_delta_e_std(stats: pd.DataFrame) -> None:
    data = stats[stats["metric"] == PERCEPTUAL_METRIC].set_index("scheme")
    means = [data.loc[s, "mean"] for s in SCHEMES]
    stds = [
        data.loc[s, "std"] if pd.notna(data.loc[s, "std"]) else 0
        for s in SCHEMES
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        [SCHEME_LABELS[s] for s in SCHEMES],
        means,
        yerr=stds,
        capsize=6,
    )
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_title(f"ΔE00 médio e desvio-padrão — {MODEL_1} × {MODEL_2}")
    ax.set_ylabel("ΔE00")
    ax.set_xlabel("Esquema de cores")
    ax.grid(axis="y", alpha=0.25)
    finish_figure(fig, "02_delta_E00_medio_com_dp.png")


def plot_delta_e_distribution(per_color: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    data = [scheme_values(per_color, s, PERCEPTUAL_METRIC) for s in SCHEMES]
    ax.boxplot(
        data,
        tick_labels=[SCHEME_LABELS[s] for s in SCHEMES],
        showfliers=True,
    )
    ax.set_title(f"Distribuição de ΔE00 — {MODEL_1} × {MODEL_2}")
    ax.set_ylabel("ΔE00")
    ax.set_xlabel("Esquema de cores")
    ax.grid(axis="y", alpha=0.25)
    finish_figure(fig, "03_distribuicao_delta_E00.png")


def plot_delta_e_percentiles(percentiles: pd.DataFrame) -> None:
    """Gráfico de linhas: uma curva por esquema ao longo dos percentis."""
    fig, ax = plt.subplots(figsize=(11, 6))

    # Valores numéricos para que o espaçamento seja proporcional
    x_values = PERCENTILES

    for scheme in SCHEMES:
        row = percentiles.loc[percentiles["scheme"] == scheme].iloc[0]
        values = [row[f"p{p}"] for p in PERCENTILES]

        ax.plot(
            x_values,
            values,
            marker="o",
            linewidth=2,
            label=SCHEME_LABELS[scheme],
        )

    ax.set_title(f"Percentis de ΔE00 — {MODEL_1} × {MODEL_2}")
    ax.set_xlabel("Percentil")
    ax.set_ylabel("ΔE00")

    # Mantém os rótulos P25, P50 etc., mas respeita as distâncias reais
    ax.set_xticks(x_values)
    ax.set_xticklabels([f"P{p}" for p in PERCENTILES])

    ax.set_xlim(24, 100)
    ax.legend(title="Esquema")
    ax.grid(alpha=0.3)

    finish_figure(fig, "04_percentis_delta_E00.png")

# ============================================================
# GRÁFICOS AGRUPADOS — MÉTRICAS ABSOLUTAS CIELAB
# Cada figura tem quatro painéis (um por esquema); dentro de cada
# painel, as métricas do grupo aparecem juntas.
# ============================================================


def plot_metric_means(stats: pd.DataFrame) -> None:
    """Grupo 1: médias das quatro magnitudes absolutas CIELAB."""
    data = stats[stats["metric"].isin(MAGNITUDE_METRICS)]
    fig, axes = create_four_panel_figure(
        f"Médias das diferenças absolutas CIELAB — {MODEL_1} × {MODEL_2}",
        "Diferença média (unidades CIELAB)",
    )

    labels = [METRIC_LABELS[m] for m in MAGNITUDE_METRICS]
    x = np.arange(len(MAGNITUDE_METRICS))

    for ax, scheme in zip(axes, SCHEMES):
        subset = data[data["scheme"] == scheme].set_index("metric")
        means = [subset.loc[m, "mean"] for m in MAGNITUDE_METRICS]
        stds = [
            subset.loc[m, "std"] if pd.notna(subset.loc[m, "std"]) else 0
            for m in MAGNITUDE_METRICS
        ]
        bars = ax.bar(x, means, yerr=stds, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Métrica")
        ax.tick_params(axis="x", labelrotation=0, labelsize=9)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
        ax.set_ylim(bottom=0)

    finish_figure(fig, "grupo_1_medias_abs_CIELAB.png")


def plot_metric_distributions(per_color: pd.DataFrame) -> None:
    """Grupo 2: distribuições das quatro magnitudes absolutas CIELAB."""
    fig, axes = create_four_panel_figure(
        f"Distribuição das diferenças absolutas CIELAB — {MODEL_1} × {MODEL_2}",
        "Diferença (unidades CIELAB)",
    )
    labels = [METRIC_LABELS[m] for m in MAGNITUDE_METRICS]

    for ax, scheme in zip(axes, SCHEMES):
        data = [scheme_values(per_color, scheme, metric) for metric in MAGNITUDE_METRICS]
        ax.boxplot(data, tick_labels=labels, showfliers=True)
        ax.set_xlabel("Métrica")
        ax.tick_params(axis="x", labelrotation=0, labelsize=9)

    finish_figure(fig, "grupo_2_distribuicoes_abs_CIELAB.png")


# ============================================================
# GRÁFICOS AGRUPADOS — MÉTRICAS ASSINADAS CIELAB
# ============================================================


def plot_signed_distributions(per_color: pd.DataFrame) -> None:
    """Grupo 3: distribuições das quatro diferenças assinadas CIELAB."""
    signed_columns = [f"{metric}_signed" for metric in SIGNED_METRICS]
    labels = [METRIC_LABELS[m] for m in SIGNED_METRICS]
    fig, axes = create_four_panel_figure(
        f"Distribuição das diferenças assinadas CIELAB — {MODEL_1} × {MODEL_2}",
        "Diferença assinada (unidades CIELAB)",
    )

    for ax, scheme in zip(axes, SCHEMES):
        data = [scheme_values(per_color, scheme, metric) for metric in signed_columns]
        ax.boxplot(data, tick_labels=labels, showfliers=True)
        ax.axhline(0, linestyle="--", linewidth=1, color="black")
        ax.set_xlabel("Métrica")
        ax.tick_params(axis="x", labelrotation=0, labelsize=9)

    finish_figure(fig, "grupo_3_distribuicoes_assinadas_CIELAB.png")


# ============================================================
# GRÁFICOS AGRUPADOS — DIFERENÇAS ANGULARES (GRAUS)
# delta_h_LCh e delta_H_HSL ficam juntas entre si, mas separadas
# das métricas CIELAB por terem unidade angular.
# ============================================================


def plot_angular_means(stats: pd.DataFrame) -> None:
    data = stats[stats["metric"].isin(ANGULAR_METRICS)]
    fig, axes = create_four_panel_figure(
        f"Médias das diferenças angulares — {MODEL_1} × {MODEL_2}",
        "Diferença angular média (graus)",
    )
    labels = [METRIC_LABELS[m] for m in ANGULAR_METRICS]
    x = np.arange(len(ANGULAR_METRICS))

    for ax, scheme in zip(axes, SCHEMES):
        subset = data[data["scheme"] == scheme].set_index("metric")
        means = [subset.loc[m, "mean"] for m in ANGULAR_METRICS]
        stds = [
            subset.loc[m, "std"] if pd.notna(subset.loc[m, "std"]) else 0
            for m in ANGULAR_METRICS
        ]
        bars = ax.bar(x, means, yerr=stds, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Métrica")
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
        ax.set_ylim(bottom=0)

    finish_figure(fig, "medias_diferencas_angulares.png")


def plot_angular_distributions(per_color: pd.DataFrame) -> None:
    fig, axes = create_four_panel_figure(
        f"Distribuição das diferenças angulares — {MODEL_1} × {MODEL_2}",
        "Diferença angular (graus)",
    )
    labels = [METRIC_LABELS[m] for m in ANGULAR_METRICS]

    for ax, scheme in zip(axes, SCHEMES):
        data = [scheme_values(per_color, scheme, metric) for metric in ANGULAR_METRICS]
        ax.boxplot(data, tick_labels=labels, showfliers=True)
        ax.set_xlabel("Métrica")

    finish_figure(fig, "distribuicoes_diferencas_angulares.png")


# ============================================================
# RELATÓRIO TEXTUAL
# ============================================================

def generate_report(
    per_color: pd.DataFrame,
    stats: pd.DataFrame,
    percentiles: pd.DataFrame,
    extremes: pd.DataFrame,
) -> None:
    report_path = OUTPUT_DIR / "relatorio_analise.txt"
    lines = [
        f"ANÁLISE: {MODEL_1} × {MODEL_2}",
        f"Arquivo de entrada: {INPUT_CSV}",
        f"Data da execução: {pd.Timestamp.now()}",
        "",
        "A unidade de análise estatística é a combinação input_hex + scheme.",
        "Quando há duas saídas no esquema, as métricas são primeiro agregadas",
        "entre as saídas da mesma entrada; assim, cada cor de entrada tem",
        "o mesmo peso na estatística do respectivo esquema.",
        "",
        "ΔE00:",
    ]

    delta_stats = stats[stats["metric"] == PERCEPTUAL_METRIC]
    for _, row in delta_stats.iterrows():
        lines.append(
            f"- {row['scheme_label']}: n={row['n']}, "
            f"média={row['mean']:.4f}, DP={row['std']:.4f}, "
            f"mediana={row['median']:.4f}, "
            f"mín={row['min']:.4f}, máx={row['max']:.4f}"
        )

    lines += ["", "Percentis de ΔE00:"]
    for _, row in percentiles.iterrows():
        values = ", ".join(
            f"P{p}={row[f'p{p}']:.4f}" for p in PERCENTILES
        )
        lines.append(f"- {row['scheme_label']}: {values}")

    lines += ["", "Extremos de ΔE00:"]
    for _, row in extremes.iterrows():
        lines.append(
            f"- {row['scheme_label']}: "
            f"mín={row['min_delta_E00']:.4f}; "
            f"máx={row['max_delta_E00']:.4f}"
        )

    lines += [
        "",
        "Observação: o desvio-padrão é uma medida de dispersão descritiva,",
        "não um intervalo de confiança.",
        "As métricas assinadas representam MODEL_1 - MODEL_2 conforme",
        "a convenção adotada na geração do CSV. As versões absolutas",
        "são usadas para avaliar magnitude sem cancelamento de sinais.",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# EXECUÇÃO
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Lendo: {INPUT_CSV}")
    df = load_data(INPUT_CSV)
    print(f"Linhas válidas: {len(df):,}")

    per_color = aggregate_per_color(df)
    stats = calculate_statistics(per_color)
    percentiles = calculate_percentiles(per_color)
    extremes = calculate_extremes(per_color)

    # CSVs estatísticos.
    per_color.to_csv(
        OUTPUT_DIR / "per_color_scheme.csv",
        index=False,
        encoding="utf-8-sig",
    )
    stats.to_csv(
        OUTPUT_DIR / "statistics_by_scheme.csv",
        index=False,
        encoding="utf-8-sig",
    )
    percentiles.to_csv(
        OUTPUT_DIR / "delta_E00_percentiles_by_scheme.csv",
        index=False,
        encoding="utf-8-sig",
    )
    extremes.to_csv(
        OUTPUT_DIR / "min_max_delta_E00_by_scheme.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Quatro gráficos individuais de ΔE00.
    plot_mean_delta_e(stats)
    plot_mean_delta_e_std(stats)
    plot_delta_e_distribution(per_color)
    plot_delta_e_percentiles(percentiles)

    # Três grupos CIELAB, cada um em uma figura com quatro painéis.
    plot_metric_means(stats)
    plot_metric_distributions(per_color)
    plot_signed_distributions(per_color)

    # As duas métricas angulares são agrupadas entre si, separadas de CIELAB.
    plot_angular_means(stats)
    plot_angular_distributions(per_color)

    generate_report(per_color, stats, percentiles, extremes)

    print(f"\nAnálise concluída. Resultados em: {OUTPUT_DIR.resolve()}")
    print("Arquivos gerados:")
    for path in sorted(OUTPUT_DIR.iterdir()):
        print(f" - {path.name}")


if __name__ == "__main__":
    main()