
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURAÇÕES
# ============================================================

FILES = {
    "M1": "comparacao_m1_comercial.csv",
    "M2": "comparacao_m2_comercial.csv",
    "M3": "comparacao_m3_comercial.csv",
    "M4": "comparacao_m4_comercial.csv",
    "M5": "comparacao_m5_comercial.csv",
    "M6": "comparacao_m6_comercial.csv",
}

OUTPUT_DIR = Path("Analise_Consolidada_Modelos_Comercial")

MODELS = list(FILES.keys())

SCHEMES = [
    "analogous",
    "complementary",
    "split_complementary",
    "triadic",
]

SCHEME_NAMES = {
    "analogous": "Análoga",
    "complementary": "Complementar",
    "split_complementary": "Complementar partida",
    "triadic": "Triádica",
}

PERCENTILES = [25, 50, 75, 90, 95, 99]

# Métricas assinadas: serão convertidas em magnitudes absolutas.
LAB_METRICS = {
    "delta_L": "|ΔL*|",
    "delta_b": "|Δb*|",
    "delta_C": "|ΔC*|",
}

# Diferenças angulares: já representam menores distâncias angulares.
ANGULAR_METRICS = {
    "delta_h_LCh": "Δh LCh",
    "delta_H_HSL": "ΔH HSL",
}

REQUIRED_COLUMNS = {
    "input_hex",
    "scheme",
    "delta_E00",
    "delta_L",
    "delta_b",
    "delta_C",
    "delta_h_LCh",
    "delta_H_HSL",
}


# ============================================================
# CARREGAMENTO E TRATAMENTO
# ============================================================

def load_model_data(model, filepath):

    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {path}"
        )

    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"{path.name}: colunas ausentes: {missing}"
        )

    df = df.copy()

    df["input_hex"] = (
        df["input_hex"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["scheme"] = (
        df["scheme"]
        .astype(str)
        .str.strip()
    )

    numeric_columns = list(REQUIRED_COLUMNS - {
        "input_hex",
        "scheme"
    })

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Cria métricas de magnitude para as diferenças Lab.
    for column in LAB_METRICS:
        df[f"{column}_abs"] = df[column].abs()

    # Uma observação por entrada e esquema.
    # Primeiro calcula a média das saídas de cada entrada.
    aggregation = {
        "delta_E00": "mean",
        "delta_L_abs": "mean",
        "delta_b_abs": "mean",
        "delta_C_abs": "mean",
        "delta_h_LCh": "mean",
        "delta_H_HSL": "mean",
    }

    per_input = (
        df.groupby(
            ["input_hex", "scheme"],
            as_index=False
        )
        .agg(aggregation)
    )

    per_input["model"] = model

    return per_input


def load_all_models():

    datasets = []

    for model, filepath in FILES.items():

        print(f"Carregando {filepath}...")

        datasets.append(
            load_model_data(model, filepath)
        )

    return pd.concat(
        datasets,
        ignore_index=True
    )


# ============================================================
# VERIFICAÇÃO DA COBERTURA
# ============================================================

def check_input_coverage(data):

    print("\nVERIFICAÇÃO DAS ENTRADAS")
    print("=" * 60)

    for scheme in SCHEMES:

        scheme_data = data[
            data["scheme"] == scheme
        ]

        input_sets = {
            model: set(
                scheme_data.loc[
                    scheme_data["model"] == model,
                    "input_hex"
                ]
            )
            for model in MODELS
        }

        reference = input_sets["M1"]

        print(f"\n{SCHEME_NAMES[scheme]}")

        for model, inputs in input_sets.items():
            print(f"{model}: {len(inputs)} entradas")

        if all(
            inputs == reference
            for inputs in input_sets.values()
        ):
            print("OK: todos os modelos possuem as mesmas entradas.")
        else:
            print(
                "ATENÇÃO: existem diferenças entre os conjuntos de entradas."
            )


# ============================================================
# ESTATÍSTICAS
# ============================================================

def generate_summary(data):

    metrics = [
        "delta_E00",
        "delta_L_abs",
        "delta_b_abs",
        "delta_C_abs",
        "delta_h_LCh",
        "delta_H_HSL",
    ]

    records = []

    for (scheme, model), group in data.groupby(
        ["scheme", "model"]
    ):

        for metric in metrics:

            values = group[metric].dropna()

            if values.empty:
                continue

            row = {
                "scheme": scheme,
                "model": model,
                "metric": metric,
                "n": len(values),
                "mean": values.mean(),
                "std": values.std(),
                "median": values.median(),
                "min": values.min(),
                "max": values.max(),
            }

            for p in PERCENTILES:
                row[f"p{p}"] = np.percentile(
                    values,
                    p
                )

            records.append(row)

    return pd.DataFrame(records)


# ============================================================
# FUNÇÃO GENÉRICA PARA GRÁFICOS DE MÉDIA ± DP
# ============================================================

def plot_mean_std_panels(
    data,
    metrics,
    metric_labels,
    filename,
    figure_title,
    y_label
):
    """
    Gera uma figura 2x2, com um painel por esquema.

    Cada modelo é representado pela média da métrica
    e por barras de erro correspondentes a um desvio padrão.
    """

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 11)
    )

    axes = axes.flatten()

    for ax, scheme in zip(axes, SCHEMES):

        scheme_data = data[
            data["scheme"] == scheme
        ]

        x = np.arange(len(metrics))
        width = 0.12

        for i, model in enumerate(MODELS):

            model_data = scheme_data[
                scheme_data["model"] == model
            ]

            means = []
            stds = []

            for metric in metrics:

                values = model_data[metric].dropna()

                if values.empty:
                    means.append(np.nan)
                    stds.append(np.nan)
                else:
                    means.append(values.mean())
                    stds.append(
                        values.std() if len(values) > 1 else 0
                    )

            positions = (
                x - (len(MODELS) - 1) * width / 2
                + i * width
            )

            ax.bar(
                positions,
                means,
                width,
                yerr=stds,
                capsize=3,
                label=model,
                error_kw={"elinewidth": 0.9}
            )

        ax.set_title(SCHEME_NAMES[scheme])
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)

        ax.set_ylabel(y_label)

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.4
        )

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=6,
        bbox_to_anchor=(0.5, 0.96)
    )

    fig.suptitle(
        figure_title,
        fontsize=15,
        y=1.02
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.90]
    )

    fig.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Figura gerada: {filename}")


# ============================================================
# FIGURA 1 — ΔE00: MÉDIA ± DESVIO PADRÃO
# ============================================================

def plot_delta_e_mean_std(data):

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15, 10)
    )

    axes = axes.flatten()

    for ax, scheme in zip(axes, SCHEMES):

        scheme_data = data[
            data["scheme"] == scheme
        ]

        means = []
        stds = []

        for model in MODELS:

            values = scheme_data.loc[
                scheme_data["model"] == model,
                "delta_E00"
            ].dropna()

            means.append(values.mean() if len(values) else np.nan)
            stds.append(
                values.std() if len(values) > 1 else 0
            )

        positions = np.arange(len(MODELS))

        ax.errorbar(
            positions,
            means,
            yerr=stds,
            fmt="o",
            capsize=5,
            markersize=7
        )

        ax.set_xticks(positions)
        ax.set_xticklabels(MODELS)

        ax.set_title(SCHEME_NAMES[scheme])
        ax.set_xlabel("Modelos")
        ax.set_ylabel("ΔE00 médio ± desvio padrão")

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.5
        )

    fig.suptitle(
        "Média e desvio padrão de ΔE00 — Modelos × Comercial",
        fontsize=15
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "01_delta_E00_media_dp.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# FIGURA 2 — ΔE00: PERCENTIS
# ============================================================

def plot_delta_e_percentiles(data):

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15, 10)
    )

    axes = axes.flatten()

    percentile_labels = [
        f"P{p}" for p in PERCENTILES
    ]

    for ax, scheme in zip(axes, SCHEMES):

        scheme_data = data[
            data["scheme"] == scheme
        ]

        for model in MODELS:

            values = scheme_data.loc[
                scheme_data["model"] == model,
                "delta_E00"
            ].dropna()

            if values.empty:
                continue

            percentile_values = [
                np.percentile(values, p)
                for p in PERCENTILES
            ]

            ax.plot(
                percentile_labels,
                percentile_values,
                marker="o",
                linewidth=1.8,
                label=model
            )

        ax.set_title(SCHEME_NAMES[scheme])
        ax.set_xlabel("Percentis")
        ax.set_ylabel("ΔE00")

        ax.grid(
            linestyle="--",
            alpha=0.4
        )

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=6,
        bbox_to_anchor=(0.5, 0.96)
    )

    fig.suptitle(
        "Percentis de ΔE00 — Modelos × Comercial",
        fontsize=15,
        y=1.02
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.90]
    )

    fig.savefig(
        OUTPUT_DIR / "02_delta_E00_percentis.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# FIGURA 3 — DIFERENÇAS EM LAB: MÉDIA ± DP
# ============================================================

def plot_lab_metrics(data):

    plot_mean_std_panels(
        data=data,
        metrics=[
            "delta_L_abs",
            "delta_b_abs",
            "delta_C_abs",
        ],
        metric_labels=[
            "|ΔL*|",
            "|Δb*|",
            "|ΔC*|",
        ],
        filename="03_diferencas_Lab_media_dp.png",
        figure_title=(
            "Magnitudes médias e desvios padrão das diferenças em CIELAB"
        ),
        y_label="Diferença absoluta média (unidades CIELAB)"
    )


# ============================================================
# FIGURA 4 — DIFERENÇAS ANGULARES: MÉDIA ± DP
# ============================================================

def plot_angular_metrics(data):

    plot_mean_std_panels(
        data=data,
        metrics=[
            "delta_h_LCh",
            "delta_H_HSL",
        ],
        metric_labels=[
            "Δh LCh",
            "ΔH HSL",
        ],
        filename="04_diferencas_angulares_media_dp.png",
        figure_title=(
            "Médias e desvios padrão das diferenças angulares"
        ),
        y_label="Diferença angular média (graus)"
    )


# ============================================================
# EXECUÇÃO
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Carrega os seis arquivos
    data = load_all_models()

    # Verifica se os modelos possuem as mesmas entradas
    check_input_coverage(data)

    # Salva os dados agregados por entrada
    data.to_csv(
        OUTPUT_DIR / "resultados_por_entrada.csv",
        index=False
    )

    # Gera tabela estatística completa
    summary = generate_summary(data)

    summary.to_csv(
        OUTPUT_DIR / "estatisticas_por_modelo_esquema.csv",
        index=False
    )

    # Gera as quatro figuras
    plot_delta_e_mean_std(data)
    plot_delta_e_percentiles(data)
    plot_lab_metrics(data)
    plot_angular_metrics(data)

    print("\n" + "=" * 65)
    print("ANÁLISE CONCLUÍDA")
    print("=" * 65)

    print(f"Pasta: {OUTPUT_DIR.resolve()}")

    print("\nArquivos gerados:")

    for path in sorted(OUTPUT_DIR.iterdir()):
        print(f" - {path.name}")


if __name__ == "__main__":
    main()
