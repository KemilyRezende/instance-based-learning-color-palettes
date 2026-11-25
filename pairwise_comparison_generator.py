
import numpy as np
import pandas as pd

from skimage.color import deltaE_ciede2000


# ============================================================
# CONFIGURAÇÃO
# ============================================================

# As fontes (source) são comercial, theory, faiss, faiss_m2, faiss_m3, faiss_m4, faiss_m5 e faiss_mg

SOURCE_1 = "comercial"
SOURCE_2 = "theory"

CSV_MODEL_1 = "comercial_raw.csv"
CSV_MODEL_2 = "theory_colors.csv"
CSV_OUTPUT = "comparacao_comercial_teoria.csv"

# Tolerância para matizes indefinidas
EPSILON = 1e-10


# ============================================================
# CONFIGURAÇÃO DOS MÉTODOS
# ============================================================

SCHEMES = {
    "analogous": {
        "methods": ["ang_left", "ang_right"],
        "output_ids": ["ang_1", "ang_2"],
    },
    "complementary": {
        "methods": ["comp"],
        "output_ids": ["comp"],
    },
    "split_complementary": {
        "methods": ["scomp_left", "scomp_right"],
        "output_ids": ["scomp_1", "scomp_2"],
    },
    "triadic": {
        "methods": ["tri_left", "tri_right"],
        "output_ids": ["tri_1", "tri_2"],
    },
}

METHOD_TO_SCHEME = {
    method: scheme
    for scheme, config in SCHEMES.items()
    for method in config["methods"]
}

REQUIRED_COLUMNS = [
    "input_hex",
    "input_r",
    "input_g",
    "input_b",
    "input_L",
    "input_a",
    "input_b_lab",
    "source",
    "method",
    "output_hex",
    "output_r",
    "output_g",
    "output_b",
    "output_L",
    "output_a",
    "output_b_lab",
]


# ============================================================
# CARREGAMENTO DOS DADOS
# ============================================================

def load_csv(path, expected_source):

    df = pd.read_csv(path)

    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Colunas ausentes em {path}: {missing}"
        )

    df["input_hex"] = df["input_hex"].str.upper()
    df["output_hex"] = df["output_hex"].str.upper()
    df["method"] = df["method"].str.lower()

    df = df[df["source"] == expected_source].copy()

    if df.empty:
        raise ValueError(
            f"Nenhum registro encontrado para "
            f"source='{expected_source}' em {path}."
        )

    unknown = set(df["method"].dropna()) - set(METHOD_TO_SCHEME)

    if unknown:
        raise ValueError(
            f"Métodos desconhecidos em {path}: {sorted(unknown)}"
        )

    df["scheme"] = df["method"].map(METHOD_TO_SCHEME)

    return df.reset_index(drop=True)


# ============================================================
# VALIDAÇÃO DOS DADOS
# ============================================================

def validate_unique_records(df, source):

    duplicated = df.duplicated(
        subset=["input_hex", "method"],
        keep=False
    )

    if duplicated.any():

        examples = df.loc[
            duplicated,
            ["input_hex", "method", "output_hex"]
        ].head(10)

        raise ValueError(
            f"Registros duplicados para {source}:\n{examples}"
        )


def validate_inputs(df1, df2):

    inputs1 = set(df1["input_hex"].dropna())
    inputs2 = set(df2["input_hex"].dropna())

    if inputs1 != inputs2:
        only_in_1 = inputs1 - inputs2
        only_in_2 = inputs2 - inputs1

        raise ValueError(
            "Os CSVs possuem conjuntos diferentes de input_hex.\n"
            f"Presentes apenas em {SOURCE_1}: {len(only_in_1)}\n"
            f"Presentes apenas em {SOURCE_2}: {len(only_in_2)}\n"
            f"Exemplos exclusivos de {SOURCE_1}: {list(only_in_1)[:5]}\n"
            f"Exemplos exclusivos de {SOURCE_2}: {list(only_in_2)[:5]}"
        )

    return inputs1

# ============================================================
# CONVERSÕES PARA MÉTRICAS
# ============================================================

def rgb_to_hsl(r, g, b):

    import colorsys

    h, l, s = colorsys.rgb_to_hls(
        r / 255.0,
        g / 255.0,
        b / 255.0
    )

    if s <= EPSILON:
        h = np.nan
    else:
        h *= 360.0

    return h, s * 100.0, l * 100.0


def lab_to_lch(L, a, b):

    C = np.sqrt(a ** 2 + b ** 2)

    if C <= EPSILON:
        h = np.nan
    else:
        h = np.degrees(np.arctan2(b, a)) % 360.0

    return L, C, h


def angular_difference(h1, h2):

    if pd.isna(h1) or pd.isna(h2):
        return np.nan

    difference = abs(h1 - h2) % 360.0

    return min(difference, 360.0 - difference)


def color_properties(row, prefix):

    H_hsl, S_hsl, L_hsl = rgb_to_hsl(
        row["output_r"],
        row["output_g"],
        row["output_b"]
    )

    L_lch, C_lch, H_lch = lab_to_lch(
        row["output_L"],
        row["output_a"],
        row["output_b_lab"]
    )

    return {
        f"{prefix}_HSL_H": H_hsl,
        f"{prefix}_HSL_S": S_hsl,
        f"{prefix}_HSL_L": L_hsl,

        f"{prefix}_LCh_L": L_lch,
        f"{prefix}_LCh_C": C_lch,
        f"{prefix}_LCh_H": H_lch,
    }


# ============================================================
# CÁLCULO DAS DIFERENÇAS
# ============================================================

def calculate_metrics(row1, row2):

    lab1 = np.array([
        row1["output_L"],
        row1["output_a"],
        row1["output_b_lab"],
    ], dtype=float)

    lab2 = np.array([
        row2["output_L"],
        row2["output_a"],
        row2["output_b_lab"],
    ], dtype=float)

    delta_e00 = float(
        deltaE_ciede2000(
            lab1.reshape(1, 1, 3),
            lab2.reshape(1, 1, 3)
        )[0, 0]
    )

    properties1 = color_properties(row1, "source_1")
    properties2 = color_properties(row2, "source_2")

    return {
        "delta_E00": delta_e00,

        "delta_L": lab1[0] - lab2[0],
        "delta_a": lab1[1] - lab2[1],
        "delta_b": lab1[2] - lab2[2],

        "delta_C": (
            properties1["source_1_LCh_C"]
            - properties2["source_2_LCh_C"]
        ),

        "delta_h_LCh": angular_difference(
            properties1["source_1_LCh_H"],
            properties2["source_2_LCh_H"]
        ),

        "delta_H_HSL": angular_difference(
            properties1["source_1_HSL_H"],
            properties2["source_2_HSL_H"]
        ),
    }


# ============================================================
# MONTAGEM DO REGISTRO
# ============================================================

def add_original_color(record, row):

    record["input_hex"] = row["input_hex"]

    for channel in ["r", "g", "b"]:
        record[f"input_rgb_{channel}"] = row[f"input_{channel}"]

    record["input_lab_L"] = row["input_L"]
    record["input_lab_a"] = row["input_a"]
    record["input_lab_b"] = row["input_b_lab"]

    H, S, L = rgb_to_hsl(
        row["input_r"],
        row["input_g"],
        row["input_b"]
    )

    L_lch, C_lch, H_lch = lab_to_lch(
        row["input_L"],
        row["input_a"],
        row["input_b_lab"]
    )

    record.update({
        "input_HSL_H": H,
        "input_HSL_S": S,
        "input_HSL_L": L,

        "input_LCh_L": L_lch,
        "input_LCh_C": C_lch,
        "input_LCh_H": H_lch,
    })


def add_output_color(record, row, prefix):

    # Todos estes valores são COPIADOS do CSV original.
    record[f"{prefix}_hex"] = row["output_hex"]

    for channel in ["r", "g", "b"]:
        record[f"{prefix}_rgb_{channel}"] = row[
            f"output_{channel}"
        ]

    record[f"{prefix}_lab_L"] = row["output_L"]
    record[f"{prefix}_lab_a"] = row["output_a"]
    record[f"{prefix}_lab_b"] = row["output_b_lab"]

    record.update(color_properties(row, prefix))


def build_record(input_row, row1, row2, scheme, output_id):

    record = {}

    add_original_color(record, input_row)

    record["scheme"] = scheme
    record["output_id"] = output_id

    record["source_1"] = row1["source"]
    record["source_2"] = row2["source"]

    add_output_color(record, row1, "source_1")
    add_output_color(record, row2, "source_2")

    record.update(calculate_metrics(row1, row2))

    return record


# ============================================================
# PAREAMENTO DAS DUAS SAÍDAS
# ============================================================

def pair_two_outputs(group1, group2):

    # Recupera exatamente as duas linhas de cada fonte.
    rows1 = {
        row["method"]: row
        for _, row in group1.iterrows()
    }

    rows2 = {
        row["method"]: row
        for _, row in group2.iterrows()
    }

    # Confere se ambas possuem os dois métodos esperados.
    expected = set(group1["method"])

    if len(rows1) != 2 or len(rows2) != 2:
        raise ValueError("Esperadas exatamente duas saídas por fonte.")

    # As duas possibilidades de pareamento.
    methods1 = list(rows1.keys())
    methods2 = list(rows2.keys())

    pairs_direct = [
        (rows1[methods1[0]], rows2[methods2[0]]),
        (rows1[methods1[1]], rows2[methods2[1]]),
    ]

    pairs_reverse = [
        (rows1[methods1[0]], rows2[methods2[1]]),
        (rows1[methods1[1]], rows2[methods2[0]]),
    ]

    direct_cost = sum(
        calculate_metrics(a, b)["delta_E00"]
        for a, b in pairs_direct
    )

    reverse_cost = sum(
        calculate_metrics(a, b)["delta_E00"]
        for a, b in pairs_reverse
    )

    if direct_cost <= reverse_cost:
        return pairs_direct

    return pairs_reverse


# ============================================================
# EXECUÇÃO DA COMPARAÇÃO
# ============================================================

def main():

    print("Carregando CSVs...")

    df1 = load_csv(CSV_MODEL_1, SOURCE_1)
    df2 = load_csv(CSV_MODEL_2, SOURCE_2)

    validate_unique_records(df1, SOURCE_1)
    validate_unique_records(df2, SOURCE_2)

    common_inputs = validate_inputs(df1, df2)

    print(f"Entradas compartilhadas: {len(common_inputs)}")

    results = []

    # Índices de saída para validar a origem de cada HEX.
    original_outputs_1 = set(
        zip(df1["input_hex"], df1["method"], df1["output_hex"])
    )

    original_outputs_2 = set(
        zip(df2["input_hex"], df2["method"], df2["output_hex"])
    )

    # Percorre cada entrada e cada esquema.
    for input_hex in sorted(common_inputs):

        for scheme, config in SCHEMES.items():

            methods = config["methods"]
            output_ids = config["output_ids"]

            group1 = df1[
                (df1["input_hex"] == input_hex)
                & (df1["method"].isin(methods))
            ]

            group2 = df2[
                (df2["input_hex"] == input_hex)
                & (df2["method"].isin(methods))
            ]

            # Verifica a quantidade esperada de registros.
            if len(group1) != len(methods) or len(group2) != len(methods):

                raise ValueError(
                    f"Quantidade incorreta de saídas: "
                    f"input={input_hex}, scheme={scheme}, "
                    f"{SOURCE_1}={len(group1)}, "
                    f"{SOURCE_2}={len(group2)}"
                )

            if len(methods) == 1:

                # Complementar: comparação direta.
                pairs = [
                    (
                        group1.iloc[0],
                        group2.iloc[0]
                    )
                ]

            else:

                # Análoga, split e triádica:
                # escolhe o pareamento de menor soma de ΔE00.
                pairs = pair_two_outputs(group1, group2)

            input_row = group1.iloc[0]

            for index, (row1, row2) in enumerate(pairs):

                # Verifica a procedência dos HEX.
                key1 = (
                    row1["input_hex"],
                    row1["method"],
                    row1["output_hex"]
                )

                key2 = (
                    row2["input_hex"],
                    row2["method"],
                    row2["output_hex"]
                )

                if key1 not in original_outputs_1:
                    raise RuntimeError(
                        f"HEX da fonte 1 não encontrado no CSV original: {key1}"
                    )

                if key2 not in original_outputs_2:
                    raise RuntimeError(
                        f"HEX da fonte 2 não encontrado no CSV original: {key2}"
                    )

                record = build_record(
                    input_row,
                    row1,
                    row2,
                    scheme,
                    output_ids[index]
                )

                results.append(record)

    # ========================================================
    # EXPORTAÇÃO
    # ========================================================

    result_df = pd.DataFrame(results)

    result_df.to_csv(
        CSV_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nComparação concluída.")
    print(f"Fonte 1: {SOURCE_1}")
    print(f"Fonte 2: {SOURCE_2}")
    print(f"Entradas processadas: {len(common_inputs)}")
    print(f"Comparações geradas: {len(result_df)}")
    print(f"Arquivo: {CSV_OUTPUT}")


if __name__ == "__main__":
    main()
