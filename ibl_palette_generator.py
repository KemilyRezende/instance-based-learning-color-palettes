import numpy as np
import pandas as pd
import faiss
import skimage.color as skc
import os

# ==============================
# CONFIG
# ==============================


'''
Modelo      K   p
faiss       50  0
faiss_m2    25  0
faiss_m3    50  1
faiss_m4    25  1
faiss_m5    50  2
faiss_m6    25  2
'''


INPUT_CSV = "soble_colors.csv"  
OUTPUT_CSV = "faiss_m6_raw.csv"
K = 25

# ==============================
# CONVERSÕES
# ==============================

def rgb_255_to_01(rgb):
    return np.array(rgb) / 255.0


def rgb_to_lab(rgb_01):
    return skc.rgb2lab(rgb_01[None, None, :])[0, 0]


def lab_to_rgb(lab):
    return skc.lab2rgb(lab[None, None, :])[0, 0]


def rgb_01_to_255(rgb):
    return np.round(np.clip(rgb, 0, 1) * 255).astype(int)


def rgb_to_hex(rgb):
    rgb_int = [int(np.clip(round(x), 0, 255)) for x in rgb]
    return "#{:02X}{:02X}{:02X}".format(*rgb_int)


# ==============================
# FAISS
# ==============================

def load_faiss_index(csv_path):
    df = pd.read_csv(csv_path)

    X = df[["L", "a", "b"]].values.astype("float32")
    deltas = df[["dL", "da", "db"]].values.astype("float32")

    index = faiss.IndexFlatL2(3)
    index.add(X)

    return index, deltas

def apply_deltas(lab, deltas):
    L2 = np.clip(lab[0] + deltas[0], 0, 100)
    a2 = lab[1] + deltas[1]
    b2 = lab[2] + deltas[2]
    return np.array([L2, a2, b2])

def find_deltas(lab, index, deltas, k=25, power=2):
    query = lab.astype("float32")[None, :]

    # FAISS retorna distâncias euclidianas ao quadrado
    distances_squared, indices = index.search(query, k)

    distances_squared = distances_squared[0]
    selected = deltas[indices[0]]

    # Evita problemas numéricos
    distances_squared = np.maximum(distances_squared, 0)

    # Tratamento de correspondências exatas
    exact_matches = distances_squared <= 1e-12

    if np.any(exact_matches):
        weights = exact_matches.astype("float64")
    else:
        distances = np.sqrt(distances_squared)

        # Ponderação pelo inverso da distância elevada a power
        weights = 1.0 / (distances + 1e-12) ** power

    # Normalização dos pesos
    weights /= weights.sum()

    # Média ponderada dos deslocamentos Lab
    weighted_deltas = np.average(
        selected,
        axis=0,
        weights=weights
    )

    return weighted_deltas


def generate_from_index(lab, index, deltas, k=25, power=2):
    mean_deltas = find_deltas(
        lab,
        index,
        deltas,
        k=k,
        power=power
    )

    lab2 = apply_deltas(lab, mean_deltas)

    rgb2 = lab_to_rgb(lab2)

    return rgb_01_to_255(rgb2), lab2


# ==============================
# LOAD SCHEMES
# ==============================

schemes = {
    "ang_left": load_faiss_index("analog_left.csv"),
    "ang_right": load_faiss_index("analog_right.csv"),
    "comp": load_faiss_index("complementary.csv"),
    "scomp_left": load_faiss_index("split_complementary_left.csv"),
    "scomp_right": load_faiss_index("split_complementary_right.csv"),
    "tri_left": load_faiss_index("triadic_left.csv"),
    "tri_right": load_faiss_index("triadic_right.csv"),
}

# ==============================
# PROCESSAMENTO
# ==============================

df_input = pd.read_csv(INPUT_CSV)

results = []

for _, row in df_input.iterrows():
    rgb_input = [row["r"], row["g"], row["b"]]

    rgb_01 = rgb_255_to_01(rgb_input)
    lab = rgb_to_lab(rgb_01)

    for method, (index, deltas) in schemes.items():
        rgb_out, lab_out = generate_from_index(lab, index, deltas, K)

        results.append({
            # INPUT
            "input_hex": row['hex'],
            "input_r": rgb_input[0],
            "input_g": rgb_input[1],
            "input_b": rgb_input[2],
            "input_L": lab[0],
            "input_a": lab[1],
            "input_b_lab": lab[2],

            # METADATA
            "source": "faiss_m6",
            "method": method,

            # OUTPUT
            "output_hex": rgb_to_hex(rgb_out),
            "output_r": int(rgb_out[0]),
            "output_g": int(rgb_out[1]),
            "output_b": int(rgb_out[2]),
            "output_L": lab_out[0],
            "output_a": lab_out[1],
            "output_b_lab": lab_out[2],
        })

# ==============================
# SAVE
# ==============================

df_out = pd.DataFrame(results)

df_out.to_csv(OUTPUT_CSV, index=False)

print(f"Salvo em {OUTPUT_CSV} com {len(df_out)} linhas.")