import numpy as np
import pandas as pd
from scipy.stats import qmc
import skimage.color as skc

# ==============================
# CONFIG
# ==============================

N_SAMPLES = 2663
BATCH_SIZE = 200  # gera em blocos para performance

# limites LAB (os que você encontrou)
L_MIN, L_MAX = 0.0, 100.0
A_MIN, A_MAX = -86.1830443, 98.2330704
B_MIN, B_MAX = -107.8572997, 94.4781341

# matriz XYZ → RGB linear (sRGB D65)
M = np.array([
    [ 3.2406, -1.5372, -0.4986],
    [-0.9689,  1.8758,  0.0415],
    [ 0.0557, -0.2040,  1.0570]
])

# ==============================
# FUNÇÕES
# ==============================

def lab_to_valid_rgb_linear(lab):
    """
    Converte LAB → XYZ (skimage) → RGB linear (manual)
    sem clipping
    """
    xyz = skc.lab2xyz(lab[None, None, :])[0, 0]  # já normalizado [0,1]
    rgb_linear = M @ xyz
    return rgb_linear


def is_valid_rgb(rgb):
    return np.all((rgb >= 0) & (rgb <= 1))


def rgb_to_hex(rgb):
    rgb_255 = (np.clip(rgb, 0, 1) * 255).astype(int)
    return '#{:02x}{:02x}{:02x}'.format(*rgb_255)


# ==============================
# SOBOL SAMPLING
# ==============================

sampler = qmc.Sobol(d=3, scramble=True)

valid_samples = []

while len(valid_samples) < N_SAMPLES:
    sobol_points = sampler.random(n=BATCH_SIZE)

    for x, y, z in sobol_points:

        # mapear Sobol → LAB
        L = L_MIN + x * (L_MAX - L_MIN)
        a = A_MIN + y * (A_MAX - A_MIN)
        b = B_MIN + z * (B_MAX - B_MIN)

        lab = np.array([L, a, b])

        # validação manual (sem clipping)
        rgb_linear = lab_to_valid_rgb_linear(lab)

        if not is_valid_rgb(rgb_linear):
            continue

        # conversão final (AGORA pode usar skimage)
        rgb = skc.lab2rgb(lab[None, None, :])[0, 0]

        hex_color = rgb_to_hex(rgb)

        rgb_255 = np.round(np.clip(rgb, 0, 1) * 255).astype(int)

        valid_samples.append([
            L, a, b,
            rgb_255[0], rgb_255[1], rgb_255[2],
            hex_color
        ])

        if len(valid_samples) >= N_SAMPLES:
            break

# ==============================
# SALVAR CSV
# ==============================

df = pd.DataFrame(valid_samples, columns=[
    "L", "a", "b",
    "r", "g", "b_rgb",
    "hex"
])

df.to_csv("soble_colors.csv", index=False)

print(f"CSV gerado com {len(df)} amostras válidas.")