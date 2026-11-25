
import pandas as pd
import colorsys
import numpy as np
from skimage.color import rgb2lab

INPUT_CSV = "soble_colors.csv"
OUTPUT_CSV = "theory_colors.csv"

SCHEMES = {
    "ang_right": -30,
    "ang_left": 30,
    "comp": 180,
    "scomp_right": 150,
    "scomp_left": -150,
    "tri_right": 120,
    "tri_left": -120,
}

COLUMNS = [
    "input_hex", "input_r", "input_g", "input_b",
    "input_L", "input_a", "input_b_lab",
    "source", "method",
    "output_hex", "output_r", "output_g", "output_b",
    "output_L", "output_a", "output_b_lab",
]


def rgb_to_lab(r, g, b):
    rgb = np.array([[[r, g, b]]], dtype=np.float64) / 255.0
    lab = rgb2lab(rgb)[0, 0]
    return tuple(float(v) for v in lab)


def rgb_to_hex(r, g, b):
    return f"#{r:02X}{g:02X}{b:02X}"


def generate_colors(row):
    r, g, b = int(row["r"]), int(row["g"]), int(row["b"])

    # colorsys retorna H, L, S — não H, S, L.
    h, lightness, saturation = colorsys.rgb_to_hls(
        r / 255.0, g / 255.0, b / 255.0
    )

    input_L, input_a, input_b_lab = rgb_to_lab(r, g, b)

    results = []

    for method, offset in SCHEMES.items():
        new_h = (h + offset / 360.0) % 1.0

        # Preserva exatamente a luminosidade e a saturação HSL.
        new_r, new_g, new_b = colorsys.hls_to_rgb(
            new_h, lightness, saturation
        )

        new_r, new_g, new_b = (
            round(c * 255) for c in (new_r, new_g, new_b)
        )

        output_L, output_a, output_b_lab = rgb_to_lab(
            new_r, new_g, new_b
        )

        results.append({
            "input_hex": row["hex"],
            "input_r": r,
            "input_g": g,
            "input_b": b,
            "input_L": input_L,
            "input_a": input_a,
            "input_b_lab": input_b_lab,
            "source": "theory",
            "method": method,
            "output_hex": rgb_to_hex(new_r, new_g, new_b),
            "output_r": new_r,
            "output_g": new_g,
            "output_b": new_b,
            "output_L": output_L,
            "output_a": output_a,
            "output_b_lab": output_b_lab,
        })

    return results


df = pd.read_csv(INPUT_CSV)

results = []
for _, row in df.iterrows():
    results.extend(generate_colors(row))

result_df = pd.DataFrame(results, columns=COLUMNS)
result_df.to_csv(OUTPUT_CSV, index=False, float_format="%.10f")

print(f"Arquivo gerado: {OUTPUT_CSV}")
print(f"Cores de entrada: {len(df)}")
print(f"Combinações geradas: {len(result_df)}")

print("\nExemplo de validação:")
print(result_df[[
    "input_hex", "method", "output_hex",
    "input_L", "output_L"
]].head(7).to_string(index=False))
