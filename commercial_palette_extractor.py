from playwright.sync_api import sync_playwright
import pandas as pd
import numpy as np
import skimage.color as skc
import re

# ==============================
# CONFIG
# ==============================

INPUT_CSV = "soble_colors.csv"
OUTPUT_CSV = "comercial_raw.csv"

# Link da ferramenta comercial

COMERCIAL_URL = ""

# ==============================
# CONVERSÕES
# ==============================

def rgb_255_to_01(rgb):
    return np.array(rgb) / 255.0


def rgb_to_lab(rgb_01):
    return skc.rgb2lab(rgb_01[None, None, :])[0, 0]


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")

    return np.array([
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16)
    ])


def rgb_to_hex(rgb):
    rgb_int = [int(np.clip(round(x), 0, 255)) for x in rgb]

    return "#{:02X}{:02X}{:02X}".format(*rgb_int)

# ==============================
# PLAYWRIGHT
# ==============================

def select_harmony(page, harmony):

    page.get_by_role(
        "radio",
        name=f"{harmony} harmonia de cores"
    ).click()

    page.wait_for_timeout(1000)


def set_base_color(page, hex_color):

    page.get_by_role(
        "button",
        name="Editar cor"
    ).first.click()

    page.wait_for_timeout(500)

    hex_input = page.get_by_role(
        "textbox",
        name="Valor de cor HEX"
    )

    hex_input.click()

    hex_input.press("Control+A")

    hex_input.fill(
        hex_color.replace("#", "")
    )

    hex_input.press("Enter")

    page.wait_for_timeout(1500)

# ==============================
# EXTRAÇÃO
# ==============================

def extract_palette(page, base_color):

    html = page.content()

    colors = re.findall(
        r'#[0-9A-Fa-f]{6}',
        html
    )

    unique = []

    for color in colors:

        color = color.upper()

        if color not in unique:
            unique.append(color)

    positions = [
        i
        for i, c in enumerate(unique)
        if c == base_color.upper()
    ]

    if not positions:
        raise Exception(
            f"Cor base {base_color} não encontrada."
        )

    idx = positions[-1]

    return unique[idx:]


def get_scheme(page, base_hex, harmony):

    select_harmony(
        page,
        harmony
    )

    set_base_color(
        page,
        base_hex
    )

    return extract_palette(
        page,
        base_hex
    )

# ==============================
# RESULT BUILDER
# ==============================

def build_row(
    input_hex,
    input_rgb,
    input_lab,
    method,
    output_hex
):

    output_rgb = hex_to_rgb(
        output_hex
    )

    output_lab = rgb_to_lab(
        rgb_255_to_01(
            output_rgb
        )
    )

    return {

        # INPUT

        "input_hex": input_hex,

        "input_r": int(input_rgb[0]),
        "input_g": int(input_rgb[1]),
        "input_b": int(input_rgb[2]),

        "input_L": input_lab[0],
        "input_a": input_lab[1],
        "input_b_lab": input_lab[2],

        # META

        "source": "comercial",
        "method": method,

        # OUTPUT

        "output_hex": output_hex,

        "output_r": int(output_rgb[0]),
        "output_g": int(output_rgb[1]),
        "output_b": int(output_rgb[2]),

        "output_L": output_lab[0],
        "output_a": output_lab[1],
        "output_b_lab": output_lab[2],
    }

# ==============================
# LOAD INPUT
# ==============================

df_input = pd.read_csv(
    INPUT_CSV
)

results = []

# ==============================
# PROCESSAMENTO
# ==============================

with sync_playwright() as p:

    browser = p.chromium.launch(
        headless=False
    )

    page = browser.new_page()

    page.goto(
        COMERCIAL_URL
    )

    page.wait_for_timeout(
        5000
    )

    total = len(df_input)

    for idx, row in df_input.iterrows():

        try:

            input_hex = row["hex"].upper()

            input_rgb = np.array([
                row["r"],
                row["g"],
                row["b_rgb"]
            ])

            input_lab = np.array([
                row["L"],
                row["a"],
                row["b"]
            ])

            print(
                f"[{idx+1}/{total}] {input_hex}"
            )

            # =====================
            # ANÁLOGO
            # =====================

            palette = get_scheme(
                page,
                input_hex,
                "Análogo"
            )

            results.append(
                build_row(
                    input_hex,
                    input_rgb,
                    input_lab,
                    "ang_left",
                    palette[1]
                )
            )

            results.append(
                build_row(
                    input_hex,
                    input_rgb,
                    input_lab,
                    "ang_right",
                    palette[2]
                )
            )

            # =====================
            # COMPLEMENTAR
            # =====================

            palette = get_scheme(
                page,
                input_hex,
                "Complementar"
            )

            results.append(
                build_row(
                    input_hex,
                    input_rgb,
                    input_lab,
                    "comp",
                    palette[1]
                )
            )

            # =====================
            # COMPLEMENTAR DIVIDIDO
            # =====================

            palette = get_scheme(
                page,
                input_hex,
                "Complementar dividido"
            )

            results.append(
                build_row(
                    input_hex,
                    input_rgb,
                    input_lab,
                    "scomp_left",
                    palette[1]
                )
            )

            results.append(
                build_row(
                    input_hex,
                    input_rgb,
                    input_lab,
                    "scomp_right",
                    palette[2]
                )
            )

            # =====================
            # TRÍADE
            # =====================

            palette = get_scheme(
                page,
                input_hex,
                "Tríade"
            )

            results.append(
                build_row(
                    input_hex,
                    input_rgb,
                    input_lab,
                    "tri_left",
                    palette[1]
                )
            )

            results.append(
                build_row(
                    input_hex,
                    input_rgb,
                    input_lab,
                    "tri_right",
                    palette[2]
                )
            )

        except Exception as e:

            print(
                f"Erro em {input_hex}: {e}"
            )

# ==============================
# SAVE
# ==============================

browser.close()

df_out = pd.DataFrame(
    results
)

df_out.to_csv(
    OUTPUT_CSV,
    index=False
)

print(
    f"\nSalvo em {OUTPUT_CSV}"
)

print(
    f"Total de linhas: {len(df_out)}"
)