import skimage.color as skc
import colorsys as cs
import numpy as np
import pandas as pd

def generator(name, dH):
    rows = []

    for h in range(0, 360, 5):
        for l in range(5, 96, 5):
            for s in range(5, 100, 5):

                # cor base em HSL → RGB → Lab
                r, g, b = cs.hls_to_rgb(h / 360.0, l / 100.0, s / 100.0)
                rgb = np.array([r, g, b])
                lab = skc.rgb2lab(rgb[None, None, :])[0, 0]

                # -------- análoga direita --------
                h2 = (h + dH) % 360
                r, g, b = cs.hls_to_rgb(h2 / 360.0, l / 100.0, s / 100.0)
                rgb2 = np.array([r, g, b])
                lab2 = skc.rgb2lab(rgb2[None, None, :])[0, 0]

                dL = lab2[0] - lab[0]
                da = lab2[1] - lab[1]
                db = lab2[2] - lab[2]
                dE = skc.deltaE_ciede2000(
                    lab[None, None, :],
                    lab2[None, None, :]
                )[0, 0]

                rows.append([
                    lab[0], lab[1], lab[2],
                    dL, da, db,
                    dE
                ])

    cols = ["L", "a", "b", "dL", "da", "db", "dE"]

    pd.DataFrame(rows, columns=cols).to_csv(
        name, index=False
    )

    print(f"Arquivo {name} criado com sucesso!")

generator("analog_left.csv", 30)
generator("analog_right.csv", -30)
generator("complementary.csv", 180)
generator("split_complementary_right.csv", 150)
generator("split_complementary_left.csv", -150)
generator("triadic_right.csv", 120)
generator("triadic.csv", -120)