"""
Patch script: rewrites lab.ipynb with the requested changes.
- 1187: gamma (no comparativa) -> CLAHE LAB -> Validacion (sin Paso 3)
- 1321, 1457, 1619: log transform -> CLAHE LAB -> Validacion (sin Paso 3)
"""

import json

path = r"c:\Users\javga\Documents\personal\workspace\unir-ia-va\act1-va\lab.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)
cells = nb["cells"]


def find(fragment):
    for i, c in enumerate(cells):
        if fragment in "".join(c["source"]):
            return i
    raise ValueError(f"Not found: {fragment!r}")


def s(text):
    lines = text.split("\n")
    return [l + "\n" if i < len(lines) - 1 else l for i, l in enumerate(lines)]


# 1187 edits
cells[find("1.1 Objetivo")]["source"] = s(
    "## 1.1 Objetivo del Proceso\n\nCapturar la informaci\u00f3n relevante de la imagen original "
    "**`1187.png`** para facilitar la extracci\u00f3n de informaci\u00f3n posterior, ya sea por un observador "
    "humano o una m\u00e1quina.\n\nEl procesado se estructura en un flujo de dos pasos encadenados:\n\n"
    "| Paso | T\u00e9cnica | Prop\u00f3sito |\n|------|---------|----------|\n"
    "| **1** | Correcci\u00f3n Gamma (\u03b3 < 1) | Expandir el rango de los p\u00edxeles oscuros |\n"
    "| **2** | CLAHE LAB | Mejora de contraste local en espacio perceptual CIE-Lab |"
)

cells[find("\u03b3=0.30 seleccionado: 1187")]["source"] = s(
    "# Paso 1: Correcci\u00f3n Gamma (s = c*r^gamma, con gamma=0.30)\n"
    "# gamma=0.30 seleccionado: 1187 es la imagen mas oscura del conjunto, con el 90%\n"
    "# de los pixels por debajo de 20. Un realce agresivo recupera estructura\n"
    "# sin quemar las (pocas) zonas ya iluminadas.\n\n"
    "step1 = np.uint8(np.clip(255 * np.power(img_rgb / 255.0, 0.30), 0, 255))\n\n"
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
    "axes[0].imshow(img_rgb); axes[0].set_title('Original'); axes[0].axis('off')\n"
    "axes[1].imshow(step1); axes[1].set_title('Gamma gamma=0.30'); axes[1].axis('off')\n"
    "plt.suptitle('1187 - Paso 1: Correcci\u00f3n Gamma', fontsize=12)\n"
    "plt.tight_layout()\nplt.show()"
)

cells[find("1.3 Paso 2")]["source"] = s("## 1.3 Paso 2 - CLAHE LAB")

cells[find("CLAHE LAB seleccionado con clipLimit=8.0")]["source"] = s(
    "# Paso 2: CLAHE LAB\n# clipLimit=2.0, tileGridSize=(8,8): parametros estandar.\n\n"
    "def clahe_lab(img_rgb_in, clip_limit=2.0, tile_grid=(8, 8)):\n"
    '    """CLAHE on L channel (CIE-Lab)."""\n'
    "    lab = cv2.cvtColor(img_rgb_in, cv2.COLOR_RGB2LAB)\n"
    "    L_ch, A_ch, B_ch = cv2.split(lab)\n"
    "    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)\n"
    "    return cv2.cvtColor(cv2.merge((clahe_obj.apply(L_ch), A_ch, B_ch)), cv2.COLOR_LAB2RGB)\n\n"
    "step2 = clahe_lab(step1, clip_limit=2.0, tile_grid=(8, 8))\n\n"
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
    "axes[0].imshow(step1); axes[0].set_title('Paso 1: Gamma 0.30'); axes[0].axis('off')\n"
    "axes[1].imshow(step2); axes[1].set_title('Paso 2: CLAHE LAB'); axes[1].axis('off')\n"
    "plt.suptitle('1187 - Paso 2: CLAHE LAB', fontsize=12)\n"
    "plt.tight_layout()\nplt.show()"
)

cells[find("1.5 Validaci")]["source"] = s(
    "## 1.4 Validaci\u00f3n y M\u00e9tricas\n\nLa validaci\u00f3n se representa sobre el canal L* de Lab. "
    "La CDF se dibuja en escalones y se elimina la referencia de uniforme ideal."
)

cells[find("stages_1187")]["source"] = s(
    "stages_1187 = {\n"
    "    'Original': img_rgb,\n"
    "    'Paso 1 - Gamma 0.30': step1,\n"
    "    'Paso 2 - CLAHE LAB': step2,\n"
    "}\nrows = [{'Etapa': name, **compute_metrics(img)} for name, img in stages_1187.items()]\n"
    "df_1187 = pd.DataFrame(rows).set_index('Etapa')\n"
    "display(df_1187.style.background_gradient(cmap='YlGn', subset=['Desv. Std', 'Entrop\u00eda'])"
    ".format(precision=2).set_caption('1187 - M\u00e9tricas por etapa (canal L*)'))\n"
    "show_validation_histograms(stages_1187, '1187')"
)

# 1321 edits
cells[find("2.1 Objetivo")]["source"] = s(
    "## 2.1 Objetivo del Proceso\n\nCapturar la informaci\u00f3n relevante de la imagen original "
    "**`1321.png`** para facilitar la extracci\u00f3n de informaci\u00f3n posterior.\n\n"
    "El procesado se estructura en un flujo de dos pasos encadenados:\n\n"
    "| Paso | T\u00e9cnica | Prop\u00f3sito |\n|------|---------|----------|\n"
    "| **1** | Transformaci\u00f3n Logar\u00edtmica | Expandir el rango de los p\u00edxeles oscuros de forma no lineal |\n"
    "| **2** | CLAHE LAB | Mejora de contraste local en espacio perceptual CIE-Lab |"
)

cells[find("2.2 Paso 1")]["source"] = s(
    "## 2.2 Paso 1 - Transformaci\u00f3n Logar\u00edtmica"
)

cells[find("\u03b3=0.45 seleccionado")]["source"] = s(
    "# Paso 1: Transformacion Logaritmica (s = c*log10(1+r))\n"
    "# c = 255/log10(256) normaliza la salida al rango [0, 255].\n\n"
    "c_log = 255 / np.log10(1 + 255)\n"
    "step1 = np.uint8(np.clip(c_log * np.log10(1 + img_rgb.astype(np.float32)), 0, 255))\n\n"
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
    "axes[0].imshow(img_rgb); axes[0].set_title('Original'); axes[0].axis('off')\n"
    "axes[1].imshow(step1); axes[1].set_title('Transformaci\u00f3n Logar\u00edtmica'); axes[1].axis('off')\n"
    "plt.suptitle('1321 - Paso 1: Transformaci\u00f3n Logar\u00edtmica', fontsize=12)\n"
    "plt.tight_layout()\nplt.show()"
)

cells[find("2.3 Paso 2")]["source"] = s("## 2.3 Paso 2 - CLAHE LAB")

cells[find("CLAHE LAB seleccionado con clipLimit=4.0")]["source"] = s(
    "# Paso 2: CLAHE LAB\n# clipLimit=2.0, tileGridSize=(8,8): parametros estandar.\n\n"
    "step2 = clahe_lab(step1, clip_limit=2.0, tile_grid=(8, 8))\n\n"
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
    "axes[0].imshow(step1); axes[0].set_title('Paso 1: Transf. Log'); axes[0].axis('off')\n"
    "axes[1].imshow(step2); axes[1].set_title('Paso 2: CLAHE LAB'); axes[1].axis('off')\n"
    "plt.suptitle('1321 - Paso 2: CLAHE LAB', fontsize=12)\n"
    "plt.tight_layout()\nplt.show()"
)

cells[find("2.5 Validaci")]["source"] = s(
    "## 2.4 Validaci\u00f3n y M\u00e9tricas\n\nLa validaci\u00f3n se representa sobre el canal L* de Lab. "
    "La CDF se dibuja en escalones y se elimina la referencia de uniforme ideal."
)

cells[find("stages_1321")]["source"] = s(
    "stages_1321 = {\n"
    "    'Original': img_rgb,\n"
    "    'Paso 1 - Transf. Log': step1,\n"
    "    'Paso 2 - CLAHE LAB': step2,\n"
    "}\nrows = [{'Etapa': name, **compute_metrics(img)} for name, img in stages_1321.items()]\n"
    "df_1321 = pd.DataFrame(rows).set_index('Etapa')\n"
    "display(df_1321.style.background_gradient(cmap='YlGn', subset=['Desv. Std', 'Entrop\u00eda'])"
    ".format(precision=2).set_caption('1321 - M\u00e9tricas por etapa (canal L*)'))\n"
    "show_validation_histograms(stages_1321, '1321')"
)

# 1457 edits
cells[find("3.1 Objetivo")]["source"] = s(
    "## 3.1 Objetivo del Proceso\n\nCapturar la informaci\u00f3n relevante de la imagen original "
    "**`1457.png`** para facilitar la extracci\u00f3n de informaci\u00f3n posterior.\n\n"
    "El procesado se estructura en un flujo de dos pasos encadenados:\n\n"
    "| Paso | T\u00e9cnica | Prop\u00f3sito |\n|------|---------|----------|\n"
    "| **1** | Transformaci\u00f3n Logar\u00edtmica | Expandir el rango de los p\u00edxeles oscuros de forma no lineal |\n"
    "| **2** | CLAHE LAB | Mejora de contraste local en espacio perceptual CIE-Lab |"
)

cells[find("3.2 Paso 1")]["source"] = s(
    "## 3.2 Paso 1 - Transformaci\u00f3n Logar\u00edtmica"
)

cells[find("curva potencia con \u03b3 < 1 expande")]["source"] = s(
    "# Paso 1: Transformacion Logaritmica (s = c*log10(1+r))\n"
    "# c = 255/log10(256) normaliza la salida al rango [0, 255].\n\n"
    "c_log = 255 / np.log10(1 + 255)\n"
    "step1 = np.uint8(np.clip(c_log * np.log10(1 + img_rgb.astype(np.float32)), 0, 255))\n\n"
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
    "axes[0].imshow(img_rgb); axes[0].set_title('Original'); axes[0].axis('off')\n"
    "axes[1].imshow(step1); axes[1].set_title('Transformaci\u00f3n Logar\u00edtmica'); axes[1].axis('off')\n"
    "plt.suptitle('1457 - Paso 1: Transformaci\u00f3n Logar\u00edtmica', fontsize=12)\n"
    "plt.tight_layout()\nplt.show()"
)

cells[find("3.3 Paso 2")]["source"] = s("## 3.3 Paso 2 - CLAHE LAB")

cells[find("CLAHE LAB seleccionado con clipLimit=3.0")]["source"] = s(
    "# Paso 2: CLAHE LAB\n# clipLimit=2.0, tileGridSize=(8,8): parametros estandar.\n\n"
    "step2 = clahe_lab(step1, clip_limit=2.0, tile_grid=(8, 8))\n\n"
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
    "axes[0].imshow(step1); axes[0].set_title('Paso 1: Transf. Log'); axes[0].axis('off')\n"
    "axes[1].imshow(step2); axes[1].set_title('Paso 2: CLAHE LAB'); axes[1].axis('off')\n"
    "plt.suptitle('1457 - Paso 2: CLAHE LAB', fontsize=12)\n"
    "plt.tight_layout()\nplt.show()"
)

cells[find("3.5 Validaci")]["source"] = s(
    "## 3.4 Validaci\u00f3n y M\u00e9tricas\n\nLa validaci\u00f3n se representa sobre el canal L* de Lab. "
    "La CDF se dibuja en escalones y se elimina la referencia de uniforme ideal."
)

cells[find("stages_1457")]["source"] = s(
    "stages_1457 = {\n"
    "    'Original': img_rgb,\n"
    "    'Paso 1 - Transf. Log': step1,\n"
    "    'Paso 2 - CLAHE LAB': step2,\n"
    "}\nrows = [{'Etapa': name, **compute_metrics(img)} for name, img in stages_1457.items()]\n"
    "df_1457 = pd.DataFrame(rows).set_index('Etapa')\n"
    "display(df_1457.style.background_gradient(cmap='YlGn', subset=['Desv. Std', 'Entrop\u00eda'])"
    ".format(precision=2).set_caption('1457 - M\u00e9tricas por etapa (canal L*)'))\n"
    "show_validation_histograms(stages_1457, '1457')"
)

# 1619 edits
cells[find("4.1 Objetivo")]["source"] = s(
    "## 4.1 Objetivo del Proceso\n\nCapturar la informaci\u00f3n relevante de la imagen original "
    "**`1619.png`** para facilitar la extracci\u00f3n de informaci\u00f3n posterior.\n\n"
    "El procesado se estructura en un flujo de dos pasos encadenados:\n\n"
    "| Paso | T\u00e9cnica | Prop\u00f3sito |\n|------|---------|----------|\n"
    "| **1** | Transformaci\u00f3n Logar\u00edtmica | Expandir el rango de los p\u00edxeles oscuros de forma no lineal |\n"
    "| **2** | CLAHE LAB | Mejora de contraste local en espacio perceptual CIE-Lab |"
)

cells[find("4.2 Paso 1")]["source"] = s(
    "## 4.2 Paso 1 - Transformaci\u00f3n Logar\u00edtmica"
)

cells[find("\u03b3=0.40 seleccionado")]["source"] = s(
    "# Paso 1: Transformacion Logaritmica (s = c*log10(1+r))\n"
    "# c = 255/log10(256) normaliza la salida al rango [0, 255].\n\n"
    "c_log = 255 / np.log10(1 + 255)\n"
    "step1 = np.uint8(np.clip(c_log * np.log10(1 + img_rgb.astype(np.float32)), 0, 255))\n\n"
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
    "axes[0].imshow(img_rgb); axes[0].set_title('Original'); axes[0].axis('off')\n"
    "axes[1].imshow(step1); axes[1].set_title('Transformaci\u00f3n Logar\u00edtmica'); axes[1].axis('off')\n"
    "plt.suptitle('1619 - Paso 1: Transformaci\u00f3n Logar\u00edtmica', fontsize=12)\n"
    "plt.tight_layout()\nplt.show()"
)

cells[find("4.3 Paso 2")]["source"] = s("## 4.3 Paso 2 - CLAHE LAB")

cells[find("CLAHE LAB seleccionado con clipLimit=5.0")]["source"] = s(
    "# Paso 2: CLAHE LAB\n# clipLimit=2.0, tileGridSize=(8,8): parametros estandar.\n\n"
    "step2 = clahe_lab(step1, clip_limit=2.0, tile_grid=(8, 8))\n\n"
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
    "axes[0].imshow(step1); axes[0].set_title('Paso 1: Transf. Log'); axes[0].axis('off')\n"
    "axes[1].imshow(step2); axes[1].set_title('Paso 2: CLAHE LAB'); axes[1].axis('off')\n"
    "plt.suptitle('1619 - Paso 2: CLAHE LAB', fontsize=12)\n"
    "plt.tight_layout()\nplt.show()"
)

cells[find("4.5 Validaci")]["source"] = s(
    "## 4.4 Validaci\u00f3n y M\u00e9tricas\n\nLa validaci\u00f3n se representa sobre el canal L* de Lab. "
    "La CDF se dibuja en escalones y se elimina la referencia de uniforme ideal."
)

cells[find("stages_1619")]["source"] = s(
    "stages_1619 = {\n"
    "    'Original': img_rgb,\n"
    "    'Paso 1 - Transf. Log': step1,\n"
    "    'Paso 2 - CLAHE LAB': step2,\n"
    "}\nrows = [{'Etapa': name, **compute_metrics(img)} for name, img in stages_1619.items()]\n"
    "df_1619 = pd.DataFrame(rows).set_index('Etapa')\n"
    "display(df_1619.style.background_gradient(cmap='YlGn', subset=['Desv. Std', 'Entrop\u00eda'])"
    ".format(precision=2).set_caption('1619 - M\u00e9tricas por etapa (canal L*)'))\n"
    "show_validation_histograms(stages_1619, '1619')"
)

# DELETIONS - collect indices and delete high to low
to_delete = set()
for frag in [
    "1.4 Paso 3",
    "1187 es la imagen m\u00e1s oscura: se aplica suma +30",
    "2.4 Paso 3",
    "1321 ya tiene zonas bien iluminadas",
    "3.4 Paso 3",
    "1457 es la imagen mejor iluminada del conjunto",
    "4.4 Paso 3",
    "1619 tiene zonas intermedias",
]:
    to_delete.add(find(frag))

for idx in sorted(to_delete, reverse=True):
    del cells[idx]

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Saved OK. Total cells: {len(nb['cells'])}")
for i, c in enumerate(nb["cells"]):
    print(
        f"{i:2d} {c['cell_type']:8s} | {''.join(c['source'])[:65].replace(chr(10),'|')}"
    )
