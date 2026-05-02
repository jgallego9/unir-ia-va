# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

# ─── Helper functions ─────────────────────────────────────────────────────────


def extract_analysis_channel(img):
    """Return the channel used for validation and metrics.

    For RGB images, the perceptual luminance channel L* from Lab is used.
    This matches the intent of the histogram-processing step better than a
    generic RGB-to-gray conversion.

    :param img: numpy uint8 image — H×W (gray) or H×W×3 (RGB)
    :returns: numpy uint8 single-channel image
    """
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[:, :, 0]


def compute_metrics(img):
    """Compute image quality metrics on the validation channel.

    Color images are evaluated on the Lab L* channel so that histogram-based
    processing and validation are aligned on the same luminance component.

    :param img: numpy uint8 image — H×W (gray) or H×W×3 (RGB)
    :returns: dict with mean, std, entropy, dynamic_range, cv
    """
    flat = extract_analysis_channel(img).flatten().astype(np.float64)
    hist, _ = np.histogram(flat, bins=256, range=(0, 256))
    hist_prob = hist / hist.sum()
    nz = hist_prob[hist_prob > 0]
    ent = float(-np.sum(nz * np.log2(nz)))
    mean_val = float(flat.mean())
    std_val = float(flat.std())
    dyn_range = int(flat.max()) - int(flat.min())
    cv_val = std_val / mean_val if mean_val > 0 else 0.0
    return {
        "Media": round(mean_val, 2),
        "Desv. Std": round(std_val, 2),
        "Entropía": round(ent, 4),
        "Rango Din.": dyn_range,
        "Coef. Var.": round(cv_val, 4),
    }


def show_grid(images, titles, figsize=(22, 4), cmap=None):
    """Display a row of images with titles.

    :param images: list of numpy arrays (H×W or H×W×3)
    :param titles: list of strings
    :param figsize: figure size tuple
    :param cmap: matplotlib colormap string
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def show_validation_histograms(stages, title):
    """Plot histogram and stepwise CDF on the Lab L* validation channel.

    The red ideal-uniform line is intentionally omitted because it is useful
    for global equalization, but misleading for CLAHE where the objective is
    local contrast enhancement.

    :param stages: ordered mapping of stage name to image
    :param title: figure title prefix
    """
    fig, axes = plt.subplots(1, len(stages), figsize=(22, 5))
    if len(stages) == 1:
        axes = [axes]

    for ax, (name, img_s) in zip(axes, stages.items()):
        channel = extract_analysis_channel(img_s)
        hist_s = np.bincount(channel.flatten(), minlength=256)
        cdf_s = hist_s.cumsum()
        cdf_overlay = (cdf_s / cdf_s[-1]) * hist_s.max()

        m = compute_metrics(img_s)
        ax.bar(np.arange(256), hist_s, width=1.0, color="steelblue", alpha=0.8)
        ax.step(
            np.arange(256),
            cdf_overlay,
            where="mid",
            color="royalblue",
            linewidth=1.5,
            label="CDF",
        )
        ax.set_title(name, fontsize=8)
        ax.set_xlabel(
            f"L*: Std={m['Desv. Std']:.1f}  Ent={m['Entropía']:.2f}", fontsize=7
        )
        ax.set_ylabel("Frecuencia")
        ax.set_xlim(0, 255)
        ax.legend(fontsize=6)

    plt.suptitle(f"{title} — Histograma + CDF del canal L*", fontsize=12)
    plt.tight_layout()
    plt.show()


# %% [markdown]
# # Imagen 1: 1187.png

# %%
# ── Carga ─────────────────────────────────────────────────────────────────────
img_bgr = cv2.imread("./images/1187.png")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# ── Visualización original + histogramas ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].imshow(img_rgb)
axes[0].set_title("Original (Color)")
axes[0].axis("off")

for i, (c, label) in enumerate(zip("rgb", ("R", "G", "B"))):
    axes[1].hist(img_rgb[:, :, i].flatten(), bins=256, color=c, alpha=0.7, label=label)
axes[1].set_title("Histograma RGB")
axes[1].set_xlabel("Valor")
axes[1].set_ylabel("Frecuencia")
axes[1].legend()

axes[2].hist(img_gray.flatten(), bins=256, color="gray", alpha=0.9)
axes[2].set_title("Histograma Escala de Grises")
axes[2].set_xlabel("Intensidad")
axes[2].set_ylabel("Frecuencia")

plt.suptitle("1187 — Imagen Original", fontsize=13)
plt.tight_layout()
plt.show()

print(
    f"Resolución: {img_gray.shape[0]}×{img_gray.shape[1]}  |  "
    f"Rango: [{img_gray.min()}, {img_gray.max()}]  |  "
    f"Media: {img_gray.mean():.1f}  |  Std: {img_gray.std():.1f}"
)


# %% [markdown]
# ## 1.1 Objetivo del Proceso
#
# Capturar la información relevante de la imagen original **`1187.png`** para facilitar la extracción de información posterior, ya sea por un observador humano o una máquina.
#
# El procesado se estructura en un flujo de dos pasos encadenados:
#
# | Paso | Técnica | Propósito |
# |------|---------|----------|
# | **1** | Corrección Gamma (γ < 1) | Expandir el rango de los píxeles oscuros |
# | **2** | CLAHE LAB | Mejora de contraste local en espacio perceptual CIE-Lab |

# %% [markdown]
# ## 1.2 Paso 1 – Ajuste de Intensidad (Corrección Gamma)

# %%
# Paso 1: Corrección Gamma (s = c*r^gamma, con gamma=0.30)
# gamma=0.30 seleccionado: 1187 es la imagen mas oscura del conjunto, con el 90%
# de los pixels por debajo de 20. Un realce agresivo recupera estructura
# sin quemar las (pocas) zonas ya iluminadas.

step1 = np.uint8(np.clip(255 * np.power(img_rgb / 255.0, 0.30), 0, 255))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].imshow(img_rgb)
axes[0].set_title("Original")
axes[0].axis("off")
axes[1].imshow(step1)
axes[1].set_title("Gamma gamma=0.30")
axes[1].axis("off")
plt.suptitle("1187 - Paso 1: Corrección Gamma", fontsize=12)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 1.3 Paso 2 - CLAHE LAB

# %%
# Paso 2: CLAHE LAB
# clipLimit=2.0, tileGridSize=(8,8): parametros estandar.


def clahe_lab(img_rgb_in, clip_limit=2.0, tile_grid=(8, 8)):
    """CLAHE on L channel (CIE-Lab)."""
    lab = cv2.cvtColor(img_rgb_in, cv2.COLOR_RGB2LAB)
    L_ch, A_ch, B_ch = cv2.split(lab)
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return cv2.cvtColor(
        cv2.merge((clahe_obj.apply(L_ch), A_ch, B_ch)), cv2.COLOR_LAB2RGB
    )


step2 = clahe_lab(step1, clip_limit=2.0, tile_grid=(8, 8))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].imshow(step1)
axes[0].set_title("Paso 1: Gamma 0.30")
axes[0].axis("off")
axes[1].imshow(step2)
axes[1].set_title("Paso 2: CLAHE LAB")
axes[1].axis("off")
plt.suptitle("1187 - Paso 2: CLAHE LAB", fontsize=12)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 1.4 Validación y Métricas
#
# La validación se representa sobre el canal L* de Lab. La CDF se dibuja en escalones y se elimina la referencia de uniforme ideal.

# %%
stages_1187 = {
    "Original": img_rgb,
    "Paso 1 - Gamma 0.30": step1,
    "Paso 2 - CLAHE LAB": step2,
}
rows = [{"Etapa": name, **compute_metrics(img)} for name, img in stages_1187.items()]
df_1187 = pd.DataFrame(rows).set_index("Etapa")
display(
    df_1187.style.background_gradient(cmap="YlGn", subset=["Desv. Std", "Entropía"])
    .format(precision=2)
    .set_caption("1187 - Métricas por etapa (canal L*)")
)
show_validation_histograms(stages_1187, "1187")

# %% [markdown]
# # Imagen 2: 1321.png

# %%
# ── Carga ─────────────────────────────────────────────────────────────────────
img_bgr = cv2.imread("./images/1321.png")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# ── Visualización original + histogramas ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].imshow(img_rgb)
axes[0].set_title("Original (Color)")
axes[0].axis("off")

for i, (c, label) in enumerate(zip("rgb", ("R", "G", "B"))):
    axes[1].hist(img_rgb[:, :, i].flatten(), bins=256, color=c, alpha=0.7, label=label)
axes[1].set_title("Histograma RGB")
axes[1].set_xlabel("Valor")
axes[1].set_ylabel("Frecuencia")
axes[1].legend()

axes[2].hist(img_gray.flatten(), bins=256, color="gray", alpha=0.9)
axes[2].set_title("Histograma Escala de Grises")
axes[2].set_xlabel("Intensidad")
axes[2].set_ylabel("Frecuencia")

plt.suptitle("1321 — Imagen Original", fontsize=13)
plt.tight_layout()
plt.show()

print(
    f"Resolución: {img_gray.shape[0]}×{img_gray.shape[1]}  |  "
    f"Rango: [{img_gray.min()}, {img_gray.max()}]  |  "
    f"Media: {img_gray.mean():.1f}  |  Std: {img_gray.std():.1f}"
)


# %% [markdown]
# ## 2.1 Objetivo del Proceso
#
# Capturar la información relevante de la imagen original **`1321.png`** para facilitar la extracción de información posterior.
#
# El procesado se estructura en un flujo de dos pasos encadenados:
#
# | Paso | Técnica | Propósito |
# |------|---------|----------|
# | **1** | Transformación Logarítmica | Expandir el rango de los píxeles oscuros de forma no lineal |
# | **2** | CLAHE LAB | Mejora de contraste local en espacio perceptual CIE-Lab |

# %% [markdown]
# ## 2.2 Paso 1 - Transformación Logarítmica

# %%
# Paso 1: Transformacion Logaritmica (s = c*log10(1+r))
# c = 255/log10(256) normaliza la salida al rango [0, 255].

c_log = 255 / np.log10(1 + 255)
step1 = np.uint8(np.clip(c_log * np.log10(1 + img_rgb.astype(np.float32)), 0, 255))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].imshow(img_rgb)
axes[0].set_title("Original")
axes[0].axis("off")
axes[1].imshow(step1)
axes[1].set_title("Transformación Logarítmica")
axes[1].axis("off")
plt.suptitle("1321 - Paso 1: Transformación Logarítmica", fontsize=12)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2.3 Paso 2 - CLAHE LAB

# %%
# Paso 2: CLAHE LAB
# clipLimit=2.0, tileGridSize=(8,8): parametros estandar.

step2 = clahe_lab(step1, clip_limit=2.0, tile_grid=(8, 8))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].imshow(step1)
axes[0].set_title("Paso 1: Transf. Log")
axes[0].axis("off")
axes[1].imshow(step2)
axes[1].set_title("Paso 2: CLAHE LAB")
axes[1].axis("off")
plt.suptitle("1321 - Paso 2: CLAHE LAB", fontsize=12)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2.4 Validación y Métricas
#
# La validación se representa sobre el canal L* de Lab. La CDF se dibuja en escalones y se elimina la referencia de uniforme ideal.

# %%
stages_1321 = {
    "Original": img_rgb,
    "Paso 1 - Transf. Log": step1,
    "Paso 2 - CLAHE LAB": step2,
}
rows = [{"Etapa": name, **compute_metrics(img)} for name, img in stages_1321.items()]
df_1321 = pd.DataFrame(rows).set_index("Etapa")
display(
    df_1321.style.background_gradient(cmap="YlGn", subset=["Desv. Std", "Entropía"])
    .format(precision=2)
    .set_caption("1321 - Métricas por etapa (canal L*)")
)
show_validation_histograms(stages_1321, "1321")

# %% [markdown]
# # Imagen 3: 1457.png

# %%
# ── Carga ─────────────────────────────────────────────────────────────────────
img_bgr = cv2.imread("./images/1457.png")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# ── Visualización original + histogramas ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].imshow(img_rgb)
axes[0].set_title("Original (Color)")
axes[0].axis("off")

for i, (c, label) in enumerate(zip("rgb", ("R", "G", "B"))):
    axes[1].hist(img_rgb[:, :, i].flatten(), bins=256, color=c, alpha=0.7, label=label)
axes[1].set_title("Histograma RGB")
axes[1].set_xlabel("Valor")
axes[1].set_ylabel("Frecuencia")
axes[1].legend()

axes[2].hist(img_gray.flatten(), bins=256, color="gray", alpha=0.9)
axes[2].set_title("Histograma Escala de Grises")
axes[2].set_xlabel("Intensidad")
axes[2].set_ylabel("Frecuencia")

plt.suptitle("1457 — Imagen Original", fontsize=13)
plt.tight_layout()
plt.show()

print(
    f"Resolución: {img_gray.shape[0]}×{img_gray.shape[1]}  |  "
    f"Rango: [{img_gray.min()}, {img_gray.max()}]  |  "
    f"Media: {img_gray.mean():.1f}  |  Std: {img_gray.std():.1f}"
)


# %% [markdown]
# ## 3.1 Objetivo del Proceso
#
# Capturar la información relevante de la imagen original **`1457.png`** para facilitar la extracción de información posterior.
#
# El procesado se estructura en un flujo de dos pasos encadenados:
#
# | Paso | Técnica | Propósito |
# |------|---------|----------|
# | **1** | Transformación Logarítmica | Expandir el rango de los píxeles oscuros de forma no lineal |
# | **2** | CLAHE LAB | Mejora de contraste local en espacio perceptual CIE-Lab |

# %% [markdown]
# ## 3.2 Paso 1 - Transformación Logarítmica

# %%
# Paso 1: Transformacion Logaritmica (s = c*log10(1+r))
# c = 255/log10(256) normaliza la salida al rango [0, 255].

c_log = 255 / np.log10(1 + 255)
step1 = np.uint8(np.clip(c_log * np.log10(1 + img_rgb.astype(np.float32)), 0, 255))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].imshow(img_rgb)
axes[0].set_title("Original")
axes[0].axis("off")
axes[1].imshow(step1)
axes[1].set_title("Transformación Logarítmica")
axes[1].axis("off")
plt.suptitle("1457 - Paso 1: Transformación Logarítmica", fontsize=12)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3.3 Paso 2 - CLAHE LAB

# %%
# Paso 2: CLAHE LAB
# clipLimit=2.0, tileGridSize=(8,8): parametros estandar.

step2 = clahe_lab(step1, clip_limit=2.0, tile_grid=(8, 8))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].imshow(step1)
axes[0].set_title("Paso 1: Transf. Log")
axes[0].axis("off")
axes[1].imshow(step2)
axes[1].set_title("Paso 2: CLAHE LAB")
axes[1].axis("off")
plt.suptitle("1457 - Paso 2: CLAHE LAB", fontsize=12)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3.4 Validación y Métricas
#
# La validación se representa sobre el canal L* de Lab. La CDF se dibuja en escalones y se elimina la referencia de uniforme ideal.

# %%
stages_1457 = {
    "Original": img_rgb,
    "Paso 1 - Transf. Log": step1,
    "Paso 2 - CLAHE LAB": step2,
}
rows = [{"Etapa": name, **compute_metrics(img)} for name, img in stages_1457.items()]
df_1457 = pd.DataFrame(rows).set_index("Etapa")
display(
    df_1457.style.background_gradient(cmap="YlGn", subset=["Desv. Std", "Entropía"])
    .format(precision=2)
    .set_caption("1457 - Métricas por etapa (canal L*)")
)
show_validation_histograms(stages_1457, "1457")

# %% [markdown]
# # Imagen 4: 1619.png

# %%
# ── Carga ─────────────────────────────────────────────────────────────────────
img_bgr = cv2.imread("./images/1619.png")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# ── Visualización original + histogramas ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].imshow(img_rgb)
axes[0].set_title("Original (Color)")
axes[0].axis("off")

for i, (c, label) in enumerate(zip("rgb", ("R", "G", "B"))):
    axes[1].hist(img_rgb[:, :, i].flatten(), bins=256, color=c, alpha=0.7, label=label)
axes[1].set_title("Histograma RGB")
axes[1].set_xlabel("Valor")
axes[1].set_ylabel("Frecuencia")
axes[1].legend()

axes[2].hist(img_gray.flatten(), bins=256, color="gray", alpha=0.9)
axes[2].set_title("Histograma Escala de Grises")
axes[2].set_xlabel("Intensidad")
axes[2].set_ylabel("Frecuencia")

plt.suptitle("1619 — Imagen Original", fontsize=13)
plt.tight_layout()
plt.show()

print(
    f"Resolución: {img_gray.shape[0]}×{img_gray.shape[1]}  |  "
    f"Rango: [{img_gray.min()}, {img_gray.max()}]  |  "
    f"Media: {img_gray.mean():.1f}  |  Std: {img_gray.std():.1f}"
)


# %% [markdown]
# ## 4.1 Objetivo del Proceso
#
# Capturar la información relevante de la imagen original **`1619.png`** para facilitar la extracción de información posterior.
#
# El procesado se estructura en un flujo de dos pasos encadenados:
#
# | Paso | Técnica | Propósito |
# |------|---------|----------|
# | **1** | Transformación Logarítmica | Expandir el rango de los píxeles oscuros de forma no lineal |
# | **2** | CLAHE LAB | Mejora de contraste local en espacio perceptual CIE-Lab |

# %% [markdown]
# ## 4.2 Paso 1 - Transformación Logarítmica

# %%
# Paso 1: Transformacion Logaritmica (s = c*log10(1+r))
# c = 255/log10(256) normaliza la salida al rango [0, 255].

c_log = 255 / np.log10(1 + 255)
step1 = np.uint8(np.clip(c_log * np.log10(1 + img_rgb.astype(np.float32)), 0, 255))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].imshow(img_rgb)
axes[0].set_title("Original")
axes[0].axis("off")
axes[1].imshow(step1)
axes[1].set_title("Transformación Logarítmica")
axes[1].axis("off")
plt.suptitle("1619 - Paso 1: Transformación Logarítmica", fontsize=12)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4.3 Paso 2 - CLAHE LAB

# %%
# Paso 2: CLAHE LAB
# clipLimit=2.0, tileGridSize=(8,8): parametros estandar.

step2 = clahe_lab(step1, clip_limit=2.0, tile_grid=(8, 8))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].imshow(step1)
axes[0].set_title("Paso 1: Transf. Log")
axes[0].axis("off")
axes[1].imshow(step2)
axes[1].set_title("Paso 2: CLAHE LAB")
axes[1].axis("off")
plt.suptitle("1619 - Paso 2: CLAHE LAB", fontsize=12)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4.4 Validación y Métricas
#
# La validación se representa sobre el canal L* de Lab. La CDF se dibuja en escalones y se elimina la referencia de uniforme ideal.

# %%
stages_1619 = {
    "Original": img_rgb,
    "Paso 1 - Transf. Log": step1,
    "Paso 2 - CLAHE LAB": step2,
}
rows = [{"Etapa": name, **compute_metrics(img)} for name, img in stages_1619.items()]
df_1619 = pd.DataFrame(rows).set_index("Etapa")
display(
    df_1619.style.background_gradient(cmap="YlGn", subset=["Desv. Std", "Entropía"])
    .format(precision=2)
    .set_caption("1619 - Métricas por etapa (canal L*)")
)
show_validation_histograms(stages_1619, "1619")

# %%
