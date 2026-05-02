from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2

OUT_DIR = Path("doc/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR = Path("images")


def extract_analysis_channel(img):
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[:, :, 0]


def clahe_lab(img_rgb_in, clip_limit=2.0, tile_grid=(8, 8)):
    lab = cv2.cvtColor(img_rgb_in, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return cv2.cvtColor(
        cv2.merge((clahe_obj.apply(l_ch), a_ch, b_ch)), cv2.COLOR_LAB2RGB
    )


def load_rgb(image_id):
    img_bgr = cv2.imread(str(IMAGES_DIR / f"{image_id}.png"))
    if img_bgr is None:
        raise FileNotFoundError(image_id)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def step1_gamma_1187(img_rgb):
    return np.uint8(np.clip(255 * np.power(img_rgb / 255.0, 0.40), 0, 255))


def step1_log(img_rgb):
    c_log = 255 / np.log10(1 + 255)
    return np.uint8(np.clip(c_log * np.log10(1 + img_rgb.astype(np.float32)), 0, 255))


def save_visual_triplet(stages, image_id):
    items = list(stages.items())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (name, img) in zip(axes, items):
        ax.imshow(img)
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(f"{image_id} - Comparativa visual por etapas", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"fig_{image_id}_visual.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_hist_cdf(stages, image_id):
    fig, axes = plt.subplots(1, len(stages), figsize=(22, 5))
    if len(stages) == 1:
        axes = [axes]

    for ax, (name, img_s) in zip(axes, stages.items()):
        channel = extract_analysis_channel(img_s)
        hist_s = np.bincount(channel.flatten(), minlength=256)
        cdf_s = hist_s.cumsum()
        cdf_overlay = (cdf_s / cdf_s[-1]) * hist_s.max()

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
        ax.set_xlabel("Canal L*", fontsize=8)
        ax.set_ylabel("Frecuencia")
        ax.set_xlim(0, 255)
        ax.legend(fontsize=6)

    fig.suptitle(f"{image_id} — Histograma + CDF del canal L*", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"fig_{image_id}_hist.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


all_stages = {}

img_1187 = load_rgb("1187")
step1_1187 = step1_gamma_1187(img_1187)
step2_1187 = clahe_lab(step1_1187)
all_stages["1187"] = {
    "Original": img_1187,
    "Paso 1 - Gamma 0.40": step1_1187,
    "Paso 2 - CLAHE LAB": step2_1187,
}

for image_id in ["1321", "1457", "1619"]:
    img = load_rgb(image_id)
    step1 = step1_log(img)
    step2 = clahe_lab(step1)
    all_stages[image_id] = {
        "Original": img,
        "Paso 1 - Transf. Log": step1,
        "Paso 2 - CLAHE LAB": step2,
    }

for image_id, stages in all_stages.items():
    save_visual_triplet(stages, image_id)
    save_hist_cdf(stages, image_id)

print("OK")
for image_id in all_stages:
    print(OUT_DIR / f"fig_{image_id}_visual.png")
    print(OUT_DIR / f"fig_{image_id}_hist.png")
