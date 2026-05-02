import json

keywords = [
    "metric",
    "entrop",
    "std",
    "mean",
    "rango",
    "valid",
    "contraste",
    "histogram",
    "compute",
    "show_valid",
]

for nb_name in ["T6_Ajuste_intensidad.ipynb", "T6_Ecualización_del_histograma.ipynb"]:
    with open(nb_name, "r", encoding="utf-8") as f:
        nb = json.load(f)
    print(f"=== {nb_name} ===")
    for i, c in enumerate(nb["cells"]):
        src = "".join(c["source"])
        if any(kw in src.lower() for kw in keywords):
            ct = c["cell_type"]
            print(f"-- Cell {i} ({ct}) --")
            print(src[:800])
            print()
