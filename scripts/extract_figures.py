#!/usr/bin/env python3
"""Extrai as imagens PNG das saídas de um .ipynb executado.

Nome do arquivo: <prefixo>_cell<NN>_<k>.png. Se a célula tiver uma linha
"# fig: nome_estavel", usa <prefixo>_<nome_estavel>[_k].png.

Uso:
    python3 extract_figures.py entregas/A1_vision_transformers.ipynb -o relatorio/figuras --prefix A1
"""
import argparse
import base64
import json
import re
from pathlib import Path

FIG_RE = re.compile(r"#\s*fig:\s*([\w\-]+)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("notebook")
    p.add_argument("-o", "--out", default="relatorio/figuras")
    p.add_argument("--prefix", default=None)
    a = p.parse_args()

    nb = json.loads(Path(a.notebook).read_text(encoding="utf-8"))
    prefix = a.prefix or Path(a.notebook).stem.split("_")[0]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    count = 0
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        pngs = [o["data"]["image/png"] for o in cell.get("outputs", []) if "image/png" in o.get("data", {})]
        if not pngs:
            continue
        m = FIG_RE.search("".join(cell.get("source", [])))
        for k, data in enumerate(pngs):
            if m:
                name = f"{prefix}_{m.group(1)}" + (f"_{k}" if len(pngs) > 1 else "")
            else:
                name = f"{prefix}_cell{i:02d}_{k}"
            raw = "".join(data) if isinstance(data, list) else data
            (out / f"{name}.png").write_bytes(base64.b64decode(raw))
            count += 1
    print(f"{count} figura(s) extraída(s) para {out}")


if __name__ == "__main__":
    main()
