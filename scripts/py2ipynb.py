#!/usr/bin/env python3
"""Converte um notebook em formato percent (.py) para .ipynb (nbformat 4), checando a sintaxe das células.

Formato de entrada:
    # %%                 -> início de célula de código
    # %% [markdown]      -> início de célula markdown (linhas prefixadas com "# ")

Uso:
    python3 py2ipynb.py notebooks/src/A3_cnn_kaggle.py -o notebooks/A3_cnn_kaggle.ipynb
    python3 py2ipynb.py --selftest
"""
import argparse
import ast
import json
import re
import sys
import tempfile
from pathlib import Path

CELL_RE = re.compile(r"^# %%(?P<rest>.*)$")
MAGIC_RE = re.compile(r"^\s*[!%]")


def parse_cells(text):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        lines = buf[:]
        while lines and not lines[-1].strip():
            lines.pop()
        while lines and not lines[0].strip():
            lines.pop(0)
        if kind == "markdown":
            lines = [l[2:] if l.startswith("# ") else l.lstrip("#") for l in lines]
        cells.append((kind, lines))

    for line in text.splitlines():
        m = CELL_RE.match(line)
        if m:
            flush()
            kind = "markdown" if "[markdown]" in m.group("rest") else "code"
            buf = []
        elif kind is not None:
            buf.append(line)
    flush()
    return cells


def check_syntax(cells):
    errors = []
    for i, (kind, lines) in enumerate(cells):
        if kind != "code":
            continue
        src = "\n".join("pass" if MAGIC_RE.match(l) else l for l in lines)
        try:
            ast.parse(src)
        except SyntaxError as e:
            errors.append(f"célula {i} linha {e.lineno}: {e.msg}")
    return errors


def to_notebook(cells):
    nb_cells = []
    for kind, lines in cells:
        source = [l + "\n" for l in lines[:-1]] + lines[-1:]
        cell = {"cell_type": kind, "metadata": {}, "source": source}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb_cells.append(cell)
    return {
        "cells": nb_cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def convert(src_path, out_path):
    cells = parse_cells(Path(src_path).read_text(encoding="utf-8"))
    if not cells:
        return ["nenhuma célula encontrada (falta '# %%')"]
    errors = check_syntax(cells)
    if errors:
        return errors
    Path(out_path).write_text(json.dumps(to_notebook(cells), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return []


def selftest():
    ok_src = "# %% [markdown]\n# # Título\n# texto\n\n# %%\n!pip install -q x\nimport os\nprint(os.name)\n"
    bad_src = "# %%\ndef f(:\n    pass\n"
    with tempfile.TemporaryDirectory() as d:
        ok, bad, out = Path(d, "ok.py"), Path(d, "bad.py"), Path(d, "ok.ipynb")
        ok.write_text(ok_src)
        bad.write_text(bad_src)
        assert convert(ok, out) == [], "conversão válida falhou"
        nb = json.loads(out.read_text())
        assert nb["nbformat"] == 4 and len(nb["cells"]) == 2
        assert nb["cells"][0]["source"] == ["# Título\n", "texto"]
        assert nb["cells"][1]["outputs"] == []
        assert convert(bad, Path(d, "bad.ipynb")), "erro de sintaxe não detectado"
    print("selftest OK")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", nargs="?")
    p.add_argument("-o", "--out")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        selftest()
        return
    if not a.src:
        p.error("informe o arquivo .py de entrada")
    out = a.out or str(Path(a.src).with_suffix(".ipynb"))
    errors = convert(a.src, out)
    if errors:
        print("ERROS de sintaxe:\n  " + "\n  ".join(errors), file=sys.stderr)
        sys.exit(1)
    print(f"OK -> {out}")


if __name__ == "__main__":
    main()
