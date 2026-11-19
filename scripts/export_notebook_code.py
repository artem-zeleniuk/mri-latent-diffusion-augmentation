"""Write the code of the notebook (Stages 1 to 4, cells 5, 7, 9 and 11) as the module nbcode.py,
which the other scripts in this folder import. With the three flags below the cells only define
their functions and print a notice.

usage: python export_notebook_code.py [NOTEBOOK] [OUTPUT]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CELLS = (5, 7, 9, 11)
HEAD = '''"""Code of the notebook oasis3_ldm_augmentation.ipynb (Stages 1 to 4) as a module.
Written by export_notebook_code.py from the notebook cells, do not edit by hand."""
LOAD_PRETRAINED_UNET = True         # the cells only define their functions and print a notice
LOAD_PRETRAINED_CLASSIFIER = True
DEMO_MODE = False

'''


def main():
    nb_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "oasis3_ldm_augmentation.ipynb"
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "nbcode.py"
    nb = json.loads(nb_path.read_text())
    src = {i: "".join(nb["cells"][i]["source"]) for i in CELLS}
    for i in CELLS:
        assert nb["cells"][i]["cell_type"] == "code", i
    out = HEAD + "\n\n".join(f"# ===== notebook cell {i} =====\n{src[i]}" for i in CELLS) + "\n"
    compile(out, "nbcode.py", "exec")
    out_path.write_text(out)
    print("wrote", out_path, len(out.splitlines()), "lines")


if __name__ == "__main__":
    main()
