"""
bundle.py — build the zip files uploaded to Google Drive for the Colab notebooks.

  python -m aml_gnn bundle train   → dist/AML_GNN_colab.zip  (code + HI-Small raw data, ~90 MB)
  python -m aml_gnn bundle model   → dist/AML_GNN_model.zip  (code + trained model + web app, <1 MB)
"""

import zipfile

from . import config
from .utils import get_logger

log = get_logger(__name__)

MODEL_FILES = ["multipna_model.pt", "feature_spec.json", "training_metrics.json", "results.json"]


def _files(kind: str) -> list:
    root = config.PROJECT_ROOT
    code = sorted(p for p in (root / "aml_gnn").glob("*.py"))
    if kind == "train":
        return code + [root / "data/raw/HI-Small_Trans.csv", root / "data/raw/HI-Small_Patterns.txt"]
    web_app = [root / "web/index.html", root / "web/_headers"] + sorted(
        p for folder in ("css", "js") for p in (root / "web" / folder).rglob("*") if p.is_file())
    return (code + [root / "artifacts" / name for name in MODEL_FILES]
            + web_app + sorted((root / "examples").glob("*")))


def build_bundle(kind: str):
    """Zip the files one Colab notebook needs; returns the zip's path."""
    files = _files(kind)
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise SystemExit("Missing: " + ", ".join(missing))

    config.DIST_DIR.mkdir(exist_ok=True)
    path = config.DIST_DIR / ("AML_GNN_colab.zip" if kind == "train" else "AML_GNN_model.zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for f in files:
            bundle.write(f, f.relative_to(config.PROJECT_ROOT).as_posix())
    log.info("Built %s (%d files, %.1f MB) — upload it to MyDrive/AML/", path, len(files), path.stat().st_size / 1e6)
    return path
