"""
Single entry point for the project.

  python -m aml_gnn prepare                    raw HI-Small CSV → graph Parquet files
  python -m aml_gnn train [options]            train Multi-PNA, write artifacts/
  python -m aml_gnn score --input X --name N   run the trained model on a new dataset → runs/N/
  python -m aml_gnn report [--name N]          quantified results → artifacts/results.md
  python -m aml_gnn export-web [--name N]      static data for the web app
  python -m aml_gnn web [--name N]             open the web app in the browser (http://127.0.0.1:8080)
  python -m aml_gnn api [--name N]             serve the REST API (needs AML_API_KEY)
  python -m aml_gnn bundle {train,model}       zip for the Colab notebooks → dist/

--name N works on the workspace runs/N/ of a scored dataset instead of HI-Small.
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from . import config


def activate(name) -> None:
    """Switch every path to the workspace of a scored dataset (no-op for HI-Small)."""
    if name:
        workspace = config.RUNS_DIR / name
        if not workspace.exists():
            sys.exit(f"No workspace {workspace} — run `python -m aml_gnn score --name {name}` first.")
        config.use_workspace(workspace, dataset_name=name)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m aml_gnn",
                                     description="Multi-PNA anti-money-laundering pipeline.")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("prepare", help="build the HI-Small transaction graph from data/raw/")

    t = commands.add_parser("train", help="train Multi-PNA and write artifacts/")
    t.add_argument("--epochs", type=int, default=config.EPOCHS)
    t.add_argument("--patience", type=int, default=config.PATIENCE)
    t.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    t.add_argument("--neighbors", type=int, nargs="+", default=config.NUM_NEIGHBORS)
    t.add_argument("--train-batches", type=int, default=config.TRAIN_BATCHES,
                   help="training batches per epoch (0 = full pass)")
    t.add_argument("--time-budget", type=float, default=config.TIME_BUDGET,
                   help="minutes of training, then keep the best epoch (0 = no limit)")
    t.add_argument("--seed", type=int, default=config.SEED)
    t.add_argument("--max-batches", type=int, default=None,
                   help="cap batches per pass, for quick smoke tests")
    t.add_argument("--no-scores", action="store_true",
                   help="skip scoring every transaction after training")

    s = commands.add_parser("score", help="run the trained model on a new dataset (inference only)")
    s.add_argument("--input", type=Path, required=True, help="transaction CSV")
    s.add_argument("--name", required=True, help="workspace name → runs/<name>/")
    s.add_argument("--columns", type=Path,
                   help="JSON mapping of your column names (see examples/column_mapping.json); "
                        "omit for the IBM AMLworld format")
    s.add_argument("--patterns", type=Path, help="AMLworld *_Patterns.txt, for recall per typology")
    s.add_argument("--time-format", help="timestamp format, e.g. %%Y-%%m-%%d %%H:%%M:%%S (default: inferred)")
    s.add_argument("--start", type=date.fromisoformat, help="first day to score (YYYY-MM-DD)")
    s.add_argument("--end", type=date.fromisoformat, help="last day to score (YYYY-MM-DD)")
    s.add_argument("--model-dir", type=Path, default=config.MODEL_DIR,
                   help="folder with multipna_model.pt and feature_spec.json")
    s.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    s.add_argument("--neighbors", type=int, nargs="+", default=config.NUM_NEIGHBORS)
    s.add_argument("--max-batches", type=int, default=None, help="cap scoring batches, for smoke tests")

    for name, text in [("report", "quantified results of the last training or scoring run"),
                       ("export-web", "precompute the static data for the web app")]:
        commands.add_parser(name, help=text).add_argument("--name", help="scored dataset workspace")

    w = commands.add_parser("web", help="open the web app in the browser")
    w.add_argument("--name", help="scored dataset workspace")
    w.add_argument("--port", type=int, default=8080)
    w.add_argument("--no-browser", action="store_true", help="serve without opening a browser")

    a = commands.add_parser("api", help="serve the REST API (set AML_API_KEY first)")
    a.add_argument("--name", help="scored dataset workspace")
    a.add_argument("--host", default="127.0.0.1")
    a.add_argument("--port", type=int, default=8000)

    commands.add_parser("bundle", help="zip for the Colab notebooks → dist/").add_argument(
        "kind", choices=["train", "model"], help="train: code + HI-Small data; model: code + trained model")

    args = parser.parse_args()
    if args.command == "prepare":
        from .preprocess import build_graph
        build_graph()
    elif args.command == "train":
        from .train import train
        train(epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
              neighbors=args.neighbors, train_batches=args.train_batches,
              time_budget=args.time_budget, seed=args.seed, max_batches=args.max_batches,
              score_all=not args.no_scores)
    elif args.command == "score":
        from .score import score_dataset
        columns = json.loads(args.columns.read_text()) if args.columns else None
        score_dataset(args.input, args.name, columns, args.patterns, args.time_format, args.start,
                      args.end, args.model_dir, args.batch_size, args.neighbors, args.max_batches)
    elif args.command == "report":
        activate(args.name)
        from .report import build_report, to_markdown
        print(to_markdown(build_report()))
    elif args.command == "export-web":
        activate(args.name)
        from .export_web import export_web
        export_web()
    elif args.command == "web":
        activate(args.name)
        from .web_server import serve
        serve(config.WEB_DATA_DIR.parent, args.port, open_browser=not args.no_browser)
    elif args.command == "api":
        activate(args.name)
        import uvicorn
        sys.path.insert(0, str(config.PROJECT_ROOT))
        uvicorn.run("api.main:app", host=args.host, port=args.port)
    else:
        from .bundle import build_bundle
        build_bundle(args.kind)


if __name__ == "__main__":
    main()
