# AML-GNN: Graph Neural Network for Anti-Money-Laundering Detection

This project detects money-laundering transactions in the IBM AMLworld **HI-Small** dataset (5.08M transactions, 515K accounts) using a **Multi-PNA** graph neural network. The model's scores feed an investigation web app with these pages:
- an overview with the headline results and comparisons
- a daily alert queue with its hit rate
- laundering-ring cases, each with an explained typology, a floating network graph and a draft suspicious transaction report (STR)
- account lookup with a floating 1-hop / 2-hop transaction network

The trained model can also score new transaction datasets without retraining.

## Results

These are results on the test period: a temporal 64/19/17 split, 863,900 transactions of which 1,611 are laundering, from a single training run.

| Model | HI-Small F1, laundering class (%) |
|---|---|
| Multi-PNA (this project) | 67.37 |
| Multi-PNA (Egressy et al., AAAI 2024) | 64.59 ± 3.60 |
| Multi-PNA + edge updates (Egressy et al., AAAI 2024) | 68.16 ± 2.65 |
| XGBoost + graph features (Altman et al., NeurIPS 2023) | 63.23 ± 0.17 |

| Metric | Value |
|---|---|
| Precision / recall at threshold 0.5 | 65.9% / 68.9% |
| F1 at the validation-tuned threshold (0.60) | 69.2% |
| AUPRC / ROC-AUC | 0.696 / 0.992 |
| Precision of the top 100 alerts per day | 81.0% |
| Recall on the 8 generated laundering typologies | 87.2% |
| Cases (score ≥ 0.5) | 1,693 transactions in 431 cases |

The full report is in [`artifacts/results.md`](artifacts/results.md). Laundering that fits none of the generated typologies ("UNLISTED") is rarely detected: recall is 1.3%.

## Quick start

These commands are for Windows PowerShell. On macOS or Linux, use `.venv/bin/python` instead.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

.\.venv\Scripts\python.exe -m aml_gnn web        # open the web app → http://127.0.0.1:8080
.\.venv\Scripts\python.exe -m pytest             # run the tests
```

The trained model, its scores and the web app data are included (`artifacts/`, `web/data/`). Only retraining needs the raw dataset.

## Commands

| Command | What it does |
|---|---|
| `python -m aml_gnn prepare` | Raw HI-Small CSV → transaction graph (`data/processed/graph/`) |
| `python -m aml_gnn train` | Train Multi-PNA (GPU), then score every transaction → `artifacts/` |
| `python -m aml_gnn report` | Results report → `artifacts/results.md` |
| `python -m aml_gnn export-web` | Static data for the web app → `web/data/` |
| `python -m aml_gnn web` | Serve the web app locally and open the browser |
| `python -m aml_gnn score --input X.csv --name N` | Score a new dataset with the trained model → `runs/N/` |
| `python -m aml_gnn api` | REST API → http://127.0.0.1:8000/docs (needs `AML_API_KEY`) |
| `python -m aml_gnn bundle train` / `bundle model` | Zips for the Colab notebooks → `dist/` |

`report`, `export-web`, `web` and `api` accept `--name N` to work on a scored dataset instead of HI-Small.

## Project structure

```
├── aml_gnn/                 # Python package: pipeline, model, investigation logic, CLI
│   ├── config.py            #   paths and settings
│   ├── preprocess.py        #   CSV → graph (+ graph_features.py: split, ports, typologies)
│   ├── data.py, sampler.py  #   tensors with a fixed feature encoding; 2-hop neighbour sampling
│   ├── model.py             #   Multi-PNA
│   ├── train.py, evaluate.py, artifacts.py, report.py
│   ├── score.py             #   run the trained model on a new dataset
│   ├── store.py             #   indexed query layer over scored transactions
│   ├── cases.py, typology.py, str_report.py
│   ├── export_web.py, web_server.py, bundle.py
│   └── __main__.py          #   command line
├── api/                     # FastAPI REST service (routes, schemas, API-key security)
├── web/                     # static web app (index.html, css/, js/, data/, _headers)
├── artifacts/               # trained model, feature spec, scores, results
├── examples/                # column mapping + sample bank-format CSV for `score`
├── notebooks/               # Colab: train on an A100; score a new dataset
├── tests/                   # pytest suite (synthetic data)
├── Dockerfile, docker-compose.yml, netlify.toml, .github/workflows/ci.yml
└── requirements.txt, requirements-serve.txt, pyproject.toml
```

## How it works

1. **Graph.** Each account is a node, identified by its `(bank, account)` pair. Each transaction is a directed edge with these features:
   - amount (log-scaled)
   - one-hot currency and payment format
   - timestamp
   - port numbers

   Labels come from the dataset's `Is Laundering` column. The split is temporal, cut at day boundaries.
2. **Model.** Multi-PNA (57K parameters) is PNA message passing with four adaptations:
   - reverse message passing
   - port numbers
   - ego IDs
   - edge updates

   Each transaction is scored with its 2-hop neighbourhood, up to 100 transactions per account per hop. Sampling runs in plain PyTorch on the GPU.
3. **Training.**
   - Loss: weighted cross-entropy.
   - Model selection: early stopping on validation F1.
   - Budget: 25 minutes of training (256 batches of 8,192 per epoch), then the best epoch is kept.
   - Scoring: the best model scores all 5.08M transactions in a batch, in about 68 s on an A100.
4. **Investigation.**
   - **Indexes:** an account-to-transaction (CSR) index and a time index serve account and neighbourhood queries, so they never scan all 5M rows.
   - **Cases:** high-risk transactions (score ≥ 0.5) are grouped into cases by shared accounts.
   - **Typology:** graph rules name each case's pattern (FAN-OUT, CYCLE, SCATTER-GATHER, …).
   - **Report:** each case gets a draft STR.
5. **Serving.**
   - The web app is static: `export-web` writes its data as JSON.
   - The REST API reads the same scores, so neither needs a GPU.

## Scoring a new dataset

```powershell
.\.venv\Scripts\python.exe -m aml_gnn score --input examples\sample_bank_transactions.csv --columns examples\column_mapping.json --name demo-bank
.\.venv\Scripts\python.exe -m aml_gnn web --name demo-bank
```

- **Input format:** IBM AMLworld CSVs work as they are. For other formats, `--columns` maps your column names (see [`examples/column_mapping.json`](examples/column_mapping.json)).
  - Required: timestamp, sender, receiver and an amount.
  - Missing bank columns default to a single institution, and missing currency or format to "Unknown".
- **Encoding:** features use the training-time encoding stored in `artifacts/feature_spec.json`. Categories the model never saw are encoded as zeros.
- **Threshold:** the alert threshold is the one tuned on HI-Small validation. It is not re-tuned on the new data.
- **Labels are optional.** With an `is_laundering` column, the report measures accuracy on the new data. Without one, it shows scores, alerts and cases.
- **Output:** `runs/<name>/` holds the graph, scores, report, cases and a self-contained copy of the web app. `runs/` is not tracked by git.
- **Larger datasets:** use [`notebooks/colab_score_new_dataset.ipynb`](notebooks/colab_score_new_dataset.ipynb) on a GPU. By default it scores a date window of IBM's separately simulated HI-Medium dataset.
- **Sample data:** the sample CSV is a slice of the HI-Small test period in a bank-style format. It demonstrates the workflow and is not a benchmark.

## Training on Colab

1. Run `python -m aml_gnn bundle train` and upload `dist/AML_GNN_colab.zip` to `MyDrive/AML/`.
2. Open [`notebooks/colab_train_a100.ipynb`](notebooks/colab_train_a100.ipynb) on an A100 or G4 runtime and run all cells (about 35 minutes).
3. Copy `multipna_model.pt`, `training_metrics.json`, `edge_scores.parquet` and `predictions.parquet` from `MyDrive/AML/output/` into `artifacts/`.
4. Run `report` and `export-web`.

For a full-length run, use `--time-budget 0 --train-batches 0 --epochs 100 --patience 100`.

## Deploying the web app

`web/` is a static site, so these hosts work:
- **Netlify:** log in at app.netlify.com and choose *Add new site → Deploy manually*, then drop the `web` folder. To deploy from GitHub instead, import the repository; `netlify.toml` publishes `web/`.
- **Cloudflare Pages:** *Upload assets*, using the `web` folder.
- **Hugging Face Static Space:** upload the contents of `web/`.

Netlify and Cloudflare apply `web/_headers`: a Content-Security-Policy, frame and content-type protection, and HSTS. CDN scripts are pinned with Subresource Integrity, and draft reports are sanitised with DOMPurify before rendering.

## REST API

```powershell
$env:AML_API_KEY = "a-long-random-secret"
.\.venv\Scripts\python.exe -m aml_gnn api        # docs at http://127.0.0.1:8000/docs
```

| Endpoint | Returns |
|---|---|
| `GET /health` | Status and model summary (public) |
| `GET /v1/model` | Training metrics |
| `GET /v1/transactions/{edge_id}` | One scored transaction |
| `GET /v1/accounts?account=…` | Accounts matching an account number |
| `GET /v1/accounts/{node_id}` | Account risk summary |
| `GET /v1/accounts/{node_id}/neighborhood?hops=1` | Scored transactions around an account |
| `GET /v1/alerts?day=2022-09-10&limit=100` | The day's highest-risk transactions |
| `GET /v1/cases` | Laundering-ring cases with their typology |
| `GET /v1/cases/{case_id}` and `/str` | One case and its draft STR (Markdown) |

**Security:**
- `/v1` routes need the `X-API-Key` header. The key is compared in constant time.
- If no key is configured, those routes return 503 instead of being left open.
- Inputs are validated and responses are typed (Pydantic).

**Docker:** run `docker compose up --build` after setting `AML_API_KEY` in `.env` (see `.env.example`). The image is slim, runs as a non-root user without PyTorch, and mounts `data/processed/` and `artifacts/` read-only.

**Latency** (measured in-process on a laptop CPU over 5.08M transactions, p50):
- account risk: 6 ms
- neighbourhood: 18 ms
- one-day alert queue: 38 ms

## Tests and CI

`python -m pytest` runs 36 tests on synthetic data, covering preprocessing, sampling, the model, scoring new datasets, cases and typologies, the report and the API. The GitHub Actions workflow runs the tests and builds the Docker image on every push.

## Credits

- Model and evaluation protocol: Egressy et al., *Provably Powerful Graph Neural Networks for Directed Multigraphs*, AAAI 2024. Reference code: [IBM/Multi-GNN](https://github.com/IBM/Multi-GNN) (Apache-2.0).
- Dataset: Altman et al., *Realistic Synthetic Financial Transactions for Anti-Money Laundering Models*, NeurIPS 2023, published on [Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering) under CDLA-Sharing-1.0. The sample CSV and `web/data/` are derived from it.
- The project started from [sikene3/Graph-Neural-Network](https://github.com/sikene3/Graph-Neural-Network). The pipeline, model and apps have since been rewritten.
