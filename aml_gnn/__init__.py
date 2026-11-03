"""
aml_gnn — Multi-PNA graph neural network for anti-money-laundering detection.

Modules (in pipeline order):
  config          every path and setting; workspaces for scored datasets
  graph_features  temporal split, port numbers, laundering typologies
  preprocess      transaction CSV → graph Parquet files
  data            graph → tensors, with a fixed feature encoding (feature spec)
  sampler         2-hop neighbour sampling around target transactions
  model           Multi-PNA network
  evaluate        prediction and metrics
  train           training loop with early stopping and a time budget
  artifacts       save/load model, feature spec, metrics, scores
  score           run the trained model on a new dataset
  store           indexed query layer over scored transactions
  cases           group high-risk transactions into laundering-ring cases
  typology        explain a case's laundering pattern
  str_report      draft suspicious transaction report for a case
  report          quantified results
  export_web      static data for the web app
  web_server      serve the web app locally
  bundle          zips for the Colab notebooks
  utils           logging

Run with:  python -m aml_gnn --help
"""
