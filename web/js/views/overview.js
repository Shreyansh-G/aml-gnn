// overview.js — the landing page: headline numbers, how the project builds on Multi-PNA,
// a laundering ring the model caught, what investigators get, and how it works.
import { load } from "../data.js";
import { networkView } from "../graph.js";
import { bars, compareBars, el, heroStat, kpi, lineChart, num, pct, table, typologyBadge } from "../ui.js";

// Measured outside the exported data (see README):
const STARTING_PROJECT = { f1: 0.0226, labels: 3209 };   // GraphSAGE repository this project started from
const API_P50_MS = { before: 221.7, after: 18.3 };       // account-network query over 5.08M transactions

export async function render() {
  const [summary, cases] = await Promise.all([load("summary.json"), load("cases.json")]);
  const isTrainingData = summary.dataset === summary.trained_on;
  return el("div", {},
    hero(summary, isTrainingData),
    isTrainingData && summary.results.test ? builtOnMultiPNA(summary) : null,
    cases.length ? await featuredRing(cases[0]) : null,
    investigation(summary),
    pipeline(summary, isTrainingData),
    runChips(summary, isTrainingData));
}

const millions = (n) => `${(n / 1e6).toFixed(2)}M`;
const scoringSeconds = (m) => Math.round(m.total_edges / m.scoring_transactions_per_second);

// --- Hero ------------------------------------------------------------------------

function hero(summary, isTrainingData) {
  const r = summary.results;
  const f = summary.facts;
  if (isTrainingData && r.test) {
    const t = r.test;
    return heroBlock(
      `Catching money laundering in ${millions(f.transactions)} bank transactions`,
      "A Multi-PNA graph neural network reads every transfer together with the accounts around it, then turns " +
      "its scores into a daily alert queue, laundering-ring cases and draft suspicious transaction reports.",
      [heroStat(pct(t["at_0.5"].f1), `F1 on the laundering class — just ${pct(t.laundering / t.transactions, 2)} of transactions`),
       heroStat(pct(r.alert_budget[0].precision, 0), `of the top ${r.alert_budget[0].alerts_per_day} daily alerts are real laundering`),
       r.cases ? heroStat(`${r.cases.alerts_per_case.toFixed(1)}×`, `less to review: ${num(r.cases.alerts)} alerts → ${num(r.cases.cases)} cases`) : null,
       heroStat(`${scoringSeconds(summary.metrics)} s`, `to score all ${millions(f.transactions)} transactions on one GPU`)]);
  }
  const s = r.scores;
  return heroBlock(
    `“${summary.dataset}” — scored by the ${summary.trained_on} model`,
    "The trained Multi-PNA model was applied without retraining, using its training-time feature encoding and " +
    "alert threshold. " + (summary.has_labels ? "The data has labels, so the accuracy shown is out-of-dataset."
                                               : "The data has no labels: scores, alerts and cases only."),
    [heroStat(num(s.transactions), "transactions scored"),
     heroStat(num(s.high_risk), `high risk (${pct(s.high_risk_share, 2)} of transactions)`),
     r.cases ? heroStat(num(r.cases.cases), "laundering-ring cases to review") : null,
     r.test ? heroStat(pct(r.test["at_0.5"].f1), "F1 on this new dataset") : null]);
}

function heroBlock(title, text, stats) {
  return el("section", { class: "hero" }, el("h1", {}, title), el("p", {}, text), el("div", { class: "hero-stats" }, stats));
}

// --- Built on Multi-PNA ------------------------------------------------------------

function builtOnMultiPNA(summary) {
  const r = summary.results;
  const t = r.test;
  const f1 = t["at_0.5"].f1;
  const published = summary.published_f1.map(([name, value]) => ({ name, text: value, value: parseFloat(value) / 100 }));
  const find = (prefix) => published.find((p) => p.name.startsWith(prefix));
  const reference = find("Multi-PNA + edge");
  const xgboost = find("XGBoost");
  const gnn = find("GIN");
  const noGraph = find("MLP");

  const others = [
    ...published.map((p) => ({ name: p.name, value: p.value, label: p.text, tip: `${p.name}: published F1 ${p.text}% (mean ± std of 5 runs)` })),
    { name: "Starting project — GraphSAGE", value: STARTING_PROJECT.f1, label: (100 * STARTING_PROJECT.f1).toFixed(2),
      tip: "The repository this project started from" },
  ].sort((a, b) => b.value - a.value);
  const rows = [
    { name: "This project — Multi-PNA", value: f1, label: (100 * f1).toFixed(2), ours: true,
      tip: `This project: F1 ${pct(f1, 2)} on the test period, from a 25-minute training run` },
    ...others,
  ];

  const headline = [
    reference && `on par with the published Multi-PNA + edge updates (${reference.text})`,
    xgboost && `+${(100 * (f1 - xgboost.value)).toFixed(1)} points over XGBoost with graph features`,
    gnn && `${(f1 / gnn.value).toFixed(1)}× a plain GNN`,
    noGraph && `${Math.round(f1 / noGraph.value)}× a model that ignores the graph`,
  ].filter(Boolean);

  return el("section", { class: "section" },
    el("h2", {}, "Built on Multi-PNA — and taken further"),
    el("p", { class: "secondary" }, `Same benchmark and metric as the research papers. At ${pct(f1)} F1 this project is `,
      headline.map((h, i) => [i ? (i === headline.length - 1 ? " and " : ", ") : "", el("b", {}, h)]), "."),
    el("div", { class: "grid two" },
      el("div", { class: "card" },
        el("h3", {}, "F1 on the laundering class, IBM AMLworld HI-Small (%)"),
        compareBars(rows),
        el("p", { class: "muted note" }, "Published results: mean ± std of 5 runs. This project: one 25-minute run on one GPU.")),
      accuracyTrap(t)),
    el("h3", { class: "subhead" }, "What this project adds"),
    el("div", { class: "grid diffs" }, additions(summary)));
}

function accuracyTrap(t) {
  const accuracy = 1 - t.laundering / t.transactions;
  return el("div", { class: "card trap" },
    el("h3", {}, "Why not accuracy?"),
    el("p", { class: "secondary" }, "A model that never flags anything at all would score:"),
    el("div", { class: "trap-grid" },
      el("div", {}, el("div", { class: "big" }, pct(accuracy, 2)), el("div", { class: "muted" }, "accuracy")),
      el("div", {}, el("div", { class: "big bad" }, `0 / ${num(t.laundering)}`), el("div", { class: "muted" }, "laundering caught"))),
    el("p", {}, "That's why this project, like the research, is judged on F1 for the laundering class — where it reaches ",
      el("b", {}, pct(t["at_0.5"].f1)), ", catching ", el("b", {}, num(t["at_0.5"].true_positives)),
      ` of ${num(t.laundering)} laundering transactions.`));
}

function additions(summary) {
  const r = summary.results;
  const f = summary.facts;
  const card = (value, title, text) => el("div", { class: "card diff" },
    el("div", { class: "diff-value" }, value), el("h4", {}, title), el("p", {}, text));
  return [
    card(`${scoringSeconds(summary.metrics)} s`, "Fast GPU pipeline",
      `All ${millions(f.transactions)} transactions scored on one GPU; neighbourhood sampling written in plain PyTorch, ` +
      "so no compiled graph extensions are needed."),
    r.cases && card(`${r.cases.alerts_per_case.toFixed(1)}×`, "Investigation workflow",
      `${num(r.cases.alerts)} high-risk alerts grouped into ${num(r.cases.cases)} laundering-ring cases, each with an ` +
      "explained typology and a one-click draft STR."),
    card(pct(r.alert_budget[0].precision, 0), "Alert queue that hits",
      `of the top ${r.alert_budget[0].alerts_per_day} daily alerts are real laundering; ` +
      `${num(r.alert_budget[1].alerts_per_day)} a day catch ${pct(r.alert_budget[1].recall, 0)} of all laundering.`),
    card(`${Math.round(API_P50_MS.before / API_P50_MS.after)}×`, "Production-ready API",
      `Faster account-network queries with a CSR index (${API_P50_MS.before} → ${API_P50_MS.after} ms), ` +
      "behind API-key security, Docker and CI."),
    card("0 retraining", "Works on any bank's data",
      "Scores a new transaction CSV with the frozen training-time encoding, verified identical across 5.08M transactions."),
    card(`+${pct(f.laundering / STARTING_PROJECT.labels - 1, 0)}`, "Complete labels",
      `More laundering examples than the starting project: all ${num(f.laundering)} labels, not ${num(STARTING_PROJECT.labels)}.`),
  ].filter(Boolean);
}

// --- Featured ring -----------------------------------------------------------------

async function featuredRing(top) {
  const c = await load(`cases/${top.case_id}.json`);
  return el("section", { class: "section" },
    el("h2", {}, "A laundering ring the model caught"),
    el("p", { class: "secondary" }, typologyBadge(c.typology), " ", c.evidence,
      ` ${num(c.num_transactions)} transactions across ${num(c.num_accounts)} accounts, model score up to ${c.max_score.toFixed(3)}. ` +
      "Drag the accounts to explore; hover a line for the transaction."),
    networkView(c.transactions, new Set(c.accounts.map((a) => a.node_id))),
    el("p", { class: "more" }, el("a", { href: `#/case/${c.case_id}` }, `Open case ${c.case_id} and its draft report →`)));
}

// --- What investigators get --------------------------------------------------------

function investigation(summary) {
  const r = summary.results;
  const cards = [
    r.alert_budget && el("div", { class: "card" },
      el("h3", {}, "Alert budget: hit rate vs coverage"),
      el("p", { class: "secondary" }, "Reviewing more alerts per day catches more laundering, while a smaller share of alerts are hits."),
      lineChart({
        categories: r.alert_budget.map((a) => num(a.alerts_per_day)),
        xTitle: "Alerts reviewed per day",
        series: [
          { name: "Precision", color: "var(--series-1)", values: r.alert_budget.map((a) => a.precision) },
          { name: "Recall", color: "var(--series-2)", values: r.alert_budget.map((a) => a.recall) },
        ],
        format: (v) => pct(v, 0),
        caption: "Precision and recall by number of alerts reviewed per day",
      }),
      el("details", {}, el("summary", {}, "Show as table"),
        table([
          { label: "Alerts / day", num: true, value: (a) => num(a.alerts_per_day) },
          { label: "Laundering caught", num: true, value: (a) => num(a.laundering_caught) },
          { label: "Precision", num: true, value: (a) => pct(a.precision) },
          { label: "Recall", num: true, value: (a) => pct(a.recall) },
        ], r.alert_budget))),
    r.recall_by_typology && typologyCard(r),
  ].filter(Boolean);

  return el("section", { class: "section" },
    el("h2", {}, "What investigators get"),
    cards.length ? el("div", { class: "grid two" }, cards) : null,
    r.cases ? caseTiles(summary) : null);
}

function typologyCard(r) {
  const known = r.recall_by_typology.filter((x) => x.typology !== "UNLISTED").sort((a, b) => b.recall - a.recall);
  if (!known.length) return null;
  const overall = known.reduce((a, x) => a + x.caught, 0) / known.reduce((a, x) => a + x.laundering, 0);
  return el("div", { class: "card" },
    el("h3", {}, `Laundering patterns caught — ${pct(overall, 0)} overall`),
    el("p", { class: "secondary" }, `Share of each of the ${known.length} known laundering patterns that the model flags.`),
    bars(known.map((x) => ({
      name: x.typology, value: x.recall, label: pct(x.recall, 0),
      tip: `${x.typology}: ${num(x.caught)} of ${num(x.laundering)} caught`,
    }))));
}

function caseTiles(summary) {
  const c = summary.results.cases;
  return el("div", { class: "grid kpis tiles" },
    kpi("Cases to review", num(c.cases), `from ${num(c.alerts)} high-risk transactions`),
    summary.has_labels ? kpi("Cases with confirmed laundering", pct(c.share_with_laundering, 0), `${num(c.cases_with_laundering)} cases`) : null,
    kpi("Draft STRs ready", num(c.cases), "one per case, generated automatically"),
    kpi("Largest ring", num(c.largest_case), "transactions in one case"));
}

// --- How it works --------------------------------------------------------------------

function pipeline(summary, isTrainingData) {
  const m = summary.metrics;
  const f = summary.facts;
  const r = summary.results;
  const steps = [
    [millions(f.transactions), "transactions", isTrainingData ? "IBM AMLworld HI-Small" : `dataset “${summary.dataset}”`],
    [num(f.accounts), "accounts", "linked into a transaction graph"],
    [`${Math.round(m.parameters / 1000)}K`, "parameter Multi-PNA", "reads each transfer's 2-hop neighbourhood"],
    [`${scoringSeconds(m)} s`, "to score everything", m.gpu === "cpu" ? "on a CPU" : "on one GPU"],
    [r.cases ? num(r.cases.cases) : "—", "cases", "alert queue · rings · draft STRs"],
  ];
  return el("section", { class: "section" },
    el("h2", {}, "How it works"),
    el("div", { class: "pipeline" }, steps.map(([value, title, sub], i) => [
      i ? el("span", { class: "pipeline-arrow", "aria-hidden": "true" }, "→") : null,
      el("div", { class: "step" },
        el("div", { class: "step-value" }, value), el("div", { class: "step-title" }, title), el("div", { class: "step-sub" }, sub))])));
}

function runChips(summary, isTrainingData) {
  const m = summary.metrics;
  const chips = isTrainingData
    ? [m.gpu, `${m.epochs_trained} epochs · best ${m.best_epoch}`, `${m.time_budget_minutes}-minute training budget`,
       `peak ${m.peak_gpu_memory_gb} GB GPU memory`, `${num(m.train_transactions_per_second)} transactions/s training`,
       `${num(m.scoring_transactions_per_second)} transactions/s scoring`]
    : [`model trained on ${summary.trained_on}`, `scored on ${m.gpu}`, `${num(m.scoring_transactions_per_second)} transactions/s`];
  return el("section", { class: "section" },
    el("h2", {}, "Model and run"),
    el("div", { class: "chips" }, chips.map((c) => el("span", { class: "chip" }, c))));
}
