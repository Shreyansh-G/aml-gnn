// accounts.js — find an account and see its risk, cases and alerts.
import { load } from "../data.js";
import { networkView } from "../graph.js";
import { el, kpi, num, riskBadge, riskOf, scoreCell, table, transactionColumns, typologyBadge } from "../ui.js";

const LIMIT = 50;

export async function render(params) {
  const accounts = Object.values(await load("accounts.json"));
  const body = el("div");
  let query = params.get("q") || "";

  const update = () => {
    history.replaceState(null, "", query ? `#/accounts?q=${encodeURIComponent(query)}` : "#/accounts");
    const q = query.trim().toLowerCase();
    const rows = accounts
      .filter((a) => !q || a.label.toLowerCase().includes(q) || String(a.node_id) === q)
      .sort((a, b) => b.max_score - a.max_score)
      .slice(0, LIMIT);
    body.replaceChildren(table([
      { label: "Account (bank | number)", value: (a) => el("a", { href: `#/account/${a.node_id}` }, a.label) },
      { label: "Risk", value: (a) => riskBadge(a.risk) },
      { label: "Max score", value: (a) => scoreCell(a.max_score) },
      { label: "Sent", num: true, value: (a) => num(a.transactions_sent) },
      { label: "Received", num: true, value: (a) => num(a.transactions_received) },
      { label: "High-risk transactions", num: true, value: (a) => num(a.high_risk_transactions) },
      { label: "Cases", num: true, value: (a) => num(a.cases.length) },
    ], rows, { onRow: (a) => { location.hash = `#/account/${a.node_id}`; }, empty: "No account matches." }));
  };

  update();
  return el("div", {},
    el("section", { class: "page-head" },
      el("h1", {}, "Accounts"),
      el("p", {}, `${num(accounts.length)} accounts that appear in an alert or a case. ` +
        `Search by account number or bank code; showing the ${LIMIT} riskiest matches.`)),
    el("div", { class: "filters" },
      el("label", {}, "Search",
        el("input", { type: "search", value: query, placeholder: "e.g. 800737690",
                      oninput: (e) => { query = e.target.value; update(); } }))),
    body);
}

export async function renderAccount(id) {
  const [accounts, alerts, cases, summary] = await Promise.all([
    load("accounts.json"), load("alerts.json"), load("cases.json"), load("summary.json")]);
  const labels = summary.has_labels;
  const a = accounts[id];
  if (!a) throw new Error(`Account ${id} does not appear in any alert or case.`);
  const graph = await load(`graphs/${a.node_id}.json`).catch(() => null);
  const caseById = new Map(cases.map((c) => [c.case_id, c]));
  const own = Object.values(alerts).flat()
    .filter((t) => t.sender_id === a.node_id || t.receiver_id === a.node_id)
    .sort((x, y) => y.score - x.score);

  return el("div", {},
    el("a", { class: "back", href: "#/accounts" }, "← All accounts"),
    el("section", { class: "page-head" },
      el("h1", {}, `${a.label} `, riskBadge(a.risk)),
      el("p", {}, "Risk from the model scores of all this account's transactions (every period).")),
    el("div", { class: "grid kpis" },
      kpi("Max score", a.max_score.toFixed(3), `mean ${a.mean_score.toFixed(3)}`),
      kpi("Sent / received", `${num(a.transactions_sent)} / ${num(a.transactions_received)}`),
      kpi("High-risk transactions", num(a.high_risk_transactions)),
      labels ? kpi("Confirmed laundering", num(a.known_laundering), "dataset labels") : null),
    graph ? accountNetwork(graph) : null,
    el("section", { class: "section" },
      el("h2", {}, `Cases (${a.cases.length})`),
      table([
        { label: "Case", value: (c) => el("a", { href: `#/case/${c.case_id}` }, c.case_id) },
        { label: "Typology", value: (c) => typologyBadge(c.typology) },
        { label: "Transactions", num: true, value: (c) => num(c.num_transactions) },
        { label: "Max score", value: (c) => scoreCell(c.max_score) },
      ], a.cases.map((cid) => caseById.get(cid)).filter(Boolean),
      { onRow: (c) => { location.hash = `#/case/${c.case_id}`; }, empty: "Not part of any case." })),
    el("section", { class: "section" },
      el("h2", {}, `In the alert queues (${own.length})`),
      table(transactionColumns({ withCase: true, withLabel: labels }), own, { empty: "Not in any daily alert queue." })));
}

/** Floating 1-hop / 2-hop network around the account, from graphs/<id>.json (see export_web.py). */
function accountNetwork(graph) {
  const holder = el("div");
  const toRecord = ([s, d, score, isLaundering, amount, currency, format, time]) => ({
    sender_id: s, receiver_id: d, sender: graph.nodes[s], receiver: graph.nodes[d], score,
    is_laundering: isLaundering, amount_paid: amount, payment_currency: currency, payment_format: format,
    timestamp: time, risk: riskOf(score),
  });
  const draw = (hops) => {
    const transactions = graph.edges.filter((e) => e[8] <= hops).map(toRecord);
    holder.replaceChildren(transactions.length
      ? networkView(transactions, new Set(), graph.seed)
      : el("p", { class: "muted" }, "No transactions in the test period."));
  };
  const toggle = el("div", { class: "segmented", role: "group", "aria-label": "Neighbourhood size" },
    [[1, "1 hop · direct transactions"], [2, "2 hops · their counterparties too"]].map(([hops, text]) =>
      el("button", {
        type: "button", class: hops === 1 ? "active" : null, "aria-pressed": String(hops === 1),
        onclick: (e) => {
          toggle.querySelectorAll("button").forEach((b) => {
            b.classList.toggle("active", b === e.currentTarget);
            b.setAttribute("aria-pressed", String(b === e.currentTarget));
          });
          draw(hops);
        },
      }, text)));
  draw(1);
  return el("section", { class: "section" },
    el("h2", {}, "Transaction network"),
    el("p", { class: "secondary" }, "The account (gold) and the money flowing around it in the test period — up to 60 direct and " +
      "120 second-hop transactions, riskiest first. Drag accounts to explore; hover a line for the transaction."),
    toggle, holder);
}
