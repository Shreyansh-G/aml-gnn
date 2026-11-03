// case.js — one case: why it is suspicious, its network, transactions and a draft STR.
import { load } from "../data.js";
import { networkView } from "../graph.js";
import { duration, el, kpi, num, table, transactionColumns, typologyBadge } from "../ui.js";

export async function render(caseId) {
  const [c, summary] = await Promise.all([
    load(`cases/${caseId}.json`).catch(() => { throw new Error(`Case ${caseId} was not found.`); }),
    load("summary.json"),
  ]);
  const labels = summary.has_labels;
  const subject = new Set(c.accounts.map((a) => a.node_id));

  return el("div", {},
    el("a", { class: "back", href: "#/cases" }, "← All cases"),
    el("section", { class: "page-head" },
      el("h1", {}, `Case ${c.case_id} `, typologyBadge(c.typology)),
      el("p", { class: "evidence" }, c.evidence)),

    el("div", { class: "grid kpis" },
      kpi("Accounts", num(c.num_accounts)),
      kpi("Transactions", num(c.num_transactions)),
      kpi("Max model score", c.max_score.toFixed(3), `mean ${c.mean_score.toFixed(3)}`),
      kpi("Time span", duration(c.first_seen, c.last_seen), String(c.first_seen).slice(0, 16)),
      labels ? kpi("Confirmed laundering", num(c.known_laundering), "dataset labels, for evaluation") : null),

    el("section", { class: "section" },
      el("h2", {}, "Transaction network"),
      networkView(c.transactions, subject)),

    el("section", { class: "section" },
      el("h2", {}, "Transactions"),
      table(transactionColumns({ withLabel: labels }), c.transactions)),

    el("section", { class: "section" },
      el("h2", {}, "Draft suspicious transaction report"),
      el("p", { class: "secondary" }, "Generated from the case for analyst review — edit, then file per the bank's STR policy."),
      el("div", { class: "actions" },
        el("button", { class: "button", type: "button", onclick: () => download(`STR-${c.case_id}.md`, c.str) }, "Download (.md)"),
        el("button", { class: "button ghost", type: "button", onclick: () => window.print() }, "Print")),
      strDocument(c.str)));
}

function strDocument(markdown) {
  const doc = el("article", { class: "str" });
  if (window.marked && window.DOMPurify) {
    doc.innerHTML = window.DOMPurify.sanitize(window.marked.parse(markdown));   // sanitised before insertion
  } else {
    doc.append(el("pre", { style: "white-space:pre-wrap" }, markdown));
  }
  return doc;
}

function download(filename, text) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/markdown" }));
  const link = el("a", { href: url, download: filename });
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}
