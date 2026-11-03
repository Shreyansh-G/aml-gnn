// cases.js — laundering-ring cases: alerts grouped by shared accounts, with an explained typology.
import { load } from "../data.js";
import { bars, duration, el, num, pct, scoreCell, table, typologyBadge, when } from "../ui.js";

const chip = (label, value) => el("span", { class: "chip" }, `${label}: `, el("b", {}, value));

export async function render() {
  const [cases, summary] = await Promise.all([load("cases.json"), load("summary.json")]);
  const stats = summary.results.cases;
  const labels = summary.has_labels;
  const typologies = [...new Set(cases.map((c) => c.typology))].sort();
  const state = { typology: "", minTx: 1, confirmedOnly: false, query: "" };
  const body = el("div");

  const update = () => {
    const query = state.query.trim().toUpperCase();
    const rows = cases.filter((c) =>
      (!state.typology || c.typology === state.typology)
      && c.num_transactions >= state.minTx
      && (!state.confirmedOnly || c.known_laundering > 0)
      && (!query || c.case_id.includes(query)));
    body.replaceChildren(
      el("div", { class: "chips" },
        chip("Showing", `${num(rows.length)} of ${num(cases.length)} cases`),
        chip("Transactions", num(rows.reduce((a, c) => a + c.num_transactions, 0)))),
      table([
        { label: "Case", value: (c) => el("a", { href: `#/case/${c.case_id}` }, c.case_id) },
        { label: "Typology", value: (c) => typologyBadge(c.typology) },
        { label: "Why", wrap: true, value: (c) => c.evidence },
        { label: "Accounts", num: true, value: (c) => num(c.num_accounts) },
        { label: "Transactions", num: true, value: (c) => num(c.num_transactions) },
        { label: "Max score", value: (c) => scoreCell(c.max_score) },
        { label: "First seen", value: (c) => when(c.first_seen) },
        { label: "Span", value: (c) => duration(c.first_seen, c.last_seen) },
        labels ? { label: "Confirmed", num: true, value: (c) => num(c.known_laundering) } : null,
      ].filter(Boolean), rows, { onRow: (c) => { location.hash = `#/case/${c.case_id}`; }, empty: "No case matches these filters." }));
  };

  const typologySelect = el("select", { onchange: (e) => { state.typology = e.target.value; update(); } },
    el("option", { value: "" }, "All typologies"),
    typologies.map((t) => el("option", { value: t }, t)));
  const pick = (typology) => { state.typology = typology; typologySelect.value = typology; update(); };

  const filters = el("div", { class: "filters" },
    el("label", {}, "Typology", typologySelect),
    el("label", {}, "Size",
      el("select", { onchange: (e) => { state.minTx = Number(e.target.value); update(); } },
        [1, 3, 5, 10].map((n) => el("option", { value: n }, n === 1 ? "Any size" : `≥ ${n} transactions`)))),
    el("label", {}, "Case ID",
      el("input", { type: "search", placeholder: "e.g. C2244742", oninput: (e) => { state.query = e.target.value; update(); } })),
    labels ? el("label", { class: "inline" },
      el("input", { type: "checkbox", onchange: (e) => { state.confirmedOnly = e.target.checked; update(); } }),
      "Confirmed laundering only") : null);

  update();
  return el("div", {},
    el("section", { class: "page-head" },
      el("h1", {}, "Cases"),
      el("p", {}, `${num(stats.alerts)} high-risk transactions (score ≥ ${stats.min_score}) grouped into ${num(stats.cases)} cases ` +
        `by shared accounts — ${stats.alerts_per_case.toFixed(1)}× less to review. ` +
        (labels ? `${pct(stats.share_with_laundering)} of cases contain confirmed laundering. ` : "") +
        "Each case comes with its laundering pattern explained and a draft STR.")),
    typologyChart(cases, pick),
    filters, body);
}

const SINGLE_TRANSFER = new Set(["PAIR", "SELF-TRANSFER"]);   // one flagged transfer: no pattern shape

/** Cases per detected multi-transaction pattern; clicking a bar filters the list. */
function typologyChart(cases, pick) {
  const counts = new Map();
  cases.filter((c) => !SINGLE_TRANSFER.has(c.typology))
    .forEach((c) => counts.set(c.typology, (counts.get(c.typology) || 0) + 1));
  const rows = [...counts].sort((a, b) => b[1] - a[1]);
  if (!rows.length) return null;
  const single = cases.filter((c) => SINGLE_TRANSFER.has(c.typology)).length;
  return el("div", { class: "card section-card" },
    el("h3", {}, "Laundering patterns detected"),
    el("p", { class: "secondary" }, "Multi-transaction cases by pattern. Click a pattern to filter the list."),
    bars(rows.map(([typology, n]) => ({
      name: typology, value: n / rows[0][1], label: num(n),
      tip: `${typology}: ${num(n)} cases — click to filter`,
      onclick: () => pick(typology),
    }))),
    single ? el("p", { class: "muted note" }, `Plus ${num(single)} single-transfer cases.`) : null);
}
