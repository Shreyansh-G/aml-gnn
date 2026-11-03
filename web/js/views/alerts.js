// alerts.js — daily alert queue: the riskiest transactions an investigator reviews first.
import { load } from "../data.js";
import { el, meter, num, pct, table, transactionColumns } from "../ui.js";

const BUDGETS = [25, 50, 100, 250, 500];
const chip = (label, value) => el("span", { class: "chip" }, `${label}: `, el("b", {}, value));

export async function render(params) {
  const [alerts, summary] = await Promise.all([load("alerts.json"), load("summary.json")]);
  const labels = summary.has_labels;
  const days = Object.keys(alerts);
  let day = days.includes(params.get("day")) ? params.get("day") : days[0];
  let budget = BUDGETS.includes(Number(params.get("n"))) ? Number(params.get("n")) : 100;
  const body = el("div");

  const update = () => {
    history.replaceState(null, "", `#/alerts?day=${day}&n=${budget}`);
    const rows = alerts[day].slice(0, budget);
    const confirmed = rows.filter((t) => t.is_laundering).length;
    body.replaceChildren(
      el("div", { class: "chips" },
        chip("Reviewed", num(rows.length)),
        chip("High risk", num(rows.filter((t) => t.risk === "HIGH").length)),
        labels ? chip("Confirmed laundering", `${num(confirmed)} (${pct(confirmed / Math.max(rows.length, 1))})`) : null,
        chip("Part of a case", num(rows.filter((t) => t.case_id).length))),
      labels ? meter("Hit rate — reviewed alerts that are real laundering", confirmed / Math.max(rows.length, 1),
                     `${num(confirmed)} of ${num(rows.length)} · ${pct(confirmed / Math.max(rows.length, 1), 0)}`) : null,
      table(transactionColumns({ withCase: true, withLabel: labels }), rows));
  };

  const filters = el("div", { class: "filters" },
    el("label", {}, "Day",
      el("select", { onchange: (e) => { day = e.target.value; update(); } },
        days.map((d) => el("option", { value: d, selected: d === day }, `${d} (${alerts[d].length} ranked)`)))),
    el("label", {}, "Alerts to review",
      el("select", { onchange: (e) => { budget = Number(e.target.value); update(); } },
        BUDGETS.map((n) => el("option", { value: n, selected: n === budget }, `Top ${n}`)))));

  update();
  return el("div", {},
    el("section", { class: "page-head" },
      el("h1", {}, "Alert queue"),
      el("p", {}, "Each day's transactions ranked by the model's laundering probability — what an investigator works " +
        "through first. Pick how many alerts the team can review" +
        (labels ? "; the confirmed-laundering share is the hit rate." : "."))),
    filters, body);
}
