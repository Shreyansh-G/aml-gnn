// ui.js — DOM helpers, formatters and shared widgets.
// Text always goes in through text nodes (never innerHTML), so data cannot inject markup.

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "style") node.style.cssText = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : String(child));
  }
  return node;
}

const SVG_NS = "http://www.w3.org/2000/svg";

export function svg(tag, attrs = {}, ...children) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "style") node.style.cssText = value;
    else node.setAttribute(key, value);
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : String(child));
  }
  return node;
}

// --- Formatters -------------------------------------------------------------

export const pct = (x, digits = 1) => `${(100 * x).toFixed(digits)}%`;
export const num = (x) => Number(x).toLocaleString("en-US");
export const money = (amount, currency) =>
  `${Number(amount).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })} ${currency}`;
export const when = (ts) => String(ts).slice(0, 16);

export function duration(from, to) {
  const h = (Date.parse(String(to).replace(" ", "T")) - Date.parse(String(from).replace(" ", "T"))) / 36e5;
  if (!Number.isFinite(h)) return "—";
  if (h < 1) return `${Math.round(h * 60)} min`;
  return h < 48 ? `${h.toFixed(1)} h` : `${(h / 24).toFixed(1)} days`;
}

// --- Widgets ----------------------------------------------------------------

export const riskOf = (score) => (score >= 0.5 ? "HIGH" : score >= 0.1 ? "MEDIUM" : "LOW");
export const riskBadge = (risk) => el("span", { class: `badge ${risk}` }, risk);
export const typologyBadge = (t) => el("span", { class: "badge typology" }, t);
export const labelBadge = (isLaundering) => isLaundering
  ? el("span", { class: "badge label", "data-tip": "Ground-truth label in the synthetic dataset" }, "Laundering")
  : el("span", { class: "muted" }, "—");

export function scoreCell(score) {
  return el("span", { class: "score num", "data-tip": `Model laundering probability: ${score.toFixed(4)}` },
    el("span", { class: "score-track" },
      el("span", { class: "score-fill", style: `width:${Math.max(2, 100 * score)}%` })),
    score.toFixed(3));
}

export const accountLink = (id, label) =>
  el("a", { href: `#/account/${id}`, onclick: (e) => e.stopPropagation() }, label);
export const caseLink = (caseId) => caseId
  ? el("a", { href: `#/case/${caseId}`, onclick: (e) => e.stopPropagation() }, caseId)
  : el("span", { class: "muted" }, "—");

export function kpi(label, value, sub) {
  return el("div", { class: "card kpi" },
    el("div", { class: "label" }, label),
    el("div", { class: "value" }, value),
    sub ? el("div", { class: "sub" }, sub) : null);
}

/** columns: [{ label, value: row => Node|string, num?: bool, wrap?: bool }] */
export function table(columns, rows, { onRow, empty = "Nothing to show." } = {}) {
  const head = el("tr", {}, columns.map((c) => el("th", { scope: "col", class: c.num ? "num" : null }, c.label)));
  const body = rows.length
    ? rows.map((row) => el("tr", {
        class: onRow ? "clickable" : null,
        tabindex: onRow ? 0 : null,
        onclick: onRow ? () => onRow(row) : null,
        onkeydown: onRow ? (e) => { if (e.key === "Enter") onRow(row); } : null,
      }, columns.map((c) => el("td", { class: [c.num && "num", c.wrap && "wrap"].filter(Boolean).join(" ") || null },
        c.value(row)))))
    : [el("tr", {}, el("td", { colspan: columns.length, class: "muted" }, empty))];
  return el("div", { class: "table-wrap" }, el("table", {}, el("thead", {}, head), el("tbody", {}, body)));
}

/** One-hue horizontal bars with direct value labels. rows: [{ name, value (0–1), label, tip, muted?, onclick? }] */
export function bars(rows) {
  return el("div", { class: "bars" }, rows.map((r) =>
    el("div", {
      class: ["bar-row", r.muted && "muted", r.onclick && "clickable"].filter(Boolean).join(" "),
      "data-tip": r.tip,
      onclick: r.onclick,
      tabindex: r.onclick ? 0 : null,
      onkeydown: r.onclick ? (e) => { if (e.key === "Enter") r.onclick(); } : null,
    },
      el("span", { class: "name" }, r.name),
      el("span", { class: "bar-track" }, el("span", { class: "bar-fill", style: `width:${Math.max(1, 100 * r.value)}%;display:block` })),
      el("span", { class: "value" }, r.label))));
}

/** Ranked comparison bars on a 0–100% scale; the row marked `ours` is highlighted. */
export function compareBars(rows) {
  return el("div", { class: "compare" }, rows.map((r) => {
    const width = `${Math.max(0.6, 100 * r.value)}%`;
    return el("div", { class: `compare-row${r.ours ? " ours" : ""}`, "data-tip": r.tip },
      el("span", { class: "name" }, r.name),
      el("span", { class: "compare-track" },
        el("span", { class: "compare-fill", style: `width:${width}` }),
        el("span", { class: "compare-value", style: `left:${width}` }, r.label)));
  }));
}

/** Line chart for 0–1 values over a few categories, with legend, end labels and hover tooltips. */
export function lineChart({ categories, series, format, xTitle, caption }) {
  const W = 560, H = 260, m = { top: 16, right: 132, bottom: 52, left: 48 };
  const iw = W - m.left - m.right, ih = H - m.top - m.bottom;
  const x = (i) => m.left + (categories.length > 1 ? (i * iw) / (categories.length - 1) : iw / 2);
  const y = (v) => m.top + ih * (1 - v);
  const chart = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": caption },
    [0, 0.25, 0.5, 0.75, 1].map((t) => [
      svg("line", { class: t === 0 ? "axis-line" : "grid-line", x1: m.left, x2: m.left + iw, y1: y(t), y2: y(t) }),
      svg("text", { class: "tick", x: m.left - 8, y: y(t) + 4, "text-anchor": "end" }, `${Math.round(100 * t)}%`)]),
    categories.map((c, i) => svg("text", { class: "tick", x: x(i), y: m.top + ih + 20, "text-anchor": "middle" }, c)),
    xTitle ? svg("text", { class: "tick", x: m.left + iw / 2, y: H - 8, "text-anchor": "middle" }, xTitle) : null,
    series.map((s) => svg("g", {},
      svg("path", { class: "series-line", style: `stroke:${s.color}`,
                    d: s.values.map((v, i) => `${i ? "L" : "M"}${x(i)},${y(v)}`).join(" ") }),
      s.values.map((v, i) => [
        svg("circle", { class: "series-dot", cx: x(i), cy: y(v), r: 4.5, style: `fill:${s.color}` }),
        svg("circle", { class: "hit", cx: x(i), cy: y(v), r: 14, "data-tip": `${s.name} at ${categories[i]}: ${format(v)}` })]),
      svg("text", { class: "series-label", x: x(s.values.length - 1) + 12, y: y(s.values.at(-1)) + 4 },
          `${s.name} ${format(s.values.at(-1))}`))));
  return el("div", { class: "chart" },
    el("div", { class: "legend-row" }, series.map((s) =>
      el("span", {}, el("span", { class: "legend-swatch", style: `background:${s.color}` }), s.name))),
    chart);
}

export function heroStat(value, label) {
  return el("div", { class: "hero-stat" }, el("div", { class: "value" }, value), el("div", { class: "label" }, label));
}

export function meter(label, value, text) {
  const percent = Math.round(100 * Math.min(1, Math.max(0, value)));
  return el("div", { class: "meter", role: "meter", "aria-label": label,
                     "aria-valuemin": 0, "aria-valuemax": 100, "aria-valuenow": percent },
    el("div", { class: "meter-head" }, el("span", {}, label), el("b", {}, text)),
    el("div", { class: "meter-track" }, el("div", { class: "meter-fill", style: `width:${percent}%` })));
}

export function transactionColumns({ withCase = false, withLabel = true } = {}) {
  const columns = [
    { label: "Time", value: (t) => when(t.timestamp) },
    { label: "From", value: (t) => accountLink(t.sender_id, t.sender) },
    { label: "To", value: (t) => accountLink(t.receiver_id, t.receiver) },
    { label: "Amount paid", num: true, value: (t) => money(t.amount_paid, t.payment_currency) },
    { label: "Format", value: (t) => t.payment_format },
    { label: "Model score", value: (t) => scoreCell(t.score) },
    { label: "Risk", value: (t) => riskBadge(t.risk) },
  ];
  if (withLabel) columns.push({ label: "Dataset label", value: (t) => labelBadge(t.is_laundering) });
  if (withCase) columns.push({ label: "Case", value: (t) => caseLink(t.case_id) });
  return columns;
}

// --- Hover tooltips for any element with data-tip ---------------------------

export function initTooltips() {
  const tip = document.getElementById("tooltip");
  document.addEventListener("mouseover", (e) => {
    const target = e.target.closest("[data-tip]");
    tip.hidden = !target;
    if (target) tip.textContent = target.dataset.tip;
  });
  document.addEventListener("mousemove", (e) => {
    if (tip.hidden) return;
    tip.style.left = `${Math.min(e.clientX + 12, window.innerWidth - tip.offsetWidth - 8)}px`;
    tip.style.top = `${e.clientY + 16}px`;
  });
}
