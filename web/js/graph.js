// graph.js — floating, draggable transaction network (vis-network, loaded from cdnjs with SRI).
import { el, money, when } from "./ui.js";

const token = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/**
 * transactions: records with sender_id, receiver_id, sender, receiver, score, risk, is_laundering, …
 * highlight:    account ids drawn filled (e.g. the accounts of a case)
 * seed:         the account under investigation, drawn gold and larger
 */
export function networkView(transactions, highlight = new Set(), seed = null) {
  const container = el("div", { class: "network", role: "img",
    "aria-label": `Transaction network of ${transactions.length} transactions` });
  const swatch = (color) => el("span", { class: "swatch", style: `background:${color}` });
  const dot = (color) => el("span", { class: "dot", style: `background:${color}` });
  const legend = el("div", { class: "legend" },
    el("span", {}, swatch(token("--critical")), "High risk (score ≥ 0.5)"),
    el("span", {}, swatch(token("--serious")), "Medium (0.1–0.5)"),
    el("span", {}, swatch(token("--axis")), "Low"),
    seed !== null ? el("span", {}, dot(token("--warning")), "Account under investigation") : null,
    el("span", {}, "Thick solid line = laundering per dataset labels · node size = number of transactions"));

  // Draw after the container is in the document so it has a size
  requestAnimationFrame(() => draw(container, transactions, highlight, seed));
  return el("div", {}, container, legend);
}

function draw(container, transactions, highlight, seed) {
  if (!window.vis) {
    container.textContent = "The network library could not be loaded.";
    return;
  }
  const risk = { HIGH: token("--critical"), MEDIUM: token("--serious"), LOW: token("--axis") };
  const labels = new Map();
  const degree = new Map();
  for (const t of transactions) {
    labels.set(t.sender_id, t.sender);
    labels.set(t.receiver_id, t.receiver);
    degree.set(t.sender_id, (degree.get(t.sender_id) || 0) + 1);
    degree.set(t.receiver_id, (degree.get(t.receiver_id) || 0) + 1);
  }
  const maxDegree = Math.max(1, ...degree.values());

  const nodes = [...labels].map(([id, label]) => {
    const isSeed = id === seed;
    const filled = isSeed || highlight.has(id);
    return {
      id,
      label: label.split(" | ")[1] ?? label,
      title: `${label} — ${degree.get(id)} transaction(s)${isSeed ? " · under investigation" : ""}`,
      value: isSeed ? maxDegree * 1.6 : degree.get(id),
      borderWidth: isSeed ? 3 : 1,
      color: {
        background: isSeed ? token("--warning") : filled ? token("--accent") : token("--surface-2"),
        border: isSeed ? token("--ink") : token("--accent"),
        highlight: { background: isSeed ? token("--warning") : token("--accent"), border: token("--ink") },
        hover: { background: isSeed ? token("--warning") : token("--accent"), border: token("--ink") },
      },
      font: { color: token("--ink"), size: isSeed ? 14 : 12, bold: isSeed },
    };
  });
  const edges = transactions.map((t, i) => ({
    id: i,
    from: t.sender_id,
    to: t.receiver_id,
    arrows: { to: { enabled: true, scaleFactor: 0.6 } },
    width: t.is_laundering ? 3 : 1.5,
    dashes: !t.is_laundering,
    color: { color: risk[t.risk], highlight: risk[t.risk], hover: risk[t.risk] },
    title: `${when(t.timestamp)} · ${money(t.amount_paid, t.payment_currency)} · ${t.payment_format} · score ${t.score.toFixed(3)}`,
  }));

  new window.vis.Network(container, { nodes, edges }, {
    nodes: { shape: "dot", scaling: { min: 8, max: 30 } },
    edges: { smooth: { type: "continuous" }, selfReference: { size: 14 } },
    physics: {
      solver: "forceAtlas2Based",
      forceAtlas2Based: { gravitationalConstant: -60, centralGravity: 0.01, springLength: 110, springConstant: 0.08 },
      maxVelocity: 40,
      timestep: 0.35,
      stabilization: { iterations: 150 },
    },
    interaction: { hover: true, tooltipDelay: 80 },
  });
}
