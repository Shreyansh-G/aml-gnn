// main.js — hash router, theme toggle and app start-up.
import { load } from "./data.js";
import { el, initTooltips } from "./ui.js";
import * as accounts from "./views/accounts.js";
import * as alerts from "./views/alerts.js";
import * as caseView from "./views/case.js";
import * as cases from "./views/cases.js";
import * as overview from "./views/overview.js";

const routes = [
  { pattern: /^\/?$/, nav: "overview", view: () => overview.render() },
  { pattern: /^\/alerts$/, nav: "alerts", view: (params) => alerts.render(params) },
  { pattern: /^\/cases$/, nav: "cases", view: () => cases.render() },
  { pattern: /^\/case\/(C\d+)$/, nav: "cases", view: (_, m) => caseView.render(m[1]) },
  { pattern: /^\/accounts$/, nav: "accounts", view: (params) => accounts.render(params) },
  { pattern: /^\/account\/(\d+)$/, nav: "accounts", view: (_, m) => accounts.renderAccount(m[1]) },
];

let renderToken = 0;

async function route() {
  const token = ++renderToken;
  const [path, query = ""] = location.hash.replace(/^#/, "").split("?");
  const app = document.getElementById("app");
  const found = routes.map((r) => [r, r.pattern.exec(path)]).find(([, m]) => m);

  document.querySelectorAll(".nav a").forEach((a) =>
    a.classList.toggle("active", Boolean(found) && a.dataset.route === found[0].nav));
  if (!found) {
    app.replaceChildren(el("p", { class: "error" }, "Page not found. "), el("a", { href: "#/" }, "Back to the overview"));
    return;
  }

  app.replaceChildren(el("p", { class: "loading" }, "Loading…"));
  try {
    const view = await found[0].view(new URLSearchParams(query), found[1]);
    if (token === renderToken) app.replaceChildren(view);       // ignore stale renders
  } catch (error) {
    if (token === renderToken) app.replaceChildren(el("p", { class: "error" }, error.message));
  }
  window.scrollTo(0, 0);
}

function initTheme() {
  const root = document.documentElement;
  try {
    const saved = localStorage.getItem("theme");
    if (saved) root.dataset.theme = saved;
  } catch { /* storage unavailable: follow the OS setting */ }

  document.getElementById("theme-toggle").addEventListener("click", () => {
    const dark = root.dataset.theme
      ? root.dataset.theme === "dark"
      : window.matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("theme", root.dataset.theme); } catch { /* ignore */ }
    route();                                                     // redraw so the graph picks up new colours
  });
}

function showDataset() {
  load("summary.json")
    .then((summary) => {
      document.getElementById("dataset-tag").textContent = summary.dataset;
      document.title = `AML Investigation Center — ${summary.dataset}`;
    })
    .catch(() => { /* the page itself reports missing data */ });
}

initTheme();
initTooltips();
showDataset();
window.addEventListener("hashchange", route);
route();
