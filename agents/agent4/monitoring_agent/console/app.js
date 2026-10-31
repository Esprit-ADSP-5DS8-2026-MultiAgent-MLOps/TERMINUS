// Console de l'Agent 4 : lit l'API de la gateway (/monitoring/*), sans framework ni build.
const API = "/monitoring";
const POLL_MS = 2500;
const SVGNS = "http://www.w3.org/2000/svg";

const ACTION = {
  no_action: "Aucune action", increase_monitoring: "Surveillance renforcée", open_incident: "Incident ouvert",
  notify_data_team: "Équipe data prévenue", request_retraining: "Réentraînement demandé",
  rollback_model: "Retour au modèle précédent",
};
const ACTION_PROPOSED = {
  no_action: "ne rien faire", increase_monitoring: "renforcer la surveillance", open_incident: "ouvrir un incident",
  notify_data_team: "prévenir l'équipe data", request_retraining: "demander un réentraînement",
  rollback_model: "revenir au modèle précédent",
};
const CAUSE = {
  covariate_shift: "Les entrées ont changé et la performance baisse",
  concept_drift: "La relation entre les entrées et la cible a changé",
  data_quality: "Les données reçues sont invalides",
  service_degradation: "Le service du modèle est dégradé",
  benign_drift: "Les entrées ont changé, sans effet sur la performance",
  no_issue: "Rien d'anormal",
};
const SIGNAL = {
  dataset_drift: "Drift sur la majorité des features", feature_drift: "Drift de features",
  prediction_drift: "Les prédictions se déplacent", data_quality: "Qualité des données",
  performance_drop_estimated: "Baisse de performance estimée", performance_drop_realized: "Baisse de performance mesurée",
  concept_drift_suspected: "Concept drift suspecté",
};
const QUALITY = {
  missing_column: "colonne absente", missing_values: "valeurs manquantes", type_mismatch: "type invalide",
  unseen_category: "catégorie inconnue", out_of_range: "hors de l'intervalle d'entraînement",
};
const STAGE = {
  prepare: ["Dossier", "code"], investigate: ["Enquête", "LLM + outils"], diagnose: ["Diagnostic", "LLM, vote"],
  verify: ["Vérification des preuves", "code"], judge: ["Juge indépendant", "autre LLM"],
  policy: ["Règles", "code"], await_human: ["Validation humaine", "humain"], act: ["Action", "code"],
};
const DETERMINISTIC = new Set(["prepare", "verify", "policy", "act"]);

const state = { status: null, windows: [], decisions: [], alerts: [], selected: null, follow: true, openId: null, seen: new Set(), first: true, ratings: {} };
const $ = (id) => document.getElementById(id);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toLocaleString("fr-FR", { maximumFractionDigits: d, minimumFractionDigits: Math.min(d, 2) }));
const int = (v) => (v ?? 0).toLocaleString("fr-FR");
const pct = (v) => `${(v * 100).toLocaleString("fr-FR", { maximumFractionDigits: 1 })} %`;
const time = (iso) => (iso ? new Date(iso).toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit", second: "2-digit" }) : "");

function deciderName(name) {
  if (!name) return "—";
  return name.replace(/^rules \(repli : LLM indisponible\)/, "à règles (LLM indisponible)").replace(/^rules$/, "à règles")
    .replace(/^règles \(aucune clé LLM\)$/, "à règles (aucune clé LLM)");
}

async function api(path, opts = {}) {
  const r = await fetch(API + path, { headers: { "content-type": "application/json" }, ...opts });
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail ?? detail; } catch { /* corps non JSON */ }
    throw new Error(detail);
  }
  const type = r.headers.get("content-type") || "";
  return type.includes("json") ? r.json() : r.text();
}

function toast(msg) {
  const t = $("toast");
  t.textContent = msg; t.hidden = false;
  clearTimeout(toast.timer); toast.timer = setTimeout(() => { t.hidden = true; }, 3800);
}

function svg(tag, attrs = {}, text) {
  const el = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (text !== undefined) el.textContent = text;
  return el;
}

// ---------------------------------------------------------------------------
// Rafraîchissement
// ---------------------------------------------------------------------------

async function refresh() {
  try {
    const [status, windows, decisions, alerts, ratings] = await Promise.all([
      api("/status"), api("/windows?limit=60"), api("/decisions?limit=60"), api("/alerts?firing=true"),
      api("/ratings").catch(() => ({ by_ref: {} })),
    ]);
    state.ratings = ratings.by_ref || {};
    const changedDecisions = JSON.stringify(decisions) !== JSON.stringify(state.decisions);
    Object.assign(state, { status, windows, decisions, alerts });
    // On suit la dernière fenêtre, sauf si l'utilisateur en a choisi une plus ancienne.
    if (state.follow || state.selected === null || !windows.some((w) => w.window.index === state.selected)) {
      state.selected = windows.length ? windows[windows.length - 1].window.index : null;
    }
    renderBar(); renderFilm(); renderHeatmap(); renderPerf(); renderWindow();
    if (changedDecisions) renderDecisions();
    renderBusy();
    renderAlerts();
    if (state.openId && changedDecisions) openDecision(state.openId, true);
    state.first = false;
  } catch (err) {
    $("agent-state").textContent = "Gateway injoignable";
    $("agent-models").textContent = err.message;
  }
}

// ---------------------------------------------------------------------------
// Barre du haut
// ---------------------------------------------------------------------------

function renderBar() {
  const s = state.status;
  document.title = `${s.project} · Agent 4`;
  $("project").textContent = s.project;
  const thr = s.min_threshold != null ? `, seuil métier ${fmt(s.min_threshold, 2)}` : "";
  $("spec").textContent = `${s.task === "classification" ? "Classification" : "Régression"} surveillée sur ${s.features.length} features, métrique ${s.metric}${thr}`;
  const up = $("v-upstream");
  up.querySelector(".dot").className = `dot ${s.upstream_up === true ? "up" : s.upstream_up === false ? "down" : ""}`;
  up.querySelector("span:last-child").textContent = s.upstream_up === true ? "en ligne" : s.upstream_up === false ? "hors ligne" : "—";
  $("v-preds").textContent = int(s.predictions);
  $("v-labels").textContent = int(s.labels);
  $("v-windows").textContent = `${int(s.windows)} × ${int(s.window_size)}`;

  const a = s.agent;
  const st = $("agent-state");
  if (!a) {
    st.textContent = "Agent désactivé"; st.classList.remove("busy");
    $("agent-models").textContent = "gateway lancée avec --no-agent";
    $("btn-decide").disabled = true;
  } else {
    const pending = state.decisions.filter((d) => d.status === "awaiting_human").length;
    st.textContent = a.busy ? "L'agent enquête" : pending ? `${pending} décision en attente de toi` : "L'agent veille";
    st.classList.toggle("busy", a.busy);
    $("agent-models").textContent = `Décideur ${deciderName(a.decider)}${a.judge ? `, juge ${a.judge}` : ", sans juge"}`;
    $("btn-decide").disabled = a.busy || !state.windows.length;
  }

  const sim = s.simulation || {};
  const bar = $("sim-progress");
  $("btn-sim").disabled = !!sim.running;
  if (sim.running || (sim.total && sim.sent < sim.total)) {
    bar.hidden = false;
    bar.querySelector(".fill").style.width = `${Math.round((100 * (sim.sent || 0)) / (sim.total || 1))}%`;
    bar.querySelector("span").textContent = `Simulation « ${$("sim-scenario").querySelector(`[value="${sim.scenario}"]`)?.textContent ?? sim.scenario} » : ${int(sim.sent)} / ${int(sim.total)} requêtes, ${int(sim.labels)} labels renvoyés`;
  } else {
    bar.hidden = true;
  }
}

// ---------------------------------------------------------------------------
// Film des fenêtres
// ---------------------------------------------------------------------------

function renderFilm() {
  const film = $("film");
  if (!state.windows.length) {
    film.innerHTML = `<li class="empty" style="max-width:none;flex:1">Aucune fenêtre analysée pour l'instant : il faut ${int(state.status.window_size)} prédictions par fenêtre. Lance une simulation avec le bouton « Simuler », ou envoie du trafic sur <code>/predict</code>.</li>`;
    return;
  }
  const decided = new Set(state.decisions.map((d) => d.window_index));
  film.innerHTML = "";
  for (const w of state.windows) {
    const i = w.window.index;
    const li = document.createElement("li");
    const b = document.createElement("button");
    b.type = "button";
    b.className = `sev-${w.severity}`;
    b.setAttribute("aria-pressed", String(i === state.selected));
    b.title = `Fenêtre ${i} : ${w.signals.map((s) => SIGNAL[s.kind]).join(", ") || "rien à signaler"}`;
    b.setAttribute("aria-label", b.title);
    if (decided.has(i)) b.insertAdjacentHTML("beforeend", '<span class="agent-mark" aria-hidden="true"></span>');
    b.addEventListener("click", () => {
      state.selected = i;
      state.follow = i === state.windows[state.windows.length - 1].window.index;
      renderFilm(); renderHeatmap(); renderPerf(); renderWindow();
    });
    li.append(b);
    li.insertAdjacentHTML("beforeend", `<span class="idx">${i}</span>`);
    film.append(li);
  }
  film.scrollLeft = film.scrollWidth;
}

// ---------------------------------------------------------------------------
// Heatmap du drift
// ---------------------------------------------------------------------------

const RAMP = [[0, [238, 242, 246]], [0.1, [240, 217, 168]], [0.25, [227, 155, 91]], [0.6, [201, 71, 47]], [2, [126, 29, 29]]];
function heatColor(psi) {
  const v = Math.max(0, Math.min(psi, 2));
  for (let k = 1; k < RAMP.length; k++) {
    const [x1, c1] = RAMP[k];
    const [x0, c0] = RAMP[k - 1];
    if (v <= x1) {
      const t = (v - x0) / (x1 - x0);
      return `rgb(${c0.map((c, j) => Math.round(c + t * (c1[j] - c))).join(",")})`;
    }
  }
  return "rgb(126,29,29)";
}

function tooltip() {
  let tip = document.querySelector(".tip");
  if (!tip) { tip = document.createElement("div"); tip.className = "tip"; tip.hidden = true; document.body.append(tip); }
  return tip;
}
function bindTip(el, html) {
  const tip = tooltip();
  el.addEventListener("mousemove", (e) => { tip.innerHTML = html; tip.hidden = false; tip.style.left = `${e.clientX + 14}px`; tip.style.top = `${e.clientY + 12}px`; });
  el.addEventListener("mouseleave", () => { tip.hidden = true; });
}

function renderHeatmap() {
  const box = $("heatmap");
  box.innerHTML = "";
  const ws = state.windows;
  if (!ws.length) { box.innerHTML = '<p class="calm">La carte apparaîtra avec la première fenêtre.</p>'; return; }
  const feats = state.status.features.slice();
  const maxPsi = Object.fromEntries(feats.map((f) => [f, 0]));
  for (const w of ws) for (const d of w.feature_drift) maxPsi[d.feature] = Math.max(maxPsi[d.feature] ?? 0, d.psi);
  feats.sort((a, b) => maxPsi[b] - maxPsi[a]);

  const avail = Math.max(260, box.clientWidth);
  const labelW = Math.min(150, Math.round(avail * 0.34)), rowH = 18, top = 6;
  const cell = Math.max(8, Math.min(34, Math.floor((avail - labelW - 4) / ws.length)));
  const W = labelW + ws.length * cell + 4, H = top + feats.length * rowH + 22;
  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": "Carte du drift : PSI par feature et par fenêtre" });
  const byWin = ws.map((w) => Object.fromEntries(w.feature_drift.map((d) => [d.feature, d])));

  const selCol = ws.findIndex((w) => w.window.index === state.selected);
  if (selCol >= 0) root.append(svg("rect", { x: labelW + selCol * cell - 1, y: 0, width: cell + 2, height: top + feats.length * rowH + 2, fill: "none", stroke: "#152238", "stroke-width": 1.5, rx: 2 }));

  feats.forEach((f, r) => {
    const y = top + r * rowH;
    root.append(svg("text", { x: labelW - 8, y: y + rowH * 0.68, "text-anchor": "end", "font-size": 11.5, style: "font-variation-settings:'wdth' 72" }, f));
    ws.forEach((w, c) => {
      const d = byWin[c][f];
      if (!d) return;
      const x = labelW + c * cell;
      const rect = svg("rect", { x: x + 1, y: y + 1, width: cell - 2, height: rowH - 2, rx: 2, fill: heatColor(d.psi) });
      root.append(rect);
      if (d.drifted) root.append(svg("rect", { x: x + 2, y: y + 2, width: cell - 4, height: rowH - 4, rx: 1.5, fill: "none", stroke: "#152238", "stroke-width": 1.4 }));
      bindTip(rect, `<b>${esc(f)}</b>, fenêtre ${w.window.index}<br>PSI ${fmt(d.psi)}${d.drifted ? " (drift confirmé)" : ""}<br>${esc(d.method)} : ${fmt(d.score, 4)} (seuil ${fmt(d.threshold, 4)})`);
    });
  });
  const step = Math.ceil(ws.length / 12);
  ws.forEach((w, c) => {
    if (c % step === 0 || c === ws.length - 1) root.append(svg("text", { x: labelW + c * cell + cell / 2, y: H - 6, "text-anchor": "middle", "font-size": 11 }, w.window.index));
  });
  box.append(root);
}

// ---------------------------------------------------------------------------
// Performance estimée vs mesurée
// ---------------------------------------------------------------------------

function renderPerf() {
  const box = $("perf");
  box.innerHTML = "";
  const ws = state.windows.filter((w) => w.performance);
  if (!ws.length) { box.innerHTML = '<p class="calm">Pas encore de performance à afficher.</p>'; return; }
  const m = ws[0].performance.metric;
  const higher = !["mae", "rmse"].includes(m);
  const ref = ws[ws.length - 1].performance.reference_mean, bound = ws[ws.length - 1].performance.bound;
  $("perf-hint").textContent = `${m} ${higher ? "(plus haut = mieux)" : "(plus bas = mieux)"} : estimée sans labels, puis mesurée quand la vérité terrain arrive.`;
  const est = ws.map((w) => w.performance.estimated?.[m] ?? null);
  const real = ws.map((w) => w.performance.realized?.[m] ?? null);
  const vals = [...est, ...real, ref, bound].filter((v) => v !== null && v !== undefined);
  let lo = Math.min(...vals), hi = Math.max(...vals);
  const pad = (hi - lo) * 0.12 || 0.01; lo -= pad; hi += pad;

  const W = Math.max(280, box.clientWidth), H = 240, L = 52, R = 12, T = 10, B = 26;
  const x = (i) => L + (ws.length === 1 ? (W - L - R) / 2 : (i * (W - L - R)) / (ws.length - 1));
  const y = (v) => T + ((hi - v) * (H - T - B)) / (hi - lo);
  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": `Performance ${m} estimée et mesurée par fenêtre` });

  // zone « sous la limite » : là où la performance n'est plus normale
  const badTop = higher ? y(bound) : T, badH = higher ? H - B - y(bound) : y(bound) - T;
  root.append(svg("rect", { x: L, y: badTop, width: W - L - R, height: Math.max(0, badH), fill: "#f7e1de", opacity: 0.7 }));
  for (let k = 0; k <= 4; k++) {
    const v = lo + ((hi - lo) * k) / 4;
    root.append(svg("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), stroke: "#dfe5ec" }));
    root.append(svg("text", { x: L - 8, y: y(v) + 4, "text-anchor": "end", "font-size": 11 }, fmt(v, 3)));
  }
  root.append(svg("line", { x1: L, x2: W - R, y1: y(ref), y2: y(ref), stroke: "#7b889b", "stroke-width": 1 }));
  root.append(svg("line", { x1: L, x2: W - R, y1: y(bound), y2: y(bound), stroke: "#b42318", "stroke-width": 1.5, "stroke-dasharray": "2 3" }));

  const path = (arr) => arr.map((v, i) => (v === null ? null : [x(i), y(v)])).reduce((acc, p) => {
    if (!p) { acc.push(null); return acc; }
    const last = acc[acc.length - 1];
    acc.push(`${last ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`); return acc;
  }, []).filter(Boolean).join(" ");
  if (est.some((v) => v !== null)) root.append(svg("path", { d: path(est), fill: "none", stroke: "#5b6b82", "stroke-width": 2, "stroke-dasharray": "5 4" }));
  if (real.some((v) => v !== null)) root.append(svg("path", { d: path(real), fill: "none", stroke: "#152238", "stroke-width": 2.6 }));

  ws.forEach((w, i) => {
    const p = w.performance;
    const sel = w.window.index === state.selected;
    if (sel) root.append(svg("line", { x1: x(i), x2: x(i), y1: T, y2: H - B, stroke: "#152238", "stroke-width": 1, opacity: 0.35 }));
    for (const [v, bad, kind] of [[est[i], p.estimated_degraded, "estimée"], [real[i], p.realized_degraded, "mesurée"]]) {
      if (v === null) continue;
      const dot = svg("circle", { cx: x(i), cy: y(v), r: sel ? 5 : 3.5, fill: bad ? "#b42318" : kind === "mesurée" ? "#152238" : "#f6f8fa", stroke: bad ? "#b42318" : kind === "mesurée" ? "#152238" : "#5b6b82", "stroke-width": 1.5 });
      bindTip(dot, `Fenêtre ${w.window.index}<br>${m} ${kind} : <b>${fmt(v, 4)}</b><br>limite ${fmt(bound, 4)}${bad ? "<br>sous la limite" : ""}`);
      root.append(dot);
    }
    const step = Math.ceil(ws.length / 12);
    if (i % step === 0 || i === ws.length - 1) root.append(svg("text", { x: x(i), y: H - 8, "text-anchor": "middle", "font-size": 11 }, w.window.index));
  });
  box.append(root);
}

// ---------------------------------------------------------------------------
// Détail de la fenêtre sélectionnée
// ---------------------------------------------------------------------------

function renderWindow() {
  const box = $("window-detail");
  const w = state.windows.find((x) => x.window.index === state.selected);
  if (!w) { $("win-title").textContent = "Fenêtre"; box.innerHTML = '<p class="calm">Choisis une fenêtre dans le film.</p>'; return; }
  $("win-title").textContent = `Fenêtre ${w.window.index}`;
  $("win-when").textContent = w.window.start ? `${int(w.window.n_rows)} prédictions, de ${time(w.window.start)} à ${time(w.window.end)}` : `${int(w.window.n_rows)} prédictions`;
  const sig = w.signals.length
    ? `<ul class="signals">${w.signals.map((s) => `<li class="s-${s.severity}"><span><b>${esc(SIGNAL[s.kind] ?? s.kind)}</b><br>${esc(s.message)}</span></li>`).join("")}</ul>`
    : '<p class="calm">Aucun signal : la fenêtre ressemble aux données d\'entraînement.</p>';
  const q = w.data_quality.length
    ? `<table class="mini"><thead><tr><th>Feature</th><th>Problème</th><th class="num">Taux</th></tr></thead><tbody>${w.data_quality.map((x) => `<tr><td>${esc(x.feature)}</td><td>${esc(QUALITY[x.kind] ?? x.kind)}</td><td class="num">${pct(x.value)}</td></tr>`).join("")}</tbody></table>`
    : "";
  const top = w.feature_drift.filter((d) => d.drifted).sort((a, b) => b.psi - a.psi).slice(0, 5);
  const drift = top.length
    ? `<table class="mini" style="margin-top:12px"><thead><tr><th>Feature en drift</th><th class="num">PSI</th><th>Référence → maintenant</th></tr></thead><tbody>${top.map((d) => `<tr><td>${esc(d.feature)}</td><td class="num">${fmt(d.psi)}</td><td>${summary(d)}</td></tr>`).join("")}</tbody></table>`
    : "";
  box.innerHTML = sig + q + drift;
}

function summary(d) {
  if (d.kind === "numeric") return `moyenne ${fmt(d.reference_summary.mean, 2)} → ${fmt(d.current_summary.mean, 2)}`;
  const [k] = Object.keys(d.current_summary);
  return k ? `« ${esc(k)} » ${pct(d.reference_summary[k] ?? 0)} → ${pct(d.current_summary[k])}` : "";
}

// ---------------------------------------------------------------------------
// Journal des décisions
// ---------------------------------------------------------------------------

function decisionLine(d) {
  if (d.status === "awaiting_human") return [`Veut ${ACTION_PROPOSED[d.action] ?? d.action}`, "attend ta validation"];
  if (d.status === "rejected_by_human") return [ACTION[d.action] ?? d.action, "rejetée par un humain"];
  if (d.status === "failed") return ["Décision échouée", "voir le détail"];
  return [ACTION[d.action] ?? d.action, d.mode === "auto" ? "décidé seul" : "validé par un humain"];
}

function renderDecisions() {
  const list = $("decisions");
  const ds = [...state.decisions].reverse();
  const auto = state.decisions.filter((d) => d.mode === "auto" && d.status === "executed").length;
  $("journal-stats").textContent = state.decisions.length ? `${state.decisions.length} décisions, dont ${auto} prises seul` : "";
  if (!ds.length) {
    list.innerHTML = '<li class="calm">L\'agent n\'a encore rien eu à décider. Il se réveille quand une fenêtre lève un signal.</li>';
    return;
  }
  list.innerHTML = "";
  renderBusy();
  for (const d of ds) {
    const [what, who] = decisionLine(d);
    const li = document.createElement("li");
    const pending = d.status === "awaiting_human";
    const cls = ["decision", pending ? "pending" : "", d.status === "rejected_by_human" ? "rejected" : "", !state.first && !state.seen.has(d.id) ? "fresh" : ""].join(" ");
    li.innerHTML = `<div class="${cls}" role="button" tabindex="0" data-id="${esc(d.id)}">
        <span class="what">${esc(what)}</span>
        <span class="why">${esc(d.signals.map((s) => SIGNAL[s] ?? s).join(", ") || "aucun signal")}</span>
        <span class="meta"><span>Fenêtre ${d.window_index}</span><span>${time(d.created_at)}</span><span class="who">${esc(who)}</span></span>
        ${pending ? `<span class="decision-actions"><button class="btn btn-approve" data-answer="approve" type="button">Approuver</button><button class="btn btn-reject" data-answer="reject" type="button">Rejeter</button></span>` : ""}
      </div>`;
    list.append(li);
    state.seen.add(d.id);
  }
}

function renderBusy() {
  const a = state.status?.agent;
  let card = $("busy-card");
  if (!a?.busy) { card?.remove(); return; }
  if (!card) {
    card = document.createElement("li");
    card.id = "busy-card";
    $("decisions").prepend(card);
  }
  const queued = a.pending_window != null ? ` La fenêtre ${a.pending_window} attend son tour.` : "";
  card.innerHTML = `<div class="decision working"><span class="what">Enquête en cours</span>
    <span class="why">L'agent analyse la fenêtre ${a.busy_window ?? "?"} : outils, diagnostics, vérification, juge.${queued}</span></div>`;
}

function renderAlerts() {
  const ul = $("alerts");
  const rank = { critical: 0, warning: 1, info: 2 };
  const al = [...state.alerts].sort((a, b) => (rank[a.severity] ?? 3) - (rank[b.severity] ?? 3));
  ul.innerHTML = al.length
    ? al.map((a) => {
      const f = a.payload?.labels?.feature;
      return `<li><span>${esc(a.alertname)}${f ? ` <span class="sev">(${esc(f)})</span>` : ""}</span><span class="sev ${esc(a.severity)}">${esc(a.severity)}</span></li>`;
    }).join("")
    : '<li class="calm">Aucune alerte active (ou Alertmanager n\'est pas lancé).</li>';
}

async function answer(id, decision, comment = "") {
  try {
    await api(`/decisions/${encodeURIComponent(id)}/${decision}`, { method: "POST", body: JSON.stringify({ by: "console", comment }) });
    toast(decision === "approve" ? "Action approuvée : l'agent l'exécute." : "Action rejetée : rien n'est exécuté.");
    await refresh();
  } catch (err) { toast(`Impossible d'enregistrer ta réponse : ${err.message}`); }
}

$("decisions").addEventListener("click", (e) => {
  const btn = e.target.closest("[data-answer]");
  const card = e.target.closest("[data-id]");
  if (!card) return;
  if (btn) { e.stopPropagation(); answer(card.dataset.id, btn.dataset.answer); return; }
  openDecision(card.dataset.id);
});
$("decisions").addEventListener("keydown", (e) => {
  const card = e.target.closest("[data-id]");
  if (card && (e.key === "Enter" || e.key === " ") && e.target === card) { e.preventDefault(); openDecision(card.dataset.id); }
});

// ---------------------------------------------------------------------------
// Enregistreur de décision : le raisonnement, étape par étape
// ---------------------------------------------------------------------------

async function openDecision(id, silent = false) {
  let run;
  try { run = await api(`/decisions/${encodeURIComponent(id)}`); } catch (err) { if (!silent) toast(err.message); return; }
  state.openId = id;
  const rec = $("recorder");
  const scroll = rec.querySelector(".recorder-panel").scrollTop;
  $("recorder-body").innerHTML = recorderHTML(run);
  rec.hidden = false;
  if (silent) rec.querySelector(".recorder-panel").scrollTop = scroll;
  else rec.querySelector(".close").focus();
  if (run.outputs?.incident) {
    try {
      const md = await api(`/decisions/${encodeURIComponent(id)}/report`);
      $("recorder-body").insertAdjacentHTML("beforeend", `<section class="report">${markdown(md)}</section>`);
    } catch { /* pas de rapport */ }
  }
  bindRating(id);
  const form = $("recorder-body").querySelector(".human-form");
  if (form) {
    form.addEventListener("click", (e) => {
      const b = e.target.closest("[data-answer]");
      if (b) answer(id, b.dataset.answer, form.querySelector("textarea").value);
    });
  }
}

function closeRecorder() { $("recorder").hidden = true; state.openId = null; }
$("recorder").addEventListener("click", (e) => { if (e.target.closest("[data-close]")) closeRecorder(); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("recorder").hidden) closeRecorder(); });

function recorderHTML(run) {
  const d = run.diagnosis, p = run.policy;
  const statusStamp = {
    executed: p?.mode === "auto" ? ["auto", "Décidé et exécuté seul"] : ["human", `Exécuté après validation (${esc(run.human?.by ?? "humain")})`],
    awaiting_human: ["human", "En attente de ta validation"], rejected_by_human: ["rejected", "Rejeté par un humain"],
    failed: ["failed", "Échec"], blocked: ["failed", "Bloqué"],
  }[run.status] ?? ["failed", run.status];
  const action = p ? (ACTION[p.final_action] ?? p.final_action) : "Pas de décision";
  const head = `
    <p class="rec-kicker" id="rec-title">Décision ${esc(run.id)}, fenêtre ${run.window_index}, ${time(run.created_at)}</p>
    <h2 class="rec-verdict">${esc(action)}</h2>
    <p class="rec-cause">${d ? esc(CAUSE[d.root_cause] ?? d.root_cause) : esc(run.error ?? "")}</p>
    <dl class="rec-figures">
      <div><dt>Statut</dt><dd><span class="stamp ${statusStamp[0]}">${statusStamp[1]}</span></dd></div>
      ${d ? `<div><dt>Confiance</dt><dd>${fmt(d.confidence, 2)}</dd></div><div><dt>Accord des diagnostics</dt><dd>${Math.round(run.agreement * 100)} %</dd></div>` : ""}
      <div><dt>Durée</dt><dd>${fmt(run.trace.reduce((s, t) => s + t.ms, 0) / 1000, 1)} s</dd></div>
      ${(() => {
        const u = run.trace.reduce((acc, t) => { const x = t.detail?.usage; if (x) { acc.calls += x.calls; acc.tok += (x.prompt_tokens || 0) + (x.completion_tokens || 0); } return acc; }, { calls: 0, tok: 0 });
        return u.calls ? `<div><dt>Appels LLM</dt><dd>${u.calls}</dd></div><div><dt>Jetons</dt><dd>${int(u.tok)}</dd></div>` : "";
      })()}
    </dl>
    <p class="hint">Décideur ${esc(deciderName(run.decider))}${run.judge ? `, juge ${esc(run.judge)}` : ", sans juge"}${run.trace_url ? ` · <a href="${esc(run.trace_url)}" target="_blank" rel="noopener">trace Langfuse</a>` : ""}</p>
    ${d ? `<p class="rec-why"><b>${esc(d.summary)}</b> ${esc(d.rationale)}</p>` : ""}`;

  const maxMs = Math.max(1, ...run.trace.map((t) => t.ms));
  const lastOf = (node) => run.trace.map((t) => t.node).lastIndexOf(node);
  const stages = run.trace.map((t, i) => {
    const [name, who] = STAGE[t.node] ?? [t.node, ""];
    const cls = ["stage", DETERMINISTIC.has(t.node) ? "det" : "", t.node === "await_human" ? "human" : "", t.node === "verify" && t.detail?.problems?.length ? "fail" : ""].join(" ");
    let body = "";
    if (t.node === "prepare" && t.detail?.retrieval?.length) {
      body = `<ul class="rules">${t.detail.retrieval.map((h) => `<li><span class="ok-mark">↺</span><span>${esc(h.id)} · ${esc(h.title)}<br><span class="r-detail">cause ${esc(h.root_cause)}, action ${esc(h.action)} · similarité texte ${fmt(h.text_sim, 2)}, chiffres ${fmt(h.sig_sim, 2)}</span></span></li>`).join("")}</ul>`;
    }
    if (t.node === "investigate") {
      const tools = t.detail?.tools ?? [];
      body = (t.detail?.transport === "mcp" ? `<p class="r-detail">via MCP · ${t.detail.tools_available} outils découverts</p>` : "") + (tools.length ? `<div class="chips">${tools.map((c) => `<span class="chip">${esc(c.tool)}(${esc(Object.values(c.args).join(", "))})</span>`).join("")}</div>` : "")
        + (t.detail?.notes ? `<p class="notes">${esc(t.detail.notes)}</p>` : "");
    }
    if (t.node === "diagnose" && t.detail?.votes) {
      const votes = Object.entries(t.detail.votes).flatMap(([a, n]) => Array(n).fill(a));
      const major = Object.entries(t.detail.votes).sort((a, b) => b[1] - a[1])[0][0];
      body = `<div class="votes">${votes.map((a) => `<span class="vote ${a === major ? "" : "minority"}">${esc(ACTION_PROPOSED[a] ?? a)}</span>`).join("")}</div>`;
    }
    if (t.node === "verify" && i === lastOf("verify") && run.verification) {
      body = `<table class="mini"><thead><tr><th></th><th>Fait cité</th><th class="num">Valeur</th><th>Ce que ça montre</th></tr></thead><tbody>${(d?.evidence ?? []).map((e) => {
        const c = run.verification.checks.find((x) => x.fact === e.fact);
        return `<tr><td>${c?.ok ? '<span class="ok-mark">✓</span>' : '<span class="ko-mark">✗</span>'}</td><td><code>${esc(e.fact)}</code></td><td class="num">${fmt(e.value, 4)}</td><td>${esc(e.claim)}</td></tr>`;
      }).join("")}</tbody></table>`;
    }
    if (t.node === "verify" && t.detail?.problems?.length) {
      body += `<p class="notes">${esc(t.detail.problems.join(" ; "))}</p>`;
    }
    if (t.node === "judge") {
      const v = run.verdicts[run.trace.slice(0, i + 1).filter((x) => x.node === "judge").length - 1];
      if (v) {
        const label = { approve: "Approuve", revise: "Demande une révision", reject: "Rejette" }[v.verdict] ?? v.verdict;
        const names = { grounding: "ancrage dans les faits", causality: "causalité", proportionality: "proportionnalité", safety: "sécurité" };
        body = `<div class="scores">${Object.entries(v.scores).map(([k, s]) => `<span>${esc(names[k] ?? k)}</span><span class="track"><i style="width:${(s / 5) * 100}%"></i></span><span>${s}/5</span>`).join("")}</div><p><b>${label}.</b> ${esc(v.critique)}</p>`;
      }
    }
    if (t.node === "policy" && p) {
      body = `<ul class="rules">${p.rules.map((r) => `<li><span class="${r.passed ? "ok-mark" : "ko-mark"}">${r.passed ? "✓" : "✗"}</span><span>${esc(r.rule)}<br><span class="r-detail">${esc(r.detail)}</span></span></li>`).join("")}</ul>`
        + (p.proposed_action !== p.final_action ? `<p class="substitution">Les règles ont remplacé « ${esc(ACTION_PROPOSED[p.proposed_action])} » par « ${esc(ACTION_PROPOSED[p.final_action])} » : ${esc(p.reasons.join(" ; "))}</p>` : "")
        + (p.mode === "human" ? `<p class="substitution">Validation humaine requise : ${esc(p.reasons.join(" ; "))}</p>` : "");
    }
    if (t.node === "act") {
      body = Object.keys(run.outputs).length ? `<ul class="files">${Object.entries(run.outputs).map(([k, v]) => `<li>${esc(k)} <code>${esc(v)}</code></li>`).join("")}</ul>` : "";
    }
    return `<li class="${cls}">
      <div class="stage-head"><span class="stage-name">${esc(name)}<small>${esc(who)}</small></span><span class="stage-time">${t.ms >= 1000 ? `${fmt(t.ms / 1000, 1)} s` : `${Math.round(t.ms)} ms`}</span></div>
      <div class="stage-bar"><i style="width:${Math.max(1.5, (100 * t.ms) / maxMs)}%"></i></div>
      <p class="stage-sum">${esc(t.summary)}${t.detail?.usage?.calls ? ` <span class="usage">${t.detail.usage.calls} appel${t.detail.usage.calls > 1 ? "s" : ""}, ${int((t.detail.usage.prompt_tokens || 0) + (t.detail.usage.completion_tokens || 0))} jetons</span>` : ""}</p>
      ${body ? `<div class="stage-body">${body}</div>` : ""}
    </li>`;
  });
  if (run.status === "awaiting_human") {
    stages.push(`<li class="stage human"><div class="stage-head"><span class="stage-name">Validation humaine<small>toi</small></span></div>
      <div class="human-form"><p>L'agent propose : <b>${esc(ACTION_PROPOSED[p.final_action])}</b>. ${esc(p.reasons.join(" ; "))}</p>
      <label for="human-comment" class="hint">Commentaire (facultatif)</label><textarea id="human-comment"></textarea>
      <div class="row"><button class="btn btn-approve" data-answer="approve" type="button">Approuver</button><button class="btn btn-reject" data-answer="reject" type="button">Rejeter</button></div></div></li>`);
  }
  return head + (d && run.status !== "awaiting_human" ? ratingHTML(run) : "") + `<ol class="rail">${stages.join("")}</ol>`;
}

// ---------------------------------------------------------------------------
// Avis sur une décision (gardés pour un futur fine-tuning)
// ---------------------------------------------------------------------------

const THUMB = (down) => `<svg viewBox="0 0 20 20" width="15" height="15" aria-hidden="true"${down ? ' style="transform:rotate(180deg)"' : ""}><path d="M6 9v8H3V9zm2 8h6.6a2 2 0 0 0 2-1.6l1-5A2 2 0 0 0 15.6 8H12V4.8A1.8 1.8 0 0 0 10.2 3L8 9z" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/></svg>`;

function ratingHTML(run) {
  const current = state.ratings[`decision:${run.id}`];
  const opt = (obj, sel) => Object.entries(obj).map(([k, v]) => `<option value="${k}"${k === sel ? " selected" : ""}>${esc(v)}</option>`).join("");
  return `<section class="rate" data-rate="${esc(run.id)}">
    <p class="rate-q">Cette décision te semble juste ?</p>
    <div class="rate-buttons">
      <button type="button" class="rate-btn" data-rating="up" aria-pressed="${current === "up"}">${THUMB(false)}Bonne décision</button>
      <button type="button" class="rate-btn" data-rating="down" aria-pressed="${current === "down"}">${THUMB(true)}À corriger</button>
    </div>
    <div class="rate-fix" hidden>
      <label>Bonne cause<select name="cause"><option value="">(inchangée)</option>${opt(CAUSE, "")}</select></label>
      <label>Bonne action<select name="action"><option value="">(inchangée)</option>${opt(ACTION_PROPOSED, "")}</select></label>
      <label class="wide">Pourquoi (facultatif)<textarea name="comment" rows="2"></textarea></label>
      <button type="button" class="btn" data-rate-send>Enregistrer la correction</button>
    </div>
    <p class="rate-done hint" ${current ? "" : "hidden"}>${current ? "Avis déjà enregistré. Tu peux le changer." : ""}</p>
  </section>`;
}

async function sendRating(body) {
  const res = await api("/ratings", { method: "POST", body: JSON.stringify(body) });
  state.ratings[`${body.kind}:${body.ref}`] = body.rating;
  return res;
}

function bindRating(id) {
  const box = $("recorder-body").querySelector("[data-rate]");
  if (!box) return;
  const fix = box.querySelector(".rate-fix");
  const done = box.querySelector(".rate-done");
  const mark = (rating) => box.querySelectorAll("[data-rating]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.rating === rating)));
  const say = (res) => {
    done.hidden = false;
    done.textContent = res.training_example
      ? `Avis enregistré : il devient un exemple d'entraînement (${res.training_examples} au total).`
      : "Avis enregistré.";
  };
  box.addEventListener("click", async (e) => {
    const b = e.target.closest("[data-rating]");
    if (b?.dataset.rating === "down") { mark("down"); fix.hidden = false; fix.querySelector("select").focus(); return; }
    try {
      if (b?.dataset.rating === "up") {
        mark("up"); fix.hidden = true;
        say(await sendRating({ kind: "decision", ref: id, rating: "up" }));
      } else if (e.target.closest("[data-rate-send]")) {
        const v = (n) => fix.querySelector(`[name="${n}"]`).value;
        say(await sendRating({ kind: "decision", ref: id, rating: "down", correct_cause: v("cause") || null, correct_action: v("action") || null, comment: v("comment") }));
        fix.hidden = true;
      }
    } catch (err) { toast(`Avis non enregistré : ${err.message}`); }
  });
}

// Markdown minimal (rapports d'incident générés par l'agent)
function markdown(md) {
  // le code passe en premier et n'est plus touché : sinon request_retraining devient de l'italique
  const inline = (s) => esc(s).split(/(`[^`]+`)/).map((part) => (part.startsWith("`") && part.endsWith("`") && part.length > 1
    ? `<code>${part.slice(1, -1)}</code>`
    : part.replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>").replace(/(^|\s)_([^_]+)_(?=\s|$|[.,:;])/g, "$1<i>$2</i>"))).join("");
  const out = []; const lines = md.split("\n"); let i = 0;
  while (i < lines.length) {
    const l = lines[i];
    if (l.startsWith("|")) {
      const rows = [];
      while (i < lines.length && lines[i].startsWith("|")) { if (!/^\|[-| ]+\|$/.test(lines[i])) rows.push(lines[i]); i++; }
      const cells = (r) => r.slice(1, -1).split("|").map((c) => c.trim());
      out.push(`<table><thead><tr>${cells(rows[0]).map((c) => `<th>${inline(c)}</th>`).join("")}</tr></thead><tbody>${rows.slice(1).map((r) => `<tr>${cells(r).map((c) => `<td>${inline(c)}</td>`).join("")}</tr>`).join("")}</tbody></table>`);
      continue;
    }
    if (l.startsWith("- ")) {
      const items = [];
      while (i < lines.length && lines[i].startsWith("- ")) { items.push(`<li>${inline(lines[i].slice(2))}</li>`); i++; }
      out.push(`<ul>${items.join("")}</ul>`); continue;
    }
    const h = l.match(/^(#{1,3}) (.*)$/);
    if (h) out.push(`<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`);
    else if (l.trim()) out.push(`<p>${inline(l)}</p>`);
    i++;
  }
  return out.join("");
}

// ---------------------------------------------------------------------------
// Commandes
// ---------------------------------------------------------------------------

$("btn-decide").addEventListener("click", async () => {
  const b = $("btn-decide");
  b.disabled = true; b.textContent = "L'agent enquête…";
  try {
    const run = await api("/decide", { method: "POST" });
    await refresh();
    openDecision(run.id);
  } catch (err) { toast(`L'agent n'a pas pu décider : ${err.message}`); }
  finally { b.textContent = "Lancer l'agent"; b.disabled = false; }
});

$("btn-sim").addEventListener("click", async () => {
  const scenario = $("sim-scenario").value;
  try {
    await api("/simulate", { method: "POST", body: JSON.stringify({ scenario }) });
    toast("Simulation lancée : 400 requêtes normales, puis le scénario choisi.");
    refresh();
  } catch (err) { toast(`Simulation impossible : ${err.message}`); }
});

let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { renderHeatmap(); renderPerf(); }, 150);
});

// ---------------------------------------------------------------------------
// « Demande à ton monitoring »
// ---------------------------------------------------------------------------

const chat = { history: [], busy: false };
const SUGGESTIONS = [
  "Est-ce que le modèle va bien en ce moment ?",
  "Quelle feature dérive le plus, et depuis quand ?",
  "Pourquoi l'agent a pris sa dernière décision ?",
  "Les données reçues ont-elles des problèmes de qualité ?",
];

function openChat() {
  $("chat").hidden = false;
  $("btn-ask").setAttribute("aria-expanded", "true");
  renderChat();
  $("chat-input").focus();
}
function closeChat() { $("chat").hidden = true; $("btn-ask").setAttribute("aria-expanded", "false"); $("btn-ask").focus(); }

function answerHTML(text) {
  // paragraphes et listes simples ; tout est échappé
  return text.split(/\n{2,}/).map((block) => {
    const lines = block.split("\n");
    if (lines.every((l) => /^\s*[-*] /.test(l))) return `<ul>${lines.map((l) => `<li>${esc(l.replace(/^\s*[-*] /, ""))}</li>`).join("")}</ul>`;
    return `<p>${lines.map(esc).join("<br>").replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")}</p>`;
  }).join("");
}

function renderChat() {
  const ol = $("chat-thread");
  if (!chat.history.length && !chat.busy) {
    ol.innerHTML = `<li class="chat-empty"><p class="hint">Quelques idées pour commencer :</p>
      <div class="suggest">${SUGGESTIONS.map((q) => `<button type="button" class="suggest-btn" data-q="${esc(q)}">${esc(q)}</button>`).join("")}</div></li>`;
    return;
  }
  ol.innerHTML = chat.history.map((m) => {
    const rated = state.ratings[`answer:${m.id}`];
    const tools = m.tools?.length ? `<div class="chips">${m.tools.map((t) => `<span class="chip">${esc(t.tool)}(${esc(Object.values(t.args || {}).join(", "))})</span>`).join("")}</div>` : "";
    return `<li class="turn">
      <p class="q">${esc(m.question)}</p>
      <div class="a">${answerHTML(m.answer)}${tools}
        <div class="a-foot">
          <span class="hint">${esc(m.model || "")} · ${fmt((m.ms || 0) / 1000, 1)} s${m.tools?.length ? ` · ${m.tools.length} outil${m.tools.length > 1 ? "s" : ""} consulté${m.tools.length > 1 ? "s" : ""}` : ""}</span>
          ${m.id && m.model !== "aucun" ? `<span class="a-rate" data-answer-id="${esc(m.id)}">
            <button type="button" class="icon-btn" data-arating="up" aria-pressed="${rated === "up"}" aria-label="Réponse utile" title="Réponse utile">${THUMB(false)}</button>
            <button type="button" class="icon-btn" data-arating="down" aria-pressed="${rated === "down"}" aria-label="Réponse pas utile" title="Réponse pas utile">${THUMB(true)}</button>
          </span>` : ""}
        </div>
      </div>
    </li>`;
  }).join("") + (chat.busy ? `<li class="turn"><p class="q">${esc(chat.pending)}</p><div class="a thinking">L'assistant consulte les outils du monitoring</div></li>` : "");
  ol.scrollTop = ol.scrollHeight;
}

async function sendQuestion(question) {
  if (!question.trim() || chat.busy) return;
  chat.busy = true; chat.pending = question;
  $("chat-input").value = ""; $("chat-send").disabled = true;
  renderChat();
  try {
    const rec = await api("/ask", { method: "POST", body: JSON.stringify({ question, history: chat.history.slice(-4).map(({ question: q, answer: a }) => ({ question: q, answer: a })) }) });
    chat.history.push(rec);
  } catch (err) {
    chat.history.push({ question, answer: `Pas de réponse : ${err.message}`, tools: [], model: "aucun", ms: 0 });
  } finally {
    chat.busy = false; $("chat-send").disabled = false;
    renderChat(); $("chat-input").focus();
  }
}

$("btn-ask").addEventListener("click", openChat);
$("chat").addEventListener("click", async (e) => {
  if (e.target.closest("[data-chat-close]")) { closeChat(); return; }
  const sug = e.target.closest("[data-q]");
  if (sug) { sendQuestion(sug.dataset.q); return; }
  const b = e.target.closest("[data-arating]");
  if (b) {
    const id = b.closest("[data-answer-id]").dataset.answerId;
    try {
      await sendRating({ kind: "answer", ref: id, rating: b.dataset.arating });
      renderChat();
      toast("Merci, ton avis est gardé pour améliorer l'assistant.");
    } catch (err) { toast(`Avis non enregistré : ${err.message}`); }
  }
});
$("chat-form").addEventListener("submit", (e) => { e.preventDefault(); sendQuestion($("chat-input").value); });
$("chat-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendQuestion($("chat-input").value); }
});
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("chat").hidden) closeChat(); });

refresh().then(() => {
  // lien d'une notification : /console/#D0006-9145fe ouvre directement la décision
  const id = decodeURIComponent(location.hash.slice(1));
  if (id.startsWith("D")) openDecision(id);
});
setInterval(refresh, POLL_MS);
