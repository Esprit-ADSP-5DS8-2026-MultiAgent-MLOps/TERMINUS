"""
Interface de la plateforme Agentic MLOps.

    streamlit run interface/streamlit_app.py

Lancer un run, suivre le pipeline sur son plan de ligne, répondre aux
validations humaines, consulter les fichiers produits et l'historique.
Les runs sont relus depuis la base (SQLite ou PostgreSQL) à chaque affichage :
un run en pause survit à la fermeture de la page.
"""

from __future__ import annotations

import getpass
import html
import json
import os
import sys
from collections import Counter
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orchestrator.checkpoint import describe_backend  # noqa: E402
from orchestrator.graph import Run, list_runs  # noqa: E402
from orchestrator.llm import default_mode  # noqa: E402

PROJECTS = ROOT / "eval" / "projects"
RUNS = ROOT / "runs"

# --- vocabulaire et couleurs ---------------------------------------------------

COLORS = {
    "line": "#2F4B7C",
    "rail": "#D5DBE4",
    "ok": "#1E8E5A",
    "wait": "#C98309",
    "fail": "#C5402F",
    "idle": "#C2CAD5",
    "ink": "#1B2533",
    "muted": "#5E6B7D",
}

STATIONS = [
    ("validator", "Validateur", "Contrat C1–C12"),
    ("agent1", "Analyse", "Agent 1"),
    ("agent2", "MLflow", "Agent 2"),
    ("agent3", "Déploiement", "Agent 3"),
    ("agent4", "Monitoring", "Agent 4"),
]

RESULT_LOOK = {  # statut d'AgentResult -> (couleur, libellé)
    "OK": ("ok", "terminé"),
    "WARNING": ("wait", "terminé avec réserves"),
    "NEEDS_APPROVAL": ("wait", "attend une décision"),
    "FAILED": ("fail", "échec"),
}

FINAL_LOOK = {
    "DONE": ("ok", "Modèle déployé et surveillé",
             "Toutes les étapes sont passées et les décisions humaines ont été prises."),
    "BLOCKED": ("fail", "Projet refusé par le validateur",
                "Le projet ne respecte pas le contrat. Corrigez les points ci-dessous, puis relancez."),
    "FAILED": ("fail", "Une étape a échoué",
               "Le pipeline s'est arrêté proprement. Le journal indique la cause."),
    "REJECTED": ("fail", "Refusé par un humain", "Rien n'a été déployé."),
    "MANIFEST_PROPOSED": ("ok", "Manifeste validé",
                          "Ajoutez le mlops.yaml proposé au projet, puis relancez le pipeline."),
    "WAITING_HUMAN": ("wait", "En attente de votre décision",
                      "Le pipeline est en pause. Il reprendra dès que vous aurez décidé."),
    "INTERRUPTED": ("wait", "Run interrompu",
                    "Le run s'est arrêté au milieu (plantage ou fermeture). Il peut reprendre là où il en était."),
}

GATES = {
    "manifest_confirmation": "Valider le manifeste proposé",
    "champion_replacement": "Mettre ce modèle en production",
    "retraining": "Réentraîner le modèle",
    "failure_review": "Relancer l'étape qui a échoué",
}

SUPERVISORS = {"rules": "Règles (sans IA)", "gemini": "Gemini (LLM)"}
FAILURES = {None: "aucune", "transient": "passagère (réseau)", "permanent": "permanente (erreur de contenu)"}
ACTION_LABELS = {
    "agent1": "lancer l'analyse", "agent2": "lancer MLflow", "agent3": "lancer le déploiement",
    "agent4": "lancer le monitoring", "propose_manifest": "proposer un manifeste",
    "repair": "tenter une réparation", "validate": "revalider le projet réparé",
    "gate_manifest": "demander une validation", "gate_champion": "demander une validation",
    "gate_retraining": "demander une validation", "gate_failure": "demander à un humain",
    "retry": "réessayer", "end_done": "terminer", "end_failed": "arrêter", "end_blocked": "arrêter",
    "end_rejected": "arrêter", "end_manifest_proposed": "terminer",
}

# --- style -------------------------------------------------------------------

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Overpass:wght@400;600;800&display=swap');
h1, h2, h3, h4 {{ font-family: 'Overpass', sans-serif !important; letter-spacing: -0.01em; }}
h1 {{ font-weight: 800 !important; }}
.block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
.subline {{ color: {COLORS['muted']}; font-size: 0.95rem; margin: -0.6rem 0 1.4rem; }}
.subline b {{ color: {COLORS['ink']}; font-weight: 600; }}
.linemap svg {{ width: 100%; height: auto; display: block; }}
.linemap .pulse {{ animation: pulse 1.6s ease-in-out infinite; transform-box: fill-box; transform-origin: center; }}
@keyframes pulse {{ 0%,100% {{ opacity: .9; transform: scale(1); }} 50% {{ opacity: .2; transform: scale(1.35); }} }}
@media (prefers-reduced-motion: reduce) {{ .linemap .pulse {{ animation: none; }} }}
.banner {{ border-left: 6px solid var(--c); padding: 0.9rem 1.2rem; margin: 0.4rem 0 1.4rem;
           background: color-mix(in srgb, var(--c) 8%, transparent); border-radius: 0 6px 6px 0; }}
.banner .t {{ font-family: 'Overpass', sans-serif; font-weight: 800; font-size: 1.35rem; color: var(--c); }}
.banner .d {{ color: {COLORS['ink']}; margin-top: 0.15rem; }}
.log {{ list-style: none; padding: 0; margin: 0 0 0 6px; border-left: 2px solid {COLORS['rail']}; }}
.log li {{ position: relative; margin: 0 !important; padding: 0 0 1rem 1.2rem !important; list-style: none; }}
.log li::before {{ content: ''; position: absolute; left: -9px; top: 0.25rem; width: 16px; height: 16px;
                   border-radius: 50%; background: var(--c); border: 3px solid white; box-sizing: border-box; }}
.log li.h::before {{ border-radius: 2px; transform: rotate(45deg); }}
.log .n {{ font-weight: 600; color: {COLORS['ink']}; }}
.log .m {{ color: {COLORS['muted']}; font-size: 0.9rem; }}
.log .r {{ font-size: 0.9rem; margin-top: 0.15rem; color: {COLORS['ink']}; }}
</style>
"""


def esc(x) -> str:
    return html.escape(str(x))


# --- lecture de l'état -----------------------------------------------------------

GATE_STATION = {"champion_replacement": "agent2", "manifest_confirmation": "agent1"}


def station_states(state: dict, pending_gate: str | None) -> dict[str, tuple[str, str]]:
    """Pour chaque station : (couleur, texte sous la station)."""
    results = dict(state.get("results", {}))
    attempts = state.get("attempts", {})
    if "propose_manifest" in results:
        results["agent1"] = results["propose_manifest"]
    out = {}
    for key, _, _ in STATIONS:
        r = results.get(key)
        if r is None:
            out[key] = ("idle", "pas lancé")
            continue
        color, label = RESULT_LOOK.get(r["status"], ("idle", r["status"]))
        if r["status"] == "NEEDS_APPROVAL" and GATE_STATION.get(pending_gate) != key:
            color, label = "ok", "terminé"  # sa décision a déjà été prise
        tries = attempts.get(key, 0)
        out[key] = (color, f"{label} · {r['duration_s']} s" + (f" · {tries + 1} essais" if tries else ""))
    return out


def gate_state(state: dict, run: Run, gate: str) -> str | None:
    """'wait', 'ok', 'fail', ou None si la gate n'a pas été atteinte."""
    q = run.question
    if q and q["gate"] == gate:
        return "wait"
    decisions = [d for d in state.get("decisions", []) if d["gate"] == gate]
    if not decisions:
        return None
    return "ok" if decisions[-1]["decision"] == "approve" else "fail"


# --- plan de ligne (SVG) ---------------------------------------------------------

def linemap(run: Run | None, state: dict) -> str:
    q = run.question if run else None
    states = station_states(state, q["gate"] if q else None)
    xs = [70, 250, 430, 680, 880]
    y = 82
    manifest_path = "propose_manifest" in state.get("results", {})
    names = [n for _, n, _ in STATIONS]
    if manifest_path:
        names[1] = "Proposition"

    parts = ['<svg viewBox="0 0 980 185" role="img" aria-label="Plan du pipeline">']

    # rails : bleu jusqu'où le run est allé, gris au-delà
    for i in range(len(xs) - 1):
        a, b = STATIONS[i][0], STATIONS[i + 1][0]
        reached = states[a][0] != "idle" and states[b][0] != "idle"
        color = COLORS["line"] if reached else COLORS["rail"]
        parts.append(f'<line x1="{xs[i]}" y1="{y}" x2="{xs[i+1]}" y2="{y}" '
                     f'stroke="{color}" stroke-width="7" stroke-linecap="round"/>')

    # boucle de réentraînement : arc Agent 4 -> Agent 2
    retrain = gate_state(state, run, "retraining") if run else None
    if retrain:
        c = COLORS[retrain]
        parts.append(f'<path d="M {xs[4]} {y-18} C {xs[4]} {y-70}, {xs[2]} {y-70}, {xs[2]} {y-18}" '
                     f'fill="none" stroke="{c}" stroke-width="3" stroke-dasharray="7 6"/>')
        mid = (xs[2] + xs[4]) / 2
        parts.append(f'<polygon points="{mid-11},{y-57} {mid+5},{y-65} {mid+5},{y-49}" fill="{c}"/>')
        label = {"wait": "réentraînement demandé", "ok": "réentraînement approuvé",
                 "fail": "réentraînement refusé"}[retrain]
        parts.append(f'<text x="{mid}" y="{y-71}" text-anchor="middle" font-size="13" fill="{c}" '
                     f'font-family="Overpass, sans-serif" font-weight="600">{label}</text>')

    # décisions humaines : losanges sur la ligne
    gates = [("manifest_confirmation", (xs[1] + xs[2]) / 2)] if manifest_path \
        else [("champion_replacement", (xs[2] + xs[3]) / 2)]
    for gate, gx in gates:
        g = gate_state(state, run, gate) if run else None
        fill = COLORS[g] if g else "white"
        stroke = COLORS[g] if g else COLORS["idle"]
        parts.append(f'<g transform="rotate(45 {gx} {y})">')
        if g == "wait":
            parts.append(f'<rect class="pulse" x="{gx-16}" y="{y-16}" width="32" height="32" rx="4" '
                         f'fill="{fill}" opacity=".5"/>')
        parts.append(f'<rect x="{gx-12}" y="{y-12}" width="24" height="24" rx="3" '
                     f'fill="{fill}" stroke="{stroke}" stroke-width="3"/></g>')
        mark = {"wait": "?", "ok": "✓", "fail": "✕"}.get(g, "")
        parts.append(f'<text x="{gx}" y="{y+5}" text-anchor="middle" font-size="14" fill="white" '
                     f'font-weight="800">{mark}</text>')
        parts.append(f'<text x="{gx}" y="{y+42}" text-anchor="middle" font-size="12.5" '
                     f'fill="{COLORS["muted"]}">décision humaine</text>')

    # stations
    for (key, _, sub), name, x in zip(STATIONS, names, xs):
        color, detail = states[key]
        c = COLORS[color]
        if color == "wait":
            parts.append(f'<circle class="pulse" cx="{x}" cy="{y}" r="22" fill="{c}" opacity=".45"/>')
        if color == "idle":
            parts.append(f'<circle cx="{x}" cy="{y}" r="15" fill="white" stroke="{c}" stroke-width="4"/>')
        else:
            parts.append(f'<circle cx="{x}" cy="{y}" r="16" fill="white" stroke="{c}" stroke-width="5"/>')
            parts.append(f'<circle cx="{x}" cy="{y}" r="8" fill="{c}"/>')
        ink = COLORS["ink"] if color != "idle" else COLORS["muted"]
        parts.append(f'<text x="{x}" y="{y+44}" text-anchor="middle" font-size="17" font-weight="800" '
                     f'fill="{ink}" font-family="Overpass, sans-serif">{esc(name)}</text>')
        parts.append(f'<text x="{x}" y="{y+62}" text-anchor="middle" font-size="12.5" '
                     f'fill="{COLORS["muted"]}">{esc(sub)}</text>')
        detail_color = c if color != "idle" else COLORS["muted"]
        parts.append(f'<text x="{x}" y="{y+80}" text-anchor="middle" font-size="12.5" '
                     f'fill="{detail_color}">{esc(detail)}</text>')

    parts.append("</svg>")
    return '<div class="linemap">' + "".join(parts) + "</div>"


def banner(kind: str, title: str, detail: str) -> str:
    return (f'<div class="banner" style="--c:{COLORS[kind]}"><div class="t">{esc(title)}</div>'
            + (f'<div class="d">{esc(detail)}</div>' if detail else "") + "</div>")


# --- détail d'une décision humaine ------------------------------------------------

def load_json(path: str | None) -> dict | None:
    if path and Path(path).exists():
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
    return None


def quality_gauge(gate: dict) -> str:
    """Jauge : score du candidat face au seuil et au modèle naïf."""
    v, t, b = gate["value"], gate["min_threshold"], gate["baseline_value"]
    hi = 1.0
    if gate["primary_metric"] in ("rmse", "mae"):  # pas d'échelle 0-1 naturelle
        hi = max(v, t, b) * 1.2 or 1.0

    def pos(x: float) -> float:
        return 40 + max(0.0, min(x / hi, 1.0)) * 840

    c = COLORS["ok"] if gate["passed"] else COLORS["fail"]
    return "".join([
        '<svg viewBox="0 0 920 96" style="width:100%;height:auto;margin:.4rem 0 .2rem">',
        f'<rect x="40" y="40" width="840" height="12" rx="6" fill="{COLORS["rail"]}"/>',
        f'<rect x="40" y="40" width="{pos(v)-40:.1f}" height="12" rx="6" fill="{c}"/>',
        f'<line x1="{pos(t):.1f}" y1="30" x2="{pos(t):.1f}" y2="62" stroke="{COLORS["ink"]}" stroke-width="2.5"/>',
        f'<text x="{pos(t):.1f}" y="22" text-anchor="middle" font-size="13" fill="{COLORS["ink"]}">'
        f'seuil {t:g}</text>',
        f'<line x1="{pos(b):.1f}" y1="34" x2="{pos(b):.1f}" y2="58" stroke="{COLORS["muted"]}" '
        f'stroke-width="2" stroke-dasharray="3 3"/>',
        f'<text x="{pos(b):.1f}" y="78" text-anchor="middle" font-size="13" fill="{COLORS["muted"]}">'
        f'modèle naïf {b:g}</text>',
        f'<circle cx="{pos(v):.1f}" cy="46" r="9" fill="white" stroke="{c}" stroke-width="4"/>',
        f'<text x="{pos(v):.1f}" y="92" text-anchor="middle" font-size="14" font-weight="800" fill="{c}" '
        f'font-family="Overpass, sans-serif">{esc(gate["primary_metric"])} {v:.3f}</text>',
        "</svg>",
    ])


def gate_evidence(q: dict) -> None:
    if q["gate"] == "failure_review":
        step = STEP_NAMES.get(q.get("failed_step", ""), q.get("failed_step", ""))
        reasons = "".join(f'<div class="d">{esc(r)}</div>' for r in q.get("reasons", []))
        st.markdown(f'<div class="banner" style="--c:{COLORS["fail"]};padding:.6rem 1rem">'
                    f'<div style="font-weight:600">{esc(step)} a échoué</div>{reasons}</div>',
                    unsafe_allow_html=True)
        if q.get("supervisor_explanation"):
            st.caption(f"Avis du superviseur : {q['supervisor_explanation']}")
        return
    data = load_json(q.get("file_to_review"))
    if data is None:
        return
    if q["gate"] == "champion_replacement" and "quality_gate" in data:
        g = data["quality_gate"]
        st.markdown(quality_gauge(g), unsafe_allow_html=True)
        st.caption(f"Quality gate {'réussi' if g['passed'] else 'échoué'}. "
                   f"Modèle {data.get('model_uri', '?')}, alias actuel : {data.get('alias', '?')}.")
    elif q["gate"] == "retraining":
        drift = data.get("drift", [])
        if drift:
            st.markdown("**Colonnes qui ont dérivé**")
            st.dataframe(
                [{"colonne": d["feature"], "méthode": d["method"], "score": d["score"],
                  "drift": "oui" if d["drifted"] else "non"} for d in drift],
                hide_index=True, width="stretch",
            )
        if data.get("diagnosis"):
            st.caption(data["diagnosis"])
    elif q["gate"] == "manifest_confirmation":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tâche", data.get("task", "?"))
        c2.metric("Cible", data.get("target", "?"))
        c3.metric("Métrique", f"{data.get('primary_metric', '?')} ≥ {data.get('min_threshold', '?')}")
        c4.metric("Confiance", f"{data.get('confidence', 0):.0%}")
        if data.get("rationale"):
            st.caption(data["rationale"])
    with st.expander("Voir le fichier complet"):
        st.json(data, expanded=False)


def show_gate(run: Run) -> None:
    q = run.question
    st.markdown(f"### {GATES.get(q['gate'], q['gate'])} ?")
    st.write(q["question"])
    gate_evidence(q)
    comment = st.text_input("Commentaire (facultatif)", key=f"comment-{run.run_id}-{len(run.state['decisions'])}",
                            placeholder="Enregistré avec votre décision")
    c1, c2, _ = st.columns([1, 1, 3])
    decision = None
    if c1.button("Approuver", type="primary", width="stretch"):
        decision = "approve"
    if c2.button("Refuser", width="stretch"):
        decision = "reject"
    if decision:
        with st.spinner("Reprise du pipeline…"):
            run.answer({"decision": decision, "by": getpass.getuser(), "comment": comment})
        st.rerun()


# --- décisions du superviseur -------------------------------------------------------

GATE_STEPS = {
    "gate_manifest": "la validation du manifeste",
    "gate_champion": "la validation de la mise en production",
    "gate_retraining": "la décision de réentraînement",
    "gate_failure": "la décision après l'échec",
}

def supervision_panel(state: dict) -> None:
    entries = state.get("supervision", [])
    if not entries:
        return
    mode = state.get("supervisor_mode", "rules")
    st.markdown(f"#### Décisions du superviseur · {SUPERVISORS.get(mode, mode)}")
    fallbacks = sum(1 for e in entries if e.get("fallback"))
    if mode != "rules":
        agree = sum(1 for e in entries if not e.get("fallback") and e["action"] == e["rule_choice"])
        st.caption(f"{len(entries)} décisions · {agree} identiques aux règles · "
                   f"{len(entries) - agree - fallbacks} différentes · {fallbacks} reprises par les règles")
    items = []
    for e in entries:
        kind = "fail" if e.get("fallback") else ("wait" if e["action"] != e["rule_choice"] else "ok")
        after = STEP_NAMES.get(e["after"]) or GATE_STEPS.get(e["after"], e["after"])
        note = (f'<div class="m" style="color:{COLORS["fail"]}">Secours : {esc(e["fallback"])}</div>'
                if e.get("fallback") else "")
        items.append(
            f'<li style="--c:{COLORS[kind]}"><div class="n">Après {esc(after)} : '
            f'{esc(ACTION_LABELS.get(e["action"], e["action"]))}</div>'
            f'<div class="r">{esc(e["explanation"])}</div>{note}</li>'
        )
    with st.expander("Voir les décisions", expanded=mode != "rules"):
        st.markdown('<ul class="log">' + "".join(items) + "</ul>", unsafe_allow_html=True)


# --- projet refusé : ce qu'il faut corriger --------------------------------------------

SEVERITY = {"BLOCKER": ("fail", "bloquant"), "CONFIRM": ("wait", "à confirmer"),
            "WARNING": ("wait", "avertissement"), "INFO": ("idle", "info")}


def show_findings(state: dict) -> None:
    report = load_json(state.get("report_path")) or {}
    findings = report.get("findings") or []
    findings = sorted(findings, key=lambda f: list(SEVERITY).index(f.get("severity", "INFO"))
                      if f.get("severity") in SEVERITY else 9)
    if not findings:
        return
    st.markdown("#### À corriger dans le projet")
    for f in findings:
        kind, label = SEVERITY.get(f.get("severity"), ("idle", f.get("severity", "")))
        where = f.get("file") or ""
        if where and f.get("line"):
            where += f":{f['line']}"
        hint = f.get("hint")
        st.markdown(
            f'<div class="banner" style="--c:{COLORS[kind]};padding:.6rem 1rem;margin:.3rem 0">'
            f'<div style="font-weight:600;color:{COLORS["ink"]}">{esc(f.get("code", ""))} · {esc(label)}'
            + (f' <span style="color:{COLORS["muted"]};font-weight:400">· {esc(where)}</span>' if where else "")
            + f'</div><div class="d">{esc(f.get("message", ""))}</div>'
            + (f'<div class="d" style="color:{COLORS["muted"]}">Correction : {esc(hint)}</div>' if hint else "")
            + "</div>",
            unsafe_allow_html=True,
        )
    st.divider()


# --- journal et fichiers ----------------------------------------------------------

STEP_NAMES = {
    "validator": "Validateur",
    "propose_manifest": "Agent 1 · proposition du manifeste",
    "repair": "Validateur v2 · réparation",
    "agent1": "Agent 1 · analyse",
    "agent2": "Agent 2 · MLflow",
    "agent3": "Agent 3 · déploiement",
    "agent4": "Agent 4 · monitoring",
}


def journal(state: dict, status: str) -> str:
    """Étapes et décisions humaines, dans l'ordre où elles ont eu lieu."""
    items = []  # (horodatage, html)
    steps = state.get("history") or [{"node": n, **r} for n, r in state.get("results", {}).items()]
    for r in steps:
        name = r["node"]
        color, label = RESULT_LOOK.get(r["status"], ("idle", r["status"]))
        if r["status"] == "NEEDS_APPROVAL" and status != "WAITING_HUMAN":
            color, label = "ok", "terminé, décision prise"
        name = STEP_NAMES.get(name, name)
        reasons = "".join(f'<div class="r">{esc(x)}</div>' for x in r["reasons"])
        warns = "".join(f'<div class="m">{esc(w)}</div>' for w in r["warnings"]
                        if not w.startswith("stub"))
        simulated = ", simulé" if any(w.startswith("stub") for w in r["warnings"]) else ""
        items.append((r.get("started_at", ""),
                      f'<li style="--c:{COLORS[color]}"><div class="n">{esc(name)}</div>'
                      f'<div class="m">{esc(label)}, {r["duration_s"]} s{simulated}</div>{reasons}{warns}</li>'))
    for d in state.get("decisions", []):
        color = "ok" if d["decision"] == "approve" else "fail"
        verb = "approuvé" if d["decision"] == "approve" else "refusé"
        comment = f' : « {esc(d["comment"])} »' if d["comment"] else ""
        items.append((d.get("at", ""),
                      f'<li class="h" style="--c:{COLORS[color]}"><div class="n">'
                      f'{esc(GATES.get(d["gate"], d["gate"]))}</div>'
                      f'<div class="m">{verb} par {esc(d["by"])}{comment}</div></li>'))
    items.sort(key=lambda item: item[0])
    return '<ul class="log">' + "".join(h for _, h in items) + "</ul>"


def show_files(state: dict) -> None:
    files = {f"{Path(p).parent.name}/{Path(p).name}": p
             for p in state.get("artifacts", {}).values() if Path(p).exists()}
    if not files:
        st.caption("Aucun fichier produit pour l'instant.")
        return
    choice = st.selectbox("Fichier", list(files), label_visibility="collapsed")
    path = Path(files[choice])
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        st.json(json.loads(text), expanded=1)
    else:
        st.code(text)
    st.download_button("Télécharger ce fichier", text, file_name=path.name, width="stretch")


# --- historique ---------------------------------------------------------------

def outcome_bars(finals: Counter, total: int) -> str:
    """Répartition des résultats : une barre par résultat, à la couleur de son état."""
    rows = []
    for status, n in finals.most_common():
        kind, title, _ = FINAL_LOOK.get(status, ("idle", status, ""))
        width = max(4, round(100 * n / total))
        rows.append(
            f'<div style="margin:0 0 .7rem"><div style="display:flex;justify-content:space-between;'
            f'font-size:.9rem;color:{COLORS["ink"]}"><span>{esc(title)}</span><b>{n}</b></div>'
            f'<div style="height:10px;border-radius:5px;background:{COLORS["rail"]}">'
            f'<div style="width:{width}%;height:10px;border-radius:5px;background:{COLORS[kind]}"></div>'
            f'</div></div>'
        )
    return "".join(rows)


def show_history(runs: list[Run]) -> None:
    if not runs:
        st.info("Aucun run pour l'instant. Lancez un premier pipeline depuis le panneau de gauche.")
        return
    rows, finals = [], Counter()
    for r in runs:
        s, status = r.state, r.status
        finals[status] += 1
        results = s.get("results", {})
        rows.append({
            "run": r.run_id,
            "projet": Path(s.get("project_dir", "?")).name,
            "résultat": FINAL_LOOK.get(status, ("", status))[1],
            "étapes": len(results),
            "durée (s)": round(sum((x.get("duration_s") or 0) for x in results.values()), 2),
            "décisions": len(s.get("decisions", [])),
            "superviseur": SUPERVISORS.get(s.get("supervisor_mode", "rules"), "?"),
        })
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Runs", len(runs))
    c2.metric("Déployés", finals["DONE"])
    c3.metric("Projets refusés", finals["BLOCKED"])
    c4.metric("À traiter", finals["WAITING_HUMAN"] + finals["INTERRUPTED"])

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.dataframe(rows, hide_index=True, width="stretch",
                     column_config={"durée (s)": st.column_config.NumberColumn(format="%.2f")})
    with right:
        st.markdown(outcome_bars(finals, len(runs)), unsafe_allow_html=True)
    labels = {r["run"]: f'{r["run"]} · {r["projet"]} · {r["résultat"]}' for r in rows}
    c1, c2 = st.columns([3, 1])
    chosen = c1.selectbox("Ouvrir un run", list(labels), format_func=labels.get,
                          label_visibility="collapsed")
    if c2.button("Ouvrir ce run", width="stretch"):
        st.session_state.run_id = chosen
        st.rerun()


# --- page ------------------------------------------------------------------------

def project_label(p: Path) -> str:
    return f"variante {p.name}" if p.parent.name == "variants" else p.name


def list_projects() -> list[Path]:
    projects = [p for p in sorted(PROJECTS.iterdir()) if p.is_dir() and p.name != "variants"]
    variants = PROJECTS / "variants"
    if variants.exists():
        projects += sorted(p for p in variants.iterdir() if p.is_dir())
    return projects


st.set_page_config(page_title="Orchestrateur MLOps", page_icon="🚉", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
st.title("Orchestrateur MLOps")
st.markdown(
    '<div class="subline">Validateur réel, agents 1 à 4 simulés en attendant leur intégration, superviseur au choix. '
    f'Sauvegarde : <b>{esc(describe_backend(RUNS))}</b></div>',
    unsafe_allow_html=True,
)

try:
    all_runs = list_runs(str(RUNS))
except RuntimeError as exc:  # base injoignable : on le dit clairement
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.markdown("### Nouveau run")
    project = st.selectbox("Projet ML", list_projects(), format_func=project_label)
    modes = list(SUPERVISORS)
    supervisor_mode = st.radio("Qui décide de la suite ?", modes, format_func=SUPERVISORS.get,
                               index=modes.index(default_mode()) if default_mode() in modes else 0,
                               horizontal=True)
    if supervisor_mode == "gemini" and not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        st.warning("Clé GEMINI_API_KEY absente : les règles prendront le relais à chaque décision.")
    drift = st.toggle("Simuler un drift au monitoring")
    failure = st.selectbox("Simuler une panne du déploiement", list(FAILURES), format_func=FAILURES.get)
    if st.button("Lancer le pipeline", type="primary", width="stretch"):
        with st.spinner("Validation du projet… jusqu'à une minute la première fois"):
            st.session_state.run_id = Run.start(
                str(project), runs_dir=str(RUNS), supervisor=supervisor_mode,
                simulate={"drift": drift, "failure": failure},
            ).run_id
        st.rerun()

    pending = [r for r in all_runs if r.status in ("WAITING_HUMAN", "INTERRUPTED")]
    st.markdown(f"### À traiter ({len(pending)})")
    if not pending:
        st.caption("Aucun run n'attend de décision.")
    for r in pending:
        label = "décision" if r.status == "WAITING_HUMAN" else "à reprendre"
        if st.button(f"{r.run_id} · {Path(r.state['project_dir']).name} · {label}",
                     key=f"open-{r.run_id}", width="stretch"):
            st.session_state.run_id = r.run_id
            st.rerun()

tab_run, tab_history = st.tabs(["Run", f"Historique ({len(all_runs)})"])

with tab_run:
    run_id = st.session_state.get("run_id")
    if run_id is None:
        st.markdown(linemap(None, {}), unsafe_allow_html=True)
        st.info("Choisissez un projet dans le panneau de gauche, puis lancez le pipeline. "
                "Chaque étape s'allumera sur la ligne.")
    else:
        run = Run.load(run_id, runs_dir=str(RUNS))
        state, status = run.state, run.status
        mode = SUPERVISORS.get(state.get("supervisor_mode", "rules"), "")
        st.markdown(f"#### Run {run.run_id} · {Path(state['project_dir']).name} · {mode}")
        st.markdown(linemap(run, state), unsafe_allow_html=True)
        kind, title, detail = FINAL_LOOK.get(status, ("idle", status, ""))
        st.markdown(banner(kind, title, detail), unsafe_allow_html=True)

        if status == "BLOCKED":
            show_findings(state)
        if status == "WAITING_HUMAN":
            show_gate(run)
            st.divider()
        elif status == "INTERRUPTED":
            st.write(f"Reprise à l'étape : **{', '.join(run.snapshot.next)}**")
            if st.button("Reprendre le run", type="primary"):
                with st.spinner("Reprise…"):
                    run.resume()
                st.rerun()
            st.divider()

        supervision_panel(state)
        left, right = st.columns([1, 1], gap="large")
        with left:
            st.markdown("#### Journal")
            st.markdown(journal(state, status), unsafe_allow_html=True)
        with right:
            st.markdown("#### Fichiers produits")
            show_files(state)
        st.caption(f"Dossier du run : {run.run_dir}")

with tab_history:
    show_history(all_runs)
