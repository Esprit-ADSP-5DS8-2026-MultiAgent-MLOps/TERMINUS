"""
Fiches d'incident et signature numérique d'une situation.

Signature = 9 nombres dans [0, 1] calculés depuis les FAITS (jamais par un LLM) :
    0 part des features en drift        1 PSI max (plafonné à 1)
    2 baisse de perf MESURÉE            3 baisse de perf ESTIMÉE    (0 = normale, 1 = 3× l'écart toléré)
    4 pire taux de problème qualité     5 labels disponibles
    6 drift des prédictions             7 signal de qualité grave    8 incident de SERVICE (latence, erreurs)
Deux situations proches ont des signatures proches, même si leurs features portent des noms différents :
c'est ce qui permet de rapprocher un incident « bank » d'un incident « credit_risk ».
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

SIG_DIM = 9
SERVICE_ALERTS = {"MLGatewayDown", "MLUpstreamDown", "MLHighErrorRate", "MLHighLatency"}   # règles Prometheus « service »


@dataclass
class IncidentCard:
    id: str                      # INC-0001…
    source: str                  # scenario | agent | rcaeval | openrca | nab
    family: str                  # ml | service
    title: str
    text: str
    root_cause: str
    action: str
    outcome: str = "vérifié"     # vérifié | confirmé | rejeté par un humain
    task: str = ""
    signals: list[str] = field(default_factory=list)
    signature: list[float] = field(default_factory=lambda: [0.0] * SIG_DIM)

    def payload(self) -> dict:
        return asdict(self)


def _drop(facts: dict, kind: str) -> float:
    metric = next((k.split(".")[-1] for k in facts if k.startswith(f"perf.{kind}.")), None)
    if metric is None or "perf.bound" not in facts or "perf.reference_mean" not in facts:
        return 0.0
    value, bound, ref = facts[f"perf.{kind}.{metric}"], facts["perf.bound"], facts["perf.reference_mean"]
    tolerance = abs(ref - bound) or 1e-9
    worse = (ref - value) if bound < ref else (value - ref)      # sens de la métrique déduit de la limite
    return float(np.clip(worse / (3 * tolerance), 0, 1))


def signature(facts: dict, signals: list[str] | set[str] = (), service: bool = False) -> list[float]:
    psi = [v for k, v in facts.items() if k.startswith("drift.") and k.endswith(".psi")]
    quality = [v for k, v in facts.items() if k.startswith("quality.")]
    return [round(float(x), 4) for x in (
        facts.get("dataset.share_drifted", 0.0), min(max(psi, default=0.0), 1.0),
        _drop(facts, "realized"), _drop(facts, "estimated"), min(max(quality, default=0.0), 1.0),
        facts.get("perf.labels_available", 0.0), facts.get("prediction_drift.drifted", 0.0),
        float("data_quality" in signals), float(service))]


def query_text(task: str, signals, summary: str) -> str:
    return f"[{task}] signaux : {', '.join(sorted(signals)) or 'aucun'}.\n{summary}"


# ---------------------------------------------------------------------------
# Fiches issues des scénarios ML (bases d'ENTRAÎNEMENT uniquement : pas de fuite vers le test)
# ---------------------------------------------------------------------------

CAUSE_TXT = {"covariate_shift": "glissement des entrées qui dégrade le modèle", "concept_drift": "la relation entrées → cible a changé",
             "data_quality": "données cassées en amont", "benign_drift": "drift sans dégradation",
             "no_issue": "rien d'anormal", "service_degradation": "dégradation du service"}
ACTION_TXT = {"request_retraining": "demande de réentraînement", "notify_data_team": "équipe data prévenue, PAS de réentraînement",
              "increase_monitoring": "surveillance renforcée", "open_incident": "incident ouvert en attendant les labels",
              "no_action": "aucune action", "rollback_model": "retour au modèle précédent"}


def _narrative(sc: dict) -> str:
    p, inj = sc["params"], sc["injection"]
    parts = []
    if "shift" in p:
        parts.append("glissement : " + ", ".join(f"{f} ({d})" for f, d in p["shift"].items()))
    if "concept" in p:
        c = p["concept"]
        parts.append(f"relation x → y modifiée ({c.get('region')})")
    if "quality" in p:
        q = p["quality"]
        parts.append(f"panne de données {q['kind']} sur {', '.join(q.get('features', []))}"
                     + (f" ({q['rate']:.0%} des lignes)" if "rate" in q else ""))
    if p.get("labels_delayed"):
        parts.append("labels pas encore arrivés")
    if p.get("ramp"):
        parts.append("apparition progressive sur 3 fenêtres")
    return "; ".join(parts) or ("fenêtre calme" if inj == "calm" else inj)


def from_scenarios(path: Path, per_group: int = 6) -> list[IncidentCard]:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.exists() else []
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for sc in rows:
        if sc["split"] == "train":
            groups[(sc["base"], sc["injection"], sc["oracle"]["action"])].append(sc)
    cards = []
    for (base, _inj, _act), items in sorted(groups.items()):
        for sc in items[:per_group]:
            o = sc["oracle"]
            cards.append(IncidentCard(
                id="", source="scenario", family="ml", task=sc["task"], signals=sc["signals"],
                title=f"{base} : {CAUSE_TXT[o['root_cause']]}",
                text=(f"[{sc['task']}] signaux : {', '.join(sc['signals']) or 'aucun'}.\n"
                      f"Projet {base}. {_narrative(sc)}. Cause : {o['root_cause']} ({CAUSE_TXT[o['root_cause']]}). "
                      f"Résolution : {o['action']} ({ACTION_TXT[o['action']]})."),
                root_cause=o["root_cause"], action=o["action"], signature=signature(sc["facts"], sc["signals"])))
    return cards


# ---------------------------------------------------------------------------
# Fiches d'incidents de SERVICE (dossier 03_Incidents_Service)
# ---------------------------------------------------------------------------

def _service_card(source, title, text) -> IncidentCard:
    sig = [0.0] * SIG_DIM
    sig[SIG_DIM - 1] = 1.0
    return IncidentCard(id="", source=source, family="service", title=title, text=text,
                        root_cause="service_degradation", action="open_incident", signals=["service"], signature=sig)


NAB_CAUSES = {
    "ec2_request_latency_system_failure": "latence des requêtes EC2 en hausse avant une panne système",
    "machine_temperature_system_failure": "température d'une machine industrielle avant une panne",
    "cpu_utilization_asg_misconfiguration": "CPU anormal causé par une mauvaise configuration d'autoscaling",
    "rogue_agent_key_hold": "agent défaillant (touche bloquée) qui sature le système",
    "rogue_agent_key_updown": "agent défaillant qui génère des événements parasites",
    "ambient_temperature_system_failure": "température ambiante anormale avant une panne",
    "nyc_taxi": "pics de demande liés à des événements (marathon, fêtes, tempête)",
}


def from_incident_data(data_dir: Path) -> list[IncidentCard]:
    """RCAEval (cases.parquet), OpenRCA (record.csv), NAB (combined_windows.json) : ce qui est présent."""
    import pandas as pd
    cards: list[IncidentCard] = []
    rca = sorted(data_dir.rglob("cases.parquet"))
    if rca:
        df = pd.read_parquet(rca[0])
        for (system, fault, desc), g in df.groupby(["system_name", "fault", "fault_description"]):
            services = ", ".join(sorted(g["root_cause_service"].astype(str).unique())[:5])
            cards.append(_service_card(
                "rcaeval", f"{system} : {desc}",
                f"[service] Panne « {desc} » ({fault}) injectée sur {services} ({system}, {len(g)} cas RCAEval). "
                "Symptômes : latence, erreurs ou saturation sur le service fautif puis ses appelants. "
                "Cause : service_degradation (infrastructure, pas le modèle). Résolution : open_incident vers l'équipe "
                "plateforme ; réentraîner ne sert à rien."))
    for rec in sorted(data_dir.rglob("record.csv"))[:1]:
        df = pd.read_csv(rec)
        for (reason, level), g in df.groupby(["reason", "level"]):
            comps = Counter(g["component"].astype(str)).most_common(4)
            cards.append(_service_card(
                "openrca", f"{reason} ({level})",
                f"[service] {len(g)} incidents OpenRCA « {reason} » au niveau {level}, composants les plus touchés : "
                f"{', '.join(c for c, _ in comps)}. Cause : service_degradation. Résolution : open_incident ; "
                "vérifier la ressource en cause avant de soupçonner le modèle."))
    for nab in sorted(data_dir.rglob("combined_windows.json"))[:1]:
        windows = json.loads(nab.read_text(encoding="utf-8"))
        for key, wins in windows.items():
            name = Path(key).stem
            if name in NAB_CAUSES:
                cards.append(_service_card(
                    "nab", NAB_CAUSES[name],
                    f"[service] Série NAB {name} : {NAB_CAUSES[name]} ({len(wins)} fenêtres d'anomalie étiquetées). "
                    "Cause : service_degradation. Résolution : open_incident ; corréler avec les déploiements."))
    return cards
