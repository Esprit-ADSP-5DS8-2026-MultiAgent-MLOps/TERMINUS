import type { Station } from "../api";
import { seconds } from "../hooks";

/** Ce que fait chaque gare, pour qu'on comprenne la ligne sans lire le code. */
const STEPS: Record<string, string[]> = {
  validator: [
    "lit la structure du projet et le manifeste mlops.yaml",
    "analyse le code sans l'exécuter : dépendances, appels dangereux, secrets, chemins",
    "contrôle les données (lignes, cible, valeurs manquantes)",
    "exécute l'entraînement dans un bac à sable isolé, deux fois (déterminisme)",
    "scelle un rapport et une copie figée du projet pour les agents suivants",
  ],
  repair: [
    "le validateur v2 refait un diagnostic avec sa propre grille",
    "s'il faut réparer : le Repair corrige une COPIE (le projet d'origine ne change jamais)",
    "le juge, un autre modèle, vérifie la réparation preuve par preuve",
    "la copie est revalidée par le v1 ; en cas de désaccord v1/v2, un humain tranche",
  ],
  agent1: [
    "lit le code et les données du projet",
    "détecte la tâche, la cible et la métrique, et les compare au manifeste",
    "relit les pratiques ML (fuite de données, graine, split, déséquilibre)",
    "produit la fiche du projet, le profil des données et le modèle de référence",
  ],
  propose_manifest: [
    "le projet n'a pas de manifeste : l'Agent 1 en propose un depuis le code",
    "un humain le confirme avant toute exécution",
  ],
  agent2: [
    "un LLM propose un plan d'expériences (modèles, budget) ; repli sur un plan par défaut",
    "compare les modèles en validation croisée (Optuna), tout est suivi dans MLflow",
    "évalue le meilleur une seule fois sur le jeu de test, face au modèle naïf",
    "rejuge le champion en production sur le même jeu de test, puis applique le quality gate",
  ],
  gate_champion: ["un humain décide de mettre le candidat en production (et de publier l'image)"],
  agent3: [
    "récupère le modèle promu dans le registre MLflow",
    "génère l'API FastAPI et son Dockerfile depuis le contrat d'entrée et de sortie",
    "teste l'API : santé, contrat, entrées invalides, mêmes prédictions que le modèle",
    "construit l'image, la teste dans le conteneur, puis la déploie sur le port 8000",
  ],
  agent4: [
    "prend les données d'entraînement du modèle servi comme référence",
    "génère et valide les règles Prometheus et le tableau Grafana",
    "la gateway observe chaque prédiction ; toutes les 200, une fenêtre est analysée",
    "si une fenêtre est anormale, l'agent enquête, décide, et demande un humain pour une action sensible",
  ],
  gate_retraining: ["un humain décide de réentraîner sur les données récentes"],
  gate_repair_review: ["la réparation touche un point sensible : un humain lit le diff et l'avis du juge"],
  gate_validator_override: ["le v1 bloque, le v2 ne confirme pas : un humain tranche"],
  gate_failure: ["une étape a échoué : un humain choisit de la relancer ou d'arrêter"],
  gate_manifest: ["un humain confirme le manifeste proposé"],
};

const VERDICT_STYLE: Record<string, string> = { confirme: "st-inverse", "ne confirme pas": "st-signal" };

export function StationPanel({ s, live }: { s: Station; live: boolean }) {
  const steps = STEPS[s.id] ?? [];
  const facts = s.details?.facts ?? [];
  const compare = s.details?.compare;
  const reasons = s.all_reasons ?? s.reasons ?? [];
  const warnings = s.all_warnings ?? s.warnings ?? [];
  return (
    <section className="panel" aria-labelledby="gare-detail" aria-live="polite">
      <h2 id="gare-detail" className={`panel-head${s.status === "waiting" ? " signal" : ""}`}>
        <span>GARE · {s.label.toUpperCase()}</span>
        <span style={{ fontWeight: 400 }}>
          {s.status === "running" ? "en cours" : s.status === "waiting" ? "attend un humain"
            : s.status === "todo" ? "à venir" : s.decided_by ? `décidé par ${s.decided_by}`
            : `${s.duration_s != null ? seconds(s.duration_s) : ""}`}
        </span>
      </h2>
      <div className="panel-body" style={{ flexDirection: "row", flexWrap: "wrap", gap: 28 }}>
        <div style={{ flex: "1 1 300px", minWidth: 0 }}>
          <h3 className="cmd">ce que fait cette gare</h3>
          <ol className="small" style={{ margin: "8px 0 0", paddingLeft: 20, display: "flex", flexDirection: "column", gap: 4 }}>
            {steps.map((t, i) => (
              <li key={i}>
                {t}
                {live && i === 0 && <span className="blink" aria-hidden="true"> ▶</span>}
              </li>
            ))}
            {steps.length === 0 && <li>—</li>}
          </ol>
        </div>
        <div style={{ flex: "1 1 340px", minWidth: 0, display: "flex", flexDirection: "column", gap: 12 }}>
          <h3 className="cmd">ce qu'elle a fait</h3>
          {facts.length > 0 ? (
            <dl className="small" style={{ margin: 0, display: "grid", gridTemplateColumns: "minmax(0, max-content) 1fr", gap: "4px 14px" }}>
              {facts.map(([k, v], i) => (
                <div key={i} style={{ display: "contents" }}>
                  <dt className="muted">{k}</dt>
                  <dd style={{ margin: 0, overflowWrap: "anywhere" }}>{v}</dd>
                </div>
              ))}
            </dl>
          ) : (
            <p className="small muted">{s.status === "todo" || live ? "pas encore de résultat" : s.comment || "—"}</p>
          )}
          {reasons.length > 0 && (
            <ul className="small" style={{ margin: 0, paddingLeft: 18 }}>
              {reasons.map((r, i) => <li key={i}><strong>{r}</strong></li>)}
            </ul>
          )}
          {warnings.length > 0 && (
            <details className="small">
              <summary>{warnings.length} avertissement{warnings.length > 1 ? "s" : ""}</summary>
              <ul style={{ margin: "6px 0 0", paddingLeft: 18 }}>{warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>
            </details>
          )}
        </div>
        {compare && (
          <div style={{ flex: "1 1 100%", minWidth: 0, display: "flex", flexDirection: "column", gap: 8 }}>
            <h3 className="cmd">v1 face au v2 · {compare.summary}</h3>
            <div className="scroll-x">
              <table className="table small" style={{ minWidth: 560 }}>
                <thead><tr><th scope="col">v1 bloque</th><th scope="col">contrôle du v2</th><th scope="col">verdict du v2</th></tr></thead>
                <tbody>
                  {compare.rows.map((r) => (
                    <tr key={r.v1}>
                      <td><strong>{r.v1}</strong><br /><span className="muted">{r.v1_message}</span></td>
                      <td>{r.v2 ?? "—"}{r.v2_status ? ` (${r.v2_status})` : ""}</td>
                      <td><span className={VERDICT_STYLE[r.verdict]}>{r.verdict}</span></td>
                    </tr>
                  ))}
                  {compare.only_v2.map((o, i) => (
                    <tr key={`v2-${i}`}>
                      <td className="muted">—</td>
                      <td>{o.v2}<br /><span className="muted">{o.description}</span></td>
                      <td>vu seulement par le v2</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="tiny muted">
              Les deux validateurs n'ont pas la même grille : l'équivalence entre leurs contrôles est indicative.
              v2 : {compare.v2_status} → {compare.v2_next_action}.
            </p>
          </div>
        )}
      </div>
    </section>
  );
}
