import { useState } from "react";
import { api, type Agent4Pending } from "../api";
import { useOperator } from "../hooks";

const ACTIONS: Record<string, string> = {
  request_retraining: "demander un réentraînement",
  rollback_model: "revenir au modèle précédent",
  open_incident: "ouvrir un incident",
  notify_data_team: "prévenir l'équipe données",
  increase_monitoring: "renforcer la surveillance",
  no_action: "ne rien faire",
};

const CAUSES: Record<string, string> = {
  concept_drift: "concept drift : la relation données → cible a changé",
  covariate_shift: "drift des entrées",
  data_quality: "données cassées",
  service_degradation: "service dégradé",
  benign_drift: "drift sans conséquence",
  no_issue: "rien d'anormal",
};

/** Une décision de l'agent de la gateway qui attend un humain : validée ici comme dans la console ou Slack. */
export function Agent4Card({ runId, d, onDecided }: { runId: string; d: Agent4Pending; onDecided: () => void }) {
  const [who, setWho] = useOperator();
  const [comment, setComment] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState<string | null>(null);

  async function answer(decision: "approve" | "reject") {
    if (!who.trim()) {
      setError("Indique ton nom : il est enregistré avec la décision.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const r = await api.agent4Answer(runId, d.id, { decision, by: who.trim(), comment });
      setDone(r.retraining ? "Approuvé : la ligne reçoit la demande, le signal « réentraînement » arrive."
        : decision === "approve" ? "Approuvé." : "Refusé : l'agent n'agit pas.");
      onDecided();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="panel" aria-labelledby={`a4-${d.id}`}>
      <h2 id={`a4-${d.id}`} className="panel-head signal">
        <span>● AGENT 4 · {ACTIONS[d.action] ?? d.action}</span>
        <span style={{ fontWeight: 400 }}>fenêtre {d.window_index ?? "?"}</span>
      </h2>
      <div className="panel-body small">
        {d.root_cause && <p><strong>{CAUSES[d.root_cause] ?? d.root_cause}</strong></p>}
        {d.summary && <p>{d.summary}{d.rationale ? ` ${d.rationale}` : ""}</p>}
        {(d.evidence ?? []).length > 0 && (
          <dl style={{ margin: 0, display: "grid", gridTemplateColumns: "minmax(0, max-content) 1fr", gap: "4px 14px" }}>
            {(d.evidence ?? []).map((e, i) => (
              <div key={i} style={{ display: "contents" }}>
                <dt className="muted" style={{ overflowWrap: "anywhere" }}>{e.claim}</dt>
                <dd style={{ margin: 0, overflowWrap: "anywhere" }}>
                  {e.fact} = <strong>{typeof e.value === "number" ? Number(e.value).toFixed(4) : String(e.value)}</strong>
                </dd>
              </div>
            ))}
          </dl>
        )}
        {(d.signals ?? []).length > 0 && <p className="muted">signaux : {(d.signals ?? []).join(", ")}</p>}
        {done ? <p role="status">{done}</p> : (
          <>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 12 }}>
              <label className="field" style={{ flex: "1 1 160px" }}>
                ton nom
                <input value={who} onChange={(e) => setWho(e.target.value)} autoComplete="name" />
              </label>
              <label className="field" style={{ flex: "2 1 220px" }}>
                commentaire (facultatif)
                <input value={comment} onChange={(e) => setComment(e.target.value)} />
              </label>
            </div>
            {error && <div className="error" role="alert">{error}</div>}
            <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
              <button className="btn primary" disabled={busy} onClick={() => answer("approve")}>[ approuver ]</button>
              <button className="btn danger" disabled={busy} onClick={() => answer("reject")}>[ refuser ]</button>
            </div>
          </>
        )}
      </div>
    </section>
  );
}
