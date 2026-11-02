import { useState } from "react";
import { Link, useNavigate } from "react-router";
import { api, GATE_TITLES, type RunSummary } from "../api";
import { ErrorLine } from "../components/Bits";
import { day, usePoll } from "../hooks";

export function Etat({ r }: { r: RunSummary }) {
  if (r.status === "WAITING_HUMAN")
    return <span className="st-signal">● {GATE_TITLES[r.waiting_gate ?? ""] ?? "décision"}</span>;
  if (r.status === "RUNNING" || r.status === "STARTING") return <span>▶ en cours<span className="blink">…</span></span>;
  if (["BLOCKED", "FAILED", "REJECTED"].includes(r.status))
    return <span className="st-inverse">✗ {({ BLOCKED: "bloqué", FAILED: "échec", REJECTED: "refusé" } as Record<string, string>)[r.status]}</span>;
  if (r.status === "CHAMPION_KEPT") return <span>■ champion gardé</span>;
  if (r.status === "MANIFEST_PROPOSED") return <span>✓ manifeste validé</span>;
  return (
    <span>
      ✓ terminé{r.retrain_count ? ` · réentraîné ${r.retrain_count}×` : ""}{r.repair_count ? " · réparé" : ""}
    </span>
  );
}

export function Lignes() {
  const runs = usePoll(() => api.runs(), 5000);
  const projects = usePoll(() => api.projects(), 60000);
  const [project, setProject] = useState("");
  const [supervisor, setSupervisor] = useState("gemini");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const navigate = useNavigate();
  const chosen = project || projects.data?.[0]?.path || "";

  async function launch() {
    setBusy(true);
    setError(null);
    try {
      const { run_id } = await api.start(chosen, supervisor);
      navigate(`/lignes/${run_id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  }

  return (
    <main className="wrap page">
      <div className="row">
        <section className="panel side" aria-labelledby="lancer">
          <h1 id="lancer" className="panel-head">$ terminus lancer</h1>
          <div className="panel-body">
            <label className="field">
              projet
              <select value={chosen} onChange={(e) => setProject(e.target.value)}>
                {(projects.data ?? []).map((p) => (
                  <option key={p.path} value={p.path}>{p.path}{p.manifest ? "" : " (sans mlops.yaml)"}</option>
                ))}
              </select>
            </label>
            <fieldset style={{ margin: 0, padding: 0, border: "none", display: "flex", flexDirection: "column", gap: 8 }}
              className="small">
              <legend style={{ paddingBottom: 6 }}>superviseur</legend>
              <label style={{ display: "flex", gap: 10, alignItems: "center", minHeight: 32 }}>
                <input type="radio" name="sup" checked={supervisor === "gemini"} onChange={() => setSupervisor("gemini")} />
                gemini · explique chaque décision
              </label>
              <label style={{ display: "flex", gap: 10, alignItems: "center", minHeight: 32 }}>
                <input type="radio" name="sup" checked={supervisor === "rules"} onChange={() => setSupervisor("rules")} />
                règles · sans IA
              </label>
            </fieldset>
            <ErrorLine error={error} />
            <button className="btn primary" disabled={busy || !chosen} onClick={launch}>
              {busy ? "[ départ… ]" : "[ lancer la ligne ]"}
            </button>
            <p className="tiny muted">
              Les validations s'affichent ici, et dans Slack si le service tourne (python -m orchestrator slack).
            </p>
          </div>
        </section>

        <section className="grow" aria-labelledby="lignes" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "space-between", alignItems: "baseline", gap: "8px 20px" }}>
            <h2 id="lignes" className="title">Lignes</h2>
            <span className="small muted">$ terminus lignes --toutes</span>
          </div>
          <ErrorLine error={runs.error} />
          {runs.data && runs.data.length === 0 && <p className="empty">Aucune ligne pour l'instant : lance la première.</p>}
          {runs.data && runs.data.length > 0 && (
            <div className="scroll-x rule">
              <table className="table" style={{ minWidth: 680 }}>
                <thead>
                  <tr><th scope="col">LIGNE</th><th scope="col">PROJET</th><th scope="col">ÉTAT</th>
                    <th scope="col">MODÈLE SERVI</th><th scope="col">MIS À JOUR</th></tr>
                </thead>
                <tbody>
                  {runs.data.map((r) => (
                    <tr key={r.run_id}>
                      <td><Link to={`/lignes/${r.run_id}`}>{r.run_id}</Link></td>
                      <td>{r.project ?? "…"}</td>
                      <td><Etat r={r} /></td>
                      <td>
                        {r.served?.model_uri
                          ? `${r.served.model_uri.split("/").pop() ? "v" + r.served.model_uri.split("/").pop() : ""}${r.served.value != null ? ` · ${r.served.value.toFixed(4)}` : ""}`
                          : <span className="muted">—</span>}
                      </td>
                      <td className="muted">{day(r.updated_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>
    </main>
  );
}
