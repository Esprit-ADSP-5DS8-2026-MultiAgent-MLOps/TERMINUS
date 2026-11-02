import { useEffect, useState } from "react";
import { Navigate, useParams } from "react-router";
import { api, GATE_TITLES, type RunDetail } from "../api";

const AGENT4 = { console: "http://127.0.0.1:8100/console/", grafana: "http://127.0.0.1:3000", alerts: "http://127.0.0.1:9090/alerts" };
import { Annonce, Cmd, ErrorLine } from "../components/Bits";
import { GateCard } from "../components/GateCard";
import { Track } from "../components/Track";
import { usePoll } from "../hooks";
import { Etat } from "./Lignes";

function announcement(r: RunDetail): string {
  const running = r.track.find((s) => s.status === "running");
  if (r.status === "STARTING") return "La ligne se forme : le validateur contrôle le projet.";
  if (r.gate) return `La ligne attend votre décision : ${GATE_TITLES[r.gate.gate] ?? r.gate.gate}.`;
  if (running) return `${running.label} en cours.`;
  if (r.status === "DONE") return `Modèle ${r.served?.model_uri ?? ""} en production, surveillé par l'Agent 4.`;
  if (r.status === "CHAMPION_KEPT") return "Le nouveau candidat ne fait pas mieux : le modèle en production reste en place.";
  if (r.status === "BLOCKED") return "Ligne arrêtée : le projet ne respecte pas le contrat.";
  if (r.status === "FAILED") return "Ligne arrêtée : une étape a échoué.";
  if (r.status === "REJECTED") return "Ligne arrêtée par une décision humaine.";
  return "La ligne avance.";
}

/** /pipeline : la ligne la plus récente. */
export function LatestPipeline() {
  const runs = usePoll(() => api.runs(), 60000);
  if (runs.error) return <main className="wrap page"><ErrorLine error={runs.error} /></main>;
  if (!runs.data) return null;
  if (runs.data.length === 0) return <Navigate to="/" replace />;
  return <Navigate to={`/lignes/${runs.data[0].run_id}`} replace />;
}

function FileViewer({ runId, artifacts }: { runId: string; artifacts: Record<string, string> }) {
  const [open, setOpen] = useState<string | null>(null);
  const [text, setText] = useState<string>("");
  useEffect(() => {
    if (!open) return;
    setText("…");
    api.file(runId, open).then((t) => setText(typeof t === "string" ? t : JSON.stringify(t, null, 2)),
      (e) => setText(String(e.message ?? e)));
  }, [runId, open]);
  const files = Object.entries(artifacts).filter(([, p]) => !p.startsWith("/"));
  return (
    <section aria-labelledby="sorties" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <h2 id="sorties"><Cmd>ls runs/{runId}/</Cmd></h2>
      <ul className="rule" style={{ listStyle: "none", margin: 0, padding: "14px 0 0", display: "grid",
        gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))", gap: "6px 24px", fontSize: 14 }}>
        {files.map(([name, path]) => (
          <li key={name}>
            <button onClick={() => setOpen(open === path ? null : path)} aria-expanded={open === path}
              style={{ border: "none", background: open === path ? "var(--signal)" : "none", padding: 0,
                textDecoration: "underline", cursor: "pointer", font: "inherit", textAlign: "left" }}>
              {path}
            </button>
          </li>
        ))}
      </ul>
      {open && <pre className="mono-pre" aria-label={`Contenu de ${open}`}>{text}</pre>}
    </section>
  );
}

/** Ce que l'Agent 4 surveille, sans quitter Terminus : état de la gateway, dernier avis, liens directs. */
function Surveillance({ r }: { r: RunDetail }) {
  const health = usePoll(() => api.health(), 15000);
  const gateway = health.data?.find((h) => h.service === "Gateway Agent 4");
  const notes = r.results?.agent4?.warnings ?? [];
  const project = r.project ?? "";
  const dashboard = `${AGENT4.grafana}/d/${`mlmon-${project}`.slice(0, 40)}`;
  return (
    <section className="panel" aria-labelledby="surveillance">
      <h2 id="surveillance" className="panel-head">
        <span>$ agent4 surveille {project}</span>
        <span style={{ fontWeight: 400, color: "var(--signal)" }}>
          {gateway ? (gateway.ok ? "gateway ✓" : "gateway ✗") : "…"}
        </span>
      </h2>
      <div className="panel-body small">
        {gateway && !gateway.ok && (
          <p>
            <span className="st-signal">!</span> Le trafic n'est pas observé. Lancer la gateway :
            <br /><code style={{ overflowWrap: "anywhere" }}>cd agents/agent4 && python -m monitoring_agent gateway {project} --host 0.0.0.0</code>
          </p>
        )}
        {notes.length > 0 && (
          <ul style={{ listStyle: "none", margin: 0, padding: 0, display: "flex", flexDirection: "column", gap: 4 }}>
            {notes.map((n, i) => <li key={i} style={{ overflowWrap: "anywhere" }}>› {n}</li>)}
          </ul>
        )}
        <span style={{ display: "flex", flexWrap: "wrap", gap: "4px 16px" }}>
          <a href={AGENT4.console} target="_blank" rel="noreferrer">→ console</a>
          <a href={dashboard} target="_blank" rel="noreferrer">→ grafana</a>
          <a href={AGENT4.alerts} target="_blank" rel="noreferrer">→ alertes</a>
        </span>
      </div>
    </section>
  );
}

export function Pipeline() {
  const { id = "" } = useParams();
  const run = usePoll(() => api.run(id), 2500, [id]);
  const [monitorError, setMonitorError] = useState<string | null>(null);
  const r = run.data;

  if (run.error && !r) return <main className="wrap page"><ErrorLine error={run.error} /></main>;
  if (!r) return <main className="wrap page"><p className="muted">$ terminus ligne {id} …</p></main>;

  async function monitor() {
    setMonitorError(null);
    try {
      await api.monitor(id);
      run.reload();
    } catch (e) {
      setMonitorError(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <main className="wrap page">
      <section style={{ display: "flex", flexWrap: "wrap", alignItems: "flex-end", justifyContent: "space-between", gap: 20 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 4, minWidth: 0 }}>
          <span className="small muted">$ terminus ligne {r.run_id}</span>
          <h1 className="title title-xl">{r.project ?? "…"}</h1>
          <span className="small muted">
            superviseur {r.supervisor ?? "?"}
            {r.repair_count ? ` · projet réparé (${r.repair_count} tentative${r.repair_count > 1 ? "s" : ""})` : ""}
            {r.retrain_count ? ` · réentraîné ${r.retrain_count}×` : ""}
            {r.served?.endpoint ? ` · API ${r.served.endpoint}` : ""}
          </span>
        </div>
        <div style={{ fontSize: 16 }}><Etat r={r} /></div>
      </section>

      <Annonce>{announcement(r)}</Annonce>
      {r.job?.error && <ErrorLine error={`${r.job.job} : ${r.job.error}`} />}

      <section className="frame" aria-labelledby="plan">
        <h2 id="plan" className="legend">plan de ligne</h2>
        <Track stations={r.track} retrainCount={r.retrain_count} />
      </section>

      <div className="row">
        <section className="grow" aria-labelledby="journal" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <h2 id="journal"><Cmd opt={`--superviseur ${r.supervisor ?? ""}`}>terminus journal</Cmd></h2>
          <ol className="rule" style={{ listStyle: "none", margin: 0, padding: 0, fontSize: 14 }}>
            {r.supervision.map((s, i) => (
              <li key={i} className="dashed" style={{ display: "grid", gridTemplateColumns: "44px 1fr", gap: 12, padding: "14px 0" }}>
                <span className="muted">[{String(i + 1).padStart(2, "0")}]</span>
                <span>
                  <strong style={{ fontWeight: 600 }}>{s.after} → {s.action}</strong>
                  {s.fallback && <span className="muted"> · secours : règle appliquée</span>}
                  <br />
                  {s.explanation}
                </span>
              </li>
            ))}
            {r.supervision.length === 0 && <li className="muted" style={{ padding: "14px 0" }}>pas encore de décision</li>}
          </ol>
        </section>

        <div className="side" style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          {r.gate && <GateCard gate={r.gate} onDecided={run.reload} />}
          {!r.gate && r.served?.model_uri && (
            <section className="panel" aria-labelledby="prod">
              <h2 id="prod" className="panel-head"><span>■ EN PRODUCTION</span></h2>
              <div className="panel-body small">
                <p style={{ overflowWrap: "anywhere" }}>{r.served.model_uri}</p>
                {r.served.value != null && <p>{r.served.metric} {r.served.value.toFixed(4)}</p>}
                <p className="muted" style={{ overflowWrap: "anywhere" }}>
                  image {r.served.image} · Docker Hub {r.served.pushed ? "publiée" : "non publiée"}
                </p>
                {["DONE", "CHAMPION_KEPT", "FAILED"].includes(r.status) && (
                  <button className="btn" onClick={monitor}>[ relancer l'Agent 4 ]</button>
                )}
                <ErrorLine error={monitorError} />
              </div>
            </section>
          )}
          {!r.gate && r.served?.endpoint && <Surveillance r={r} />}
        </div>
      </div>

      <FileViewer runId={r.run_id} artifacts={r.artifacts ?? {}} />
    </main>
  );
}
