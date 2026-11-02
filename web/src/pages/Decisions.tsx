import { useSearchParams } from "react-router";
import { api, GATE_TITLES } from "../api";
import { Annonce, Cmd, ErrorLine } from "../components/Bits";
import { GateCard } from "../components/GateCard";
import { day, usePoll } from "../hooks";

const FRONT_ONLY = new Set(["repair_review", "validator_override"]);

function Selected({ runId, onDecided }: { runId: string; onDecided: () => void }) {
  const run = usePoll(() => api.run(runId), 3000, [runId]);
  if (run.error) return <ErrorLine error={run.error} />;
  if (!run.data) return <p className="muted">…</p>;
  if (!run.data.gate) return <p className="empty">Cette ligne n'attend plus de décision.</p>;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <span className="small muted">ligne {run.data.run_id}</span>
        <h2 className="title">{run.data.project}</h2>
      </div>
      <GateCard gate={run.data.gate} onDecided={() => { run.reload(); onDecided(); }} />
    </div>
  );
}

export function Decisions() {
  const data = usePoll(() => api.decisions(), 4000);
  const [params, setParams] = useSearchParams();
  const pending = data.data?.pending ?? [];
  const selected = params.get("ligne") ?? pending[0]?.run_id ?? null;

  return (
    <main className="wrap page">
      <div className="row">
        <aside className="side" aria-labelledby="file" style={{ display: "flex", flexDirection: "column", gap: 24 }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <span className="small muted">$ terminus signaux --attente</span>
            <h1 id="file" className="title">Signaux en attente</h1>
            <Annonce>
              {pending.length === 0 ? "Aucune ligne n'attend de décision."
                : `${pending.length} ligne${pending.length > 1 ? "s" : ""} arrêtée${pending.length > 1 ? "s" : ""} à un signal.`}
            </Annonce>
          </div>
          <ErrorLine error={data.error} />
          <div className="rule" role="list" style={{ fontSize: 14 }}>
            <div aria-hidden="true" className="tiny muted"
              style={{ display: "grid", gridTemplateColumns: "34px 1fr 96px", gap: 10, padding: "8px 10px", borderBottom: "1px solid var(--ink)" }}>
              <span>N°</span><span>SIGNAL · LIGNE</span><span>OÙ</span>
            </div>
            {pending.map((p, i) => {
              const active = p.run_id === selected;
              return (
                <button key={p.key + p.run_id} role="listitem" aria-current={active || undefined}
                  onClick={() => setParams({ ligne: p.run_id })}
                  style={{ display: "grid", gridTemplateColumns: "34px 1fr 96px", gap: 10, padding: "12px 10px", width: "100%",
                    textAlign: "left", border: "none", borderBottom: "1px dashed var(--soft-line)", cursor: "pointer",
                    font: "inherit", background: active ? "var(--ink)" : "transparent", color: active ? "var(--paper)" : "var(--ink)" }}>
                  <span>{String(i + 1).padStart(2, "0")}</span>
                  <span><strong>{GATE_TITLES[p.gate] ?? p.gate}</strong><br /><span className="tiny">{p.project} · {p.run_id}</span></span>
                  <span className="tiny">{FRONT_ONLY.has(p.gate) ? "terminus" : "+ slack"}</span>
                </button>
              );
            })}
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <h2><Cmd>terminus signaux --recents</Cmd></h2>
            <ul style={{ listStyle: "none", margin: 0, padding: 0 }} className="small muted">
              {(data.data?.recent ?? []).slice(0, 8).map((d, i) => (
                <li key={i}>
                  {d.decision === "approve" ? "✓" : "✗"} {GATE_TITLES[d.gate] ?? d.gate} · {d.project} · {d.by} · {day(d.at)}
                </li>
              ))}
            </ul>
          </div>
        </aside>
        <section className="grow" aria-label="Signal sélectionné">
          {selected ? <Selected runId={selected} onDecided={data.reload} />
            : <p className="empty">Rien à décider : toutes les lignes roulent ou sont arrivées.</p>}
        </section>
      </div>
    </main>
  );
}
