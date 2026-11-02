import { useState } from "react";
import { api } from "../api";
import { Cmd, ErrorLine } from "../components/Bits";
import { day, usePoll } from "../hooks";

function History({ name }: { name: string }) {
  const versions = usePoll(() => api.model(name), 15000, [name]);
  if (versions.error) return <ErrorLine error={versions.error} />;
  if (!versions.data) return <p className="muted">…</p>;
  return (
    <ol style={{ listStyle: "none", margin: 0, padding: 0, display: "flex", flexDirection: "column" }}>
      {versions.data.map((v, i) => {
        const champion = v.aliases.includes("champion");
        const metrics = Object.entries(v.metrics).filter(([k]) => k.startsWith("test_"));
        const base = Object.entries(v.metrics).find(([k]) => k.startsWith("baseline_test_"));
        return (
          <li key={v.version} style={{ display: "grid", gridTemplateColumns: "36px 1fr", gap: 16 }}>
            <div aria-hidden="true" style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
              <span style={{ width: 20, height: 20, border: "3px solid var(--ink)", background: champion ? "var(--signal)" : "transparent" }} />
              {i < versions.data!.length - 1 && <span style={{ flex: 1, borderLeft: "3px solid var(--ink)" }} />}
            </div>
            <article className="panel" style={{ marginBottom: 20 }}>
              <div className={`panel-head${champion ? " signal" : ""}`} style={champion ? undefined : { background: "transparent", color: "var(--ink)", borderBottom: "2px solid var(--ink)" }}>
                <strong>v{v.version}{v.aliases.length ? " · " + v.aliases.map((a) => "@" + a).join(" ") : " · retirée"}</strong>
                <span style={{ fontWeight: 400 }}>{day(v.created_at)}</span>
              </div>
              <dl style={{ margin: 0, padding: 16, display: "grid", gridTemplateColumns: "170px 1fr", gap: "6px 16px", fontSize: 14 }}>
                <dt className="muted">modèle</dt><dd style={{ margin: 0 }}>{v.model ?? "—"}</dd>
                {metrics.map(([k, m]) => (
                  <div key={k} style={{ display: "contents" }}>
                    <dt className="muted">{k.replace("test_", "")} test</dt>
                    <dd style={{ margin: 0 }}><strong>{m.toFixed(4)}</strong>{base ? <span className="muted"> · modèle naïf {base[1].toFixed(4)}</span> : null}</dd>
                  </div>
                ))}
                <dt className="muted">run MLflow</dt><dd style={{ margin: 0, overflowWrap: "anywhere" }}>{v.run_id ?? "—"}</dd>
              </dl>
            </article>
          </li>
        );
      })}
    </ol>
  );
}

export function Registre() {
  const models = usePoll(() => api.registry(), 15000);
  const [chosen, setChosen] = useState<string | null>(null);
  const name = chosen ?? models.data?.[0]?.name ?? null;

  return (
    <main className="wrap page">
      <div className="row">
        <aside className="side" aria-labelledby="modeles" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <h1 id="modeles" className="title">Registre</h1>
          <span className="small muted">$ terminus modeles · MLflow</span>
          <ErrorLine error={models.error} />
          <ul className="rule" style={{ listStyle: "none", margin: "8px 0 0", padding: 0, fontSize: 14 }}>
            {(models.data ?? []).map((m) => {
              const active = m.name === name;
              return (
                <li key={m.name}>
                  <button onClick={() => setChosen(m.name)} aria-current={active || undefined}
                    style={{ display: "flex", justifyContent: "space-between", gap: 12, padding: "12px 10px", width: "100%",
                      border: "none", borderBottom: "1px dashed var(--soft-line)", font: "inherit", cursor: "pointer",
                      background: active ? "var(--ink)" : "transparent", color: active ? "var(--paper)" : "var(--ink)" }}>
                    <span>{m.name}</span><span>{m.champion ? `v${m.champion} ★` : "—"}</span>
                  </button>
                </li>
              );
            })}
          </ul>
          <p className="tiny muted">★ = alias champion : la version servie en production.</p>
        </aside>
        <section className="grow" aria-labelledby="historique" style={{ display: "flex", flexDirection: "column", gap: 18 }}>
          {name ? (
            <>
              <h2 id="historique"><Cmd opt="--historique">terminus modele {name}</Cmd></h2>
              <History name={name} />
            </>
          ) : <p className="empty">Aucun modèle enregistré pour l'instant.</p>}
        </section>
      </div>
    </main>
  );
}
