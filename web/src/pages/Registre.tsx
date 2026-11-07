import { useState } from "react";
import { api, type ModelVersion } from "../api";
import { Cmd, ErrorLine } from "../components/Bits";
import { day, usePoll } from "../hooks";

/** Code-barres décoratif, tiré de l'identifiant du run MLflow : chaque version a le sien. */
function Barcode({ seed }: { seed: string }) {
  const bars = [...(seed || "0000000000000000").slice(0, 24)].map((c) => parseInt(c, 16) || 0);
  return (
    <span className="barcode" aria-hidden="true">
      {bars.map((b, i) => <span key={i} style={{ width: 1 + (b % 4), marginRight: 1 + ((b >> 2) % 3) }} />)}
    </span>
  );
}

const CLASSE: Record<string, string> = { champion: "CHAMPION", challenger: "CHALLENGER", candidate: "CANDIDAT" };

function Ticket({ v, project }: { v: ModelVersion; project: string }) {
  const champion = v.aliases.includes("champion");
  // une version peut porter plusieurs alias (candidate puis champion) : le plus fort s'affiche
  const alias = ["champion", "challenger", "candidate"].find((a) => v.aliases.includes(a)) ?? v.aliases[0];
  const metrics = Object.entries(v.metrics).filter(([k]) => k.startsWith("test_"));
  const base = Object.entries(v.metrics).find(([k]) => k.startsWith("baseline_test_"));
  return (
    <article className={`ticket${champion ? " champion" : ""}`} aria-label={`Version ${v.version}`}>
      <div className="ticket-main">
        <div className="ticket-band">
          <span>TERMINUS · BILLET DE MODÈLE</span>
          <span>{day(v.created_at)}</span>
        </div>
        <p className="ticket-route">
          <span className="muted">DE</span> entraînement <span aria-hidden="true">━━▶</span>{" "}
          <span className="muted">À</span> {champion ? "production" : "registre"}
        </p>
        <dl className="ticket-fields">
          <dt>modèle</dt><dd>{v.model ?? "—"}</dd>
          {metrics.map(([k, m]) => (
            <div key={k} style={{ display: "contents" }}>
              <dt>{k.replace("test_", "")} test</dt>
              <dd><strong>{m.toFixed(4)}</strong>{base ? <span className="muted"> · modèle naïf {base[1].toFixed(4)}</span> : null}</dd>
            </div>
          ))}
          <dt>run MLflow</dt><dd className="ticket-id">{v.run_id ?? "—"}</dd>
        </dl>
        <span className={`stamp${champion ? " on" : ""}`}>{CLASSE[alias] ?? (alias ? alias.toUpperCase() : "RETIRÉE")}</span>
      </div>
      <div className="ticket-stub">
        <span className="notch top" aria-hidden="true" />
        <span className="notch bottom" aria-hidden="true" />
        <span className="tiny muted">{project}</span>
        <span className="tiny muted">N°</span>
        <span className="ticket-number">v{v.version}</span>
        <Barcode seed={v.run_id ?? ""} />
      </div>
    </article>
  );
}

function History({ name }: { name: string }) {
  const versions = usePoll(() => api.model(name), 15000, [name]);
  if (versions.error) return <ErrorLine error={versions.error} />;
  if (!versions.data) return <p className="muted">…</p>;
  return (
    <ol style={{ listStyle: "none", margin: 0, padding: 0, display: "flex", flexDirection: "column", gap: 22 }}>
      {versions.data.map((v) => <li key={v.version}><Ticket v={v} project={name} /></li>)}
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
