import { useState } from "react";
import { api, GATE_TITLES, type Gate } from "../api";
import { bar, useOperator } from "../hooks";

type Content = Record<string, any>;

function num(v: unknown, digits = 4): string {
  return typeof v === "number" ? v.toFixed(digits) : "—";
}

const LOWER_IS_BETTER = new Set(["rmse", "mae"]);

function Champion({ c }: { c: Content }) {
  const g: Content = c.quality_gate ?? {};
  const lower = LOWER_IS_BETTER.has(g.primary_metric);
  const rows: [string, number | null | undefined, boolean][] = [
    [`${g.primary_metric ?? "score"} candidat`, g.value, true],
    ["modèle naïf", g.baseline_value, false],
    ["champion actuel", g.champion_value, false],
    [lower ? "erreur max. du manifeste" : "seuil du manifeste", g.min_threshold, false],
  ];
  // barres à l'échelle des valeurs : 0..1 pour roc_auc, f1… ; jusqu'à la plus grande erreur pour rmse / mae
  const scale = Math.max(1, ...rows.map(([, v]) => (typeof v === "number" ? Math.abs(v) : 0)));
  return (
    <>
      <dl style={{ margin: 0, display: "grid", gridTemplateColumns: "max-content max-content 1fr", gap: "6px 16px",
        fontSize: 14, alignItems: "baseline" }}>
        {rows.map(([label, v, main]) => (
          <div key={label} style={{ display: "contents" }}>
            <dt className={main ? undefined : "muted"}>{label}</dt>
            <dd style={{ margin: 0, fontWeight: main ? 600 : 400 }}>
              {v == null ? (label === "champion actuel" ? "aucun" : "—") : num(v, label.includes("manifeste") ? 2 : 4)}
            </dd>
            <dd aria-hidden="true" className={main ? undefined : "muted"} style={{ margin: 0, letterSpacing: 1 }}>
              {typeof v === "number" ? bar(v, scale) : ""}
            </dd>
          </div>
        ))}
      </dl>
      {lower && <p className="small muted">{g.primary_metric} : plus bas = meilleur (erreur, dans l'unité de la cible).</p>}
      <div className="small muted" style={{ overflowWrap: "anywhere" }}>
        modèle {c.params?.model ?? "?"} · {c.model_uri ?? ""}
      </div>
    </>
  );
}

function Retraining({ c }: { c: Content }) {
  const perf: Content = c.current_performance ?? {};
  return (
    <>
      <p className="small">
        raison <strong>{c.reason}</strong> · modèle {c.model_uri}
        {Object.keys(perf).length > 0 && (
          <> · en production : {Object.entries(perf).slice(0, 3).map(([k, v]) => `${k} ${num(v, 3)}`).join(", ")}</>
        )}
      </p>
      {c.diagnosis && <p className="small" style={{ borderLeft: "3px solid var(--ink)", paddingLeft: 12 }}>{c.diagnosis}</p>}
    </>
  );
}

function Manifest({ c }: { c: Content }) {
  return (
    <p className="small">
      tâche <strong>{c.task}</strong> · cible <code>{c.target}</code> · {c.primary_metric} ≥ {c.min_threshold} ·
      confiance {num(c.confidence, 2)}
      {c.rationale && <><br />{c.rationale}</>}
    </p>
  );
}

export function Diff({ text }: { text: string }) {
  const lines = text.split("\n").filter((l, i, all) => i < all.length - 1 || l !== "");
  return (
    <div className="diff" role="region" aria-label="Diff de la réparation">
      {lines.map((l, i) => {
        if (l.startsWith("---") || l.startsWith("+++")) return <div key={i} className="hunk">{l}</div>;
        if (l.startsWith("@@")) return <div key={i} className="hunk">{l}</div>;
        const kind = l.startsWith("+") ? "add" : l.startsWith("-") ? "del" : "";
        return (
          <div key={i} className={`line ${kind}`}>
            <span>{kind === "add" ? "+" : kind === "del" ? "−" : " "}</span>
            <span>{kind ? l.slice(1) : l.slice(1)}</span>
          </div>
        );
      })}
    </div>
  );
}

function Judge({ j }: { j: Content }) {
  const criteria: Content[] = j.criteria ?? [];
  const flagged = criteria.filter((c) => c.result !== "PASS");
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <p className="small">
        <span className="st-inverse">{j.verdict ?? "?"}</span>{" "}
        <span className="muted">{j.llm?.model ?? ""}</span>
      </p>
      <p className="small">
        {criteria.length - flagged.length} critères PASS
        {flagged.map((c) => <span key={c.name}> · <span className="st-signal">{c.name} {c.result}</span></span>)}
      </p>
      {criteria.length > 0 && (
        <details className="small">
          <summary style={{ cursor: "pointer", padding: "4px 0" }}>voir le détail des critères</summary>
          <ul style={{ listStyle: "none", margin: "6px 0 0", padding: 0 }}>
            {criteria.map((c) => (
              <li key={c.name}>{c.name.padEnd(20, ".")} {c.result}{c.rationale ? ` · ${c.rationale}` : ""}</li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

function Override({ gate }: { gate: Gate }) {
  const v1: Content = (gate.validator_v1 as Content)?.handoff ?? {};
  const v2: Content = gate.content ?? {};
  const issues: Content[] = v2.issues ?? [];
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 20 }}>
      <div style={{ flex: "1 1 260px", minWidth: 0 }}>
        <h3 className="cmd">v1 · contrôle d'entrée</h3>
        <p className="small"><span className="st-inverse">{v1.status ?? "BLOCKED"}</span> {v1.reason ?? ""}</p>
      </div>
      <div style={{ flex: "1 1 260px", minWidth: 0 }}>
        <h3 className="cmd">v2 · diagnostic</h3>
        <p className="small"><strong>{v2.status ?? "?"}</strong> · prochaine action {v2.next_action ?? "?"}</p>
        <ul className="small" style={{ margin: "6px 0 0", paddingLeft: 18 }}>
          {issues.slice(0, 5).map((i) => <li key={i.issue_id}>{i.check_id} · {i.description}</li>)}
          {issues.length === 0 && <li>aucun problème relevé</li>}
        </ul>
      </div>
    </div>
  );
}

const LABELS: Record<string, [string, string]> = {
  failure_review: ["relancer l'étape", "arrêter la ligne"],
  repair_review: ["valider la réparation", "refuser"],
  validator_override: ["passer outre le v1", "arrêter la ligne"],
};

export function GateCard({ gate, onDecided }: { gate: Gate; onDecided: () => void }) {
  const [who, setWho] = useOperator();
  const [comment, setComment] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const c = gate.content as Content;
  const [yes, no] = LABELS[gate.gate] ?? ["approuver", "refuser"];

  async function decide(decision: "approve" | "reject", publish?: boolean) {
    if (!who.trim()) {
      setError("Indique ton nom : il est enregistré avec la décision.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await api.decide(gate.run_id, { decision, by: who.trim(), comment, key: gate.key, publish });
      onDecided();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="panel" aria-labelledby={`gate-${gate.key}`}>
      <h2 id={`gate-${gate.key}`} className="panel-head signal">
        <span>● SIGNAL · {GATE_TITLES[gate.gate] ?? gate.gate}</span>
        <span style={{ fontWeight: 400 }}>en attente</span>
      </h2>
      <div className="panel-body">
        <p className="small">{gate.question}</p>
        {gate.gate === "champion_replacement" && <Champion c={c} />}
        {gate.gate === "retraining" && <Retraining c={c} />}
        {gate.gate === "manifest_confirmation" && <Manifest c={c} />}
        {gate.gate === "failure_review" && (
          <div className="small">
            <p>étape <strong>{gate.failed_step}</strong></p>
            <ul style={{ margin: "6px 0", paddingLeft: 18 }}>{(gate.reasons ?? []).map((r, i) => <li key={i}>{r}</li>)}</ul>
            {gate.supervisor_explanation && <p className="muted">superviseur : {gate.supervisor_explanation}</p>}
          </div>
        )}
        {gate.gate === "repair_review" && (
          <>
            <p className="small">{(gate.reasons ?? [])[0]}</p>
            <h3 className="cmd">git diff</h3>
            {gate.diff ? <Diff text={gate.diff} /> : <p className="small muted">diff indisponible</p>}
            <h3 className="cmd">terminus juge</h3>
            <Judge j={(gate.judge ?? {}) as Content} />
          </>
        )}
        {gate.gate === "validator_override" && <Override gate={gate} />}

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
          {gate.gate === "champion_replacement" ? (
            <>
              <button className="btn primary" disabled={busy} onClick={() => decide("approve", true)}>[ approuver + publier ]</button>
              <button className="btn" disabled={busy} onClick={() => decide("approve", false)}>[ sans publier ]</button>
            </>
          ) : (
            <button className="btn primary" disabled={busy} onClick={() => decide("approve")}>[ {yes} ]</button>
          )}
          <button className="btn danger" disabled={busy} onClick={() => decide("reject")}>[ {no} ]</button>
        </div>
        {gate.gate === "champion_replacement" && (
          <p className="tiny muted">publier = image sur Docker Hub (si AGENT3_PUSH=1 dans agents/agent3/.env)</p>
        )}
      </div>
    </section>
  );
}
