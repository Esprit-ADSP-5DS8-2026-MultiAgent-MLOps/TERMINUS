import { useEffect, useState } from "react";
import { api } from "../api";
import { Cmd, ErrorLine } from "../components/Bits";
import { usePoll } from "../hooks";

const GARES = [
  { n: 0, name: "ORCHESTRATEUR", who: "Ahmed", role: "Décide de la suite après chaque gare, arrête la ligne aux signaux." },
  { n: 1, name: "VALIDATEUR + RÉPARATION", who: "Sarra", role: "Le v1 contrôle l'entrée ; le v2, le Repair et le juge corrigent une copie quand le v1 bloque." },
  { n: 2, name: "ANALYSE", who: "Yacine", role: "Lit le code et les données : tâche, cible, métrique, profil, modèle de référence." },
  { n: 3, name: "MLFLOW", who: "Aziz", role: "Entraîne et compare les modèles en validation croisée, applique le quality gate." },
  { n: 4, name: "API DOCKER", who: "Syrine", role: "Sert le modèle MLflow dans une API testée, construite en image Docker et déployée." },
  { n: 5, name: "MONITORING", who: "Mohamed", role: "Surveille la production, diagnostique et demande un réentraînement quand il le faut." },
];

const LINKS: Record<number, [string, string][]> = {
  3: [["MLflow", "http://127.0.0.1:5001"]],
  5: [["console", "http://127.0.0.1:8100/console/"], ["grafana", "http://127.0.0.1:3000"]],
};

function TryModel() {
  const runs = usePoll(() => api.runs(), 30000);
  const live = runs.data?.find((r) => r.served?.endpoint);
  const [body, setBody] = useState("");
  const [answer, setAnswer] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!live || body) return;
    api.example(live.run_id).then((row) => setBody(JSON.stringify(row, null, 2)), () => setBody("{}"));
  }, [live, body]);

  async function send() {
    setError(null);
    try {
      const res = await api.predict(JSON.parse(body));
      setAnswer(JSON.stringify(res.body, null, 2));
    } catch (e) {
      setError(e instanceof SyntaxError ? "JSON invalide" : e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <section className="panel" aria-labelledby="essai">
      <h2 id="essai" className="panel-head">
        <span>$ curl -X POST /predict</span>
        <span style={{ fontWeight: 400, color: "var(--signal)" }}>{live ? `${live.project} · ${live.served?.model_uri}` : "aucun modèle déployé"}</span>
      </h2>
      <div className="panel-body" style={{ flexDirection: "row", flexWrap: "wrap", gap: 24 }}>
        <label className="field" style={{ flex: "1 1 420px", minWidth: 0 }}>
          requête (une ligne des données du projet, sans la cible)
          <textarea rows={9} value={body} onChange={(e) => setBody(e.target.value)} style={{ resize: "vertical" }} />
        </label>
        <div style={{ flex: "1 1 280px", minWidth: 0, display: "flex", flexDirection: "column", gap: 12 }}>
          <button className="btn primary" onClick={send} disabled={!live}>[ envoyer ]</button>
          <ErrorLine error={error} />
          <pre className="mono-pre" aria-live="polite">{answer || "← la réponse du modèle"}</pre>
          <p className="tiny muted">La requête passe par la gateway de l'Agent 4 : elle compte dans le monitoring.</p>
        </div>
      </div>
    </section>
  );
}

export function Agents() {
  return (
    <main className="wrap page">
      <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "space-between", alignItems: "baseline", gap: "8px 20px" }}>
        <h1 className="title">Les gares de la ligne</h1>
        <span className="small muted">$ terminus agents</span>
      </div>
      <ol style={{ listStyle: "none", margin: 0, padding: 0, display: "grid",
        gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))", gap: 20 }}>
        {GARES.map((g) => (
          <li key={g.n} className="panel" style={{ display: "flex", flexDirection: "column" }}>
            <div className="small" style={{ padding: "8px 14px", borderBottom: "2px solid var(--ink)", display: "flex",
              justifyContent: "space-between", gap: 10 }}>
              <strong>GARE {g.n} · {g.name}</strong><span className="muted">{g.who}</span>
            </div>
            <div style={{ padding: 14, display: "flex", flexDirection: "column", gap: 8, fontSize: 14 }}>
              <p>{g.role}</p>
              {LINKS[g.n] && (
                <span className="small" style={{ display: "flex", flexWrap: "wrap", gap: "4px 14px" }}>
                  {LINKS[g.n].map(([label, url]) => <a key={url} href={url} target="_blank" rel="noreferrer">→ {label}</a>)}
                </span>
              )}
            </div>
          </li>
        ))}
      </ol>
      <h2><Cmd>terminus essayer</Cmd></h2>
      <TryModel />
    </main>
  );
}
