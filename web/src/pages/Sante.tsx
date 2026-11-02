import { api } from "../api";
import { Cmd, ErrorLine } from "../components/Bits";
import { usePoll } from "../hooks";

const FIX: Record<string, string> = {
  Docker: "lancer Docker Desktop",
  "base des runs": "docker compose -f infra/docker-compose.yml up -d",
  "Service Slack": "python -m orchestrator slack",
  "Gateway Agent 4": "python -m monitoring_agent gateway <projet>",
  Prometheus: "docker compose -f agents/agent4/infra/docker-compose.yml up -d",
  Grafana: "docker compose -f agents/agent4/infra/docker-compose.yml up -d",
  Alertmanager: "docker compose -f agents/agent4/infra/docker-compose.yml up -d",
  "API du modèle": "lancer une ligne jusqu'à l'Agent 3, ou docker start mlops-<projet>",
  "Décideur local MLX": "mlx_lm.server, ou MONITORING_DECIDER=groq:… dans agents/agent4/.env",
};

export function Sante() {
  const health = usePoll(() => api.health(), 15000);
  const down = (health.data ?? []).filter((h) => !h.ok);

  return (
    <main className="wrap page">
      <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "space-between", alignItems: "flex-end", gap: "12px 20px" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <span className="small muted">$ terminus sante</span>
          <h1 className="title">Tableau des services</h1>
        </div>
        <button className="btn primary" onClick={health.reload}>[ tout revérifier ]</button>
      </div>
      <ErrorLine error={health.error} />
      <div className="board">
        <table>
          <thead><tr><th scope="col">SERVICE</th><th scope="col">VOIE</th><th scope="col">ÉTAT</th><th scope="col">DÉTAIL</th></tr></thead>
          <tbody>
            {(health.data ?? []).map((h) => (
              <tr key={h.service}>
                <td>{h.service}</td>
                <td>{h.where}</td>
                <td>{h.ok ? "✓" : h.optional ? <span className="st-signal">! éteint</span> : <span className="st-signal">✗ arrêté</span>}</td>
                <td className="detail">{h.detail}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <section aria-labelledby="demo" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        <h2 id="demo"><Cmd opt="--avant-demo">terminus sante</Cmd></h2>
        <ol className="rule small" style={{ listStyle: "none", margin: 0, padding: "14px 0 0", display: "flex", flexDirection: "column", gap: 8 }}>
          {down.length === 0 && health.data && <li>✓ tout est prêt pour la démo</li>}
          {down.map((h) => (
            <li key={h.service}>
              <span className="st-signal">!</span> {h.service} : <code>{FIX[h.service] ?? h.detail}</code>
            </li>
          ))}
        </ol>
      </section>
    </main>
  );
}
