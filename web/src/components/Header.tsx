import { Link, NavLink, useLocation } from "react-router";
import { api } from "../api";
import { usePoll } from "../hooks";
import { PassingLoco } from "./Loco";

const TABS = [
  { to: "/", label: "lignes", end: true },
  { to: "/pipeline", label: "pipeline" },
  { to: "/decisions", label: "décisions" },
  { to: "/agents", label: "agents" },
  { to: "/registre", label: "registre" },
  { to: "/sante", label: "santé" },
];

const MINI = ["base des runs", "Service Slack", "MLflow · registre", "Docker"];
const SHORT: Record<string, string> = {
  "base des runs": "base", "Service Slack": "slack", "MLflow · registre": "mlflow", Docker: "docker",
};

export function Header() {
  const { pathname } = useLocation();
  const decisions = usePoll(() => api.decisions(), 5000);
  const health = usePoll(() => api.health(), 20000);
  const pending = decisions.data?.pending.length ?? 0;
  const onPipeline = pathname.startsWith("/lignes/") || pathname === "/pipeline";

  return (
    <header className="topbar">
      {!onPipeline && <PassingLoco delay={3 + (pathname.length % 9)} />}
      <div className="wrap">
        <Link to="/" className="logo">TERMINUS</Link>
        <nav aria-label="Navigation principale" className="tabs">
          {TABS.map((t) => (
            <NavLink
              key={t.to}
              to={t.to}
              end={t.end}
              className={({ isActive }) =>
                `tab${isActive || (t.to === "/pipeline" && pathname.startsWith("/lignes/")) ? " active" : ""}`
              }
            >
              [ {t.label}
              {t.to === "/decisions" && pending > 0 && <> <span className="badge">{pending}</span></>} ]
            </NavLink>
          ))}
        </nav>
        <div className="status-mini" aria-label="État des services">
          {health.error && <span>api ✗</span>}
          {health.data
            ?.filter((h) => MINI.includes(h.service))
            .map((h) => (
              <span key={h.service}>
                {SHORT[h.service]} {h.ok ? "✓" : "✗"}
              </span>
            ))}
        </div>
      </div>
    </header>
  );
}
