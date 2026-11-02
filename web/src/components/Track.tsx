import { useEffect, useRef, useState, type CSSProperties } from "react";
import type { Station } from "../api";
import { seconds } from "../hooks";
import { Terrain } from "./Bits";
import { Loco } from "./Loco";

const GLYPH: Record<Station["status"], string> = {
  passed: "✓", blocked: "✗", failed: "✗", stopped: "■", refused: "✗", waiting: "●", running: "▶", todo: "·",
};

function sub(s: Station): string {
  if (s.status === "waiting") return "attend un humain";
  if (s.status === "running") return "en cours";
  if (s.status === "todo") return "à venir";
  if (s.decided_by) return `${s.status === "refused" ? "refusé" : "validé"} par ${s.decided_by}`;
  if (s.status === "blocked" || s.status === "failed" || s.status === "stopped") {
    return (s.reasons?.[0] ?? "").slice(0, 70);
  }
  const warn = s.warnings?.length ? ` · ${s.warnings.length} avert.` : "";
  return `${seconds(s.duration_s)}${warn}`;
}

function nameStyle(s: Station): CSSProperties {
  if (["blocked", "failed", "refused"].includes(s.status))
    return { background: "var(--ink)", color: "var(--paper)", padding: "2px 6px" };
  if (s.status === "waiting") return { background: "var(--signal)", padding: "2px 6px" };
  if (s.status === "todo") return { color: "var(--muted)" };
  return {};
}

/** Où se trouve la loco : au signal ou à la gare en cours, sinon à la dernière gare franchie. */
function locoIndex(stations: Station[]): number {
  const live = stations.findIndex((s) => s.status === "waiting" || s.status === "running");
  if (live >= 0) return live;
  let last = 0;
  stations.forEach((s, i) => {
    if (s.status !== "todo") last = i;
  });
  return last;
}

export function Track({ stations, retrainCount = 0 }: { stations: Station[]; retrainCount?: number }) {
  const n = Math.max(stations.length, 1);
  const idx = locoIndex(stations);
  const prev = useRef(idx);
  const scroller = useRef<HTMLDivElement>(null);
  const [moving, setMoving] = useState(false);
  const here = stations[idx];

  useEffect(() => {
    if (prev.current === idx) return;
    prev.current = idx;
    setMoving(true);
    const t = window.setTimeout(() => setMoving(false), 1600);
    return () => window.clearTimeout(t);
  }, [idx]);

  // Sur petit écran le plan défile : on garde la loco visible.
  useEffect(() => {
    const el = scroller.current;
    if (!el || el.scrollWidth <= el.clientWidth) return;
    el.scrollLeft = (el.scrollWidth * (idx + 0.5)) / n - el.clientWidth / 2;
  }, [idx, n]);

  const label =
    here?.status === "waiting" ? `Locomotive arrêtée au ${here.label}` :
    here?.status === "running" ? `Locomotive en route : ${here.label}` : `Locomotive en gare : ${here?.label ?? ""}`;

  return (
    <>
      <div className="scroll-x" ref={scroller}>
        <div style={{ minWidth: n * 132 }}>
          {retrainCount > 0 && (
            <div className="retour" style={{ marginLeft: `${(100 / n) * 2}%` }}>
              <span>retour · Agent 4 → signal → Agent 2 · {retrainCount} réentraînement{retrainCount > 1 ? "s" : ""}</span>
            </div>
          )}
          <div className="loco-lane">
            <div className="loco-slot" style={{ left: `${((idx + 0.5) / n) * 100}%` }}>
              <Loco label={label} rolling={moving || here?.status === "running"} />
              {here?.status === "waiting" && (
                <div className="mast" aria-hidden="true">
                  <span className="lamp blink" />
                  <span className="pole" />
                </div>
              )}
            </div>
          </div>
          <div className="rail" aria-hidden="true" />
          <ol className="track" style={{ gridTemplateColumns: `repeat(${n}, minmax(0, 1fr))`, listStyle: "none",
            margin: "8px 0 0", padding: 0 }}>
            {stations.map((s, i) => (
              <li key={`${s.id}-${i}`} className="station">
                <span className="name" style={nameStyle(s)}>
                  {GLYPH[s.status]} {s.label.toUpperCase()}
                  {s.status === "running" && <span className="blink" aria-hidden="true">…</span>}
                </span>
                <span className="sub">{sub(s)}</span>
              </li>
            ))}
          </ol>
        </div>
      </div>
      <Terrain moving={moving} />
    </>
  );
}
