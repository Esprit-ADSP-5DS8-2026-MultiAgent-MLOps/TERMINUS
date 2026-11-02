import { useCallback, useEffect, useRef, useState } from "react";

/** Interroge l'API à intervalle régulier ; s'arrête quand la page est cachée. */
export function usePoll<T>(fetcher: () => Promise<T>, everyMs: number, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const ref = useRef(fetcher);
  ref.current = fetcher;

  const reload = useCallback(async () => {
    try {
      setData(await ref.current());
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    setData(null);
    reload();
    const id = window.setInterval(() => {
      if (!document.hidden) reload();
    }, everyMs);
    return () => window.clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [everyMs, reload, ...deps]);

  return { data, error, reload };
}

/** Nom de la personne qui décide, gardé dans ce navigateur. */
export function useOperator(): [string, (v: string) => void] {
  const [name, setName] = useState<string>(() => {
    try {
      return localStorage.getItem("terminus.operateur") ?? "";
    } catch {
      return "";
    }
  });
  const save = (v: string) => {
    setName(v);
    try {
      localStorage.setItem("terminus.operateur", v);
    } catch {
      /* stockage indisponible : le nom reste pour cette session */
    }
  };
  return [name, save];
}

export function seconds(s?: number | null): string {
  if (s == null) return "";
  return s < 60 ? `${s.toFixed(1)} s` : `${Math.round(s)} s`;
}

export function day(iso?: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return `${String(d.getDate()).padStart(2, "0")}/${String(d.getMonth() + 1).padStart(2, "0")} · ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

/** Barre en caractères, comme dans un terminal : ████████░░ */
export function bar(value: number, max = 1, width = 10): string {
  const n = Math.max(0, Math.min(width, Math.round((value / max) * width)));
  return "█".repeat(n) + "░".repeat(width - n);
}
