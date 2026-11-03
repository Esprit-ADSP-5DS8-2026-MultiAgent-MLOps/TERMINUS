// Client de l'API de Terminus (orchestrator/api.py). En développement, Vite relaie /api vers :8090.

export type StationStatus =
  | "passed" | "blocked" | "failed" | "stopped" | "refused" | "waiting" | "running" | "todo";

export interface Station {
  id: string;
  label: string;
  status: StationStatus;
  duration_s?: number | null;
  reasons?: string[];
  warnings?: string[];
  decided_by?: string;
  comment?: string;
  gate?: string;
  at?: string | null;
}

export interface Served {
  model_uri?: string | null;
  endpoint?: string | null;
  image?: string | null;
  pushed?: boolean | null;
  metric?: string | null;
  value?: number | null;
}

export interface RunSummary {
  run_id: string;
  project: string;
  project_dir?: string;
  status: string;
  waiting_gate?: string | null;
  retrain_count?: number;
  repair_count?: number;
  supervisor?: string;
  served?: Served;
  updated_at?: string | null;
  job?: Job | null;
}

export interface Gate {
  gate: string;
  key: string;
  question: string;
  run_id: string;
  file_to_review?: string | null;
  content: Record<string, unknown>;
  reasons?: string[];
  failed_step?: string;
  supervisor_explanation?: string;
  diff?: string;
  judge?: Record<string, unknown>;
  validator_v1?: Record<string, unknown>;
}

export interface Supervision {
  after: string;
  action: string;
  explanation: string;
  decider: string;
  fallback?: string | null;
  at?: string;
}

export interface Decision {
  gate: string;
  decision: "approve" | "reject";
  by: string;
  comment?: string;
  at?: string;
  run_id?: string;
  project?: string;
}

export interface Job { job: string; state: string; error: string | null }

export interface RunDetail extends RunSummary {
  track: Station[];
  gate: Gate | null;
  supervision: Supervision[];
  decisions: Decision[];
  results: Record<string, { status: string; reasons: string[]; warnings: string[]; duration_s: number }>;
  artifacts: Record<string, string>;
  job?: Job | null;
}

export interface Health { service: string; where: string; ok: boolean; detail: string; optional?: boolean }
export interface Project { name: string; path: string; manifest: boolean }
export interface Pending { run_id: string; project: string; gate: string; key: string; question: string }
export interface ModelEntry { name: string; champion?: string | null; latest?: number | null }
export interface ModelVersion {
  version: number;
  aliases: string[];
  created_at: string;
  metrics: Record<string, number>;
  model?: string | null;
  run_id?: string | null;
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...init });
  } catch {
    throw new ApiError(0, "API Terminus injoignable : lancer « python -m orchestrator api »");
  }
  const text = await res.text();
  if (!res.ok) {
    let detail = text;
    try {
      detail = JSON.parse(text).detail ?? text;
    } catch {
      /* texte brut */
    }
    throw new ApiError(res.status, typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  const type = res.headers.get("content-type") ?? "";
  return (type.includes("application/json") ? JSON.parse(text) : text) as T;
}

const post = <T,>(path: string, body: unknown) => call<T>(path, { method: "POST", body: JSON.stringify(body) });

export const api = {
  health: () => call<Health[]>("/api/health"),
  projects: () => call<Project[]>("/api/projects"),
  runs: () => call<RunSummary[]>("/api/runs"),
  run: async (id: string): Promise<RunDetail> => ({
    // même si l'API répond pendant que la ligne se forme : jamais de champ manquant (page blanche)
    track: [], gate: null, supervision: [], decisions: [], results: {}, artifacts: {},
    ...(await call<Partial<RunDetail> & { run_id: string; status: string }>(`/api/runs/${id}`)),
  } as RunDetail),
  file: (id: string, path: string) => call<string>(`/api/runs/${id}/file?path=${encodeURIComponent(path)}`),
  start: (project_dir: string, supervisor?: string) =>
    post<{ run_id: string }>("/api/runs", { project_dir, supervisor }),
  decide: (id: string, body: { decision: "approve" | "reject"; by: string; comment: string; key: string; publish?: boolean }) =>
    post<{ accepted: boolean }>(`/api/runs/${id}/decision`, body),
  monitor: (id: string) => post<{ accepted: boolean }>(`/api/runs/${id}/monitor`, {}),
  decisions: () => call<{ pending: Pending[]; recent: Decision[] }>("/api/decisions"),
  registry: () => call<ModelEntry[]>("/api/registry"),
  model: (name: string) => call<ModelVersion[]>(`/api/registry/${encodeURIComponent(name)}`),
  example: (id: string) => call<Record<string, unknown>>(`/api/runs/${id}/example`),
  predict: (payload: unknown) => post<{ status: number; body: unknown }>("/api/predict", payload),
};

export const GATE_TITLES: Record<string, string> = {
  manifest_confirmation: "manifeste proposé",
  champion_replacement: "nouveau champion",
  retraining: "réentraînement",
  failure_review: "étape en échec",
  repair_review: "réparation à valider",
  validator_override: "désaccord v1 / v2",
};
