"""
Cible de monitoring : QUOI surveiller et OÙ.

`monitoring.json` est écrit une fois (commande `init`, ou par l'agent dans le pipeline à partir
de project_spec.json + deployment_result.json) puis relu par la gateway, le moniteur et le
générateur de configuration Prometheus / Grafana.

La référence est construite en faisant passer les données d'entraînement par l'API DÉPLOYÉE :
on obtient les prédictions et probabilités du vrai modèle servi, sans exécuter le code du projet
dans notre processus (il reste isolé dans son conteneur).
"""

from __future__ import annotations

import asyncio
import math
import os
from pathlib import Path

import httpx
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict

from ..datasets import PACKAGE_ROOT
from ..schemas import ColumnMapping

PRED_COL, PROBA_COL = "y_pred", "y_pred_proba"


def runs_dir() -> Path:
    return Path(os.environ.get("MONITORING_RUNS_DIR", PACKAGE_ROOT / "runs")).expanduser()


class MonitoringTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project: str
    upstream_url: str                       # API du modèle (Agent 3)
    mapping: ColumnMapping                  # features, cible, métrique, seuil
    window_size: int = 200
    response_prediction_field: str = "prediction"
    response_proba_field: str | None = "proba"
    reference_rows: int = 0
    model_uri: str = "unknown"              # ex. models:/credit-risk/3 (Agent 2), repris dans les demandes

    @property
    def dir(self) -> Path:
        return runs_dir() / self.project

    @property
    def reference_path(self) -> Path:
        return self.dir / "reference.parquet"

    @property
    def store_path(self) -> Path:
        return self.dir / "monitoring.db"

    def save(self) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / "monitoring.json"
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, project: str) -> MonitoringTarget:
        path = runs_dir() / project / "monitoring.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} absent : lancer d'abord `python -m monitoring_agent init ...`")
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Construction depuis un projet conforme au contrat (mlops.yaml + data.csv)
# ---------------------------------------------------------------------------


def mapping_from_project(project_dir: Path, data: pd.DataFrame) -> ColumnMapping:
    manifest = yaml.safe_load((project_dir / "mlops.yaml").read_text(encoding="utf-8"))
    target = manifest["data"]["target"]
    ev = manifest.get("evaluation", {})
    task = manifest["project"]["task"]
    return ColumnMapping(
        task=task,
        features=[c for c in data.columns if c != target],
        target=target,
        prediction=PRED_COL,
        prediction_proba=PROBA_COL if task == "classification" else None,
        timestamp="timestamp",
        id="request_id",
        primary_metric=ev.get("primary_metric", "roc_auc" if task == "classification" else "rmse"),
        min_threshold=ev.get("min_threshold"),
    )


def _json_row(row: dict) -> dict:
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()}


async def score_through_api(rows: list[dict], upstream_url: str, pred_field: str, proba_field: str | None,
                            concurrency: int = 16, timeout_s: float = 10.0) -> list[tuple[float | None, float | None]]:
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(base_url=upstream_url, timeout=timeout_s) as client:
        async def one(row: dict):
            async with sem:
                try:
                    r = await client.post("/predict", json=_json_row(row))
                    r.raise_for_status()
                    body = r.json()
                    return body.get(pred_field), body.get(proba_field) if proba_field else None
                except (httpx.HTTPError, ValueError):
                    return None, None
        return await asyncio.gather(*(one(r) for r in rows))


def build_target(project_dir: str | Path, upstream_url: str, name: str | None = None,
                 window_size: int = 200, data_file: str = "data.csv") -> tuple[MonitoringTarget, dict]:
    project_dir = Path(project_dir).resolve()
    data = pd.read_csv(project_dir / data_file)
    mapping = mapping_from_project(project_dir, data)
    target = MonitoringTarget(project=name or project_dir.name, upstream_url=upstream_url.rstrip("/"),
                              mapping=mapping, window_size=window_size)

    features = data[mapping.features].to_dict(orient="records")
    scored = asyncio.run(score_through_api(features, target.upstream_url, target.response_prediction_field,
                                           target.response_proba_field))
    ref = data.copy()
    ref[PRED_COL] = [p for p, _ in scored]
    if mapping.prediction_proba:
        ref[PROBA_COL] = [s for _, s in scored]
    failed = int(ref[PRED_COL].isna().sum())
    ref = ref[ref[PRED_COL].notna()].reset_index(drop=True)
    if ref.empty:
        raise RuntimeError(f"aucune prédiction obtenue de {upstream_url} : l'API du modèle est-elle lancée ?")

    target.reference_rows = len(ref)
    target.dir.mkdir(parents=True, exist_ok=True)
    ref.to_parquet(target.reference_path, index=False)
    target.save()
    return target, {"rows": len(data), "scored": len(ref), "failed": failed}
