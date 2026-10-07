"""Niveau 1b - Manifest mlops.yaml (C8) : absent -> confirmation humaine, invalide -> rejet."""
from __future__ import annotations

import yaml
from pydantic import ValidationError

from .contract import CLASSIFICATION_METRICS, REGRESSION_METRICS
from .fs import is_inside
from .models import Manifest, Severity as S


def run(ctx):
    path = ctx.root / ctx.cfg.manifest_name
    if not path.exists():
        ctx.manifest_state = "missing"
        ctx.add("C8", "C8.MISSING", S.CONFIRM, f"{ctx.cfg.manifest_name} absent : cible, tâche et métrique doivent être confirmées.",
                file=ctx.cfg.manifest_name,
                hint="Créez mlops.yaml (project.name/task, data.path/target, evaluation.primary_metric/min_threshold) ou faites-le proposer par l'agent d'analyse puis validez-le.")
        return
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, UnicodeDecodeError) as e:
        ctx.manifest_state = "invalid"
        ctx.add("C8", "C8.INVALID_YAML", S.BLOCKER, f"YAML illisible : {e}", file=ctx.cfg.manifest_name)
        return
    try:
        m = Manifest.model_validate(raw or {})
    except ValidationError as e:
        ctx.manifest_state = "invalid"
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"])
            ctx.add("C8", "C8.SCHEMA", S.BLOCKER, f"Champ `{loc}` : {err['msg']}", file=ctx.cfg.manifest_name,
                    hint="Voir le schéma Manifest (validator/models.py).")
        return

    ctx.manifest, ctx.manifest_state = m, "ok"
    task, metric, thr = m.project.task, m.evaluation.primary_metric, m.evaluation.min_threshold
    allowed = CLASSIFICATION_METRICS if task == "classification" else REGRESSION_METRICS
    if metric not in allowed:
        ctx.add("C8", "C8.METRIC_TASK_MISMATCH", S.BLOCKER, f"Métrique `{metric}` incompatible avec la tâche `{task}`.",
                file=ctx.cfg.manifest_name, hint=f"Métriques autorisées : {sorted(allowed)}")
    elif task == "classification" and not 0 <= thr <= 1:
        ctx.add("C8", "C8.THRESHOLD_RANGE", S.BLOCKER, "min_threshold doit être compris entre 0 et 1.", file=ctx.cfg.manifest_name)
    data_path = ctx.root / m.data.path
    if m.data.path.startswith(("/", "\\")) or ".." in m.data.path.replace("\\", "/").split("/") or not is_inside(data_path, ctx.root):
        ctx.add("C10", "C10.DATA_PATH_OUTSIDE", S.BLOCKER, "data.path doit être un chemin relatif à l'intérieur du projet.", file=ctx.cfg.manifest_name, evidence=m.data.path)
    for key in ("train", "predict"):
        ep = getattr(m.entrypoints, key)
        if not (ctx.root / ep).is_file():
            ctx.add("C8", "C8.ENTRYPOINT_MISSING", S.BLOCKER, f"Entrypoint `{key}` introuvable : {ep}", file=ctx.cfg.manifest_name)
    ctx.facts["manifest"] = m.model_dump()
