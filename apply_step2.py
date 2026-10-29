"""Étape 2 : déduire la spec (données, cible, tâche, métrique) quand il n'y a pas de mlops.yaml, et la VÉRIFIER par Python.
À lancer à la racine du dépôt, APRÈS avoir copié validator_agent/spec_infer.py et tests/test_spec_infer.py. Idempotent."""
import json
from pathlib import Path

ROOT = Path(".")


def edit(path, pairs, marker):
    p = ROOT / path
    s = p.read_text(encoding="utf-8")
    if marker in s:
        print(f"= {path} : déjà patché")
        return
    for old, new in pairs:
        assert old in s, f"{path} : bloc introuvable (fichier modifié localement ?) :\n{old[:140]}"
        s = s.replace(old, new, 1)
    p.write_text(s, encoding="utf-8")
    print(f"+ {path}")


assert (ROOT / "validator_agent" / "spec_infer.py").is_file(), "copie d'abord validator_agent/spec_infer.py"

# ----------------------------------------------------------------------------- schéma : traçabilité de ce qui a été déduit
edit("agentic_core/schemas.py", [
    ("    min_threshold: Optional[float] = None\n    data_path: Optional[str] = None\n\n\nclass PolicyOutcome",
     "    min_threshold: Optional[float] = None\n    data_path: Optional[str] = None\n"
     "    deduced: dict[str, str] = Field(default_factory=dict)   # slot -> preuve, pour tout ce que le manifeste ne donnait pas\n\n\nclass PolicyOutcome"),
], marker="deduced: dict")

# ----------------------------------------------------------------------------- harnais : la métrique principale est LUE dans le dict renvoyé
edit("agentic_core/sandbox_exec.py", [
    ("def num(d):\n    return {k: v for k, v in (d or {}).items() if isinstance(v, (int, float)) and not isinstance(v, bool)}\n",
     '''def num(d):
    return {k: v for k, v in (d or {}).items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


# Ordre de préférence UNIQUEMENT parmi les clés réellement renvoyées par train() (jamais une métrique inventée).
METRIC_PREFERENCE = ("roc_auc", "auc", "f1", "f1_macro", "f1_weighted", "accuracy", "r2", "rmse", "mae", "mse")


def pick_metric(metrics: dict | None) -> tuple[str | None, str, list[str]]:
    """Métrique principale quand le manifeste n'en déclare pas : lue dans le dict renvoyé par l'entraînement réel."""
    nm = {k: v for k, v in num(metrics).items() if math.isfinite(v)}
    keys = list(nm)
    if not keys:
        return None, "train() ne renvoie aucune métrique numérique", []
    if len(keys) == 1:
        return keys[0], "seule métrique numérique renvoyée par train()", keys
    low = {str(k).lower(): k for k in keys}
    for pref in METRIC_PREFERENCE:
        if pref in low:
            return low[pref], f"plusieurs métriques renvoyées {keys} : préférence par convention", keys
    return keys[0], f"plusieurs métriques renvoyées {keys} : première clé du dict", keys
'''),
    ('''        metric_ok = isinstance(num(m1).get(primary_metric), (int, float)) and math.isfinite(num(m1)[primary_metric])
''', '''        if not primary_metric:                       # métrique non déclarée : on la LIT dans ce que train() a réellement renvoyé
            primary_metric, how, cands = pick_metric(m1)
            res["metric_discovery"] = {"chosen": primary_metric, "rule": how, "candidates": cands}
        res["primary_metric"] = primary_metric
        metric_ok = isinstance(num(m1).get(primary_metric), (int, float)) and math.isfinite(num(m1)[primary_metric])
'''),
], marker="pick_metric")

# ----------------------------------------------------------------------------- outil : primary_metric devient facultatif
edit("validator_agent/tools.py", [
    ('''            if not all(isinstance(p.get(k), str) and p.get(k) for k in ("data_path", "target", "primary_metric")):
                raise ToolError("harness exige data_path, target et primary_metric (déduis-les du manifeste ou du code, ou déclare EXEC.RUN NOT_CHECKED)")
            res = self.runner.run("harness", profile=self.contract.harness, data_path=p["data_path"], target=p["target"], primary_metric=p["primary_metric"])''',
     '''            if not all(isinstance(p.get(k), str) and p.get(k) for k in ("data_path", "target")):
                raise ToolError("harness exige data_path et target (déduis-les du manifeste ou du code, ou déclare EXEC.RUN NOT_CHECKED)")
            pm = p.get("primary_metric") if isinstance(p.get("primary_metric"), str) and p.get("primary_metric") else None   # None : lue dans le dict renvoyé par train()
            res = self.runner.run("harness", profile=self.contract.harness, data_path=p["data_path"], target=p["target"], primary_metric=pm)'''),
], marker="lue dans le dict renvoyé par train()")

# ----------------------------------------------------------------------------- contrôles
edit("validator_agent/checks.py", [
    ("from validator.entrypoints import DEFAULT_FILE, ROLES_BY_HARNESS\n",
     "from validator.entrypoints import DEFAULT_FILE, ROLES_BY_HARNESS\nfrom validator_agent.spec_infer import deduce_spec\n"),
    ("    out.entrypoints = eps.summary()\n",
     "    out.entrypoints = eps.summary()\n    out.spec = deduce_spec(tb, out.spec, eps)      # comble les trous du manifeste (ou son absence) avec des valeurs VÉRIFIÉES\n"),
    # STRUCT.MANIFEST
    ('''        elif not man:
            ev = "absent:mlops.yaml"
            tb.observe(ev)
            add(_mk(s, CheckStatus.FAIL, evidence=ev, note="manifeste absent → revue humaine"),
                _issue(s, "Aucun mlops.yaml/yml ; tâche/cible/métrique non confirmées.", evidence=ev, suggested="Ajouter un manifeste mlops.yaml."))
        elif out.spec and (out.spec.data_path or out.spec.target):
            add(_mk(s, CheckStatus.PASS, file=man.get("file"), note="manifeste ou champs déduits"))''',
     '''        elif not man and out.spec and out.spec.data_path and out.spec.target and {"data_path", "target"} <= set(out.spec.deduced):
            d = out.spec.deduced
            add(_mk(s, CheckStatus.WARNING, note="mlops.yaml absent : spec DÉDUITE puis vérifiée par Python · "
                    + " · ".join(f"{k}={getattr(out.spec, k)} ({d[k]})" for k in ("data_path", "target") if k in d),
                    origin="validator"))
        elif not man:
            ev = "absent:mlops.yaml"
            tb.observe(ev)
            why = (out.spec.deduced.get("target?") if out.spec else None) or "données ou cible non déductibles avec certitude"
            add(_mk(s, CheckStatus.FAIL, evidence=ev, note=f"manifeste absent et spec non déductible ({why}) → revue humaine"),
                _issue(s, "Aucun mlops.yaml/yml ; tâche/cible/métrique non confirmées.", evidence=ev, suggested="Ajouter un manifeste mlops.yaml."))
        elif out.spec and (out.spec.data_path or out.spec.target):
            filled = sorted(k for k in out.spec.deduced if not k.endswith("?"))
            if filled:
                add(_mk(s, CheckStatus.WARNING, file=man.get("file"), origin="validator",
                        note=f"manifeste incomplet : {filled} déduits et vérifiés par Python"))
            else:
                add(_mk(s, CheckStatus.PASS, file=man.get("file"), note="manifeste complet"))'''),
    # harnais : précondition et métrique lue
    ('''    if not spec_p or not (spec_p.data_path and spec_p.target and spec_p.primary_metric):
        upsert(_mk(s_exec, CheckStatus.NOT_CHECKED,
                   note="harnais impossible : data_path, target ou primary_metric inconnus", origin="validator"))''',
     '''    if not spec_p or not (spec_p.data_path and spec_p.target):
        upsert(_mk(s_exec, CheckStatus.NOT_CHECKED,
                   note="harnais impossible : data_path ou target inconnus (ni manifeste, ni déduction vérifiable)", origin="validator"))'''),
    ('''    raw = tb.call("run_in_sandbox", {"command": "harness", "data_path": spec_p.data_path,
                                     "target": spec_p.target, "primary_metric": spec_p.primary_metric})''',
     '''    raw = tb.call("run_in_sandbox", {"command": "harness", "data_path": spec_p.data_path, "target": spec_p.target,
                                     **({"primary_metric": spec_p.primary_metric} if spec_p.primary_metric else {})})'''),
    ('''    ok = bool(res.get("executed_ok"))
    outcome.executed_ok = ok
    ev = res.get("_summary")''', '''    ok = bool(res.get("executed_ok"))
    outcome.executed_ok = ok
    md = res.get("metric_discovery")
    if md and md.get("chosen") and not spec_p.primary_metric:       # métrique confirmée par l'exécution réelle de train()
        spec_p.primary_metric = md["chosen"]
        spec_p.deduced["primary_metric"] = f"clé du dict renvoyé par train() à l'exécution : {md['rule']}"
    ev = res.get("_summary")'''),
], marker="deduce_spec")

# ----------------------------------------------------------------------------- annotations : on RESSERRE ce que l'étape 2 doit tenir
ap = ROOT / "eval" / "real" / "annotations.json"
if ap.is_file():
    ann = json.loads(ap.read_text(encoding="utf-8"))
    c = ann.get("src_layout_no_manifest")
    if c and c.get("acceptable_statuses") != ["ACCEPTED"]:
        c["acceptable_statuses"] = ["ACCEPTED"]
        c.pop("known_gaps", None)
        c.pop("target_after", None)
        c["rationale"] += " [Étape 2 livrée : ACCEPTED exigé, avec avertissement STRUCT.MANIFEST (spec déduite et vérifiée).]"
        for k in ("notebook_only", "ambiguous_trainers"):
            if k in ann:
                ann[k]["known_gaps"] = {"STRUCT.MANIFEST": "spec non déductible avec certitude -> revue humaine (comportement voulu)"}
        ap.write_text(json.dumps(ann, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("+ eval/real/annotations.json (src_layout_no_manifest : ACCEPTED exigé)")

# ----------------------------------------------------------------------------- le nouveau champ change le schéma JSON exporté
import subprocess
import sys

r = subprocess.run([sys.executable, "-m", "agentic_core.export_schemas", "docs/schemas"], capture_output=True, text=True)
print("+ docs/schemas régénérés" if r.returncode == 0 else f"! export des schémas échoué : {r.stderr[-300:]}")

print("\nÉtape 2 appliquée. Lance : python -m pytest -q --ignore=tests/test_validator.py --ignore=tests/test_handoff.py --ignore=tests/test_integration_slow.py")
