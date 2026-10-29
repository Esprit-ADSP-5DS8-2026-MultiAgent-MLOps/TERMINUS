"""Étape 1 : résoudre les entrypoints (manifeste -> AST -> convention) au lieu des noms figés.
À lancer à la racine du dépôt, APRÈS avoir créé validator/entrypoints.py. Idempotent : relançable sans risque."""
from pathlib import Path

ROOT = Path(".")


def edit(path, pairs, marker):
    p = ROOT / path
    s = p.read_text(encoding="utf-8")
    if marker in s:
        print(f"= {path} : déjà patché")
        return
    for old, new in pairs:
        assert old in s, f"{path} : bloc introuvable (fichier modifié localement ?) :\n{old[:120]}"
        s = s.replace(old, new, 1)
    p.write_text(s, encoding="utf-8")
    print(f"+ {path}")


# ----------------------------------------------------------------------------- harnais : charge des module:fonction résolus
HELPERS = '''
LEGACY = {
    "train": {"module": "train", "function": "train", "file": "train.py", "sys_path": "."},
    "load_model": {"module": "predict", "function": "load_model", "file": "predict.py", "sys_path": "."},
    "predict": {"module": "predict", "function": "predict", "file": "predict.py", "sys_path": "."},
}


def _entry(man, role):
    return (man.get("entrypoints") or {}).get(role) or LEGACY[role]


def _load_entry(project, e):
    """Importe le module de l'entrypoint (sys.path = dossier d'import résolu + racine du projet) et retourne la fonction."""
    for d in (e.get("sys_path") or "."), ".":
        full = os.path.normpath(os.path.join(project, d))
        if full not in sys.path:
            sys.path.insert(0, full)
    if e.get("module"):
        mod = importlib.import_module(e["module"])
    else:  # chemin non importable (tiret, etc.) : chargement par fichier
        spec = importlib.util.spec_from_file_location("_entry_" + os.path.basename(e["file"]).replace(".", "_"), os.path.join(project, e["file"]))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return getattr(mod, e["function"])

'''
IMPORTS = ("import importlib\nimport json", "import importlib\nimport importlib.util\nimport json")

edit("validator/harness.py", [
    IMPORTS,
    ("\n\ndef main():", "\n" + HELPERS + "\ndef main():"),
    ('''        train_mod = step("import_train", lambda: importlib.import_module("train"))
        pred_mod = step("import_predict", lambda: importlib.import_module("predict"))
''', '''        train_fn = step("import_train", lambda: _load_entry(project, _entry(man, "train")))

        def _load_pred():
            return _load_entry(project, _entry(man, "load_model")), _load_entry(project, _entry(man, "predict"))
        pred = step("import_predict", _load_pred)
        load_fn, predict_fn = pred if pred else (None, None)
'''),
    ('step("train", lambda: train_mod.train(data_path, a.out1)) if train_mod else None', 'step("train", lambda: train_fn(data_path, a.out1)) if train_fn else None'),
    ('step("load_model", lambda: pred_mod.load_model(str(artifact))) if (pred_mod and artifact.exists()) else None', 'step("load_model", lambda: load_fn(str(artifact))) if (load_fn and artifact.exists()) else None'),
    ('pred_mod.predict(model, X.head(5))', 'predict_fn(model, X.head(5))'),
    ('pred_mod.predict(model, X.iloc[0].to_dict())', 'predict_fn(model, X.iloc[0].to_dict())'),
    ('if train_mod and metrics is not None and not a.no_determinism:', 'if train_fn and metrics is not None and not a.no_determinism:'),
    ('step("train_2", lambda: train_mod.train(data_path, a.out2))', 'step("train_2", lambda: train_fn(data_path, a.out2))'),
], marker="_load_entry")

HELPERS_TRAIN_ONLY = HELPERS.replace('''    "load_model": {"module": "predict", "function": "load_model", "file": "predict.py", "sys_path": "."},
    "predict": {"module": "predict", "function": "predict", "file": "predict.py", "sys_path": "."},
''', "")
edit("agentic_core/harness_train_only.py", [
    IMPORTS,
    ("\n\ndef main():", "\n" + HELPERS_TRAIN_ONLY + "\ndef main():"),
    ('mod = step("import_train", lambda: importlib.import_module("train"))\n        metrics = step("train", lambda: mod.train(data_path, a.out1)) if mod else None',
     'train_fn = step("import_train", lambda: _load_entry(project, _entry(man, "train")))\n        metrics = step("train", lambda: train_fn(data_path, a.out1)) if train_fn else None'),
    ('if mod and metrics is not None and not a.no_determinism:\n            m2 = step("train_2", lambda: mod.train(data_path, a.out2))',
     'if train_fn and metrics is not None and not a.no_determinism:\n            m2 = step("train_2", lambda: train_fn(data_path, a.out2))'),
], marker="_load_entry")

# ----------------------------------------------------------------------------- SandboxRunner : résolution par Python sur la copie
edit("agentic_core/sandbox_exec.py", [
    ("from validator.check_dynamic import _ignore, _snapshot\n",
     "from validator.check_dynamic import _ignore, _snapshot\nfrom validator.entrypoints import ROLES_BY_HARNESS, read_manifest_body, resolve_entrypoints\nfrom validator.fs import walk_project\n"),
    ("    def _harness(self, work, proj, profile, data_path, target, primary_metric, check_determinism=True):\n",
     "    def _harness(self, work, proj, profile, data_path, target, primary_metric, check_determinism=True, entrypoints=None):\n"),
    ('''        shutil.copy(h["script"], work / "harness" / "_harness.py")
        (work / "manifest.json").write_text(json.dumps({"data_path": data_path, "target": target, "task": "n/a", "primary_metric": primary_metric}))
        before = _snapshot(proj)
        main = self._sb.run(["python", "main.py"], work, proj, {"OUTPUT_DIR": "{W}/out_main"}, self.timeout) if h["run_main"] and (proj / "main.py").is_file() else None
''', '''        shutil.copy(h["script"], work / "harness" / "_harness.py")
        # entrypoints : manifeste d'abord, AST ensuite (validator/entrypoints.py). Résolu PAR PYTHON sur la copie du projet.
        roles = ROLES_BY_HARNESS.get(profile, ("train",))
        files, _links = walk_project(proj)
        eps = resolve_entrypoints(proj, files, read_manifest_body(proj), profile)
        if entrypoints is None and eps.ambiguous(roles):
            return {"error": "entrypoints ambigus : " + "; ".join(f"{p.role} -> {p.candidates[:4]}" for p in eps.ambiguous(roles)),
                    "not_run": True, "entrypoints": eps.summary()}
        ep_map = entrypoints if entrypoints is not None else eps.harness_map(roles)
        (work / "manifest.json").write_text(json.dumps({"data_path": data_path, "target": target, "task": "n/a", "primary_metric": primary_metric,
                                                         "entrypoints": ep_map}))
        before = _snapshot(proj)
        main_file = eps.main if eps.main and (proj / eps.main).is_file() else None
        main = self._sb.run(["python", main_file], work, proj, {"OUTPUT_DIR": "{W}/out_main"}, self.timeout) if h["run_main"] and main_file else None
'''),
    ('''        res = {"command": "harness", "profile": profile, "sandbox": {"backend": self._sb.name, "network_isolated": self._sb.network_isolated},
               "contract_steps": steps,''', '''        res = {"command": "harness", "profile": profile, "sandbox": {"backend": self._sb.name, "network_isolated": self._sb.network_isolated},
               "entrypoints": {r: f"{e.get('file')}:{e.get('function')}" for r, e in ep_map.items()},
               "contract_steps": steps,'''),
], marker="resolve_entrypoints")

# ----------------------------------------------------------------------------- boîte à outils : résolution mise en cache + faits
edit("validator_agent/tools.py", [
    ("        self._facts: dict | None = None\n        self.tools.update(", "        self._facts: dict | None = None\n        self._eps = None\n        self.tools.update("),
    ('''    def facts(self) -> dict:
        if self._facts is None:
            self._facts = static_facts(self.root)
        return self._facts
''', '''    def entrypoint_resolution(self):
        """Entrypoints résolus par Python (manifeste -> AST -> convention). Jamais décidés par le LLM."""
        if self._eps is None:
            from validator.entrypoints import read_manifest_body, resolve_entrypoints
            self._eps = resolve_entrypoints(self.root, sorted(self.known_files), read_manifest_body(self.root), self.contract.harness)
        return self._eps

    def facts(self) -> dict:
        if self._facts is None:
            self._facts = static_facts(self.root)
            self._facts["entrypoints"] = self.entrypoint_resolution().summary()
        return self._facts
'''),
], marker="entrypoint_resolution")

# ----------------------------------------------------------------------------- faits : src/ et sous-paquets ne sont pas des paquets tiers
edit("agentic_core/facts.py", [
    ("    local = {Path(p).stem for p in srcs} | {Path(p).parts[0] for p in srcs if len(Path(p).parts) > 1}\n",
     "    local = {Path(p).stem for p in srcs} | {part for p in srcs for part in Path(p).parts[:-1]}   # tous les dossiers : src/, src/pkg/…\n"),
], marker="tous les dossiers")

# ----------------------------------------------------------------------------- contrôles déterministes
p = ROOT / "validator_agent" / "checks.py"
s = p.read_text(encoding="utf-8")
if "required_files_for" in s:
    print("= validator_agent/checks.py : déjà patché")
else:
    def rep(old, new):
        global s
        assert old in s, f"checks.py : bloc introuvable :\n{old[:120]}"
        s = s.replace(old, new, 1)

    def between(start, end, new):
        global s
        a, b = s.index(start), s.index(end)
        s = s[:a] + new + s[b:]

    rep("from validator.check_structure import _is_pinned, parse_requirements\n",
        "from validator.check_structure import _is_pinned, parse_requirements\nfrom validator.entrypoints import DEFAULT_FILE, ROLES_BY_HARNESS\n")
    rep("    security_blocked: bool = False\n    executed_ok: bool | None = None\n\n\ndef _mk(",
        "    security_blocked: bool = False\n    executed_ok: bool | None = None\n    entrypoints: dict | None = None\n\n\ndef _mk(")
    rep("def _is_numeric_const(node: ast.AST) -> bool:", '''FIXED_ROLE_FILES = frozenset(DEFAULT_FILE.values()) | {"main.py"}


def required_files_for(contract: AdmissionContract, eps) -> tuple[list[str], list]:
    """Fichiers requis DÉDUITS des entrypoints résolus (src/, fit(), manifeste…). Le contrat ne garde que ce qui n'est pas un
    fichier de rôle (ex. requirements.txt). Sans résolution possible : noms par défaut du contrat. Retourne (fichiers, ambiguïtés)."""
    base = list(required_files_of(contract))
    roles = ROLES_BY_HARNESS.get(contract.harness or "")
    if eps is None or not roles:
        return base, []
    derived = eps.required_files(roles)
    return derived + [f for f in base if f not in FIXED_ROLE_FILES and f not in derived], eps.ambiguous(roles)


def _is_numeric_const(node: ast.AST) -> bool:''')
    rep('def inspect_train_metrics(text: str) -> tuple[CheckStatus, str | None, str]:',
        'def inspect_train_metrics(text: str, fn_name: str = "train") -> tuple[CheckStatus, str | None, str]:')
    rep('''    train_fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train"), None)
    if train_fn is None:
        return CheckStatus.NOT_CHECKED, None, "pas de fonction train()"''',
        '''    train_fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn_name), None)
    if train_fn is None:
        return CheckStatus.NOT_CHECKED, None, f"pas de fonction {fn_name}()"''')
    rep('''    declared, dep_rows = _deps(tb)

    def spec(cid):
        return contract.check(cid)
''', '''    declared, dep_rows = _deps(tb)
    eps = tb.entrypoint_resolution()
    out.entrypoints = eps.summary()
    roles = ROLES_BY_HARNESS.get(contract.harness or "", ())

    def spec(cid):
        return contract.check(cid)
''')

    between("    # --- STRUCT.REQUIRED_FILES", "    # --- STRUCT.LIMITS", '''    # --- STRUCT.REQUIRED_FILES : déduits des entrypoints résolus (manifeste -> AST -> convention)
    if (s := spec("STRUCT.REQUIRED_FILES")):
        req, amb = required_files_for(contract, eps)
        missing = [n for n in req if n not in files]
        invalid = [p for p in eps.problems if p.kind == "invalid" and p.role in roles]
        if not req:
            add(_mk(s, CheckStatus.NOT_CHECKED, note="contrat sans required_files", origin="validator"))
        elif invalid:
            ev = "; ".join(f"entrypoint invalide:{p.detail}" for p in invalid)
            tb.observe(ev)
            add(_mk(s, CheckStatus.FAIL, evidence=ev, note=ev),
                _issue(s, f"Entrypoint déclaré invalide : {invalid[0].detail}", evidence=ev, suggested="Corriger `entrypoints` dans mlops.yaml (chemin relatif au projet)."))
        elif missing:
            ev = "; ".join(f"absent:{n}" for n in missing)
            tb.observe(ev)
            add(_mk(s, CheckStatus.FAIL, evidence=ev, note=f"absents : {missing}"),
                _issue(s, f"Fichiers requis absents : {missing}", evidence=ev, suggested=f"Ajouter {missing[0]} (ou déclarer le bon chemin dans `entrypoints` de mlops.yaml)."))
        elif amb:
            ev = "; ".join(f"{p.role}: {', '.join(p.candidates[:4])}" for p in amb)
            tb.observe(ev)
            add(_mk(s, CheckStatus.NOT_CHECKED, evidence=ev, origin="validator",
                    note="entrypoints ambigus : déclarez-les dans mlops.yaml (entrypoints.train / entrypoints.predict) ou revue humaine"))
        else:
            add(_mk(s, CheckStatus.PASS, note=f"présents : {req} · entrypoints {', '.join(f'{r}={e.file}:{e.function} ({e.source})' for r, e in eps.entries.items())}"))

''')

    between('    if (s := spec("ML.TRAIN_INTERFACE")):\n        train_fns', "    # --- ML.METRICS_GENUINE", '''    if (s := spec("ML.TRAIN_INTERFACE")):
        e, pr = eps.entries.get("train"), eps.problem("train")
        if pr and pr.kind == "ambiguous":
            ev = ", ".join(pr.candidates[:4])
            tb.observe(ev)
            add(_mk(s, CheckStatus.NOT_CHECKED, evidence=ev, origin="validator", note=f"fonction d'entraînement ambiguë : {pr.detail}"))
        elif e is None:
            f = (pr.file if pr and pr.file else DEFAULT_FILE["train"])
            if f not in files:
                ev = f"absent:{f}"
                tb.observe(ev)
                add(_mk(s, CheckStatus.FAIL, evidence=ev, file=f), _issue(s, f"{f} absent.", evidence=ev, file=f))
            else:
                ev = j(fns.get(f) or [])
                tb.observe(ev)
                add(_mk(s, CheckStatus.FAIL, evidence=ev, file=f),
                    _issue(s, f"{f} n'expose pas de fonction train(data_path, output_dir) ({pr.detail if pr else 'introuvable'}).", evidence=ev, file=f))
        else:
            params = e.params
            named = "data_path" in params and "output_dir" in params
            declared_ok = e.source == "manifest" and ("*" in params or len(params) >= 2)
            if named or declared_ok:
                add(_mk(s, CheckStatus.PASS, file=e.file, note=f"{e.function}{tuple(params)} résolu par {e.source}"))
            else:
                ev = j({"name": e.function, "params": params, "line": e.line})
                tb.observe(ev)
                add(_mk(s, CheckStatus.FAIL, evidence=ev, file=e.file, line=e.line),
                    _issue(s, f"{e.function}() doit exposer data_path et output_dir (params={params}).", evidence=ev, file=e.file,
                           suggested=f"def {e.function}(data_path, output_dir): ..."))

    if (s := spec("ML.PREDICT_INTERFACE")):
        le, pe = eps.entries.get("load_model"), eps.entries.get("predict")
        amb = [p for p in eps.problems if p.kind == "ambiguous" and p.role in ("load_model", "predict")]
        if amb:
            ev = ", ".join(amb[0].candidates[:4])
            tb.observe(ev)
            add(_mk(s, CheckStatus.NOT_CHECKED, evidence=ev, origin="validator", note=f"couple load_model/predict ambigu : {amb[0].detail}"))
        elif le and pe:
            add(_mk(s, CheckStatus.PASS, file=pe.file, note=f"load_model={le.file}:{le.function} · predict={pe.file}:{pe.function} ({pe.source})"))
        else:
            pr = eps.problem("predict") or eps.problem("load_model")
            f = (pr.file if pr and pr.file else DEFAULT_FILE["predict"])
            if f not in files:
                ev = f"absent:{f}"
                tb.observe(ev)
                add(_mk(s, CheckStatus.FAIL, evidence=ev, file=f), _issue(s, f"{f} absent.", evidence=ev, file=f))
            else:
                ev = j(fns.get(f) or [])
                tb.observe(ev)
                add(_mk(s, CheckStatus.FAIL, evidence=ev, file=f), _issue(s, "Le module de prédiction doit exposer load_model et predict.", evidence=ev, file=f))

''')

    between("    # --- ML.METRICS_GENUINE", "    # --- ML.REPRODUCIBILITY", '''    # --- ML.METRICS_GENUINE : ce que la fonction d'entraînement RÉSOLUE renvoie (un import sklearn.metrics ne prouve rien)
    if (s := spec("ML.METRICS_GENUINE")):
        te = eps.entries.get("train")
        tfile, tfn = (te.file, te.function) if te else (DEFAULT_FILE["train"], "train")
        p = tb.root / tfile
        text = p.read_text(encoding="utf-8", errors="replace") if p.is_file() else ""
        status, ev, note = inspect_train_metrics(text, tfn) if text else (
            CheckStatus.NOT_CHECKED, None, f"{tfile} absent ou entrypoint d'entraînement non résolu")
        if ev:
            tb.observe(ev)
        origin = "candidate" if status == CheckStatus.FAIL else "validator"
        add(_mk(s, status, evidence=ev, file=tfile if text else None, note=note, origin=origin),
            _issue(s, "Les métriques renvoyées par l'entraînement sont des littéraux, pas un calcul sur des données.",
                   evidence=ev, file=tfile) if status == CheckStatus.FAIL else None)

''')

    rep('''    steps = res.get("contract_steps") or {}
    if (s := spec("ML.TRAIN_INTERFACE")):''', '''    steps = res.get("contract_steps") or {}
    ep_files = {r: str(v).partition(":")[0] for r, v in (res.get("entrypoints") or {}).items()}
    train_file, predict_file = ep_files.get("train", "train.py"), ep_files.get("predict", "predict.py")
    if (s := spec("ML.TRAIN_INTERFACE")):''')
    rep('upsert(_mk(s, CheckStatus.PASS, source="harness", file="train.py"))', 'upsert(_mk(s, CheckStatus.PASS, source="harness", file=train_file))')
    rep('upsert(_mk(s, CheckStatus.FAIL, evidence=evs, source="harness", file="train.py"),\n                       _issue(s, "train() a échoué dans le harnais.", evidence=evs, file="train.py"))',
        'upsert(_mk(s, CheckStatus.FAIL, evidence=evs, source="harness", file=train_file),\n                       _issue(s, "train() a échoué dans le harnais.", evidence=evs, file=train_file))')
    rep('upsert(_mk(s, CheckStatus.PASS, source="harness", file="predict.py"))', 'upsert(_mk(s, CheckStatus.PASS, source="harness", file=predict_file))')
    rep('upsert(_mk(s, CheckStatus.FAIL, evidence=evs, source="harness", file="predict.py"),\n                       _issue(s, "Interface predict échouée dans le harnais.", evidence=evs, file="predict.py"))',
        'upsert(_mk(s, CheckStatus.FAIL, evidence=evs, source="harness", file=predict_file),\n                       _issue(s, "Interface predict échouée dans le harnais.", evidence=evs, file=predict_file))')
    rep('"limits": list(out.limits), "security_blocked": out.security_blocked,\n            "executed_ok": out.executed_ok}',
        '"limits": list(out.limits), "security_blocked": out.security_blocked,\n            "executed_ok": out.executed_ok, "entrypoints": out.entrypoints}')
    rep('        security_blocked=bool(data.get("security_blocked")),\n        executed_ok=data.get("executed_ok"))',
        '        security_blocked=bool(data.get("security_blocked")),\n        executed_ok=data.get("executed_ok"), entrypoints=data.get("entrypoints"))')
    p.write_text(s, encoding="utf-8")
    print("+ validator_agent/checks.py")

# ----------------------------------------------------------------------------- agent : extrait de l'entrypoint résolu
edit("validator_agent/agent.py", [
    ('''    p = tb.root / "train.py"
    if not p.is_file():
        return ""''', '''    te = tb.entrypoint_resolution().entries.get("train")
    p = tb.root / (te.file if te else "train.py")
    if not p.is_file():
        return ""'''),
    ('f"Extrait train.py (non fiable, untrusted) :\\n{excerpt}\\n\\n"', 'f"Extrait de l\'entrypoint d\'entraînement (non fiable, untrusted) :\\n{excerpt}\\n\\n"'),
], marker="entrypoint_resolution")

print("\nÉtape 1 appliquée. Lance : python -m pytest -q --ignore=tests/test_validator.py --ignore=tests/test_handoff.py --ignore=tests/test_integration_slow.py")
