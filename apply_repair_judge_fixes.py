#!/usr/bin/env python3
"""apply_repair_judge_fixes.py : applique 4 correctifs au Repair et au Judge, puis crée tests/test_repair_judge_fixes.py.

Usage (depuis la racine du projet) :  python apply_repair_judge_fixes.py
Sûr : refuse de s'exécuter deux fois (détecte les correctifs déjà présents) et ne modifie rien si une ancre est introuvable.
"""
import sys
from pathlib import Path

ROOT = Path.cwd()
EDITS = [
    # 1. Judge : un Repair en échec ne peut jamais être accepté
    ("repair_agent/judge/tools.py",
     "        r, out = self.repair, []\n        if self.flags:",
     "        r, out = self.repair, []\n"
     "        if r.failure or r.run_status != RunStatus.OK or r.status != \"REPAIRED\":\n"
     "            out.append(f\"le Repair n'est pas en succès complet (statut {r.status}, run_status {getattr(r.run_status, 'value', r.run_status)}\"\n"
     "                       + (f\", échec : {r.failure.kind.value}\" if r.failure else \"\") + \")\")\n"
     "        if self.flags:"),
    ("repair_agent/judge/tools.py",
     "from agentic_core.schemas import RepairReport, ScopeFlag, ValidatorReport",
     "from agentic_core.schemas import RepairReport, RunStatus, ScopeFlag, ValidatorReport"),
    # 2. Judge : artefacts binaires retirés et déclarés = pas un « changement scientifique »
    ("repair_agent/judge/tools.py",
     "            changes.append({\"file\": rel, \"change\": kind})\n",
     "            changes.append({\"file\": rel, \"change\": kind})\n"
     "            if kind == \"removed\" and rel in set(self.repair.removed_artifacts):\n"
     "                continue                      # artefact binaire non copié par le Repair : déclaré dans `removed_artifacts`\n"),
    # 3. Repair : revert_patch ne retire que les signalements des patchs annulés
    ("repair_agent/tools.py",
     "        self.flags: list[ScopeFlag] = []\n        self.last_checks: list = []",
     "        self.flags: list[ScopeFlag] = []\n"
     "        self._flag_owner: dict[int, str] = {}          # id(flag) -> patch_id qui l'a provoqué (signalements non bloquants)\n"
     "        self.last_checks: list = []"),
    ("repair_agent/tools.py",
     "        self.flags += scientific\n        out = {",
     "        self.flags += scientific\n"
     "        for f in scientific:\n"
     "            self._flag_owner[id(f)] = patch.patch_id\n"
     "        out = {"),
    ("repair_agent/tools.py",
     "        self.flags = [f for f in self.flags if f.blocked]   # les signalements de changement scientifique annulés disparaissent ; les tentatives bloquées restent",
     "        # seuls les signalements des patchs réellement annulés disparaissent ; les tentatives bloquées et les patchs encore appliqués restent signalés\n"
     "        self.flags = [f for f in self.flags if f.blocked or self._flag_owner.get(id(f)) not in set(undone)]"),
    # 4. Périmètre : préfixe « ./ » + tests/données imbriqués
    ("repair_agent/scope.py",
     "    p = path.lstrip(\"./\")\n",
     "    p = path.replace(\"\\\\\", \"/\")\n    while p.startswith(\"./\"):\n        p = p[2:]\n    p = p.lstrip(\"/\")\n"),
    ("policy/repair_policy.yaml",
     "  - \"tests/**\"\n  - \"test/**\"\n",
     "  - \"tests/**\"\n  - \"test/**\"\n"
     "  - \"**/tests/**\"                                   # tests imbriqués (pkg/tests/…)\n"
     "  - \"**/test/**\"\n  - \"**/*.npy\"\n  - \"**/*.npz\"\n  - \"**/*.feather\"\n  - \"**/test_*.ipynb\"\n"),
]

TEST_FILE = '"""Non-régression : 4 correctifs de logique du Repair et du Judge."""\nimport pytest\n\nfrom agentic_core.schemas import RepairReport, seal\nfrom policy import load_repair_policy\nfrom repair_agent.judge.tools import JudgeToolBox\nfrom repair_agent.patches import PatchManager\nfrom repair_agent.scope import ScopeGuard, is_protected\nfrom repair_agent.tools import RepairToolBox\nfrom repair_agent.workspace import Workspace\n\n\ndef _report(**kw):\n    base = dict(repair_id="r", created_at="x", project="p", attempt=1, status="REPAIRED", next_action="JUDGE",\n                tests=[{"name": "syntax", "status": "PASS", "command": ""}])\n    return seal(RepairReport(**{**base, **kw}))\n\n\n@pytest.fixture\ndef src(tmp_path):\n    s = tmp_path / "src"\n    s.mkdir()\n    (s / "train.py").write_text("from sklearn.ensemble import RandomForestClassifier\\nm = RandomForestClassifier(n_estimators=10)\\n")\n    (s / "main.py").write_text("print(\'a\')\\n")\n    return s\n\n\n# 1. un Repair en échec ne peut jamais être accepté\n@pytest.mark.parametrize("kw", [\n    dict(status="FAILED", next_action="ESCALATE", run_status="FAILED_CLOSED", failure={"kind": "INTERNAL", "message": "x"}),\n    dict(status="PARTIAL"), dict(status="ESCALATED", next_action="ESCALATE"),\n])\ndef test_judge_blocks_accept_when_repair_not_fully_successful(src, tmp_path, kw):\n    ws = Workspace.create(src, tmp_path / "w")\n    assert any("pas en succès" in b for b in JudgeToolBox(src, ws.root, _report(**kw)).blockers_for_accept())\n\n\ndef test_judge_does_not_block_a_clean_successful_repair(src, tmp_path):\n    ws = Workspace.create(src, tmp_path / "w")\n    assert not any("pas en succès" in b for b in JudgeToolBox(src, ws.root, _report()).blockers_for_accept())\n\n\n# 2. artefact binaire retiré volontairement : pas de faux « changement scientifique »\ndef test_declared_removed_binary_artifact_is_not_a_scope_flag(src, tmp_path):\n    (src / "model.pkl").write_bytes(b"\\x80\\x04\\x95\\xff\\xfe\\x00binary")\n    ws = Workspace.create(src, tmp_path / "w")\n    assert ws.removed_artifacts == ["model.pkl"]\n    declared = JudgeToolBox(src, ws.root, _report(removed_artifacts=["model.pkl"]))\n    assert declared.flags == []\n    undeclared = JudgeToolBox(src, ws.root, _report())                    # suppression NON déclarée : toujours signalée\n    assert any(f.file == "model.pkl" for f in undeclared.flags)\n\n\n# 3. revert_patch ne retire que les signalements des patchs annulés\ndef test_revert_keeps_scientific_flag_of_patches_still_applied(src, tmp_path):\n    ws = Workspace.create(src, tmp_path / "w")\n    pm = PatchManager(ws.root, ws.meta)\n    tb = RepairToolBox(ws, pm, ScopeGuard(load_repair_policy()), None)\n    tb.replace_in_file("train.py", "n_estimators=10", "n_estimators=500", "I1", "r")      # p0001 : scientifique\n    tb.replace_in_file("main.py", "print(\'a\')", "print(\'b\')", "I2", "r")                  # p0002 : anodin\n    tb.revert_patch("p0002")\n    assert [f.kind for f in tb.flags] == ["scientific_change"] and [p.patch_id for p in pm.applied] == ["p0001"]\n    tb.revert_patch("p0001")\n    assert tb.flags == []\n\n\ndef test_revert_of_scientific_patch_removes_its_flag_but_keeps_blocked_attempts(src, tmp_path):\n    ws = Workspace.create(src, tmp_path / "w")\n    tb = RepairToolBox(ws, PatchManager(ws.root, ws.meta), ScopeGuard(load_repair_policy()), None)\n    tb.replace_in_file("train.py", "n_estimators=10", "n_estimators=500", "I1", "r")\n    from agentic_core.toolkit import ToolError\n    with pytest.raises(ToolError):\n        tb.write_file("tests/test_x.py", "assert True\\n", "I1", "r")\n    tb.revert_patch("p0001")\n    assert [f.kind for f in tb.flags] == ["protected_path"]\n\n\n# 4. chemins protégés\n@pytest.mark.parametrize("path", ["tests/a.py", "./tests/a.py", "pkg/tests/helpers.py", "src/test/util.py", "x/y/test_a.py",\n                                  "data/raw.json", "d.csv", "emb.npy", "notebooks/test_x.ipynb", "tests\\\\a.py"])\ndef test_protected(path):\n    assert is_protected(path, load_repair_policy())\n\n\n@pytest.mark.parametrize("path", ["train.py", ".env", ".github/ci.yml", "pkg/model.py", "requirements.txt", "attests.py", "latest/model.py"])\ndef test_not_protected(path):\n    assert not is_protected(path, load_repair_policy())\n'


def main() -> int:
    if (ROOT / "tests" / "test_repair_judge_fixes.py").exists():
        print("déjà appliqué (tests/test_repair_judge_fixes.py existe) : rien à faire")
        return 0
    texts = {}
    for rel, old, new in EDITS:                       # 1re passe : tout vérifier avant d'écrire quoi que ce soit
        p = ROOT / rel
        if not p.is_file():
            print(f"ERREUR : {rel} introuvable. Lancez ce script depuis la racine du projet.")
            return 1
        t = texts.setdefault(rel, p.read_text(encoding="utf-8"))
        if t.count(old) != 1:
            print(f"ERREUR : ancre introuvable ou ambiguë dans {rel} (trouvée {t.count(old)} fois) : {old[:60]!r}\nAucun fichier modifié.")
            return 1
        texts[rel] = t.replace(old, new)
    for rel, t in texts.items():                      # 2e passe : écriture
        (ROOT / rel).write_text(t, encoding="utf-8")
        print(f"modifié : {rel}")
    (ROOT / "tests" / "test_repair_judge_fixes.py").write_text(TEST_FILE, encoding="utf-8")
    print("créé    : tests/test_repair_judge_fixes.py\nOK. Lancez : pytest -q tests/test_repair_judge_fixes.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
