"""Non-régression : 4 correctifs de logique du Repair et du Judge."""
import pytest

from agentic_core.schemas import RepairReport, seal
from policy import load_repair_policy
from repair_agent.judge.tools import JudgeToolBox
from repair_agent.patches import PatchManager
from repair_agent.scope import ScopeGuard, is_protected
from repair_agent.tools import RepairToolBox
from repair_agent.workspace import Workspace


def _report(**kw):
    base = dict(repair_id="r", created_at="x", project="p", attempt=1, status="REPAIRED", next_action="JUDGE",
                tests=[{"name": "syntax", "status": "PASS", "command": ""}])
    return seal(RepairReport(**{**base, **kw}))


@pytest.fixture
def src(tmp_path):
    s = tmp_path / "src"
    s.mkdir()
    (s / "train.py").write_text("from sklearn.ensemble import RandomForestClassifier\nm = RandomForestClassifier(n_estimators=10)\n")
    (s / "main.py").write_text("print('a')\n")
    return s


# 1. un Repair en échec ne peut jamais être accepté
@pytest.mark.parametrize("kw", [
    dict(status="FAILED", next_action="ESCALATE", run_status="FAILED_CLOSED", failure={"kind": "INTERNAL", "message": "x"}),
    dict(status="PARTIAL"), dict(status="ESCALATED", next_action="ESCALATE"),
])
def test_judge_blocks_accept_when_repair_not_fully_successful(src, tmp_path, kw):
    ws = Workspace.create(src, tmp_path / "w")
    assert any("pas en succès" in b for b in JudgeToolBox(src, ws.root, _report(**kw)).blockers_for_accept())


def test_judge_does_not_block_a_clean_successful_repair(src, tmp_path):
    ws = Workspace.create(src, tmp_path / "w")
    assert not any("pas en succès" in b for b in JudgeToolBox(src, ws.root, _report()).blockers_for_accept())


# 2. artefact binaire retiré volontairement : pas de faux « changement scientifique »
def test_declared_removed_binary_artifact_is_not_a_scope_flag(src, tmp_path):
    (src / "model.pkl").write_bytes(b"\x80\x04\x95\xff\xfe\x00binary")
    ws = Workspace.create(src, tmp_path / "w")
    assert ws.removed_artifacts == ["model.pkl"]
    declared = JudgeToolBox(src, ws.root, _report(removed_artifacts=["model.pkl"]))
    assert declared.flags == []
    undeclared = JudgeToolBox(src, ws.root, _report())                    # suppression NON déclarée : toujours signalée
    assert any(f.file == "model.pkl" for f in undeclared.flags)


# 3. revert_patch ne retire que les signalements des patchs annulés
def test_revert_keeps_scientific_flag_of_patches_still_applied(src, tmp_path):
    ws = Workspace.create(src, tmp_path / "w")
    pm = PatchManager(ws.root, ws.meta)
    tb = RepairToolBox(ws, pm, ScopeGuard(load_repair_policy()), None)
    tb.replace_in_file("train.py", "n_estimators=10", "n_estimators=500", "I1", "r")      # p0001 : scientifique
    tb.replace_in_file("main.py", "print('a')", "print('b')", "I2", "r")                  # p0002 : anodin
    tb.revert_patch("p0002")
    assert [f.kind for f in tb.flags] == ["scientific_change"] and [p.patch_id for p in pm.applied] == ["p0001"]
    tb.revert_patch("p0001")
    assert tb.flags == []


def test_revert_of_scientific_patch_removes_its_flag_but_keeps_blocked_attempts(src, tmp_path):
    ws = Workspace.create(src, tmp_path / "w")
    tb = RepairToolBox(ws, PatchManager(ws.root, ws.meta), ScopeGuard(load_repair_policy()), None)
    tb.replace_in_file("train.py", "n_estimators=10", "n_estimators=500", "I1", "r")
    from agentic_core.toolkit import ToolError
    with pytest.raises(ToolError):
        tb.write_file("tests/test_x.py", "assert True\n", "I1", "r")
    tb.revert_patch("p0001")
    assert [f.kind for f in tb.flags] == ["protected_path"]


# 4. chemins protégés
@pytest.mark.parametrize("path", ["tests/a.py", "./tests/a.py", "pkg/tests/helpers.py", "src/test/util.py", "x/y/test_a.py",
                                  "data/raw.json", "d.csv", "emb.npy", "notebooks/test_x.ipynb", "tests\\a.py"])
def test_protected(path):
    assert is_protected(path, load_repair_policy())


@pytest.mark.parametrize("path", ["train.py", ".env", ".github/ci.yml", "pkg/model.py", "requirements.txt", "attests.py", "latest/model.py"])
def test_not_protected(path):
    assert not is_protected(path, load_repair_policy())
