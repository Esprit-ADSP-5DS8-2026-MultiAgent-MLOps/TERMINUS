"""Collecte de données de fine-tuning du Repair : projets sains, défauts, enregistreur, filtres, export, et chaîne complète (faux LLM + vrai Validator)."""
import json
import re
import shutil
from pathlib import Path

import pytest
from helpers import tc

from agentic_core.llm_client import LLMResponse, ScriptedClient
from finetune.bases import REQUIREMENTS, make_base
from finetune.collect import CollectConfig, collect
from finetune.export import export, load_runs, split_of
from finetune.faults import FAULTS, inject
from finetune.filters import FilterConfig, changed_lines, evaluate, invented_paths
from finetune.recorder import RecordingClient, rebuild_trajectory, redact, sanitize, secret_values


def snapshot(d: Path) -> dict:
    return {p.relative_to(d).as_posix(): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}


# ----------------------------------------------------------------------------- projets sains et défauts
def test_bases_are_deterministic_and_diverse(tmp_path):
    make_base(tmp_path / "a", 1), make_base(tmp_path / "b", 1)
    assert snapshot(tmp_path / "a") == snapshot(tmp_path / "b")
    metas = [make_base(tmp_path / f"s{i}", i) for i in range(10)]
    assert len({m["domain"] for m in metas}) >= 3 and len({tuple(m["models"]) for m in metas}) >= 4
    assert set(snapshot(tmp_path / "a")) == {"data.csv", "train.py", "preprocessing.py", "predict.py", "main.py", "mlops.yaml", "requirements.txt"}
    assert (tmp_path / "a" / "data.csv").read_text().splitlines()[0].endswith(metas[1]["target"])


@pytest.mark.parametrize("fid", list(FAULTS))
def test_each_fault_only_touches_its_declared_files_and_is_deterministic(tmp_path, fid):
    meta = make_base(tmp_path / "base", 3)
    for name in ("x", "y"):
        shutil.copytree(tmp_path / "base", tmp_path / name)
        inject(fid, tmp_path / name, 5, meta)
    assert snapshot(tmp_path / "x") == snapshot(tmp_path / "y")
    before, after = snapshot(tmp_path / "base"), snapshot(tmp_path / "x")
    touched = {f for f in before.keys() | after.keys() if before.get(f) != after.get(f)}
    assert touched and touched <= set(FAULTS[fid].gold_files)


def test_registry_is_repair_only_with_expected_checks():
    assert all(f.kind == "repair" and f.expected_checks and f.gold_files for f in FAULTS.values()) and len(FAULTS) >= 12


# ----------------------------------------------------------------------------- enregistreur
def _msg(role, content="", **kw):
    return {"role": role, "content": content, **kw}


def _call(i, name="read_file"):
    return {"id": f"c{i}", "function": {"name": name, "arguments": "{}"}}


def test_recording_client_records_and_delegates():
    inner = ScriptedClient([tc("list_files", {}, 1)], model="m1")
    c = RecordingClient(inner)
    c.chat([_msg("system", "s"), _msg("user", "u")], [{"type": "function"}])
    assert c.model == "m1" and len(c.records) == 1 and c.tools == [{"type": "function"}]
    assert c.records[0]["response"]["tool_calls"][0]["function"]["name"] == "list_files"


def test_rebuild_uses_first_appearance_so_compaction_is_undone():
    big = "x" * 400
    a1, a2 = _msg("assistant", tool_calls=[_call(1)]), _msg("assistant", tool_calls=[_call(2)])
    r1 = {"messages": [_msg("system"), _msg("user")], "response": a1, "usage": {"total_tokens": 10}, "model": "m"}
    r2 = {"messages": [_msg("system"), _msg("user"), a1, _msg("tool", big, tool_call_id="c1")], "response": a2, "usage": {"total_tokens": 20}, "model": "m"}
    archived = _msg("tool", "[archivé] sortie de read_file (400 car.) retirée", tool_call_id="c1")
    r3 = {"messages": [_msg("system"), _msg("user"), a1, archived, a2, _msg("tool", "ok", tool_call_id="c2")], "response": _msg("assistant", "fin"),
          "usage": {"total_tokens": 30}, "model": "m"}
    t = rebuild_trajectory([r1, r2, r3])
    assert t["messages"][3]["content"] == big and not t["compacted"] and t["steps"] == 3 and t["tokens"] == 60 and t["messages"][-1]["content"] == "fin"
    assert rebuild_trajectory([r1, {**r2, "messages": r2["messages"][:3] + [archived]}])["compacted"]      # déjà archivé à la 1re apparition : signalé
    assert rebuild_trajectory([]) is None


def test_sanitize_keeps_chat_format_only():
    m = sanitize({"role": "assistant", "content": None, "reasoning": "secret chain of thought", "tool_calls": [_call(1)]})
    assert set(m) == {"role", "content", "tool_calls"} and m["content"] == "" and m["tool_calls"][0]["type"] == "function"


def test_secrets_are_detected_and_redacted():
    env = {"GROQ_API_KEY": "gsk_abcdef1234567890", "PATH": "/usr/bin", "SHORT_TOKEN": "abc"}
    assert secret_values(env) == ["gsk_abcdef1234567890"]
    assert redact("clé gsk_abcdef1234567890 ici", secret_values(env)) == "clé [REDACTED] ici"


# ----------------------------------------------------------------------------- filtres
def good_run(**over):
    run = {"run_id": "0-unpinned_dep-0-s0", "base_seed": 0, "fault_id": "unpinned_dep", "baseline_checks": ["ENV.SANDBOX_ISOLATION"], "baseline_status": "ACCEPTED",
           "baseline_issue_ids": [], "gold": {"files": ["requirements.txt"], "changed_lines": 1},
           "repair": {"status": "REPAIRED", "failure": None, "tests": [{"name": "syntax", "status": "PASS"}], "scope_flags": [], "requires_human_approval": False,
                      "behavior_changed": False, "unresolved_ids": [], "files_changed": ["requirements.txt"], "added": 1, "removed": 1, "removed_artifacts": [],
                      "steps": 4, "invented_paths": []},
           "revalidation": {"status": "ACCEPTED", "issue_checks": []}, "judge": {"verdict": "ACCEPT", "independent_model": True, "failure": None},
           "trajectory": {"messages": [], "compacted": False}}
    for k, v in over.items():
        run[k] = {**run[k], **v} if isinstance(v, dict) and isinstance(run.get(k), dict) else v
    return run


def test_a_clean_run_passes_every_filter():
    assert evaluate(good_run()) == []


@pytest.mark.parametrize("over, reason", [
    ({"repair": {"status": "FAILED", "failure": "LLM_ERROR"}}, "repair_status:FAILED"),
    ({"repair": {"tests": [{"name": "import", "status": "FAIL"}]}}, "tests_failed"),
    ({"repair": {"scope_flags": ["scientific_change"]}}, "scope_flags"),
    ({"repair": {"behavior_changed": True}}, "behavior_changed"),
    ({"repair": {"files_changed": ["requirements.txt", "tests/test_a.py"]}}, "off_target_files"),
    ({"repair": {"added": 40, "removed": 40}}, "diff_too_large"),
    ({"repair": {"invented_paths": ["data/credit.csv"]}}, "invented_path"),
    ({"repair": {"steps": 30}}, "too_many_steps"),
    ({"repair": {"files_changed": []}}, "no_change"),
    ({"repair": {"status": "PARTIAL", "unresolved_ids": ["I2"]}}, "not_all_fixed"),
    ({"revalidation": None}, "not_revalidated"),
    ({"revalidation": {"status": "REPAIR_REQUIRED"}}, "revalidation:REPAIR_REQUIRED"),
    ({"revalidation": {"issue_checks": ["IO.PATHS"]}}, "issues_remaining"),
    ({"judge": None}, "no_judge"),
    ({"judge": {"verdict": "REVISE"}}, "judge:REVISE"),
    ({"judge": {"independent_model": False}}, "judge_not_independent"),
    ({"trajectory": {"compacted": True}}, "compacted_trajectory"),
    ({"trajectory": None}, "no_trajectory"),
])
def test_each_filter_rejects_its_failure(over, reason):
    assert any(w.startswith(reason) for w in evaluate(good_run(**over)))


def test_baseline_problems_of_the_healthy_project_are_not_held_against_the_repair():
    amb = good_run(repair={"status": "PARTIAL", "unresolved_ids": ["I2"]}, baseline_issue_ids=["I2"], baseline_status="NEEDS_REVIEW",
                   revalidation={"status": "NEEDS_REVIEW", "issue_checks": ["ENV.SANDBOX_ISOLATION"]})
    assert evaluate(amb) == []
    assert evaluate(good_run(judge=None), FilterConfig(require_judge=False)) == []


def test_invented_paths_flags_nonexistent_and_absolute_paths(tmp_path):
    (tmp_path / "data.csv").write_text("a\n")
    diff = '+BACKUP = "data/credit.csv"\n+X = "data.csv"\n+Y = "/home/user/x.csv"\n+  path: data.csv\n+  path: data/train.csv\n+++ b/train.py\n-OLD = "nope.csv"\n'
    assert invented_paths(diff, tmp_path) == ["/home/user/x.csv", "data/credit.csv", "data/train.csv"]


def test_changed_lines_counts_a_unified_diff():
    d = "--- a/train.py\n+++ b/train.py\n@@ -1,2 +1,2 @@\n-a\n+b\n c\n--- a/x\n+++ b/x\n+n\n"
    assert changed_lines(d) == (2, 1, ["train.py", "x"])


# ----------------------------------------------------------------------------- export
def mk_run(rid, base, fault="unpinned_dep", accepted=True, rejected_by=(), prompt=("S", "U"), steps=4, extra=None):
    msgs = [_msg("system", prompt[0]), _msg("user", prompt[1]), _msg("assistant", tool_calls=[_call(1)]), _msg("tool", "out", tool_call_id="c1"), _msg("assistant", extra or "fin")]
    return {"run_id": rid, "base_seed": base, "fault_id": fault, "accepted": accepted, "rejected_by": list(rejected_by),
            "repair": {"steps": steps, "added": 1, "removed": 1, "model": "m", "files_changed": ["requirements.txt"]}, "judge": {"verdict": "ACCEPT"},
            "trajectory": {"messages": msgs, "tools": [{"type": "function"}], "tokens": 100, "compacted": False}}


def test_split_is_by_base_project_and_roughly_balanced():
    assert split_of(7) == split_of(7)
    shares = [split_of(i) for i in range(2000)]
    assert 0.05 < shares.count("test") / 2000 < 0.15 and 0.05 < shares.count("val") / 2000 < 0.15


def test_export_builds_sft_and_preference_pairs_without_leaks(tmp_path):
    env = {"GROQ_API_KEY": "gsk_abcdef1234567890"}
    runs = [mk_run("1-unpinned_dep-0-s0", 1, steps=6), mk_run("1-unpinned_dep-0-s1", 1, steps=3, extra="clé gsk_abcdef1234567890"),
            mk_run("1-unpinned_dep-0-s2", 1, accepted=False, rejected_by=["invented_path:data/x.csv"], extra="mauvais"),
            mk_run("1-unpinned_dep-0-s3", 1, accepted=False, rejected_by=["compacted_trajectory"]),
            mk_run("1-unpinned_dep-0-s4", 1, accepted=False, rejected_by=["judge:REVISE"], prompt=("S", "AUTRE")),
            mk_run("2-typo_import-0-s0", 2, fault="typo_import"), mk_run("3-unpinned_dep-0-s0", 3, accepted=False, rejected_by=["no_judge"])]
    stats = export(runs, tmp_path, per_task=1, holdout_faults=["typo_import"], environ=env)
    assert stats["runs"] == 7 and stats["accepted"] == 3 and stats["tasks"] == 3 and stats["tasks_with_a_good_run"] == 2
    sft = {p.name: [json.loads(l) for l in p.read_text().splitlines()] for p in tmp_path.glob("sft_*.jsonl")}
    allsft = [x for v in sft.values() for x in v]
    assert {x["id"] for x in allsft} == {"1-unpinned_dep-0-s1", "2-typo_import-0-s0"}              # per_task=1 : la trajectoire la moins coûteuse (3 étapes)
    assert "sft_test_unseen.jsonl" in sft and [x["id"] for x in sft["sft_test_unseen.jsonl"]] == ["2-typo_import-0-s0"]
    assert all("gsk_abcdef1234567890" not in p.read_text() for p in tmp_path.glob("*.jsonl")) and any("[REDACTED]" in p.read_text() for p in tmp_path.glob("sft_*.jsonl"))
    dpo = [json.loads(l) for p in tmp_path.glob("dpo_*.jsonl") for l in p.read_text().splitlines()]
    assert [d["id"] for d in dpo] == ["1-unpinned_dep-0-s1__vs__1-unpinned_dep-0-s2"]              # ni trajectoire compactée, ni prompt différent
    assert dpo[0]["rejected_by"] == ["invented_path:data/x.csv"] and dpo[0]["prompt"][1]["content"] == "U" and dpo[0]["rejected"][-1]["content"] == "mauvais"
    assert stats["rejection_reasons"]["invented_path"] == 1 and (tmp_path / "stats.json").is_file()


# ----------------------------------------------------------------------------- chaîne complète : faux LLM + vrai Validator
class PromptScripted:
    """Faux Repair : lit le message du Validator pour citer les bons identifiants, puis déroule un plan."""
    model = "fake-repair"

    def __init__(self, plan):
        self.plan, self.i = plan, 0

    def chat(self, messages, tools=None):
        name, args = self.plan(messages[1]["content"])[self.i]
        self.i += 1
        return LLMResponse(message=tc(name, args, self.i), usage={"total_tokens": 100}, model=self.model)


def _issue(user, check):
    return re.search(rf'"issue_id": "(I\d+)", "check_id": "{re.escape(check)}"', user).group(1)


def fix_pins(user):
    i = _issue(user, "ENV.DEPENDENCIES_PINNED")
    return [("read_file", {"path": "requirements.txt"}), ("write_file", {"path": "requirements.txt", "content": REQUIREMENTS, "issue_id": i, "reason": "épingler"}),
            ("run_checks", {}), ("submit_repair", {"results": [{"issue_id": i, "outcome": "fixed", "patch_ids": ["p0001"], "explanation": "ok"}]})]


def invent_path(user):
    i = _issue(user, "IO.PATHS")
    old = re.search(r"/home/user/[\w.]+\.csv", user).group(0)
    return [("read_file", {"path": "train.py"}), ("replace_in_file", {"path": "train.py", "old_str": old, "new_str": "data/credit.csv", "issue_id": i, "reason": "chemin relatif"}),
            ("run_checks", {}), ("submit_repair", {"results": [{"issue_id": i, "outcome": "fixed", "patch_ids": ["p0001"], "explanation": "ok"}]})]


@pytest.fixture
def need_ml():
    pytest.importorskip("sklearn"), pytest.importorskip("pandas")


def run_collect(tmp_path, fault, plan, **kw):
    cfg = CollectConfig(out_dir=tmp_path / "out", bases=1, faults=[fault], env="current", filters=FilterConfig(require_judge=False), **kw)
    stats = collect(cfg, lambda: PromptScripted(plan), log=lambda *a, **k: None)
    return stats, load_runs(cfg.out_dir / "runs.jsonl"), cfg


def test_full_chain_keeps_a_genuine_repair_and_exports_it(need_ml, tmp_path):
    stats, runs, cfg = run_collect(tmp_path, "unpinned_dep", fix_pins)
    assert stats["runs"] == 1 and stats["accepted"] == 1, runs[0]["rejected_by"]
    r = runs[0]
    assert r["accepted"] and r["repair"]["files_changed"] == ["requirements.txt"] and r["revalidation"]["issue_checks"] == ["ENV.SANDBOX_ISOLATION"]
    t = r["trajectory"]
    assert [m["role"] for m in t["messages"]][:2] == ["system", "user"] and t["messages"][-1]["tool_calls"][0]["function"]["name"] == "submit_repair" and t["tools"]
    assert "unpinned" not in t["messages"][1]["content"] and "Projet «" in t["messages"][1]["content"]        # le prompt ne révèle pas le défaut injecté
    assert collect(cfg, lambda: PromptScripted(fix_pins), log=lambda *a, **k: None)["runs"] == 0              # reprise : rien n'est refait
    ex = export(load_runs(cfg.out_dir / "runs.jsonl"), tmp_path / "ex")
    assert sum(ex["sft"].values()) == 1


def test_full_chain_rejects_the_cosmetic_invented_path_repair(need_ml, tmp_path):
    stats, runs, _ = run_collect(tmp_path, "abs_path_dead", invent_path)
    assert stats["runs"] == 1 and stats["accepted"] == 0
    assert any(w.startswith("invented_path:data/credit.csv") for w in runs[0]["rejected_by"])


def test_unrouted_or_check_mode_never_calls_a_llm(need_ml, tmp_path):
    cfg = CollectConfig(out_dir=tmp_path / "c", bases=1, faults=["syntax_error"], env="current", check_faults=True)
    assert collect(cfg, None, log=lambda *a, **k: None)["runs"] == 0
    row = json.loads((tmp_path / "c" / "faults_check.jsonl").read_text().splitlines()[0])
    assert row["routed"] and "CODE.SYNTAX" in row["new_checks"]
