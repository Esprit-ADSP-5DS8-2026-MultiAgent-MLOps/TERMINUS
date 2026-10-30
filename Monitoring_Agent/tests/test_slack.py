"""Slack : notifications avec boutons, suite dans le fil, approbation réservée, sans jamais appeler Slack."""

from __future__ import annotations

import json
from types import SimpleNamespace

from monitoring_agent.agent import notify
from monitoring_agent.runtime.slack import SlackBridge, to_mrkdwn


class FakeSlack:
    def __init__(self):
        self.posts, self.updates = [], []

    def chat_postMessage(self, **kw):
        self.posts.append(kw)
        return {"ts": f"{len(self.posts)}.0", "channel": kw["channel"]}

    def chat_update(self, **kw):
        self.updates.append(kw)


def run(status="awaiting_human", by=None):
    diag = SimpleNamespace(root_cause="covariate_shift", confidence=0.55, summary="x dérive.", action="rollback_model")
    pol = SimpleNamespace(final_action="rollback_model", mode="human", reasons=["action à haut risque"])
    return SimpleNamespace(id="D0004-abc", window_index=4, diagnosis=diag, policy=pol, status=status,
                           human={"by": by} if by else None)


def test_awaiting_message_has_buttons_and_the_follow_up_goes_in_its_thread(tmp_path, monkeypatch):
    fake = FakeSlack()
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("SLACK_CHANNEL", "C123")
    monkeypatch.setattr(notify, "_slack_client", lambda token: fake)

    assert notify.send(run(), "demo", "awaiting_human", tmp_path) == "slack"
    buttons = [e["action_id"] for b in fake.posts[0]["blocks"] if b["type"] == "actions" for e in b["elements"]]
    assert buttons[:2] == [notify.APPROVE, notify.REJECT]

    assert notify.send(run("executed", by="Mohamed (Slack)"), "demo", "executed", tmp_path) == "slack"
    assert fake.updates[0]["ts"] == "1.0"                      # le message d'origine perd ses boutons
    assert fake.posts[1]["thread_ts"] == "1.0"                 # la suite arrive dans son fil
    assert "validée par Mohamed (Slack)" in json.dumps(fake.posts[1]["blocks"], ensure_ascii=False)
    lines = (tmp_path / "notifications.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(x)["channel"] for x in lines] == ["slack", "slack"]


def test_slack_failure_falls_back_to_the_file(tmp_path, monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("SLACK_CHANNEL", "C123")

    def broken(token):
        raise RuntimeError("not_in_channel")

    monkeypatch.setattr(notify, "_slack_client", broken)
    assert notify.send(run(), "demo", "awaiting_human", tmp_path) == "fichier (envoi en échec)"


class FakeSupervisor:
    def __init__(self):
        self.calls, self.done = [], set()

    def resume(self, run_id, answer, by, comment):
        if run_id in self.done:
            raise RuntimeError("n'attend pas d'humain")
        self.done.add(run_id)
        self.calls.append((run_id, answer, by))
        return SimpleNamespace(policy=SimpleNamespace(final_action="rollback_model"), status="executed")


def test_only_listed_people_can_approve_and_the_first_answer_wins():
    sup = FakeSupervisor()
    bridge = SlackBridge(None, None, sup, settings={"SLACK_APPROVERS": "U1, U2"})
    assert "autorisées" in bridge.on_decision("U9", "intrus", "D1", "approve") and not sup.calls
    assert bridge.on_decision("U1", "mohamed", "D1", "approve").startswith("Approuvé")
    assert sup.calls == [("D1", "approve", "mohamed (Slack)")]
    assert "n'attend plus" in bridge.on_decision("U2", "collegue", "D1", "reject")


def test_answer_is_converted_to_slack_format_with_rating_buttons():
    bridge = SlackBridge(None, None, None, settings={})
    blocks = bridge.answer_blocks("ça va ?", {"id": "Q1", "answer": "Un **concept drift**.", "model": "m",
                                              "tools": [{"tool": "previous_decisions"}]})
    assert "*concept drift*" in blocks[0]["text"]["text"] and to_mrkdwn("**a**") == "*a*"
    assert [e["value"] for e in blocks[-1]["elements"]] == ["Q1", "Q1"]
    assert not bridge.configured
