from agents.agent1.nodes.detect_problem_type import detect_problem_type


def test_target_comes_from_stratify_not_first_column(tmp_path):
    (tmp_path / "train.py").write_text(
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "df = pd.read_csv('data.csv')\n"
        "a, b, c, d = train_test_split(df['text'], df['label'], test_size=0.2, stratify=df['label'])\n"
    )
    rows = [("please reset my password for account number " + str(i), "account" if i % 2 else "billing") for i in range(40)]
    csv = tmp_path / "data.csv"
    csv.write_text("text,label\n" + "\n".join(f'"{t}",{l}' for t, l in rows) + "\n")
    profile = {"rows": 40, "columns": {
        "text": {"unique_count": 40, "numeric": False, "value_counts": {}},
        "label": {"unique_count": 2, "numeric": False, "value_counts": {"account": 20, "billing": 20}},
    }}
    out = detect_problem_type({"workspace_path": str(tmp_path), "csv_path": str(csv), "profile": profile})
    assert out["target"] == "label"
    assert out["problem_type"] == "classification"
    assert out["features"].get("text") == ["text"]
    assert "text" not in out["features"]["categorical"]


def test_long_category_labels_are_not_free_text(tmp_path):
    labels = ["all credits at this bank paid back duly", "existing credits paid back duly till now",
              "delay in paying off in the past", "critical account or other credits elsewhere"]
    csv = tmp_path / "data.csv"
    csv.write_text("history,target\n" + "\n".join(f'"{labels[i % 4]}",{i % 2}' for i in range(60)) + "\n")
    profile = {"rows": 60, "columns": {
        "history": {"unique_count": 4, "numeric": False, "value_counts": {}},
        "target": {"unique_count": 2, "numeric": True, "value_counts": {"0": 30, "1": 30}},
    }}
    out = detect_problem_type({"workspace_path": str(tmp_path), "csv_path": str(csv), "profile": profile})
    assert out["features"]["categorical"] == ["history"]
    assert "text" not in out["features"]
