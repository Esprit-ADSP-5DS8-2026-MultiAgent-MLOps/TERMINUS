from train import train


def test_train_returns_metric(tmp_path):
    assert "roc_auc" in train("data.csv", str(tmp_path))
