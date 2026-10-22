import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dp_model.service import load_artifact, score  # noqa: E402

EXAMPLE = {'age': 0.038076, 'sex': 0.05068, 'bmi': 0.061696, 'bp': 0.021872, 's1': -0.044223, 's2': -0.034821, 's3': -0.043401, 's4': -0.002592, 's5': 0.019907, 's6': -0.017646}
EXPECTED = 205.5  # progression de la maladie predite pour EXAMPLE


def test_score_example():
    artifact = load_artifact(Path(__file__).resolve().parents[1] / "model")
    assert score(artifact, EXAMPLE) == [pytest.approx(EXPECTED, abs=0.5)]
