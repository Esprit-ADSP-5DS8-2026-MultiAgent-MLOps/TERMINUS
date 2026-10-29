"""Judge : composant du Repair Agent qui évalue la QUALITÉ d'une réparation (lecture seule, preuves vérifiables).

Il ne décide jamais de l'admission : après ACCEPT (next_action=REVALIDATE), le Validator revalide et décide.
"""
from .evaluator import JudgeConfig, run_judge  # noqa: F401
