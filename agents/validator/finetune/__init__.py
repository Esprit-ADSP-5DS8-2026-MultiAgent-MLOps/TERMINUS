"""Données de fine-tuning du Repair Agent : projets sains variés -> injection de défauts -> trajectoires vérifiées -> JSONL.

    bases.py     projets ML sains, générés (aucune donnée réelle)         faults.py    défauts injectés (réparables / à escalader)
    recorder.py  enregistre chaque appel LLM et reconstruit la trajectoire filters.py   filtres de qualité (purs, testés)
    collect.py   chaîne Validator -> Repair -> (Judge) -> revalidation    export.py    runs -> SFT / préférences, découpage par projet
"""
