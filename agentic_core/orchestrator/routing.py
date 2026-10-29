"""Table de routage par `next_action` (contrat des schémas v2). Données seulement."""
ROUTES: dict[str, dict[str, str]] = {
    "validator": {"PROCEED": "END_ADMITTED", "REPAIR": "repair", "HUMAN_REVIEW": "END_HUMAN_REVIEW", "STOP": "END_REJECTED"},
    "repair": {"JUDGE": "judge", "ESCALATE": "END_HUMAN_REVIEW", "NONE": "END_NO_CHANGE"},
    "judge": {"REVALIDATE": "validator", "RETRY": "repair", "ESCALATE": "END_HUMAN_REVIEW", "ROLLBACK": "rollback"},
}
