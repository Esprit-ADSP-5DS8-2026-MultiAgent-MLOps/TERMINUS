"""Correspondance codes du validateur déterministe (C1…C12) -> contrôles du catalogue v2. Sert à comparer baseline et agent sur la même échelle."""
RULES = [  # (préfixe de code, check_id) ; premier qui correspond
    ("C1.", "STRUCT.REQUIRED_FILES"), ("C2.UNSUPPORTED_FRAMEWORK", "FRAMEWORK.SCOPE"), ("C2.NO_", "FRAMEWORK.SCOPE"), ("C2.SYNTAX_ERROR", "CODE.SYNTAX"),
    ("C3.TRAIN_FAILED", "EXEC.RUN"), ("C3.HARNESS_FAILED", "EXEC.RUN"), ("C3.", "ML.TRAIN_INTERFACE"), ("C4.", "ML.PREDICT_INTERFACE"),
    ("C5.MAIN_FAILED", "EXEC.RUN"), ("C5.MAIN_NO_ARTIFACT", "EXEC.RUN"), ("C5.", "ML.NON_INTERACTIVE"),
    ("C6.UNDECLARED_IMPORT", "ENV.DEPENDENCIES_DECLARED"), ("C6.INSTALL_FAILED", "EXEC.RUN"), ("C6.ENV_MISMATCH", "EXEC.RUN"), ("C6.", "ENV.DEPENDENCIES_PINNED"),
    ("C7.", "DATA.USABLE"), ("C8.", "STRUCT.MANIFEST"), ("C9.", "ML.REPRODUCIBILITY"), ("C10.", "IO.PATHS"),
    ("C11.DANGEROUS_CALL", "SEC.DANGEROUS_CALLS"), ("C11.BANDIT_HIGH", "SEC.DANGEROUS_CALLS"), ("C11.SECRET", "SEC.SECRETS"), ("C11.NETWORK", "SEC.NETWORK"),
    ("C11.", "SEC.COMMITTED_ARTIFACTS"), ("C12.SANDBOX_FALLBACK", None), ("C12.", "STRUCT.LIMITS"),
]


def check_for(code: str):
    for prefix, cid in RULES:
        if code.startswith(prefix):
            return cid
    return None
