"""Niveau 4 - Données (C7) : lecture du CSV et contrôles de qualité. Pandas uniquement, pas d'exécution du projet."""
from __future__ import annotations

import pandas as pd
from pandas.api.types import is_numeric_dtype

from .models import Severity as S


def run(ctx):
    cfg, m = ctx.cfg, ctx.manifest
    if m is not None:
        rel = m.data.path
        path = ctx.root / rel
        if not path.is_file():
            ctx.add("C7", "C7.DATA_NOT_FOUND", S.BLOCKER, f"Fichier de données introuvable : {rel}", file=rel)
            return
    else:  # sans manifest : on tente data.csv, sinon on demande confirmation
        path = ctx.root / cfg.default_data_file
        rel = cfg.default_data_file
        if not path.is_file():
            csvs = [str(f) for f in ctx.files if f.suffix.lower() == ".csv"]
            ctx.add("C7", "C7.DATA_NOT_FOUND", S.CONFIRM, "Fichier de données non identifiable sans manifest.", file=cfg.default_data_file,
                    hint=f"CSV candidats : {csvs}" if csvs else "Aucun CSV dans le projet.")
            return
    size_mb = path.stat().st_size / 1e6
    if size_mb > cfg.max_data_mb:
        ctx.add("C12", "C12.DATA_TOO_LARGE", S.BLOCKER, f"Données de {size_mb:.0f} Mo (max {cfg.max_data_mb}).", file=rel)
        return
    try:
        df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        ctx.add("C7", "C7.ENCODING", S.BLOCKER, "CSV non encodé en UTF-8.", file=rel, hint="Convertissez le fichier en UTF-8.")
        return
    except Exception as e:
        ctx.add("C7", "C7.UNREADABLE", S.BLOCKER, f"CSV illisible : {type(e).__name__}: {e}"[:200], file=rel)
        return

    num = [c for c in df.columns if is_numeric_dtype(df[c])]
    ctx.facts["data"] = {"path": rel, "rows": len(df), "columns": len(df.columns), "numeric_cols": len(num),
                         "categorical_cols": len(df.columns) - len(num), "size_mb": round(size_mb, 3)}
    if len(df) < cfg.min_rows:
        ctx.add("C7", "C7.TOO_FEW_ROWS", S.BLOCKER, f"{len(df)} lignes (minimum {cfg.min_rows}).", file=rel)
    if len(df.columns) < 2:
        ctx.add("C7", "C7.TOO_FEW_COLUMNS", S.BLOCKER, "Au moins 2 colonnes sont nécessaires (features + cible).", file=rel)
        return

    # --- cible ------------------------------------------------------------------------------------
    target = m.data.target if m else None
    if target is not None:
        if target not in df.columns:
            ctx.add("C7", "C7.TARGET_NOT_FOUND", S.BLOCKER, f"Colonne cible `{target}` absente du CSV.", file=rel,
                    hint=f"Colonnes disponibles : {list(df.columns)[:15]}")
            target = None
        else:
            y = df[target]
            if y.isna().any():
                ctx.add("C7", "C7.TARGET_HAS_NULLS", S.WARNING, f"{int(y.isna().sum())} valeurs manquantes dans la cible.", file=rel)
            if m.project.task == "classification":
                counts = y.value_counts(dropna=True)
                ctx.facts["data"]["class_balance"] = {str(k): int(v) for k, v in counts.items()}
                if len(counts) < 2:
                    ctx.add("C7", "C7.SINGLE_CLASS", S.BLOCKER, "La cible ne contient qu'une seule classe.", file=rel)
                else:
                    if len(counts) > 50 and is_numeric_dtype(y):
                        ctx.add("C7", "C7.TARGET_LOOKS_CONTINUOUS", S.WARNING, f"{len(counts)} valeurs distinctes : la tâche est-elle une régression ?", file=rel)
                    if counts.min() < 5:
                        ctx.add("C7", "C7.RARE_CLASS", S.WARNING, f"Classe très rare ({int(counts.min())} exemples) : split stratifié fragile.", file=rel)
            elif not is_numeric_dtype(y):
                ctx.add("C7", "C7.TARGET_NOT_NUMERIC", S.BLOCKER, "Cible non numérique pour une tâche de régression.", file=rel)

    # --- qualité des features ------------------------------------------------------------------------
    feats = df.drop(columns=[target]) if target else df
    const = [c for c in feats.columns if feats[c].nunique(dropna=True) <= 1]
    if const:
        ctx.add("C7", "C7.CONSTANT_COLUMNS", S.WARNING, f"Colonnes constantes : {const[:8]}", file=rel)
    miss = feats.isna().mean()
    bad = miss[miss > cfg.missing_rate_warn]
    if len(bad):
        ctx.add("C7", "C7.HIGH_MISSING", S.WARNING, f"Colonnes avec > {cfg.missing_rate_warn:.0%} de valeurs manquantes : {list(bad.index)[:8]}", file=rel)
    ctx.facts["data"]["missing_rate_total"] = round(float(feats.isna().mean().mean()), 4)
    dup = float(df.duplicated().mean())
    if dup > cfg.duplicate_rate_warn:
        ctx.add("C7", "C7.DUPLICATES", S.WARNING, f"{dup:.0%} de lignes dupliquées.", file=rel)
    if target is not None and is_numeric_dtype(df[target]):  # fuite évidente
        for c in feats.columns:
            if is_numeric_dtype(feats[c]) and feats[c].nunique() > 1:
                corr = feats[c].corr(df[target])
                if pd.notna(corr) and abs(corr) > 0.999:
                    ctx.add("C7", "C7.POSSIBLE_LEAKAGE", S.WARNING, f"`{c}` est quasi identique à la cible (corr={corr:.4f}).", file=rel)
