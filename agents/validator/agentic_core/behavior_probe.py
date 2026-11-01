"""Sonde comportementale de ML.METRICS_GENUINE : on ne LIT pas le code qui calcule la métrique, on MESURE ce qu'elle fait.

Principe : le harnais ré-entraîne le projet sur une copie des données dont les ÉTIQUETTES SONT PERMUTÉES, en enregistrant
ce que le code fait de ses étiquettes. Trois constats, sans aucune liste de bibliothèques ni de noms de métriques :
  1. métrique IDENTIQUE avec des étiquettes détruites -> elle ne dépend ni des étiquettes ni du fichier fourni
     (constante, lue d'un fichier, calculée ailleurs) ;
  2. la valeur renvoyée provient d'un appel de métrique dont `y_true` est EXACTEMENT le tableau passé à `fit` -> calculée sur les
     données d'entraînement, pas une performance de généralisation (fuite) ;
  3. sinon la métrique réagit aux étiquettes (écart > SENSITIVE_REL) -> comportement d'une vraie mesure.
Ce module est copié dans le sandbox sous le nom `_probe.py` (bibliothèque standard seule à l'import ; numpy/pandas/sklearn
sont importés à la demande) et il est aussi importé côté Validator pour `judge_metric` (pur Python).
"""
import functools
import hashlib
import inspect
import math
import sys

SENSITIVE_REL = 0.02          # écart relatif minimal pour dire que la métrique réagit aux étiquettes
EXACT_REL = 1e-9
MAX_ROWS_FP = 500_000
MAX_CALLS = 200

_ACTIVE = None                # enregistreur courant (None : les enveloppes ne font que relayer l'appel)
_ARMED = []                   # (module, nom, original) pour pouvoir restaurer


def _fp(y):
    """Empreinte d'un tableau d'étiquettes (taille + hash du contenu) ; None si non calculable."""
    try:
        import numpy as np
        a = np.asarray(y).ravel()
        if a.size == 0 or a.size > MAX_ROWS_FP:
            return None
        if a.dtype == object:
            a = a.astype(str)
        return f"{a.size}:{hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]}"
    except Exception:
        return None


def arm():
    """À appeler AVANT l'import du code du candidat : enveloppe les fonctions de `sklearn.metrics` pour que même
    `from sklearn.metrics import accuracy_score` (lié à l'import) passe par l'enregistreur. Sans effet si scikit-learn est absent."""
    if _ARMED:
        return
    try:
        import sklearn.metrics as M
    except Exception:
        return
    for name in dir(M):
        fn = getattr(M, name, None)
        if name.startswith("_") or not inspect.isfunction(fn):
            continue

        def make(name, fn):
            @functools.wraps(fn)
            def wrapper(*a, **k):
                r = fn(*a, **k)
                rec = _ACTIVE
                if rec is not None:
                    rec.metric_call(name, a[0] if a else k.get("y_true"), r)
                return r
            return wrapper
        setattr(M, name, make(name, fn))
        _ARMED.append((M, name, fn))


class Recorder:
    """Enregistre, pendant UN entraînement : les étiquettes vues par chaque `fit(X, y)` et chaque appel de métrique (y_true, valeur)."""

    def __enter__(self):
        global _ACTIVE
        self.fits, self.calls = [], []
        _ACTIVE = self
        sys.setprofile(self._profile)
        return self

    def __exit__(self, *exc):
        global _ACTIVE
        sys.setprofile(None)
        _ACTIVE = None
        return False

    def _profile(self, frame, event, arg):
        if event != "call":
            return
        co = frame.f_code
        if co.co_name == "fit" and "y" in co.co_varnames[:co.co_argcount]:     # protocole estimateur : fit(X, y)
            y = frame.f_locals.get("y")
            fp = _fp(y) if y is not None else None
            if fp:
                self.fits.append(fp)

    def metric_call(self, name, y_true, result):
        if len(self.calls) >= MAX_CALLS or isinstance(result, bool):
            return
        try:
            value = float(result)
        except (TypeError, ValueError):
            return
        if math.isfinite(value):
            self.calls.append({"fn": name, "y": _fp(y_true), "value": value})

    def summary(self):
        return {"fit_label_hashes": sorted(set(self.fits))[:50], "n_fits": len(self.fits), "metric_calls": self.calls[:100]}


def permuted_copy(data_path, target, dest, seed=0):
    """Copie des données où la colonne cible est mélangée (graine fixe) : le lien entre variables et étiquettes est détruit."""
    import pandas as pd
    sep = "\t" if str(data_path).lower().endswith(".tsv") else ","
    df = pd.read_csv(data_path, sep=sep)
    if target not in df.columns:
        raise KeyError(f"colonne cible absente : {target}")
    df[target] = df[target].sample(frac=1.0, random_state=seed).to_numpy()
    df.to_csv(dest, index=False, sep=sep)
    return dest


# ------------------------------------------------------------------------------------------------ côté Validator (pur Python)
def _num(metrics, name):
    v = (metrics or {}).get(name) if isinstance(metrics, dict) else None
    if isinstance(v, bool) or v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def judge_metric(metric, real, perm, probe=None, probe_error=None):
    """Verdict comportemental : label_independent | leak | weak | genuine | inconclusive. Ne regarde que des mesures."""
    if not metric:
        return {"verdict": "inconclusive", "why": "aucune métrique principale connue"}
    if probe_error and perm is None:
        return {"verdict": "inconclusive", "why": f"entraînement sur étiquettes permutées impossible ({str(probe_error)[:80]})"}
    r, p = _num(real, metric), _num(perm, metric)
    if r is None or p is None:
        return {"verdict": "inconclusive", "why": "métrique absente de l'entraînement sur étiquettes permutées"}
    ev = f"{metric}: réel={r:.6g} ; étiquettes permutées={p:.6g}"
    gap, scale = abs(r - p), max(abs(r), 1e-9)
    calls = [c for c in (probe or {}).get("metric_calls") or [] if abs(c.get("value", math.inf) - p) <= EXACT_REL * max(1.0, abs(p))]
    fits = set((probe or {}).get("fit_label_hashes") or [])
    on_train = [c for c in calls if c.get("y") and c["y"] in fits]
    if calls and len(on_train) == len(calls) and fits:           # preuve directe : prioritaire sur l'égalité (un modèle qui mémorise score 1.0 des deux côtés)
        return {"verdict": "leak", "evidence": ev + f" ; appel {on_train[0]['fn']}() sur les étiquettes passées à fit",
                "why": f"la métrique « {metric} » est calculée sur les étiquettes d'entraînement (le même tableau est passé à fit) : "
                       f"ce n'est pas une performance de généralisation"}
    if gap <= EXACT_REL * max(1.0, scale):
        return {"verdict": "label_independent", "evidence": ev,
                "why": f"la métrique « {metric} » est identique quand les étiquettes sont permutées : elle ne dépend ni des étiquettes ni du fichier "
                       f"de données fourni (valeur constante, lue d'un fichier ou calculée ailleurs)"}
    if gap <= SENSITIVE_REL * scale:
        return {"verdict": "weak", "evidence": ev,
                "why": f"la métrique « {metric} » varie de moins de {SENSITIVE_REL:.0%} quand les étiquettes sont détruites : classes très "
                       f"déséquilibrées ou métrique peu informative (revue humaine)"}
    return {"verdict": "genuine", "evidence": ev, "holdout_confirmed": bool(calls) and not on_train,
            "why": f"la métrique « {metric} » réagit aux étiquettes ({r:.6g} -> {p:.6g} une fois permutées)"
                   + (" et provient d'un appel évalué hors des données d'entraînement" if calls and not on_train else "")}
