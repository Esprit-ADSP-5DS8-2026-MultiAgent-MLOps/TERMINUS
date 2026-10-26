"""
Mémoire d'incidents : Qdrant en mode local (un dossier, aucun serveur) + recherche HYBRIDE.

Pourquoi pas un RAG « naïf » (un embedding, top-k) : deux incidents se ressemblent par leur DESCRIPTION
(« catégorie inconnue sur salary_range » ≈ « nouvelle valeur dans job ») ET par leurs CHIFFRES
(perf mesurée -40 %, estimée stable). Un seul de ces deux angles se trompe souvent.

Recherche, pour un dossier :
1. FILTRE structuré : famille (ml / service) — un incident de latence ne doit pas servir d'exemple
   à un problème de drift ;
2. deux classements indépendants : sémantique (embedding multilingue du texte, cosinus) et numérique
   (signature de 9 nombres, distance euclidienne) — 2 vecteurs nommés par fiche dans Qdrant ;
3. FUSION par rangs (Reciprocal Rank Fusion, k = 60) : robuste, sans calibrer des scores hétérogènes ;
4. SEUIL de pertinence : une fiche n'est gardée que si au moins un des deux angles la juge proche.
   Mieux vaut aucun exemple qu'un mauvais exemple ;
5. les identifiants retenus sont transmis au vérificateur : citer un INC-xxxx non fourni = hallucination.

Écriture : une décision de l'agent n'entre en mémoire que si elle est CONFIRMÉE (exécutée avec l'accord
du juge, ou approuvée par un humain) ; un rejet humain est gardé comme contre-exemple.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path

import numpy as np

from .cards import SERVICE_ALERTS, SIG_DIM, IncidentCard, query_text, signature

log = logging.getLogger("monitoring_agent.memory")
COLLECTION = "incidents"
DEFAULT_EMBEDDER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"   # 384 dim, FR/EN, 220 Mo
CITATION = re.compile(r"INC-\d{4}")


class HashEmbedder:
    """Repli hors-ligne (tests, machine sans accès au modèle) : sac de n-grammes de caractères haché."""

    name, dim = "hash-ngrams-384", 384

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            t = f"  {t.lower()}  "
            for j in range(len(t) - 2):
                h = int.from_bytes(hashlib.blake2b(t[j:j + 3].encode(), digest_size=4).digest(), "little")
                out[i, h % self.dim] += 1.0
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


class FastEmbedder:
    def __init__(self, model: str, cache_dir: Path):
        from fastembed import TextEmbedding
        self.model = TextEmbedding(model, cache_dir=str(cache_dir))
        self.name = model
        self.dim = len(next(iter(self.model.embed(["test"]))))

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(list(self.model.embed(texts)), dtype=np.float32)


def make_embedder(cache_dir: Path):
    choice = os.environ.get("MONITORING_EMBEDDER", DEFAULT_EMBEDDER)
    if choice == "hash":
        return HashEmbedder()
    try:
        return FastEmbedder(choice, cache_dir)
    except Exception as exc:  # noqa: BLE001 — pas de réseau : la mémoire reste utilisable
        log.warning("modèle d'embedding %s indisponible (%s) : repli sur %s", choice, exc, HashEmbedder.name)
        return HashEmbedder()


def rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, pid in enumerate(ranking):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
    return scores


class IncidentMemory:
    def __init__(self, path: Path, embedder=None):
        from qdrant_client import QdrantClient
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or make_embedder(self.path / "models")
        self.client = QdrantClient(path=str(self.path / "qdrant"))
        meta_path = self.path / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        if meta.get("embedder") not in (None, self.embedder.name) and self.client.collection_exists(COLLECTION):
            log.warning("embedder changé (%s -> %s) : la mémoire est réindexée", meta.get("embedder"), self.embedder.name)
            cards = self.all()
            self.client.delete_collection(COLLECTION)
            self._create()
            self.add(cards)
        if not self.client.collection_exists(COLLECTION):
            self._create()
        meta_path.write_text(json.dumps({"embedder": self.embedder.name, "dim": self.embedder.dim}))

    def _create(self) -> None:
        from qdrant_client import models
        self.client.create_collection(COLLECTION, vectors_config={
            "text": models.VectorParams(size=self.embedder.dim, distance=models.Distance.COSINE),
            "sig": models.VectorParams(size=SIG_DIM, distance=models.Distance.EUCLID)})

    # -- écriture -------------------------------------------------------------------

    def count(self) -> int:
        return self.client.count(COLLECTION).count

    def add(self, cards: list[IncidentCard]) -> list[str]:
        from qdrant_client import models
        if not cards:
            return []
        start = self.count()
        for i, c in enumerate(cards):
            c.id = c.id or f"INC-{start + i + 1:04d}"
        vecs = self.embedder.embed([c.text for c in cards])
        self.client.upsert(COLLECTION, points=[
            models.PointStruct(id=int(c.id[4:]), vector={"text": v.tolist(), "sig": c.signature}, payload=c.payload())
            for c, v in zip(cards, vecs, strict=True)])
        return [c.id for c in cards]

    def all(self) -> list[IncidentCard]:
        points, _ = self.client.scroll(COLLECTION, limit=100_000, with_payload=True)
        return [IncidentCard(**p.payload) for p in points]

    def remember(self, run, case) -> str | None:
        """Décision de l'agent -> fiche, seulement si l'issue est connue (confirmée ou rejetée)."""
        from ..agent.context import case_facts, case_summary
        if run.diagnosis is None or run.policy is None:
            return None
        verdict = run.verdicts[-1].verdict if run.verdicts else None
        if run.status == "executed" and (run.human or verdict == "approve" or run.decider.startswith("rules")):
            outcome = "confirmé par un humain" if run.human else "confirmé (juge + policy)"
        elif run.status == "rejected_by_human":
            outcome = "rejeté par un humain : " + ((run.human or {}).get("comment") or "sans commentaire")
        else:
            return None
        d, p = run.diagnosis, run.policy
        kinds = sorted(case.signal_kinds())
        card = IncidentCard(
            id="", source="agent", family="ml", task=case.mapping.task, signals=kinds,
            title=f"{case.project} fenêtre {case.current.window.index} : {d.root_cause}",
            text=(f"{query_text(case.mapping.task, kinds, case_summary(case))}\n"
                  f"Diagnostic : {d.summary} Action : {p.final_action}. Raison : {d.rationale}"),
            root_cause=d.root_cause, action=p.final_action, outcome=outcome,
            signature=signature(case_facts(case), kinds))
        return self.add([card])[0]

    # -- lecture --------------------------------------------------------------------

    def search(self, case, k: int = 3, candidates: int = 20, min_text: float = 0.55, min_sig: float = 0.75,
               mode: str = "hybrid") -> list[dict]:
        """Fiches les plus proches du dossier. mode : hybrid | text | sig (pour l'évaluation)."""
        from ..agent.context import case_facts, case_summary
        kinds = case.signal_kinds()
        service = bool({a.get("alertname") for a in case.alerts} & SERVICE_ALERTS)
        q_text = self.embedder.embed([query_text(case.mapping.task, kinds, case_summary(case))])[0]
        q_sig = signature(case_facts(case), kinds, service)
        return self.search_vectors(q_text, q_sig, "service" if service else "ml", k, candidates, min_text, min_sig, mode)

    def search_vectors(self, q_text, q_sig, family: str, k: int = 3, candidates: int = 20, min_text: float = 0.55,
                       min_sig: float = 0.75, mode: str = "hybrid", exclude: set[str] | None = None) -> list[dict]:
        from qdrant_client import models
        flt = models.Filter(must=[models.FieldCondition(key="family", match=models.MatchValue(value=family))])
        by_text = self.client.query_points(COLLECTION, query=list(map(float, q_text)), using="text", query_filter=flt,
                                           limit=candidates, with_payload=True).points
        by_sig = self.client.query_points(COLLECTION, query=list(map(float, q_sig)), using="sig", query_filter=flt,
                                          limit=candidates, with_payload=True).points
        exclude = exclude or set()
        text_sim = {p.payload["id"]: p.score for p in by_text}
        sig_sim = {p.payload["id"]: 1.0 / (1.0 + p.score) for p in by_sig}        # distance -> similarité (0..1]
        payloads = {p.payload["id"]: p.payload for p in [*by_text, *by_sig]}
        rankings = {"hybrid": [list(text_sim), list(sig_sim)], "text": [list(text_sim)], "sig": [list(sig_sim)]}[mode]
        fused = rrf(rankings)
        hits = []
        for pid, score in sorted(fused.items(), key=lambda kv: -kv[1]):
            if pid in exclude:
                continue
            ts, ss = text_sim.get(pid, 0.0), sig_sim.get(pid, 0.0)
            if mode == "hybrid" and ts < min_text and ss < min_sig:
                continue                                                     # ni le texte ni les chiffres ne suivent
            hits.append({"id": pid, "rrf": round(score, 4), "text_sim": round(ts, 3), "sig_sim": round(ss, 3),
                         **{x: payloads[pid][x] for x in ("title", "root_cause", "action", "outcome", "source", "text")}})
            if len(hits) == k:
                break
        return hits


def format_similar(hits: list[dict]) -> str:
    """Bloc <cas_similaires> : des EXEMPLES (few-shot dynamique), pas des ordres."""
    return "\n".join(f"{h['id']} (texte {h['text_sim']:.2f}, chiffres {h['sig_sim']:.2f}) — {h['title']} -> "
                     f"cause {h['root_cause']}, action {h['action']} [{h['outcome']}]" for h in hits)


def memory_dir() -> Path:
    from ..datasets import PACKAGE_ROOT
    return Path(os.environ.get("MONITORING_MEMORY_DIR", PACKAGE_ROOT / "runs" / "memory")).expanduser()


_OPEN: dict[Path, IncidentMemory] = {}


def default_memory() -> IncidentMemory | None:
    """La mémoire si elle a été construite (`python -m monitoring_agent memory build`), sinon None.
    Qdrant local verrouille son dossier : une seule instance par processus."""
    path = memory_dir()
    if os.environ.get("MONITORING_MEMORY", "1") == "0" or not (path / "qdrant").exists():
        return None
    if path not in _OPEN:
        try:
            _OPEN[path] = IncidentMemory(path)
        except Exception:
            log.warning("mémoire d'incidents indisponible (%s)", path, exc_info=True)
            return None
    return _OPEN[path]


def evaluate_retrieval(memory: IncidentMemory, scenarios: list[dict], k: int = 3) -> dict:
    """Sur des scénarios jamais mémorisés (bases de test) : la mémoire ramène-t-elle des cas de MÊME cause ?
    hit@k = au moins un des k premiers a la bonne cause ; précision@k ; MRR. Comparaison des 3 modes."""
    out = {}
    for mode in ("text", "sig", "hybrid"):
        hits_at, prec, rr, empty = [], [], [], 0
        for sc in scenarios:
            summary = re.search(r"<dossier>\n(.*?)\n</dossier>", sc["messages"][1]["content"], re.DOTALL).group(1)
            q_text = memory.embedder.embed([query_text(sc["task"], sc["signals"], summary)])[0]
            res = memory.search_vectors(q_text, signature(sc["facts"], sc["signals"]), "ml", k=k, mode=mode)
            good = [h["root_cause"] == sc["oracle"]["root_cause"] for h in res]
            empty += not res
            hits_at.append(any(good))
            prec.append(sum(good) / k)
            rr.append(next((1 / (i + 1) for i, g in enumerate(good) if g), 0.0))
        n = len(scenarios) or 1
        out[mode] = {"hit@k": round(sum(hits_at) / n, 3), "precision@k": round(sum(prec) / n, 3),
                     "mrr": round(sum(rr) / n, 3), "no_result": round(empty / n, 3)}
    return {"k": k, "queries": len(scenarios), "modes": out}
