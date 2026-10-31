"""
Mémoire d'incidents (RAG hybride) : l'agent retrouve des cas passés similaires et peut s'en servir comme
exemples (few-shot dynamique), en citant leur identifiant - citation vérifiée par le code.

    cards.py   fiches d'incident : signature numérique + texte ; sources : scénarios ML vérifiés,
               décisions confirmées de l'agent, incidents de service (RCAEval, OpenRCA, NAB)
    store.py   Qdrant local (2 vecteurs nommés par fiche) + recherche hybride + écriture des décisions
"""

from .cards import IncidentCard, signature
from .store import IncidentMemory

__all__ = ["IncidentCard", "IncidentMemory", "signature"]
