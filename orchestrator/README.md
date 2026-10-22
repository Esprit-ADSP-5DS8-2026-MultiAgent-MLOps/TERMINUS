# Orchestrateur — Agentic MLOps

L'orchestrateur est le **chef d'orchestre** du pipeline. Il lance chaque étape dans l'ordre,
lit le résultat, décide de la suite, et demande l'avis d'un humain pour les décisions sensibles.

- Il **ne modifie jamais** le code du projet.
- Il **ne manipule pas les données** : il transmet seulement des chemins de fichiers et des statuts.
- Il est construit avec **LangGraph** (graphe à état, pauses humaines, boucles).

## Le pipeline

```mermaid
flowchart TD
    P([Projet ML]) --> V[Contract Validator]

    V -->|ACCEPTÉ| A1[Agent 1<br/>Analyse]
    V -->|À CONFIRMER| PM[Agent 1<br/>propose un mlops.yaml]
    V -->|REJETÉ| B([BLOCKED])

    PM --> G1{{Humain :<br/>valider le manifeste ?}}
    G1 -->|oui| MP([MANIFEST_PROPOSED<br/>ajouter mlops.yaml et relancer])
    G1 -->|non| R([REJECTED])

    A1 --> A2[Agent 2<br/>MLflow + quality gate]
    A2 -->|gate échoué| F([FAILED])
    A2 -->|gate réussi| G2{{Humain :<br/>promouvoir en champion ?}}
    G2 -->|non| R
    G2 -->|oui| A3[Agent 3<br/>Packaging + déploiement]

    A3 --> A4[Agent 4<br/>Monitoring]
    A4 -->|pas de drift| D([DONE])
    A4 -->|drift| G3{{Humain :<br/>réentraîner ?}}
    G3 -->|oui| A2
    G3 -->|non| D

    classDef human fill:#fff4d6,stroke:#c99a00
    classDef stop fill:#eee,stroke:#999
    classDef ok fill:#dff5e3,stroke:#2e8b57
    class G1,G2,G3 human
    class B,R,F,MP stop
    class D ok
```

Un agent qui plante, à n'importe quelle étape, mène à `FAILED` sans faire planter le pipeline.
Le réentraînement est limité à **une fois par run**.

## Les fichiers

| Fichier | Rôle |
|---|---|
| `../common/schemas.py` | Le **contrat** commun : ce que chaque agent reçoit et renvoie |
| `state.py` | La **mémoire** du pipeline : chemins, résultats, décisions |
| `nodes.py` | Les **étapes** : le validateur et l'enveloppe commune des agents |
| `routing.py` | Les **aiguillages** : quelle étape après chaque résultat |
| `gates.py` | Les **pauses humaines** (`interrupt` de LangGraph) |
| `checkpoint.py` | La **sauvegarde** de l'état : SQLite par défaut, PostgreSQL si `ORCH_DATABASE_URL` est défini |
| `graph.py` | Le **plan** du graphe et l'exécution d'un run |
| `__main__.py` | Les **commandes** `run`, `status`, `approve`, `reject`, `resume` |
| `stubs/agents.py` | Les **faux agents**, en attendant les vrais |

## Lancer une démo

Avec l'interface web :

```bash
source .venv/bin/activate
streamlit run interface/streamlit_app.py
```

Ou dans le terminal :

```bash
python -m orchestrator run eval/projects/credit_risk                       # cas normal
ORCH_SIMULATE_DRIFT=1 python -m orchestrator run eval/projects/credit_risk  # avec drift
python -m orchestrator run eval/projects/variants/02_unpinned_deps         # projet rejeté
python -m pytest tests/orchestrator -q                                     # 11 tests
```

## Persistance : reprendre un run plus tard

L'état est sauvegardé après chaque étape dans `runs/checkpoints.sqlite`.
Un run en pause peut donc être décidé plus tard, depuis un autre terminal :

```bash
python -m orchestrator run eval/projects/credit_risk --no-wait   # s'arrête à la pause
python -m orchestrator status                                    # liste des runs
python -m orchestrator status <run_id>                           # détail
python -m orchestrator approve <run_id> -m "commentaire"         # ou reject
python -m orchestrator resume <run_id>                           # après un plantage
```

Après un plantage, `resume` reprend à l'étape interrompue, sans relancer les précédentes.

### Utiliser PostgreSQL (base prévue par l'architecture)

Par défaut, l'état est sauvegardé dans SQLite (`runs/checkpoints.sqlite`), sans rien installer.
Pour utiliser PostgreSQL :

```bash
docker compose -f infra/docker-compose.yml up -d            # démarre PostgreSQL
export ORCH_DATABASE_URL=postgresql://mlops:mlops@localhost:5432/orchestrator
python -m orchestrator status                               # affiche « Base : PostgreSQL · ... »
```

Seul `checkpoint.py` connaît la base utilisée : le reste de l'orchestrateur ne change pas.
Dans l'interface Streamlit, les runs en pause apparaissent dans le panneau de gauche.

Chaque exécution crée `runs/<run_id>/` : un dossier par étape, la copie figée du projet
(`snapshot/`) et `run_summary.json` (statut, durée, décisions humaines).

## Brancher un vrai agent

Chaque agent expose une seule fonction :

```python
from common.schemas import AgentInput, AgentResult

def run(inp: AgentInput) -> AgentResult:
    ...  # lire inp.inputs, écrire dans inp.workdir, renvoyer un AgentResult
```

Puis, dans `graph.py`, on remplace le stub :

```python
from agents.agent1_analysis.agent import run as agent1_run
AGENTS["agent1"] = agent1_run
```

## État d'avancement

| Fait | À faire |
|---|---|
| Contrat commun (J1) | Brancher les vrais agents (J3) |
| Graphe complet avec stubs (J2) | Ajouter MLflow, MinIO, Prometheus… à `infra/` |
| 3 gates humaines + interface Streamlit | Décisions humaines à distance (plusieurs utilisateurs) |
| Boucle de réentraînement limitée | Langfuse |
| Traces de chaque run | Mode baseline sans LLM (évaluation) |
| Persistance SQLite ou PostgreSQL, reprise des runs | |
| Interface : plan de ligne, décisions, journal, historique | |
