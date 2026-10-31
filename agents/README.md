# Les agents de la plateforme

Chaque agent vit dans son propre dossier, avec son code, ses tests, son évaluation, sa documentation et
ses dépendances. Les noms des paquets Python ne changent pas : l'intégration n'a réécrit aucun import.

| Dossier | Agent | Paquet Python | Responsable | État |
|---|---|---|---|---|
| `validator/` (racine, pour l'instant) | Validateur de contrat v1 | `validator` | Sarra | intégré (nœud `validate`) |
| `agents/validator/` | Validateur v2, réparation, juge | `validator_agent`, `repair_agent`, `agentic_core`, `policy` | Sarra | nœud `repair` réservé, à intégrer (étape 6) |
| `agents/agent1/` | Analyse du code et des données | `agents.agent1` | Yacine | à intégrer (étape 2) |
| `agents/agent2/` | MLflow : expériences, quality gate, registre | `mlflow_agent` | Aziz | à intégrer (étape 3) |
| `agents/agent3/` | API et image Docker | `agent_api_docker` | Syrine | à intégrer (étape 4) |
| `agents/agent4/` | Monitoring, diagnostic, réentraînement contrôlé | `monitoring_agent` | Mohamed | intégré |

## Contrat avec l'orchestrateur

Chaque agent expose une seule fonction, déclarée dans `orchestrator/registry.py` :

```python
def run(inp: AgentInput) -> AgentResult: ...      # common/schemas.py
```

- il lit ses entrées dans `inp.project_dir` (copie figée du projet validé), `inp.report_path` (rapport du
  validateur) et `inp.inputs` (fichiers produits par les agents précédents) ;
- il écrit uniquement dans `inp.workdir` (`runs/<run_id>/<agent>/`) ;
- il renvoie ses fichiers de sortie dans `AgentResult.outputs`, avec les noms prévus par le contrat
  (`project_spec`, `data_profile`, `baseline`, `mlflow_metadata`, `deployment_result`, `retraining_request`).

Si l'agent a son propre format d'entrée ou de sortie, on ne le réécrit pas : on ajoute un petit adaptateur
dans son dossier (`<paquet>/orchestrator_adapter.py`) qui traduit le contrat commun dans son format, et
inversement. Point de départ : `agents/adapter_template.py`. Les fonctions `read_output` et `write_output`
de `common/schemas.py` lisent et écrivent les fichiers du contrat en les validant.

| Agent | Lit | Doit produire |
|---|---|---|
| agent1 | rapport du validateur, projet | `project_spec`, `data_profile`, `baseline` |
| propose_manifest (Agent 1) | rapport du validateur, projet | `proposed_manifest` |
| agent2 | `project_spec`, `data_profile`, `baseline` | `mlflow_metadata` |
| agent3 | `project_spec`, `data_profile`, `mlflow_metadata`, rapport | `deployment_result` |
| agent4 | `project_spec`, `deployment_result` | `retraining_request` si réentraînement demandé |
| repair | rapport du validateur, projet | `repair_result`, `repaired_project` |

## Vérifier son agent avant l'intégration

```bash
python -m orchestrator check-agent agent2 mlflow_agent.orchestrator_adapter:run
```

La commande lance le vrai validateur, fabrique les entrées avec les stubs des agents précédents, appelle
l'agent comme le fera l'orchestrateur et vérifie sa réponse (schémas, sorties obligatoires, fichiers écrits
dans le workdir, projet laissé intact). Quand elle affiche « Conforme », il reste une ligne à ajouter
dans `orchestrator/registry.py`.

## Configuration : un `.env` par agent

- `.env` à la racine : orchestrateur et infrastructure commune seulement (`ORCH_*`, `GEMINI_API_KEY`,
  `ORCH_DATABASE_URL`), chargé dans le terminal avec `set -a; source .env; set +a`.
- `agents/<agent>/.env` : les clés et réglages de l'agent, avec un `agents/<agent>/.env.example` vide commité.
  L'agent le charge lui-même au démarrage (ex. `dotenv_values`), sans écraser une variable déjà définie
  et non vide dans le terminal.
- Chaque membre utilise ses propres clés ; aucun `.env` n'est commité.

## Installer un agent dans l'environnement commun

```bash
uv pip install -e agents/agent4            # puis ses dépendances : -r agents/agent4/requirements.txt
```
