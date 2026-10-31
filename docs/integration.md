# Intégration de la plateforme

Branche de travail : `integration` (créée depuis `feature/orchestrator`). Quand le pipeline complet tourne,
elle devient `main` et remplace `feature/contract-validator` comme branche par défaut.

## Règles

- On prend dans chaque branche **uniquement les dossiers de l'agent** (`git checkout <branche> -- <dossier>`),
  jamais un merge complet : plusieurs branches portent une copie du validateur v1, et deux sont orphelines
  (`feature/api-docker-build`, `feature/validaor-repair-agents`).
- Le contrat `common/schemas.py` ne change qu'avec l'accord du groupe, et on incrémente `SCHEMA_VERSION`.
- Un agent intégré = une ligne dans `orchestrator/registry.py`, ses tests qui passent, et un run complet
  `python -m orchestrator run eval/projects/credit_risk` qui passe par lui.
- Les tests de l'orchestrateur tournent avec les stubs (`ORCH_AGENTS=stub`, fixé dans `tests/conftest.py`).

## Étapes

| # | Avec | Objet | État |
|---|---|---|---|
| 0 | Mohamed | branche `integration`, dossier `agents/`, registre des agents, Agent 4 branché | fait |
| 1 | Ahmed | structure, contrat v0.2, place de la boucle validateur → réparation → juge | à faire |
| 2 | Yacine | Agent 1 | à faire |
| 3 | Aziz | Agent 2 (MLflow) | à faire |
| 4 | Syrine | Agent 3 (API, Docker) | à faire |
| 5 | Mohamed | Agent 4 sur le vrai `endpoint`, run complet | à faire |
| 6 | Sarra | validateur v2, réparation, juge | à faire |
| 7 | tout le groupe | démo complète, README, `main` | à faire |

## Fiche de chaque meet

### 1. Orchestrateur (Ahmed)

À décider ensemble :
- les entrées de l'Agent 2 : `project_spec` + `baseline` du contrat (l'adaptateur construit son rapport v0.1) ;
- l'entrée de l'Agent 3 : le rapport d'analyse v1.0 qu'il attend aujourd'hui, ou les fichiers du contrat ;
- ce que l'Agent 3 déploie : le modèle enregistré par MLflow (`mlflow_metadata.model_uri`, prévu par le
  contrat) ou l'artefact du projet ;
- un nœud réservé (stub) pour la boucle de réparation du validateur v2 ;
- `SCHEMA_VERSION` 0.2 avec ces changements.

### 2. Agent 1 (Yacine) : `feature/agent-code-data-analysis`

À prendre : `agents/agent1/`, `eval/diabetes_progression/`, `eval/ml-iris/`, `eval/ticket_classifier/`.

Écart avec le contrat :
- l'agent lit `(project_path, csv_path)` et renvoie un dictionnaire (`problem_type`, `target`, `features`,
  `code_quality`) ;
- il écrit dans `agents/agent1/.workspace/` au lieu de `inp.workdir`.

Adaptateur `agents/agent1/orchestrator_adapter.py` : `run(AgentInput)` lance l'agent sur `inp.project_dir`
et le `data_path` du handoff, puis écrit `project_spec.json`, `data_profile.json` et `baseline.json`
(schémas du contrat) dans `inp.workdir`. Cas `AWAITING_CONFIRMATION` : produire `proposed_manifest.json`.

### 3. Agent 2 (Aziz) : `mlflow_agent`

À prendre : `mlflow_agent/`, `mlflow_agent.toml`, `benchmark_results/`, `docs/benchmark_report.md`,
`examples/`, ses tests (`test_benchmark`, `test_config`, `test_executor`, `test_gemini_pool`,
`test_mlflow_agent_input`, `test_planner`, `test_plans`, `test_prompts`). Pas son `README.md` racine.

Écart avec le contrat :
- il lit un « AnalysisReport v0.1 » (`project`, `task`, `dataset_analysis`) et renvoie `ExperimentResult` ;
- `mlflow` et `optuna` ne sont déclarés dans aucune dépendance.

Adaptateur : construire le rapport v0.1 depuis `project_spec` + `data_profile` + `baseline`, lancer
`run_experiment`, écrire `mlflow_metadata.json` (avec `quality_gate` et `champion_replacement_requested`).
Infrastructure : serveur MLflow dans `infra/docker-compose.yml` (ou SQLite local pour commencer).

### 4. Agent 3 (Syrine) : `feature/api-docker-build`

À prendre : `agent_api_docker/`, `fixes/`, ses tests (`test_agent_api_docker`, `test_docker_stage`,
`test_llm_fallback`, `test_validate_and_run`), `eval/make_ml_projects.py`, ses projets dans `eval/projects/`.
Pas `validator/`, `tests/test_validator.py`, `tests/test_handoff.py` (copies du v1), ni `workspaces/`
(sorties générées).

Écart avec le contrat :
- il lit un « AnalysisReport v1.0 » (`python`, `dependencies`, `io`, `inference`) depuis `workspaces/` ;
- il package le code du projet, pas le modèle MLflow ;
- `pyproject.toml` fixe pandas 3 / numpy 2.4 / scikit-learn 1.8 (versions des images, pas de l'environnement commun).

Adaptateur : rapport v1.0 depuis les fichiers du contrat, build et tests Docker, puis `deployment_result.json`
avec `endpoint` (l'API qui tourne) : c'est l'entrée de l'Agent 4.

### 5. Agent 4 (Mohamed) : déjà conforme

`monitoring_agent.pipeline:run` lit `project_spec` et `deployment_result`. Avec un `endpoint`, il prépare
le monitoring (référence notée par l'API, règles Prometheus, dashboard Grafana) ; avec du trafic observé, il
décide et peut produire `retraining_request.json`, qui déclenche la gate `retraining` de l'orchestrateur.

### 6. Validateur v2 (Sarra) : `feature/validaor-repair-agents`

À prendre : `validator_agent/`, `repair_agent/`, `agentic_core/`, `policy/`, `finetune/`, `docs/schemas/`,
son `eval/`, ses tests, `data/pilot/`.
À écarter : les scripts de rustine à la racine (`apply_*.py`, `apply_refactor.sh`, `patch_*.py`,
`fix_test_step1.py`), `validator_agent/tools.py.bak`, le lien de sous-module `Multi-Agent-MLOps-Pipeline`.

Points à vérifier : son `validator/` diffère du v1 (413 lignes) alors que le README le dit inchangé ;
`requires-python >= 3.12` alors que le code compile en 3.11 ; le `handoff` reste compatible avec le contrat
(il réutilise le modèle `Handoff` du v1).
