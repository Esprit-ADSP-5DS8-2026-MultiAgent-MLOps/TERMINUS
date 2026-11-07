# Intégration de la plateforme

Branche de travail : `integration` (créée depuis `feature/orchestrator`). Quand le pipeline complet tourne,
elle devient `main` et remplace `feature/contract-validator` comme branche par défaut.

## Règles

- On prend dans chaque branche **uniquement les dossiers de l'agent** (`git checkout <branche> -- <dossier>`),
  jamais un merge complet : plusieurs branches portent une copie du validateur v1, et deux sont orphelines
  (`feature/api-docker-build`, `feature/validaor-repair-agents`).
- Le contrat `common/schemas.py` ne change qu'avec l'accord du groupe, et on incrémente `SCHEMA_VERSION`.
- Un agent ne lit que les fichiers du contrat, jamais le format interne d'un autre agent. S'il a son propre
  format, son adaptateur traduit (`agents/adapter_template.py` à copier en `<paquet>/orchestrator_adapter.py`).
- Un agent intégré = `python -m orchestrator check-agent <nom> <module:fonction>` qui affiche « Conforme »,
  une ligne dans `orchestrator/registry.py`, ses tests qui passent, et un run complet
  `python -m orchestrator run eval/projects/credit_risk` qui passe par lui.
- Configuration : `.env` racine pour l'orchestrateur, `agents/<agent>/.env` pour chaque agent (voir
  agents/README.md). Un agent ne met aucune de ses variables dans le `.env` racine.
- Les tests de l'orchestrateur tournent avec les stubs et le superviseur à règles (`tests/conftest.py`) :
  jamais de vrai LLM dans les tests.
- Après le meet 1, `feature/orchestrator` est gelée : chacun part d'`integration` et ouvre une PR vers elle.

## Étapes

| # | Avec | Objet | État |
|---|---|---|---|
| 0 | Mohamed | branche `integration`, dossier `agents/`, registre des agents, Agent 4 branché | fait |
| 1 | Ahmed | contrat v0.2, nœud `repair` réservé, superviseur Gemini par défaut, `check-agent` | fait |
| 2 | Yacine | Agent 1 : adaptateur, détection de la cible corrigée | intégré, à valider avec Yacine |
| 3 | Aziz | Agent 2 (MLflow) : adaptateur, registre, champion | intégré, à valider avec Aziz |
| 4 | Syrine | Agent 3 (API, Docker) : modèle MLflow servi, conteneur déployé, champion | intégré, à valider avec Syrine |
| 5 | Mohamed | Agent 4 sur le vrai `endpoint`, boucle de réentraînement (`monitor`) | intégré |
| 6 | Sarra | validateur v2, réparation, juge : nœud `repair` | intégré, à valider avec Sarra |
| 6b | tout le groupe | validations humaines dans Slack (`python -m orchestrator slack`, docs/slack.md) | intégré, listes de validation à remplir |
| 6c | Mohamed | Terminus : API (`python -m orchestrator api`, docs/terminus.md), validations réparation et désaccord v1/v2 | intégré, front React à faire |
| 7 | tout le groupe | démo complète, README, `main` | à faire |

## Fiche de chaque meet

### 1. Orchestrateur (Ahmed) : décisions prises

- Principe : chaque agent lit les fichiers du contrat ; les adaptateurs traduisent vers les formats internes.
- Agent 2 : lit `project_spec`, `data_profile`, `baseline` (son adaptateur construit le rapport v0.1).
- Agent 3 : lit `project_spec`, `data_profile`, `mlflow_metadata` et le rapport du validateur (son adaptateur
  construit le rapport v1.0). `ProjectSpec` porte maintenant `model_artifact`, `interfaces`, `python_version`.
- Agent 3 déploie le modèle promu par MLflow (`model_uri`) ; repli sur l'artefact du projet
  (`DeploymentResult.model_source = "project_artifact"`). À confirmer avec Aziz et Syrine.
- Boucle de réparation : projet `BLOCKED` ou `INCOMPLETE` → nœud `repair` (copie corrigée, jugée) →
  revalidation de la copie, 2 tentatives au plus, sans gate humaine. Le projet d'origine n'est jamais
  modifié. Nombre de tentatives et gate à confirmer avec Sarra.
- Superviseur : `gemini` par défaut (`ORCH_SUPERVISOR`), règles si aucune clé.
- `SCHEMA_VERSION = "0.3"` (validé avec Ahmed) : gates `repair_review` et `validator_override`, décidées dans
  Terminus ; après `validator_override` approuvé, revalidation avec les bloquants du v1 levés (voir
  `docs/terminus.md`).

### 2. Agent 1 (Yacine) : `feature/agent-code-data-analysis`

À prendre : `agents/agent1/`, `eval/diabetes_progression/`, `eval/ml-iris/`, `eval/ticket_classifier/`.

Écart avec le contrat :
- l'agent lit `(project_path, csv_path)` et renvoie un dictionnaire (`problem_type`, `target`, `features`,
  `code_quality`) ;
- il écrit dans `agents/agent1/.workspace/` au lieu de `inp.workdir`.

Adaptateur `agents/agent1/orchestrator_adapter.py` : `run(AgentInput)` lance l'agent sur `inp.project_dir`
et le `data_path` du handoff, puis écrit `project_spec.json` (avec `model_artifact`, `interfaces` et
`python_version` repris du handoff), `data_profile.json` et `baseline.json` dans `inp.workdir`.
Cas `AWAITING_CONFIRMATION` : une 2e fonction `propose_manifest(AgentInput)` produit `proposed_manifest.json`.

Vérification : `python -m orchestrator check-agent agent1 agents.agent1.orchestrator_adapter:run`
puis `check-agent propose_manifest agents.agent1.orchestrator_adapter:propose_manifest`.

Fait (branche `integ/agent1`) :
- `agents/agent1/` et ses projets `eval/` repris de sa branche ; `orchestrator_adapter.py` (`run`,
  `propose_manifest`) ; les deux lignes dans `registry.py` ;
- `run_agent1(..., artifact_dir=...)` : ses artefacts vont dans le workdir du run (avant : `agents/agent1/.workspace`) ;
- détection de la cible : `X, y = df.drop(...), df[TARGET]` est maintenant compris (avant : la cible
  devenait `amount_per_month`, colonne fabriquée par le prétraitement, et l'agent abandonnait).

Choix à valider avec Yacine :
- le manifeste déclaré (vérifié par le validateur) fait foi ; sa détection sert de contrôle (avertissement si désaccord) ;
- ses améliorations de code (`code_patch`, `ml_patch`, `improved_project`) sont des propositions : la suite du
  pipeline continue sur le projet validé, puisque le code modifié n'a pas été revalidé ;
- `baseline.naive_metrics` : modèle naïf calculé par l'adaptateur (classe majoritaire / moyenne) ;
- sans manifeste : métrique et seuil proposés par défaut (binaire roc_auc 0.7, multiclasse f1 0.6, régression r2 0.5).

### 3. Agent 2 (Aziz) : `mlflow_agent`

À prendre : `mlflow_agent/`, `mlflow_agent.toml`, `benchmark_results/`, `docs/benchmark_report.md`,
`examples/`, ses tests (`test_benchmark`, `test_config`, `test_executor`, `test_gemini_pool`,
`test_mlflow_agent_input`, `test_planner`, `test_plans`, `test_prompts`). Pas son `README.md` racine.

Écart avec le contrat :
- il lit un « AnalysisReport v0.1 » (`project`, `task`, `dataset_analysis`) et renvoie `ExperimentResult` ;
- `mlflow` et `optuna` ne sont déclarés dans aucune dépendance.

Adaptateur : construire le rapport v0.1 depuis `project_spec` + `data_profile` + `baseline`, lancer
`run_experiment`, écrire `mlflow_metadata.json` (avec `quality_gate` et `champion_replacement_requested`).
Statut `NEEDS_APPROVAL` quand le quality gate passe (gate humaine champion), `FAILED` sinon.
Vérification : `python -m orchestrator check-agent agent2 mlflow_agent.orchestrator_adapter:run`.

Fait (branche `integ/agent2`) :
- `agents/agent2/` : paquet `mlflow_agent`, ses tests, benchmarks, exemples, `mlflow_agent.toml`,
  `pyproject.toml` et `requirements.txt` (mlflow et optuna sont maintenant déclarés) ;
- `orchestrator_adapter.py` : rapport v0.1 construit depuis `project_spec` et le CSV, plan du LLM (Gemini,
  `mlflow_agent.toml`) ou plan par défaut, expérience MLflow, `mlflow_metadata.json` ;
- MLflow : `MLFLOW_TRACKING_URI`, sinon SQLite dans `runs/mlflow/` (`mlflow ui --backend-store-uri
  sqlite:///runs/mlflow/mlflow.db`) ; une expérience et un modèle enregistré par projet ;
- le candidat reçoit l'alias `candidate` ; il doit battre le seuil, le modèle naïf ET le champion actuel
  (alias `champion`) ; sinon `FAILED` avec la raison ;
- le champion est rejugé sur **le même jeu de test** que le candidat (rechargé depuis le registre) ; son ancien
  score, ou sa performance en production, ne sert que si on ne peut pas le faire prédire (colonnes ou classes
  différentes, avertissement dans le résultat) ;
- classification : les classes de l'encodeur (`label_classes`) sont enregistrées avec le modèle (metadata du
  MLmodel et `mlflow_metadata.params`) ; l'Agent 3 s'en sert pour rendre l'étiquette d'origine, même texte ;
- réentraînement : relancé par l'orchestrateur après la gate `retraining`, sur `new_data_path` s'il existe.

Choix à valider avec Aziz :
- `mlflow_agent.toml` cherché dans `agents/agent2/` (avant : dossier courant) ;
- budget par défaut : 12 essais, 10 minutes (`AGENT2_MAX_RUNS`, `AGENT2_MAX_MINUTES`) ;
- `instrumentation_diff_path` : une note (l'agent n'exécute pas le code du projet, rien n'est instrumenté).

À décider au meet 4 (Syrine) : qui pose l'alias `champion`. Proposition : l'adaptateur de l'Agent 3 appelle
`mlflow_agent.orchestrator_adapter.promote(model_name, version)` une fois le déploiement réussi.
Tant que ce n'est pas fait, il n'y a jamais de champion et chaque candidat qui passe le gate est proposé.
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

Adaptateur : rapport v1.0 depuis les fichiers du contrat (`python` ← `project_spec.python_version`,
`dependencies` ← `pinned_dependencies`, `ml.model_file` ← `model_artifact`, `inference` ← `interfaces`,
`io.input_schema` ← `data_profile.columns`), build et tests Docker sur le modèle `model_uri`, puis
`deployment_result.json` avec `model_source` et `endpoint` (l'API qui tourne) : c'est l'entrée de l'Agent 4.
Vérification : `python -m orchestrator check-agent agent3 agent_api_docker.orchestrator_adapter:run`.

Fait (branche `integ/agent3`) :
- `agents/agent3/` : paquet `agent_api_docker` (inchangé), `fixes/`, ses projets `eval/projects/`, ses tests ;
  `pyproject.toml` et `requirements.txt` sans les versions ML de l'image (elles viennent du rapport) ;
- `orchestrator_adapter.py` :
  1. modèle de l'Agent 2 chargé depuis MLflow (`model_uri`) et écrit en `model/model.joblib`, avec
     `mlflow_predict.py` (rend les étiquettes d'origine) ; repli : modèle du projet entraîné au build ;
  2. rapport v1.0 construit depuis le contrat et le CSV ; Python et librairies = celles qui ont entraîné le modèle ;
  3. `run_pipeline` de l'agent tel quel : génération, validation, tests locaux, image et tests dans le conteneur ;
  4. conteneur `mlops-<projet>` démarré sur `AGENT3_PORT` (8000) : `deployment_result.endpoint` ;
  5. alias `champion` posé sur la version MLflow déployée (`promote()` de l'Agent 2).
- sans Docker : API testée en local, `endpoint` vide, statut WARNING.

À voir avec Syrine :
- deux tests échouaient déjà sur sa branche (Syrine : à laisser tels quels) : `test_low_confidence_stops` (dossier `eval/expected_reports/`
  absent) et `test_push_requires_approval` (`docker_push` a un argument `source_ref` en plus) ;
- régression servie : `prediction` est un `float` (et une chaîne pour des étiquettes texte), le type vient du
  rapport (`io.output_schema`) ; les tests générés comparent les nombres à 1e-6 près (aller-retour JSON) ;
- publication Docker Hub : `AGENT3_PUSH=1` + `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN` (et `DOCKER_IMAGE_NAMESPACE`)
  dans `agents/agent3/.env` ; l'approbation est la gate `champion_replacement`. Dépôt public par défaut sur Docker Hub.
  En ligne de commande interactive, comme le CLI de Syrine : destination affichée, choix du compte (Entrée = compte
  du `.env`, sinon un autre compte et son token), confirmation du push.

Encodeur des étiquettes : corrigé à la source (Agent 2) ; l'adaptateur lit les classes enregistrées avec le modèle.

#### Régression de bout en bout

Projet de test : `eval/projects/house_prices` (prix de logements, données synthétiques générées par
`eval/make_house_prices.py`, métrique `rmse`, `min_threshold` = erreur maximale acceptée). Ce qui a changé :
- validateur v1 : pour `rmse` / `mae`, un score SOUS le seuil est bon (avant : avertissement à tort) ;
- Agent 1 : pas de `NO_STRATIFY` sur une régression ;
- Agent 3 : `prediction: float` ;
- Agent 4 : le scénario `concept` augmente la cible de 50 % au-delà de la médiane (au lieu de `1 - prix`) ;
- Terminus : barres à l'échelle de l'erreur, « plus bas = meilleur ».

Boucle vérifiée : validateur → Agents 1, 2, 3 → Agent 4 (gateway, concept drift, réentraînement demandé) →
Agent 2 sur les nouvelles données (champion rejugé sur le même jeu de test) → nouveau champion → Agent 3.

### 5. Agent 4 (Mohamed) : déjà conforme (`check-agent agent4 monitoring_agent.pipeline:run`)

`monitoring_agent.pipeline:run` lit `project_spec` et `deployment_result`. Avec un `endpoint`, il prépare
le monitoring (référence notée par l'API, règles Prometheus, dashboard Grafana) ; avec du trafic observé, il
décide et peut produire `retraining_request.json`, qui déclenche la gate `retraining` de l'orchestrateur.

Fait (branche `integ/agent4-boucle`) : la boucle retour.
- l'orchestrateur n'appelle l'Agent 4 qu'une fois, juste après le déploiement : pas encore de trafic. La
  décision se prend ensuite dans la gateway (`python -m monitoring_agent gateway <projet>`), qui écrit
  `agents/agent4/runs/<projet>/actions/retraining_request.json` (risque moyen : décidé seul, la
  validation humaine est celle de l'orchestrateur) ;
- `python -m orchestrator monitor <run_id>` relance l'Agent 4 sur un run DONE : il remet cette demande une
  seule fois (copiée dans le workdir avec ses nouvelles données), puis gate `retraining` -> Agent 2 ->
  gate `champion_replacement` -> Agent 3 (redéploie) -> Agent 4. Sans demande, le run reste DONE ;
- Agent 4 : les nouvelles données couvrent l'épisode en cours (drift des entrées OU baisse de performance
  mesurée : un concept drift ne fait pas dériver les entrées) ;
- Agent 2 : lit les nouvelles données en parquet ; en réentraînement, le champion est jugé sur sa
  performance mesurée en production (`current_performance`), pas sur son ancien jeu de test.

- un candidat refusé pendant un réentraînement (quality gate raté) termine en CHAMPION_KEPT, pas FAILED :
  le modèle déployé continue de servir et `monitor` reste possible ;
- simulateur : le scénario `concept` inverse la cible au-delà de la médiane de la feature numérique la plus
  variée (règle apprenable). Avant : 35 % des labels inversés au hasard, du bruit qu'aucun réentraînement
  ne rattrape (essai réel : candidat à 0,61, sous le seuil de 0,70).

- après un réentraînement, la référence du monitoring = les données d'entraînement du modèle déployé
  (Agent 2 : `mlflow_metadata.params.training_data`, copie `train_v<version>.csv`), et une nouvelle période
  de monitoring commence (historique du modèle précédent archivé en `monitoring_<modèle>_v<n>_<date>.db`).
  Avant : référence = `data.csv` du projet, faux incidents « baisse de performance estimée » sur le v2.

Démo : API déployée, `docker compose -f agents/agent4/infra/docker-compose.yml up -d`, gateway,
`python -m monitoring_agent simulate <projet> --scenario concept`, puis `monitor <run_id>`.
Après le redéploiement, relancer la gateway (elle garde l'ancienne référence en mémoire).

À voir avec Aziz : idéalement, évaluer le champion sur le jeu de test du candidat (mêmes lignes).

### 6. Validateur v2 (Sarra) : `feature/validaor-repair-agents`

À prendre : `validator_agent/`, `repair_agent/`, `agentic_core/`, `policy/`, `finetune/`, `docs/schemas/`,
son `eval/`, ses tests, `data/pilot/`.
À écarter : les scripts de rustine à la racine (`apply_*.py`, `apply_refactor.sh`, `patch_*.py`,
`fix_test_step1.py`), `validator_agent/tools.py.bak`, le lien de sous-module `Multi-Agent-MLOps-Pipeline`.

Adaptateur `repair_agent/orchestrator_adapter.py` : lit `inp.report_path` et `inp.project_dir`, écrit une
copie corrigée DANS `inp.workdir`, la fait juger, puis renvoie `repair_result` (schéma `RepairResult`) et
`repaired_project` (le dossier). L'orchestrateur refuse une copie hors du workdir ou non approuvée par le juge.
Vérification : `python -m orchestrator check-agent repair repair_agent.orchestrator_adapter:run`.

Fait (branche `integ/validator-v2`) :
- `agents/validator/` : ses paquets inchangés (`agentic_core`, `validator_agent`, `repair_agent` + `judge`,
  `policy`, `finetune`), `eval/`, `docs/`, `data/pilot/`, ses tests (sans les copies `test_validator.py`,
  `test_handoff.py` du v1) ; `pyproject.toml` en Python ≥ 3.11 et sans le paquet `validator` (celui de la racine) ;
- `validator/` racine : sa version de `harness.py` et son `entrypoints.py` (entrées LEGACY, `_audit`, `_probe`) ;
  notre `sandbox.py` garde la correction macOS (rlimit) ;
- `repair_agent/orchestrator_adapter.py`, appelé quand le v1 bloque le projet (au plus 2 tentatives) :
  1. validateur v2 : diagnostic ; il ne continue que si `next_action = REPAIR` (sinon WARNING avec la raison :
     PROCEED = écart v1/v2 à examiner, STOP = rejet, HUMAN_REVIEW = décision humaine) ;
  2. Repair sur une copie `runs/<id>/repair/attempt_<n>/project` (le projet d'origine n'est jamais touché) ;
     à la tentative 2, le verdict REVISE du Judge lui est transmis ;
  3. Judge : la copie n'est retenue que sur ACCEPT et sans `requires_human_approval` ;
  4. l'orchestrateur revalide la copie avec le v1 puis continue (Agent 1...).
  Sorties : `validator_v2.json`, `repair.json`, `judge.json` (rapports scellés), `repair.diff`,
  `repair_result.json`, `repaired_project`.
- `common/envscope.py` : le `.env` d'un agent n'est appliqué que pendant son appel (Sarra et Yacine chargeaient
  leur `GROQ_API_KEY` dans tout le processus) ;
- `agentic_core/env_check.py` : liste des paquets via `importlib.metadata` quand le venv n'a pas `pip` (uv) ;
- `check-agent` : un agent qui s'arrête proprement (WARNING, `can_proceed = false`) n'a pas à fournir ses sorties.

À voir avec Sarra :
- son README dit `validator/` inchangé, mais `harness.py` a changé et `entrypoints.py` est nouveau : repris tels quels ;
- `requires-python >= 3.12` passé à 3.11 (environnement commun) : tout passe en 3.11 ;
- `agentic_core/__init__.py` charge aussi le `.env` du dossier courant : sans effet dans le pipeline (envscope) ;
- `REPAIR_ENV=current` ajoute un problème `ENV.SANDBOX_ISOLATION` que le Judge sanctionne (REVISE) : garder `venv` ;
- modèles par défaut (`openai/gpt-oss-120b`, `qwen/qwen3.8-27b`) : à confirmer sur son compte Groq et son quota.
