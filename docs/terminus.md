# Terminus : l'API de l'interface web

```bash
uv pip install -e .                     # fastapi, uvicorn, httpx
set -a; source .env; set +a
python -m orchestrator api              # http://127.0.0.1:8090/api · documentation : /docs
```

L'interface React est dans `web/` (voir `web/README.md`) : `cd web && npm install && npm run dev`,
puis http://localhost:5173.

L'API ne contient aucune logique de pipeline : elle passe par `Run` (graph.py), comme la ligne de
commande et le service Slack. Une ligne (run) ne s'exécute qu'à un endroit à la fois : verrou
`runs/<run_id>/.busy` (runlock.py), partagé par l'API, le service Slack et `approve` / `reject` / `monitor`.

| Méthode | Chemin | Rôle |
|---|---|---|
| GET | `/api/health` | état des services (base, Docker, MLflow, API :8000, gateway :8100, Prometheus, Grafana, Slack, clés « présente / absente », MLX) |
| GET | `/api/projects` | projets lançables (`eval/projects`, variantes du validateur) |
| GET | `/api/runs` | lignes : projet, état, modèle servi, réentraînements |
| POST | `/api/runs` | lancer `{project_dir, supervisor?}` → `{run_id}` (exécution en arrière-plan) |
| GET | `/api/runs/{id}` | détail : plan de ligne (`track`), validation en attente (`gate`), journal du superviseur, décisions, résultats, fichiers |
| GET | `/api/runs/{id}/file?path=` | un fichier de la ligne (jamais en dehors de `runs/<id>/`) |
| POST | `/api/runs/{id}/decision` | `{decision, by, comment?, key, publish?}` ; `key` évite un double clic, `publish` = Docker Hub |
| POST | `/api/runs/{id}/monitor` | relancer l'Agent 4 sur une ligne déployée |
| GET | `/api/decisions` | validations en attente sur toutes les lignes, décisions récentes |
| GET | `/api/registry`, `/api/registry/{modèle}` | registre MLflow : versions, alias, métriques |
| GET | `/api/runs/{id}/example` | une ligne des données du projet (sans la cible) pour essayer le modèle |
| POST | `/api/predict` | requête au modèle en production, via la gateway de l'Agent 4 (compte dans le monitoring) |

`track` : une gare par étape réellement passée (réparations et réentraînements compris), dans l'ordre,
avec `status` = `passed`, `blocked`, `failed`, `stopped`, `refused`, `waiting` (signal), `running`, `todo`.

## Deux validations réservées à Terminus

| Gate | Quand | Approuver | Refuser |
|---|---|---|---|
| `repair_review` | le juge accepte la réparation, mais elle touche un périmètre sensible (`requires_human_approval`) | la copie est revérifiée (dans le workdir de la tentative) puis revalidée par le v1 | ligne arrêtée (BLOCKED) |
| `validator_override` | le v1 bloque, le v2 répond PROCEED ou HUMAN_REVIEW | le v1 revalide le projet d'origine avec ses bloquants levés (`--waive`, en avertissement « levé par <nom> »), puis la ligne continue | ligne arrêtée (BLOCKED) |

Dans Slack, elles apparaissent sans boutons, avec « à décider dans Terminus ». Contrat : `HumanDecision.gate`
accepte ces deux valeurs (`SCHEMA_VERSION` 0.3).

Après un désaccord tranché pour le v2, l'Agent 1 et les suivants reçoivent un rapport complet (tâche, cible,
métrique, exécution de contrôle, métriques observées), scellé, qui garde la trace de la décision : les bloquants
levés sont dans `carry_forward` et dans l'état du run (`waived_codes`). Jamais levés, même par un humain : appel
dangereux, secret, réseau, suppression de fichiers, pickle, bandit high, lien symbolique sortant, plantage du
validateur (`NEVER_WAIVED`). S'il ne reste que ceux-là, la ligne s'arrête sans reboucler sur la réparation.
