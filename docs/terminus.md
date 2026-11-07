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
| GET | `/api/runs/{id}` | détail : plan de ligne (`track`), validation en attente (`gate`), journal du superviseur, décisions, résultats, fichiers, ce que voit la gateway de l'Agent 4 (`monitoring`) |
| GET | `/api/runs/{id}/file?path=` | un fichier de la ligne (jamais en dehors de `runs/<id>/`) |
| POST | `/api/runs/{id}/decision` | `{decision, by, comment?, key, publish?}` ; `key` évite un double clic, `publish` = Docker Hub |
| POST | `/api/runs/{id}/monitor` | relancer l'Agent 4 sur une ligne déployée |
| POST | `/api/runs/{id}/agent4/{décision}` | `{decision, by, comment?}` : répondre à une décision de l'agent de la gateway (comme la console) ; un réentraînement approuvé est remis à la ligne |
| GET | `/api/decisions` | validations en attente sur toutes les lignes, décisions récentes |
| GET | `/api/registry`, `/api/registry/{modèle}` | registre MLflow : versions, alias, métriques |
| GET | `/api/runs/{id}/example` | une ligne des données du projet (sans la cible) pour essayer le modèle |
| POST | `/api/predict` | requête au modèle en production, via la gateway de l'Agent 4 (compte dans le monitoring) |

`track` : une gare par étape réellement passée (réparations et réentraînements compris), dans l'ordre,
avec `status` = `passed`, `blocked`, `failed`, `stopped`, `refused`, `waiting` (signal), `running`, `todo`.
Chaque gare porte `details.facts` (ce qu'elle a fait, lu dans ses fichiers : verdict et niveaux du validateur,
tâche et cible, modèles essayés et scores, image et contrôles, modèle surveillé…). Un agent relancé réécrit
ses fichiers : seule sa dernière exécution est détaillée. À la gare de réparation, `details.compare` met chaque
bloquant du v1 face au contrôle équivalent du v2 (confirme, ne confirme pas, non vérifié) et liste ce que
seul le v2 a vu : c'est la mesure v1/v2 de la ligne. Dans l'interface, un clic sur une gare l'ouvre.

## L'Agent 4 dans Terminus

`monitoring` (quand la gateway surveille le projet de la ligne) : prédictions, labels, fenêtres, gravité de la
dernière fenêtre, décideur, et les décisions de l'agent qui attendent un humain (cause, résumé, preuves).
Elles s'affichent dans le panneau « agent4 surveille » et dans **Décisions** ; les valider ici revient à les
valider dans la console ou dans Slack. Un réentraînement approuvé relance aussitôt l'Agent 4 de la ligne, qui
remet la demande : le signal « réentraînement » suit.

« Relancer l'Agent 4 » quand la gateway tourne : c'est l'agent de la gateway qui décide
(`MONITORING_GATEWAY_URL`), pas un second agent. La mémoire d'incidents (Qdrant local) ne s'ouvre que dans
un processus à la fois : avant, l'agent relancé décidait sans mémoire.

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
