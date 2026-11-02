# Validations humaines dans Slack

Le pipeline pose ses validations dans `#mlops-alertes`, avec des boutons. Deux apps Slack dans le même
espace de travail, chacune avec ses jetons :

| App | Qui l'utilise | Ce qu'elle poste |
|---|---|---|
| Agent 4 Monitoring (`agents/agent4/infra/slack/manifest.yml`) | la gateway de l'Agent 4 | alertes, décisions de l'agent, `/monitoring` |
| MLOps Orchestrateur (`infra/slack/orchestrator-manifest.yml`) | `python -m orchestrator slack` | les validations du pipeline, début et fin des runs |

Deux apps, parce qu'en Socket Mode deux programmes connectés avec le même jeton se partagent les clics au
hasard : un clic sur « Approuver le champion » pourrait arriver à la gateway.

## Ce qui se valide dans Slack

| Validation | Message | Boutons |
|---|---|---|
| `manifest_confirmation` | tâche, cible, métrique, seuil, confiance de l'Agent 1 | Approuver / Refuser |
| `champion_replacement` | score du candidat, du modèle naïf, du champion, seuil ; destination Docker Hub | Approuver et publier sur Docker Hub / Approuver sans publier / Refuser |
| `retraining` | raison, performance mesurée en production, diagnostic de l'Agent 4 | Approuver / Refuser |
| `failure_review` | étape en échec, raisons, avis du superviseur | Relancer l'étape / Arrêter le run |

- Chaque run a son fil : début, validations (aussi montrées dans le canal), fin.
- Après un clic, les boutons disparaissent et le message indique qui a décidé ; la décision est enregistrée
  dans le run (`par <nom> (Slack)`).
- Quand la gateway de l'Agent 4 demande un réentraînement, le service relance lui-même `monitor` sur le
  dernier run déployé du projet : la validation `retraining` arrive dans le fil de ce run.
- Docker Hub : seul le choix « publier ou non » passe par Slack. Le compte et le jeton restent dans
  `agents/agent3/.env`, jamais dans Slack.
- Réparation à valider et désaccord v1/v2 : pas dans Slack, prévus pour le front.

## Qui valide

Identifiants Slack (U...) dans le `.env` racine : une liste par validation, sinon `ORCH_SLACK_APPROVERS`,
sinon tout le canal. La publication Docker Hub a sa propre liste (`ORCH_SLACK_APPROVERS_DOCKER_PUSH`) : une
personne autorisée pour le champion mais pas pour Docker Hub ne voit marcher que « Approuver sans publier ».

Proposition : chacun valide ce qui touche son agent, avec un suppléant.

| Variable | Qui |
|---|---|
| `ORCH_SLACK_APPROVERS_MANIFEST_CONFIRMATION` | Yacine (Agent 1) + suppléant |
| `ORCH_SLACK_APPROVERS_CHAMPION_REPLACEMENT` | Aziz (Agent 2) + suppléant |
| `ORCH_SLACK_APPROVERS_DOCKER_PUSH` | Syrine (Agent 3) |
| `ORCH_SLACK_APPROVERS_RETRAINING` | Mohamed (Agent 4) + Aziz |
| `ORCH_SLACK_APPROVERS_FAILURE_REVIEW` | Ahmed (orchestrateur) + suppléant |

Identifiant d'une personne : son profil Slack > ⋮ > « Copier l'ID de membre ».

## Mise en place (une fois)

1. api.slack.com/apps > Create New App > From a manifest > l'espace de travail du groupe > coller
   `infra/slack/orchestrator-manifest.yml` > Create.
2. Basic Information > App-Level Tokens > Generate : nom `socket`, scope `connections:write` > copier le
   jeton `xapp-...` dans `ORCH_SLACK_APP_TOKEN`.
3. Install App > Install to Workspace > copier le jeton `xoxb-...` dans `ORCH_SLACK_BOT_TOKEN`.
4. Dans `#mlops-alertes` : `/invite @mlops-orchestrateur`. ID du canal (détails du canal, en bas) dans
   `ORCH_SLACK_CHANNEL`.
5. Remplir les listes de validation.

Les jetons ne vont que dans le `.env` de la machine qui fait tourner le pipeline : jamais dans git, jamais
dans un message Slack. Les collègues n'en ont pas besoin, ils cliquent avec leur propre compte.

## Utilisation

```bash
set -a; source .env; set +a
python -m orchestrator slack                                           # terminal 1 : le service
python -m orchestrator run eval/projects/credit_risk --slack           # terminal 2 : le run
python -m monitoring_agent gateway credit_risk                         # terminal 3 : la gateway de l'Agent 4
```

Le service exécute la suite du pipeline après chaque clic (Agent 2, Agent 3, Agent 4) : il doit tourner sur
la machine qui a Docker, MLflow et les `.env` des agents. Sans le service, `approve` / `reject` en ligne de
commande marchent toujours.
