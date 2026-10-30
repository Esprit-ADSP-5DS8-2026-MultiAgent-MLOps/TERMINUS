# Architecture technique de l'Agent 4

Ce document complète le README avec les détails techniques : les méthodes de calcul, les seuils, la gateway, le graphe de décision, MCP, la mémoire RAG, le fine-tuning et le suivi Langfuse.

## 1. Moteur d'analyse (aucun LLM)

Code : `monitoring_agent/analysis/`. Seuils : `monitoring_agent/config.py`.

Chaque fenêtre de production est comparée à la référence (les données d'entraînement).

| Contrôle | Méthode | Seuil |
|---|---|---|
| Drift numérique | Wasserstein normalisée par l'écart-type (n > 1000), sinon test KS | 0,1 / p < 0,05 |
| Drift catégoriel | Jensen-Shannon (n > 1000), sinon chi² | 0,1 / p < 0,05 |
| Sévérité du drift | PSI, toujours calculé | 0,1 modéré, 0,25 majeur |
| Drift du dataset | Part des features en drift | 50 % |
| Drift des prédictions | Même méthode, sur le score du modèle | idem |
| Qualité | Manquants, types invalides, catégories inconnues, valeurs hors intervalle | voir `config.py` |
| Performance mesurée | Avec les labels | moyenne de référence ± 3 écarts-types, ou seuil métier |
| Performance estimée | CBPE (classification), DLE (régression), sans labels | idem |

Quelques choix à connaître :

- Sur de gros échantillons, les tests statistiques trouvent une différence partout (la p-value tend vers 0 pour des écarts sans importance). On utilise donc des distances au-delà de 1000 lignes, comme Evidently.
- Avec 20 features testées à p < 0,05, on aurait en moyenne un faux positif par fenêtre. Les tests KS et chi² utilisent la correction de Bonferroni (0,05 divisé par le nombre de features).
- Concept drift suspecté : la performance mesurée chute alors que l'estimée reste normale. CBPE suppose que la relation entre les features et la cible ne change pas. Si la mesure le contredit, c'est justement cette relation qui a changé.
- Chaque fenêtre produit des signaux. `trigger_agent` passe à vrai à partir de la sévérité `warning`, et c'est le seul moment où un LLM est appelé.

Exemple sur Car Loan (NannyML) : drift détecté à la fenêtre 5, ROC AUC estimée 0,961 contre 0,960 mesurée, sans un seul label. Sur Car Price, 5 features sur 7 dérivent mais la MAE baisse : un drift n'est pas une dégradation.

## 2. Gateway et observabilité

```
client ──► gateway :8100 ──► API du modèle :8000 (Agent 3, non modifiée)
              │  enregistre entrées, prédiction et latence (SQLite), reçoit les labels (/feedback)
              │  moniteur : toutes les W prédictions, analyse de la fenêtre
              ▼
         /metrics ──► Prometheus :9090 ──► règles générées ──► Alertmanager :9093 ──► webhook /alerts
                          │
                          └──► Grafana :3000 (dashboard généré)
```

La gateway est un proxy : la réponse de l'API est renvoyée telle quelle, avec un en-tête `X-Request-ID`. Le client réutilise cet identifiant pour envoyer le vrai label plus tard sur `/feedback`.

Routes principales :

| Route | Rôle |
|---|---|
| `POST /predict` | Proxy vers l'API du modèle, avec enregistrement |
| `POST /feedback` | Labels reçus après coup |
| `POST /alerts` | Webhook d'Alertmanager |
| `GET /metrics` | Métriques Prometheus (préfixe `mlmon_`) |
| `GET /monitoring/status`, `/windows`, `/decisions` | Données de la console |
| `POST /monitoring/decide` | Force une décision sur la dernière fenêtre |
| `POST /monitoring/decisions/{id}/approve` ou `/reject` | Réponse humaine |
| `POST /monitoring/ask` | Chat « Demande à ton monitoring » |
| `GET`, `POST /monitoring/ratings` | Avis sur les décisions et les réponses |
| `/mcp/` | Serveur MCP (Streamable HTTP) |
| `/console/` | La console web |

Les règles d'alerte et le dashboard sont générés pour chaque projet (`provision`) : ils connaissent la métrique principale et son sens (`roc_auc` plus haut = mieux, `mae` l'inverse), le seuil métier de `mlops.yaml` et la limite de variabilité calculée sur la référence. Les règles sont validées par `promtool` avant d'être chargées.

Alertes : gateway ou modèle injoignable, erreurs 5xx, entrées invalides (4xx), latence p95, drift du dataset, drift majeur d'une feature, drift des prédictions, qualité des données, baisse de performance estimée, mesurée, ou sous le seuil métier. Chaque alerte porte `severity`, `component` (service, data, model) et `project`.

Alertmanager regroupe les alertes (une notification par alerte et par projet) et fait taire les conséquences quand la cause est connue : des données cassées font taire les alertes de drift, un modèle tombé fait taire les erreurs et la latence.

Anti-rafale : une fenêtre arrive toutes les dix secondes environ. L'agent n'est relancé que si les types de signaux changent, ou toutes les 5 fenêtres, et jamais pendant qu'une décision attend un humain. Un réveil reçu pendant que l'agent travaille est gardé et rejoué ensuite.

## 3. Graphe de décision

Code : `monitoring_agent/agent/graph.py`.

```
prepare ─► investigate ─► diagnose x3 ─► verify ─► judge ─► policy ─┬─ auto ──► act
              (outils)     (vote)     (preuves)  (LLM)    (règles) └─ human ─► interrupt ─► act | fin
                              ▲           │          │
                              └─ une révision si une preuve est fausse ou si le juge demande « revise »
```

| Nœud | Qui | Détail |
|---|---|---|
| prepare | code | Faits chiffrés de la fenêtre + recherche de cas similaires (RAG) |
| investigate | LLM | Boucle ReAct bornée sur les outils MCP, puis une note d'enquête |
| diagnose | LLM x3 | Trois tirages en parallèle, vote majoritaire (self-consistency) |
| verify | code | Chaque chiffre cité doit exister dans les faits, à 2 % près ; libellés « mesurée / estimée » corrigés s'ils sont inversés |
| judge | autre LLM | Grille : ancrage dans les faits, causalité, proportionnalité, sécurité |
| policy | code | Règles R1 à R9 (`guards.py`) |
| await_human | humain | Pause LangGraph (`interrupt`), reprise par la console ou l'API |
| act | code | Fichiers, demande de réentraînement, notification |

Si le diagnostiqueur est le modèle local fine-tuné, il reçoit uniquement les faits (le format appris à l'entraînement). La note d'enquête et les cas similaires vont à l'enquêteur et au juge.

Si les preuves restent invérifiables après une révision, l'agent n'escalade pas : il applique le décideur à règles, dont les preuves sont exactes par construction.

### Prompts

Code : `monitoring_agent/agent/prompts.py`. Chaque modification change `PROMPT_VERSION`, enregistrée avec chaque décision.

- rôle et périmètre précis ;
- vocabulaire fermé : 6 causes, 6 actions ;
- procédure de décision ordonnée, identique à celle du décideur à règles (`rules.procedure`) ;
- chaque source dans sa balise (`<dossier>`, `<faits>`, `<note_enquete>`, `<cas_similaires>`, `<critique>`), et tout ce qui vient des données est traité comme une donnée, jamais comme une consigne ;
- sortie JSON contrainte par un JSON Schema quand le fournisseur le permet, validée par Pydantic dans tous les cas ;
- self-consistency, révision guidée par la critique, LLM-as-judge.

## 4. Outils MCP

Code : `monitoring_agent/agent/mcp_server.py` (SDK `mcp` 2.x).

- 8 outils en lecture seule : `feature_history`, `performance_history`, `quality_details`, `prediction_drift`, `active_alerts`, `previous_decisions`, `window_facts_at`, `service_health` (requêtes PromQL) ;
- ressources : `monitoring://status`, `monitoring://policy`, `monitoring://decisions/{id}` ;
- prompts : `diagnose`, `explain_decision`.

Le même serveur est utilisé de trois façons :

- en interne par l'enquêteur et par le chat (client MCP en mémoire : `tools/list` puis `tools/call`) ;
- en HTTP sur `http://localhost:8100/mcp/` pour un autre agent ou l'orchestrateur ;
- en stdio (`python -m monitoring_agent mcp <projet>`) pour Claude Desktop ou MCP Inspector.

Aucun outil n'agit. Agir passe toujours par la vérification, le juge et les règles.

## 5. Mémoire d'incidents (RAG hybride)

Code : `monitoring_agent/memory/`.

- Qdrant en mode local (un dossier, aucun serveur), deux vecteurs nommés par fiche :
  - le texte de l'incident, encodé par un modèle multilingue (fastembed, 384 dimensions, cosinus) ;
  - une signature de 9 nombres calculée depuis les faits (part de features en drift, PSI max, baisse mesurée, baisse estimée, pire problème de qualité, labels disponibles, drift des prédictions, signal qualité, incident de service), distance euclidienne.
- Filtre par famille (ML ou service), deux classements fusionnés par Reciprocal Rank Fusion (k = 60), puis un seuil de pertinence.
- Le poids du texte dans la fusion est calibré sur le jeu de validation. Il vaut 0 : comparer les mots égare la recherche, comparer les mesures fonctionne.
- Un identifiant `INC-xxxx` cité par le LLM doit faire partie des cas réellement fournis, sinon la preuve est refusée.
- Boucle d'apprentissage : une décision exécutée, ou rejetée par un humain, devient une nouvelle fiche.

Évaluation sur les bases de test (jamais en mémoire) :

| Méthode | Bon cas dans les 3 premiers | Précision sur les 3 premiers |
|---|---|---|
| Texte seul | 59 % | - |
| Signature seule | 100 % | - |
| Hybride, poids 1:1 | 97 % | 71 % |
| Hybride calibré | 99 % | 95 % (MRR 0,98) |

## 6. Fine-tuning du diagnostiqueur local

Code : `monitoring_agent/finetune/`.

```bash
python -m monitoring_agent finetune bases
python -m monitoring_agent finetune scenarios --per-base 220
python -m monitoring_agent finetune teach --max-calls 400
python -m monitoring_agent finetune build
mlx_lm.lora --config runs/finetune/data/lora_fast.yaml
mlx_lm.server --model mlx-community/Qwen3-4B-Instruct-2507-4bit --adapter-path runs/finetune/data/adapters --port 8080
python -m monitoring_agent finetune eval --models "mlx:default_model=qwen-finetune"
```

- Scénarios : on casse la production de façon contrôlée (shift, concept, pannes de données, labels en retard), puis le vrai moteur d'analyse calcule les signaux. Un scénario n'est gardé que si l'injection et la procédure concluent la même chose (1650 acceptés sur 1805).
- Découpage par base, sans fuite : 6 bases pour l'entraînement, `mc_prepaid` pour la validation, `car_loan` et `car_price` pour le test.
- Distillation vérifiée : un grand modèle (Groq) répond sans indice. Sa réponse est gardée si la cause, l'action et toutes les preuves sont justes. Sinon on lui donne la bonne réponse en indice et on lui demande de la justifier (rationalisation, méthode STaR) ; l'exemple d'entraînement garde le prompt sans l'indice.
- 694 exemples d'entraînement (388 du professeur, 306 des règles), équilibrés par action.
- QLoRA sur Qwen3-4B-Instruct-2507 quantifié en 4 bits : rang 16, alpha 32, 16 dernières couches, learning rate 1e-4 avec warmup et décroissance cosinus, batch effectif 8, perte calculée seulement sur la réponse, gradient checkpointing (obligatoire sur 24 Go). Meilleur checkpoint à l'itération 150 (perte de validation de 1,707 à 0,140).

### Hébergement

Code : `monitoring_agent/finetune/export.py`.

1. `mlx_lm.fuse` fusionne l'adaptateur dans le modèle de base : une version MLX 4 bits (Mac) et une version f16.
2. `convert_hf_to_gguf.py` et `llama-quantize` (llama.cpp) produisent un GGUF Q4_K_M, avec son `Modelfile` Ollama.
3. `finetune publish` crée un dépôt de modèle privé sur Hugging Face (adaptateur, GGUF, MLX, fiche du modèle) et un
   Space privé Docker basé sur l'image `llama.cpp:server` : il télécharge le GGUF au démarrage (secret `HF_TOKEN`)
   et expose `/v1/chat/completions`, avec 3 requêtes en parallèle pour les 3 diagnostics.
4. Côté agent, le fournisseur `hf` (`MONITORING_DECIDER=hf:agent4`) appelle ce Space avec `HF_TOKEN`. Comme `mlx` et
   `ollama`, il est traité comme le diagnostiqueur fine-tuné : il reçoit les faits seuls, et l'enquête reste au
   grand modèle. Le délai se règle avec `MONITORING_HF_TIMEOUT_S`.

Les avis de la console (`ratings.jsonl`) fournissent de nouveaux exemples réels au même format, pour un futur réentraînement.

## 7. Notifications

Code : `monitoring_agent/agent/notify.py` (envoi) et `monitoring_agent/runtime/slack.py` (réception des clics et des questions).

Une notification part quand l'agent exécute `notify_data_team`, `open_incident`, `request_retraining` ou `rollback_model`, quand une décision attend un humain, et quand un humain la rejette.

- App Slack (prioritaire) : message en Block Kit. Une décision en attente a des boutons Approuver / Rejeter ; la suite (exécution ou rejet) est postée dans le fil et le message d'origine perd ses boutons. Le `ts` Slack est gardé dans `notifications.jsonl` pour retrouver le fil, même si la validation vient de la console.
- Réception en Socket Mode (`slack-bolt`) : clic Approuver / Rejeter -> `Supervisor.resume`, comme `POST /monitoring/decisions/{id}/approve` ; `/monitoring` et `@agent4` -> le chat ; boutons Utile / Pas utile et Bonne décision -> `ratings.jsonl`.
- Contrôles : seules les personnes de `SLACK_APPROVERS` peuvent valider ; un verrou dans `MonitoringAgent.resume` garantit que la première réponse gagne.
- Webhook entrant (`MONITORING_NOTIFY_WEBHOOK`) si l'app n'est pas configurée : Slack, Teams, Mattermost (champ `text`), Discord (champ `content`).
- Toujours une ligne dans `notifications.jsonl`. Un échec d'envoi ne bloque jamais la décision.

`MONITORING_ALWAYS_HUMAN` (règle R10) liste les actions que l'équipe veut toujours valider elle-même.

## 8. Suivi des appels LLM (Langfuse)

Code : `monitoring_agent/observability.py`. Optionnel : sans clés, rien n'est envoyé.

- une trace par décision, avec un span par nœud du graphe et une génération par appel LLM (modèle, prompt, réponse, jetons, latence) ;
- des scores sur la trace : preuves vérifiées, notes du juge, accord du vote, autonomie, réponse humaine ;
- les prompts versionnés (`langfuse push-prompts`) et un dataset de test (`langfuse push-dataset`).

## 9. Tests

```bash
python -m pytest -q
```

Les tests n'appellent jamais un vrai LLM ni un vrai webhook : les LLM sont remplacés par des faux qui répondent selon un script. Les tests qui ont besoin des jeux de données sont sautés s'ils sont absents.
