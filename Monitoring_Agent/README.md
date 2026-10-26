# Agent 4 — Monitoring & fiabilité

L'agent surveille un modèle déployé par l'Agent 3, comprend ce qui change, décide et agit.
**Les outils calculent, le LLM interprète** : aucun chiffre de ce dossier n'est produit par un LLM.

```
observer  ──►  comprendre  ──►  décider  ──►  agir
(analyzer)     (diagnostic LLM)  (juge + policy)  (actions annulables / humain)
```

## État d'avancement

| Étape | Contenu | État |
|---|---|---|
| 1 | Moteur d'analyse déterministe : drift, qualité, performance mesurée et estimée | ✅ |
| 2 | Gateway devant l'API de l'Agent 3 + Prometheus / Alertmanager / Grafana | ✅ |
| 3 | Agent LangGraph autonome : enquête à outils, diagnostic ×3, vérification, juge, policy, actions | ✅ |
| 4 | Console de l'agent (servie par la gateway) | ✅ |
| 5 | Évaluation chiffrée (règles / LLM seul / agent) + dataset de fine-tuning | ✅ |

## Installation

```bash
source ~/AI_projects/ai_env311/bin/activate      # ou n'importe quel env Python 3.11
pip install -r Monitoring_Agent/requirements.txt
```

Les données (≈ 26 Go) ne sont **pas** dans git : elles vont dans `Monitoring_Agent/data/`
(ou ailleurs, avec `export MONITORING_DATA_DIR=/chemin/vers/data`). Les fichiers sont
retrouvés par leur nom, quelle que soit l'arborescence.

## Étape 1 — Moteur d'analyse

```bash
cd Monitoring_Agent
python -m monitoring_agent datasets
python -m monitoring_agent analyze car_loan                       # drift + perf estimée vs réelle
python -m monitoring_agent analyze car_loan --label-delay 3       # les 3 dernières fenêtres sans labels
python -m monitoring_agent analyze car_loan_dq                    # problèmes de qualité de données
python -m monitoring_agent analyze car_price --window 6000        # régression : drift SANS dégradation
python -m monitoring_agent analyze car_loan --out runs/report.json
python -m pytest -q
```

Pour chaque fenêtre de production, comparée à la référence (données d'entraînement) :

| Contrôle | Méthode | Seuil |
|---|---|---|
| Drift numérique | Wasserstein / écart-type (n > 1000), sinon KS | 0.1 / p < 0.05 |
| Drift catégoriel | Jensen-Shannon (n > 1000), sinon chi² | 0.1 / p < 0.05 |
| Sévérité du drift | PSI, toujours calculé | 0.1 modéré, 0.25 majeur |
| Drift du dataset | part des features en drift | ≥ 50 % |
| Drift des prédictions | même méthode, sur le score du modèle | idem |
| Qualité | manquants, type, catégories inconnues, hors intervalle | voir `config.py` |
| Performance mesurée | labels disponibles | moyenne ± 3σ des fenêtres de référence, ou seuil métier |
| Performance **estimée** | CBPE (classification), DLE (régression), sans labels | idem |

Chaque fenêtre produit des **signaux** ; `trigger_agent` passe à `true` à partir de `warning`.
C'est le seul moment où le LLM sera appelé (étape 3).

Résultat sur Car Loan : drift détecté à la fenêtre 5, ROC AUC estimée 0.961 contre 0.960 réelle,
sans utiliser un seul label. Sur `car_price`, 5 features sur 7 dérivent mais la MAE **baisse** :
un drift n'est pas une dégradation, et l'agent ne doit pas réentraîner.

## Étape 2 — Monitoring en production

```
client ──► gateway :8100 ──► API du modèle :8000 (Agent 3, inchangée)
              │  enregistre entrées, prédiction, latence (SQLite) · reçoit les labels (/feedback)
              │  moniteur : toutes les W prédictions -> analyse de la fenêtre (étape 1)
              ▼
         /metrics ──► Prometheus :9090 ──► règles générées ──► Alertmanager :9093 ──► webhook /alerts
                          │
                          └──► Grafana :3000 (dashboard généré)
```

La gateway est un **proxy** : on ne modifie pas l'API générée par l'Agent 3, la réponse est renvoyée
telle quelle, avec un en-tête `X-Request-ID` que le client réutilise pour envoyer le vrai label plus tard.

### Démo complète (4 terminaux)

```bash
# 1. L'API du modèle (Agent 3) — l'image publiée par l'Agent 3 (amd64 : émulée sur Mac Apple Silicon)
docker run --rm --platform linux/amd64 -p 8000:8000 syrineboh/credit-risk:20261021-5992db83

# 2. Monitoring : référence notée par l'API + règles + dashboard, puis la gateway
cd Monitoring_Agent
python -m monitoring_agent init ../eval/projects/credit_risk --upstream http://127.0.0.1:8000 --name credit_risk
python -m monitoring_agent gateway credit_risk --host 0.0.0.0   # 0.0.0.0 : joignable par Prometheus (Docker)

# 3. Prometheus + Alertmanager + Grafana
docker compose -f infra/docker-compose.yml up -d
#    Grafana http://localhost:3000 (admin/admin) · Prometheus http://localhost:9090/alerts

# 4. Trafic simulé (scénarios : normal, covariate, quality, concept)
python -m monitoring_agent simulate credit_risk --scenario covariate
```

| Commande | Rôle |
|---|---|
| `init <dossier>` | lit `mlops.yaml`, fait noter `data.csv` par l'API déployée (référence), écrit `runs/<projet>/` |
| `provision <projet>` | génère `infra/prometheus/rules/<projet>.rules.yml` (validé par `promtool`) et `infra/grafana/dashboards/<projet>.json` |
| `gateway <projet>` | proxy + `/metrics` + `/feedback` + `/alerts` + `/monitoring/*` + moniteur en tâche de fond |
| `simulate <projet>` | trafic de production avec drift injecté et labels envoyés en retard |

**Ce qui est généré pour chaque projet** (pas copié) : les règles d'alerte connaissent la métrique
principale et son sens (`roc_auc` : plus haut = mieux ; `mae` : l'inverse), le seuil de `mlops.yaml`,
et la limite de variabilité calculée sur la référence colore les seuils du dashboard.

**Alertes** (étiquetées `severity` et `component` = service / data / model) : gateway ou modèle injoignable,
erreurs 5xx, entrées invalides (4xx), latence p95, drift du dataset, drift majeur d'une feature,
prediction drift, qualité des données, baisse de performance estimée, mesurée, sous le seuil métier.
Alertmanager **regroupe** (une notification par alerte et par projet) et **inhibe les conséquences**
quand la cause est connue : données cassées ⇒ on fait taire les alertes de drift ; modèle tombé ⇒
on fait taire erreurs et latence.

**Deux détections qui découlent des résultats :**
- *Concept drift suspecté* : la performance MESURÉE chute alors que l'ESTIMÉE reste normale. CBPE suppose
  que la relation features → cible est stable ; si la mesure le contredit, c'est cette relation qui a changé.
- *Tests multiples* : avec 20 features testées à p < 0.05, on attend un faux positif par fenêtre.
  Les tests (KS, chi²) utilisent la correction de Bonferroni (0.05 / nombre de features).

Limite connue : la référence est construite sur `data.csv`, qui contient les données d'entraînement ;
la performance de référence est donc optimiste. La bonne référence est le jeu de test, que l'Agent 2
pourra enregistrer dans MLflow.

## Étape 3 — L'agent qui décide

Le moteur DÉTECTE ; l'agent DÉCIDE. Il est réveillé par une fenêtre `trigger_agent` (gateway) ou appelé par
l'orchestrateur (`monitoring_agent.pipeline.run`). Graphe LangGraph (`agent/graph.py`) :

```
prepare ─► investigate ─► diagnose ×3 ─► verify ─► judge ─► policy ─┬─ auto ──► act
              (outils)     (vote)     (preuves)  (Gemini)  (règles) └─ human ─► interrupt ─► act | fin
                              ▲           │          │
                              └─ 1 révision si preuve fausse ou si le juge dit « revise »
```

| Étape | Qui | Rôle |
|---|---|---|
| investigate | LLM décideur (Groq) | appelle des **outils en lecture seule** : historique d'une feature, de la performance, qualité, prédictions, alertes, décisions passées |
| diagnose | LLM décideur ×3 | 3 diagnostics indépendants → **vote** ; l'accord mesure l'incertitude (self-consistency) |
| verify | code | chaque chiffre cité doit exister dans les FACTS calculés, à 2 % près ; sinon une révision |
| judge | LLM juge (Gemini) | **autre famille de modèle** ; grille grounding / causality / proportionality / safety |
| policy | code | substitue une action plus sûre et décide **auto** ou **humain** (règles R1–R9, `agent/guards.py`) |
| act | code | écrit un fichier pour chaque action : incident Markdown, `retraining_request.json` (contrat commun) + données labellisées à réentraîner |

**Autonomie graduée par le risque :** surveiller, ouvrir un incident, signaler un problème de données → seul.
Demander un réentraînement → seul si la policy, les preuves et le juge sont d'accord (puis gate de l'orchestrateur
et quality gate de l'Agent 2). Revenir à l'ancien modèle → toujours un humain.

**Règles de la policy :** R1 pas de réentraînement sans baisse de performance (drift ≠ dégradation) ·
R2 pas sur des données cassées (on signale la source) · R3 il faut des labels récents · R4 24 h entre deux demandes ·
R5 haut risque = humain · R6 preuves exactes · R7 juge d'accord · R8 confiance ≥ 0.6 et accord ≥ 2/3 ·
R9 disjoncteur : 10 actions automatiques / 24 h.

**Sans clé LLM**, un décideur à règles (`agent/rules.py`) prend le relais : c'est aussi la *baseline sans LLM*
de l'évaluation.

```bash
cp .env.example .env                                   # GROQ_API_KEY, GEMINI_API_KEY
python -m monitoring_agent llm-check                   # clés, modèles, JSON
python -m monitoring_agent decide car_loan --at 5      # l'agent raisonne sur une fenêtre NannyML
python -m monitoring_agent decide car_price --window 6000 --at 6   # drift sans dégradation
python -m monitoring_agent decide car_loan --at 5 --rules          # baseline sans LLM
python -m monitoring_agent gateway credit_risk --host 0.0.0.0      # en production : l'agent décide seul
curl localhost:8100/monitoring/decisions                           # décisions, trace complète par id
curl -X POST localhost:8100/monitoring/decisions/<id>/approve -H 'content-type: application/json' -d '{"by":"Mohamed"}'
```

Anti-rafale : l'agent n'est relancé que si les signaux CHANGENT ou toutes les 5 fenêtres, jamais pendant qu'une
décision attend un humain. Prompts versionnés (`agent/prompts.py`, `PROMPT_VERSION`), température 0 pour
l'enquête et le juge.

**Orchestrateur :** `from monitoring_agent.pipeline import run` remplace `agent4_stub` (même contrat).

## Étape 4 — La console

`http://localhost:8100/console/` (servie par la gateway, sans installation ni build).

- **Film des fenêtres** : une case par fenêtre, couleur = sévérité, point bleu = l'agent a décidé.
- **Drift par feature** : heatmap PSI features × fenêtres ; encadré = drift confirmé par le test.
- **Performance** : mesurée (labels) vs estimée (CBPE/DLE), niveau habituel et limite ± 3σ.
- **Décisions de l'agent** : ce qu'il a fait, seul ou validé ; boutons Approuver / Rejeter si besoin.
- **Enregistreur de décision** (clic sur une décision) : le raisonnement rejoué étape par étape — outils appelés,
  votes, preuves vérifiées, notes du juge, règles, action, rapport d'incident.
- **Simuler** : envoie du trafic (drift des entrées, concept drift, données cassées, normal) à la gateway.

## Étape 5 — Évaluation

```bash
python -m monitoring_agent evaluate --method rules            # baseline sans LLM (instantané)
python -m monitoring_agent evaluate --method both --per-dataset 2   # + LLM seul + agent complet (~10 min)
```

Vérité terrain NannyML : `car_loan` 5-9 = vraie baisse (réentraîner) · `car_price` 5-9 = drift sans dégradation
(surveiller, ne PAS réentraîner) · `car_loan_dq` 5-9 = données cassées (prévenir l'équipe data) · fenêtres 0-4 =
calmes (l'agent ne doit pas être réveillé). Mesures : exactitude, actions dangereuses, problèmes manqués,
autonomie, latence. Rapport : `runs/eval/<date>/report.md`.

`sft.jsonl` : exemples pour le fine-tuning (QLoRA), conservés seulement si la décision est correcte, ses preuves
vérifiées et le juge d'accord — distillation vérifiée d'un grand modèle vers un petit.

**Autonomie.** L'humain n'est sollicité que lorsque c'est logiquement nécessaire : action à haut risque (rollback),
rejet explicite du juge, désaccord entre le LLM et le second avis déterministe sur une action à risque, ou
disjoncteur. Si le juge est indisponible ou hésite, LLM + règles concordants suffisent ; si les preuves du LLM
restent invérifiables, l'agent applique le décideur à règles au lieu d'escalader.

## Étape 6 — Outils en MCP

Les outils d'enquête (lecture seule) sont servis par un **serveur MCP** (`agent/mcp_server.py`, SDK `mcp` 2.x) :
8 outils (`feature_history`, `performance_history`, `quality_details`, `prediction_drift`, `active_alerts`,
`previous_decisions`, `window_facts_at`, `service_health` via Prometheus), des ressources
(`monitoring://status`, `monitoring://policy`, `monitoring://decisions/{id}`) et des prompts (`diagnose`, `explain_decision`).

- l'enquêteur de l'agent est un **client MCP** : il découvre les outils (`tools/list`) puis les appelle (`tools/call`) ;
- la gateway monte le même serveur sur **`http://localhost:8100/mcp/`** (Streamable HTTP) ;
- en stdio pour Claude Desktop / MCP Inspector : `python -m monitoring_agent mcp credit_risk`.

Aucun outil MCP n'agit : agir reste derrière la vérification, le juge et la policy.

## Étape 7 — Fine-tuning du diagnostiqueur local (Qwen3-4B-Instruct-2507, MLX)

```bash
python -m monitoring_agent finetune bases                    # bases disponibles (data/)
python -m monitoring_agent finetune scenarios --per-base 220 # scénarios injectés + étiquetés (reprenable)
python -m monitoring_agent finetune teach --max-calls 400    # distillation vérifiée (Groq), reprenable, à relancer
python -m monitoring_agent finetune build                    # train/valid/test + lora.yaml
mlx_lm.lora --config runs/finetune/data/lora.yaml            # QLoRA sur Mac (Apple Silicon)
mlx_lm.server --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
              --adapter-path runs/finetune/data/adapters --port 8080
python -m monitoring_agent finetune eval --models mlx:default_model,groq:openai/gpt-oss-120b
```

- **Scénarios** : on casse la production d'une façon connue (shift, concept, pannes de données, labels en retard)
  et le vrai moteur d'analyse calcule les signaux. Un scénario n'est gardé que si l'injection et l'oracle
  (la procédure, `rules.procedure`) concluent la même chose.
- **Découpage par base** : train = 6 bases, valid = `mc_prepaid`, test = `car_loan` / `car_price` (jamais vues).
- **Professeur** : réponse à l'aveugle gardée si cause + action + preuves vérifiées ; sinon rationalisation (STaR)
  avec la bonne réponse en indice ; l'exemple d'entraînement garde le prompt sans l'indice.
- **LoRA** : rang 16, alpha 32, toutes les couches, lr 1e-4 cosinus + warmup, batch effectif 8, `mask_prompt`.
- **En production** : `MONITORING_DECIDER=mlx:default_model` dans `.env` → le SLM diagnostique (faits seuls),
  l'enquête reste à `MONITORING_INVESTIGATOR` (Groq), repli automatique sur Groq si le serveur local est arrêté.

## Structure

```
Monitoring_Agent/
├── monitoring_agent/
│   ├── schemas.py          # rapports produits (Pydantic, extra="forbid")
│   ├── config.py           # tous les seuils, versionnés
│   ├── datasets.py         # jeux NannyML (référence / production / labels)
│   ├── agent/
│   │   ├── graph.py        # graphe LangGraph : enquête -> diagnostic -> vérif -> juge -> policy -> action
│   │   ├── context.py      # dossier d'enquête : FACTS citables + outils en lecture seule
│   │   ├── guards.py       # vérification des preuves + policy-as-code
│   │   ├── llm.py          # Groq / Gemini / Ollama (protocole OpenAI), repli, JSON validé
│   │   ├── prompts.py      # prompts versionnés
│   │   ├── rules.py        # décideur sans LLM (repli + baseline)
│   │   └── actions.py      # incident, retraining_request.json, rollback (après humain)
│   ├── console/            # index.html, style.css, app.js (servis sur /console)
│   ├── evaluation.py       # règles vs LLM seul vs agent complet + sft.jsonl
│   ├── pipeline.py         # run(AgentInput) -> AgentResult pour l'orchestrateur
│   ├── runtime/
│   │   ├── gateway.py      # proxy FastAPI devant l'API du modèle
│   │   ├── store.py        # SQLite : prédictions, labels, fenêtres, alertes
│   │   ├── metrics.py      # métriques Prometheus (mlmon_*)
│   │   ├── monitor.py      # flux de prédictions -> fenêtres analysées
│   │   ├── supervisor.py   # fenêtre -> agent (anti-rafale, thread dédié, reprise humaine)
│   │   ├── target.py       # monitoring.json + référence notée par l'API
│   │   └── simulator.py    # trafic de production simulé
│   ├── provisioning/
│   │   ├── prometheus.py   # règles d'alerte du projet + validation promtool
│   │   └── grafana.py      # dashboard du projet
│   ├── analysis/
│   │   ├── stats.py        # PSI, Wasserstein, KS, Jensen-Shannon, chi² (fonctions pures)
│   │   ├── reference.py    # profil appris une fois sur la référence
│   │   ├── drift.py        # drift par colonne + drift du dataset
│   │   ├── quality.py      # contrôles de qualité
│   │   ├── performance.py  # métriques, CBPE, DLE
│   │   └── analyzer.py     # fenêtres -> rapports -> signaux
│   └── __main__.py         # ligne de commande
├── infra/                  # docker-compose + configs Prometheus / Alertmanager / Grafana
├── tests/
├── runs/                   # état local par projet (ignoré par git)
├── data/                   # ignoré par git
├── requirements.txt
└── pyproject.toml
```

Contrat avec l'orchestrateur : `common/schemas.py` (`AgentInput` → `AgentResult`, `RetrainingRequest`).
