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
| 3 | Agent LangGraph autonome : diagnostic à outils, vérification, juge, policy | à faire |
| 4 | API de monitoring + console | à faire |
| 5 | Scénarios de drift injecté, évaluation, fine-tuning | à faire |

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

## Structure

```
Monitoring_Agent/
├── monitoring_agent/
│   ├── schemas.py          # rapports produits (Pydantic, extra="forbid")
│   ├── config.py           # tous les seuils, versionnés
│   ├── datasets.py         # jeux NannyML (référence / production / labels)
│   ├── runtime/
│   │   ├── gateway.py      # proxy FastAPI devant l'API du modèle
│   │   ├── store.py        # SQLite : prédictions, labels, fenêtres, alertes
│   │   ├── metrics.py      # métriques Prometheus (mlmon_*)
│   │   ├── monitor.py      # flux de prédictions -> fenêtres analysées
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
