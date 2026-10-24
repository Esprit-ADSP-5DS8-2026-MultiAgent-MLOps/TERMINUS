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
| 2 | Gateway devant l'API de l'Agent 3 + Prometheus / Alertmanager / Grafana | à faire |
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

## Structure

```
Monitoring_Agent/
├── monitoring_agent/
│   ├── schemas.py          # rapports produits (Pydantic, extra="forbid")
│   ├── config.py           # tous les seuils, versionnés
│   ├── datasets.py         # jeux NannyML (référence / production / labels)
│   ├── analysis/
│   │   ├── stats.py        # PSI, Wasserstein, KS, Jensen-Shannon, chi² (fonctions pures)
│   │   ├── reference.py    # profil appris une fois sur la référence
│   │   ├── drift.py        # drift par colonne + drift du dataset
│   │   ├── quality.py      # contrôles de qualité
│   │   ├── performance.py  # métriques, CBPE, DLE
│   │   └── analyzer.py     # fenêtres -> rapports -> signaux
│   └── __main__.py         # ligne de commande
├── tests/
├── data/                   # ignoré par git
├── requirements.txt
└── pyproject.toml
```

Contrat avec l'orchestrateur : `common/schemas.py` (`AgentInput` → `AgentResult`, `RetrainingRequest`).
