# Agentic MLOps — Validator · Repair · Judge

Trois agents pilotés par LLM, un **schéma commun v2**, un **moteur de politique déterministe**, et le validateur
déterministe d'origine conservé comme **baseline** (`validator/`, inchangé).

```
                        ┌──────────── contrat d'admission (YAML, par profil) ────────────┐
                        │                                                                  │
 projet ──► VALIDATOR ──┴─► rapport v2 ──► REPAIR ──► copie corrigée + patchs ──► JUDGE ──► verdict
            (LLM + moteur de politique)    (LLM, périmètre gardé)                 (LLM, lecture seule)
                 ▲                                                                           │
                 └────────────── revalidation (nouveau rapport, lignage complet) ◄── REVALIDATE
```

| Qui | Fait |
|---|---|
| **Les LLM** | comprennent le projet, choisissent les contrôles, produisent résultats **avec preuves vérifiées**, classent les problèmes, réparent, jugent |
| **Le contrat** (`policy/contracts/*.yaml`) | liste les contrôles du catalogue, obligatoires ou recommandés, et quand un contrôle peut être « non applicable » |
| **Le moteur de politique** (`policy/engine.py`) | calcule le statut. Contrôle obligatoire en `FAIL` → bloque. Obligatoire `NOT_CHECKED` (ou `NOT_APPLICABLE` injustifié) → pas d'`ACCEPTED` |
| **L'infrastructure** | confinement, masquage des secrets, balisage anti-injection, vérification des preuves, périmètre du Repair, hash avant/après, sandbox |

Un LLM ne peut donc **ni décider du statut, ni « sauter » un contrôle gênant, ni inventer une preuve**.

---

## 1. Installation

Prérequis : Linux (Ubuntu 24.04 / WSL2), **Python ≥ 3.12**, accès réseau pour installer et pour construire l'environnement des projets testés.

```bash
unzip agentic-mlops-validator-agent-v3.zip && cd agentic-mlops-llmValidator
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
export GROQ_API_KEY="gsk_..."          # clé gratuite : console.groq.com (jamais dans le code ni dans un message)
```

Facultatif : Docker (isolation réseau complète). Sans Docker, repli sur un sous-processus restreint (le rapport le signale : `sandbox`).

## 2. Exécuter la chaîne complète

Les trois agents communiquent **uniquement par des rapports JSON scellés** :

```bash
P=eval/projects/variants/04_hardcoded_path        # un projet à chemin absolu en dur
mkdir -p out

# 1) Validator -> rapport v2 (statut calculé par le moteur de politique)
python -m validator_agent $P --json out/v1.json

# 2) Repair -> copie corrigée isolée (l'original n'est jamais modifié) + patchs + tests relancés
python -m repair_agent $P --report out/v1.json --output out/fixed --json out/r1.json

# 3) Judge -> verdict sur la réparation (jamais une décision d'admission)
python -m repair_agent.judge $P --repair out/r1.json --validator out/v1.json --json out/j1.json

# 4) Revalidation de la copie corrigée -> NOUVEAU rapport, lignage validator -> repair -> judge
python -m validator_agent out/fixed --parent out/j1.json --json out/v2.json

# Vérifier un rapport (version, sceau, empreinte du projet) et afficher son lignage
python -m agentic_core.verify out/v2.json out/fixed
```

Options utiles : `--contract training-only` (autre profil), `--env current` (interpréteur courant au lieu d'un venv),
`--env /chemin/python` (interpréteur de votre choix), `--llm-cache DIR [--replay]` (cassettes : rejeu gratuit), `--model`, `-q`.
Retry après un `REVISE` : `python -m repair_agent ... --attempt 2 --feedback out/j1.json --output out/fixed2`.

Codes de sortie : Validator `0` ACCEPTED · `10` NEEDS_REVIEW · `20` REPAIR_REQUIRED · `30` REJECTED — Repair `0` REPAIRED/NOTHING_TO_DO · `10` PARTIAL · `20` ESCALATED · `30` FAILED — Judge `0` ACCEPT · `10` REVISE · `20` REJECT · `30` échec fermé.

**Un modèle par agent** (le Judge est volontairement d'une autre famille que le Repair, pour réduire le biais d'auto-approbation) :
`GROQ_MODEL_VALIDATOR`, `GROQ_MODEL_REPAIR`, `GROQ_MODEL_JUDGE`, `GROQ_REASONING_EFFORT_<AGENT>` (`low|medium|high|none`).
Défauts dans `agentic_core/llm_client.py` (`AGENT_DEFAULTS`) — **vérifiez les noms de modèles sur console.groq.com**.

## 3. Tests

```bash
pytest -q --ignore=tests/test_validator.py --ignore=tests/test_handoff.py   # 131 tests, hors réseau, ~3 s : noyau, politique, 3 agents, évaluation
pytest -q                                                                   # + baseline (41 tests ; 10 construisent des venvs : RÉSEAU REQUIS)
CV_RUN_SLOW=1 pytest tests/test_integration_slow.py                         # chaîne complète avec exécution réelle (LLM scripté)
CV_RUN_SLOW=1 CV_PYTHON=/chemin/python-avec-sklearn pytest tests/test_integration_slow.py   # idem sans réseau
```

## 4. Évaluation (`eval/`)

```bash
python eval/make_annotations.py                      # régénère eval/annotations.json (29 projets : 14 variantes, 3 pièges, 10 multi-frameworks/profils)
python eval/compare.py --mode baseline               # baseline déterministe (réseau requis pour les venvs)
python eval/compare.py --mode agent --runs 3 --out results_agent.json
python eval/compare.py --mode agent --split holdout  # jeu JAMAIS vu
python eval/judge_eval.py --scope-only               # garde de périmètre seule, sans LLM
python eval/judge_eval.py --out results_judge.json   # le Judge LLM sur 10 réparations annotées (3 bonnes, 7 mauvaises dont 6 détectables mécaniquement)
python eval/repair_eval.py --max-attempts 2          # taux de réparation, tentatives, régressions
```

Métriques : rappel, précision, taux de fausses alertes, fiabilité des décisions, et le **faux accepté** (la plus grave). Le statut attendu est
calculé par le moteur de politique à partir des contrôles attendus en échec : relisez/corrigez `eval/annotations.json` si votre jugement diffère.
Discipline du jeu `holdout` : ne l'utilisez **ni** pour régler le prompt, **ni** pour écrire le contrat.

## 5. Ce qui a été fait, par rapport à la spécification

| Spécification | Réalisation |
|---|---|
| `validator/` inchangé (baseline) | **vérifié identique** à l'archive d'origine. Les « extensions » de `handoff.py`/`engine.py` sont faites dans de **nouveaux** modules qui importent l'existant (`agentic_core/handoff.py`, `policy/engine.py`) |
| `agentic_core/` | schémas v2 (+ export JSON dans `docs/schemas/`), boucle générique, client LLM, progression, outils communs, sandbox, handoff |
| `policy/` | format de contrat (`model.py`), 2 profils (`sklearn_tabular`, `training_only`), moteur déterministe, politique du Repair (`repair_policy.yaml`) |
| `validator_agent/` | découverte sans structure imposée, notebooks (cellules de code), pyproject/environment.yml/requirements, inspecteur de données multi-formats (limites signalées), commandes sandbox en liste blanche, invariants (preuves, cohérence, exécution obligatoire, recoupement faits↔LLM) |
| `repair_agent/` | copie isolée + hash avant/après, patchs/diffs/historique/rollback, lanceur de tests, garde de périmètre, comparaison de métriques avant/après |
| `repair_agent/judge/` | 5 critères à preuves vérifiées, outils en lecture seule, **diff et périmètre recalculés indépendamment**, invariants, modèle distinct enregistré |
| `llm_client` | tokens en cache déduits de l'anticipation du quota, `reasoning_effort`, température, modèle par agent, erreurs typées |
| `check_*.py` recyclés | `agentic_core/facts.py` réutilise leurs constantes et règles (AST, secrets, réseau…) et ne renvoie que des **faits** |
| Erreurs = statuts | `run_status`/`failure` dans les 3 rapports ; quota, timeout, budget → `FAILED_CLOSED`, jamais d'exception ni de décision favorable |

Note : l'archive reçue avait déjà renommé `llm_validator` en `validator_agent` (mais tests et README pointaient encore l'ancien nom, et `pyproject.toml` était invalide) ; j'ai corrigé ces trois points.

### Les deux points « à faire trancher » — décisions par défaut (modifiables par données)
- **Frameworks V1 : scikit-learn** (+ pandas/numpy/scipy/xgboost/lightgbm). PyTorch/TensorFlow sont `FRAMEWORK.SCOPE` → `REJECTED` (déclarés, jamais exécutés). Changer = éditer `frameworks_in_scope` du contrat.
- **Réparations autonomes : `mechanical` et `configuration`.** `logic` et `methodological` ne sont jamais réparées seules (le LLM n'est même pas appelé) : `needs_human`. Les problèmes de **sécurité** ne sont jamais réparés par un LLM. Changer = `auto_repair_natures` dans `policy/repair_policy.yaml`.

### Ajouter un profil ou un framework = ajouter une donnée
Copiez `policy/contracts/training_only.yaml`, changez `id` et la liste de contrôles ; `--contract chemin.yaml` suffit. Un test prouve qu'un même jeu de contrôles donne `REPAIR_REQUIRED` en `sklearn-tabular` et `ACCEPTED` en `training-only` sans toucher au code.

## 6. Garanties… et limites (à lire)

**Vérifié ici** : les 131 tests hors réseau passent ; la chaîne Validator → Repair → Judge → revalidation a été exécutée **pour de vrai** (sandbox, copie, patch, tests, métriques, lignage) avec un LLM **scripté** ; les projets annotés « conformes » ont été exécutés (credit_risk, NLP, série temporelle, entraînement seul).

**Non vérifié** :
- **Aucun appel réel à Groq n'a été fait** (pas de clé ni de réseau dans l'environnement de développement). Le comportement des vrais modèles — qualité des contrôles, des réparations, des verdicts, stabilité, **consommation de tokens** — reste à mesurer avec `eval/compare.py`, `eval/judge_eval.py`, `eval/repair_eval.py`. Les noms de modèles par défaut sont à confirmer.
- **Docker n'est toujours pas testé.** Sans Docker, le repli sous-processus n'isole pas le réseau (signalé dans le rapport) : le garde-fou statique (refus d'exécuter si appels dangereux/réseau/secrets) reste la protection principale — il ne voit pas toutes les obfuscations (le cas `A2` est couvert via `getattr(os, …)`, pas toutes les variantes).
- Les 10 tests dynamiques de la **baseline** construisent des venvs (`pip install`) : ils échouent sans réseau (`C6.INSTALL_FAILED`), indépendamment de ces changements.
- **V1 d'exécution** : seuls les projets installables depuis `requirements.txt` sont exécutés. `pyproject.toml`/`environment.yml` sont **lus**, mais l'exécution est déclarée `NOT_CHECKED` (→ `NEEDS_REVIEW`/`REPAIR_REQUIRED` selon le contrat).
- Le Judge peut se tromper : sa garde de périmètre déterministe attrape 6 des 7 mauvaises réparations du jeu annoté sans LLM ; la 7ᵉ (réparation **incomplète**) exige un jugement sémantique — c'est ce que `judge_eval.py` mesure.
- Le scellé (`report_sha256`) détecte une altération, ce n'est **pas** une signature cryptographique.

## 7. Points d'intégration pour l'orchestrateur (hors périmètre, préparés)

- Chaque agent est une **fonction** `(rapports, projet) -> rapport` sans état caché : `run_validator`, `run_repair`, `run_judge` ; rappelables, reprenables.
- Chaque rapport porte `next_action` (Validator : `PROCEED|REPAIR|HUMAN_REVIEW|STOP` ; Repair : `JUDGE|ESCALATE|NONE` ; Judge : `REVALIDATE|RETRY|ESCALATE|ROLLBACK`) et `run_status`/`failure`.
- Schémas JSON exportés : `docs/schemas/*.schema.json` (régénérer : `python -m agentic_core.export_schemas docs/schemas` ; un test vérifie qu'ils sont à jour). L'équipe LangGraph peut développer avec des agents factices dès maintenant.
- Le « paquet revue humaine » (rapport, diff, preuves) est produit par les agents : `RepairReport.diff`, `patches`, `scope_flags`, `tests`, `metrics_comparison` ; `JudgeVerdict.criteria[].evidence`.
- Action `ROLLBACK` du Judge : `repair_agent.agent.rollback_workspace(report)` supprime la copie corrigée (l'original n'a jamais bougé).

## 8. Arborescence

```
agentic_core/     orchestrator/ (squelette), schémas v2, boucle générique, client LLM, progression, outils communs, faits statiques, sandbox, handoff, export/verify
policy/           contrats d'admission (YAML), moteur de politique, politique du Repair
validator_agent/  Agent 1        repair_agent/  Agent 2        (+ judge/ : évaluateur intégré)
validator/        BASELINE déterministe (inchangée ; voir docs/BASELINE.md)
eval/             annotations, projets (variantes, pièges, multi-frameworks), cas du Judge, métriques, compare / judge_eval / repair_eval
tests/            131 tests hors réseau + baseline + intégration lente
docs/schemas/     schémas JSON des 3 rapports
```
