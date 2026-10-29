# Validateur déterministe (baseline) — documentation d'origine

> Archive de l'ancien README. Le validateur déterministe `python -m validator` est conservé tel quel comme **baseline**. Les références à `llm_validator` sont obsolètes : voir le README principal (`validator_agent`).

Le **Contract Validator** est la porte d'entrée de la plateforme. Il vérifie qu'un projet de Machine Learning respecte un **contrat** (structure, dépendances, code sûr, données exploitables…) avant que les agents ne le prennent en charge.

- **Sans LLM** : le même projet donne toujours le même verdict.
- **Sûr** : le code du projet n'est exécuté que s'il a passé toutes les analyses sans exécution, puis dans un environnement isolé.
- **Il produit un rapport JSON** (`compatibility_report.json`) qui sert d'entrée à l'agent suivant.

```
Projet ML ──► Contract Validator ──► rapport JSON ──► Agent 1 ──► …
              (accepté / à confirmer / rejeté)        (si le passage de relais est autorisé)
```

---

## 1. Prérequis

| | |
|---|---|
| Système | **Ubuntu 24.04 LTS** recommandé · compatible **WSL2** (Windows 10/11) · architecture x86_64 / amd64 |
| Python | **≥ 3.12** (version validée : 3.12.3). Python 3.8 n'est **pas** supporté |
| Outils | `git`, `python3-venv`, `python3-pip`, `build-essential` |
| Réseau | Internet pour l'installation, et au **premier** lancement d'une validation (téléchargement des dépendances du projet testé) |

Windows sans WSL n'est pas supporté (le validateur utilise des fonctions Linux).

Versions de référence utilisées pendant le développement : Ubuntu 24.04.5 LTS · Python 3.12.3 · pip 24.0+ · Git 2.43.0+

---

## 2. Installation

**1. Installer les outils système** (une seule fois) :

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip git build-essential
```

**2. Récupérer le code** :

```bash
mkdir -p ~/projects && cd ~/projects
git clone <URL_DU_DEPOT> agentic-mlops
cd agentic-mlops
git checkout feature/contract-validator      # si la branche n'est pas encore fusionnée dans main
```

**3. Créer l'environnement virtuel du validateur** :

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

> À refaire à chaque nouveau terminal : `source .venv/bin/activate` (le prompt affiche `(.venv)`).

**4. Installer le projet et ses dépendances de développement** :

```bash
python -m pip install -e ".[dev]"
```

**5. Vérifier** :

```bash
python --version        # doit afficher 3.12 ou plus
python -m pip --version
git --version
python -m pytest -q     # 62 tests, tous doivent passer
```

Les tests demandent **1 à 3 minutes la première fois** : le validateur télécharge scikit-learn, pandas, etc. pour le projet d'essai, puis les garde en cache (`~/.cache/contract-validator`). Les fois suivantes : environ 30 secondes.

> **WSL2 — erreurs TLS ou téléchargements qui échouent ?** Sur certains réseaux, il faut réduire le MTU à 1400 :
> `sudo ip link set dev eth0 mtu 1400` (valable jusqu'au redémarrage de WSL).

---

## 3. Utilisation en 3 commandes

Toujours depuis la racine du dépôt, avec l'environnement activé.

```bash
# 1) Valider un projet
python -m validator eval/projects/credit_risk

# 2) Valider et produire le rapport pour l'agent suivant (+ copie figée du projet)
python -m validator eval/projects/credit_risk \
    --json compatibility_report.json --snapshot-dir ~/validated

# 3) Vérifier qu'un rapport peut être utilisé (c'est le contrôle de l'agent suivant)
python -m validator.handoff verify compatibility_report.json
```

Le premier lancement sur un nouveau projet prend 20 à 60 secondes (création de son environnement isolé), les suivants environ 5 secondes.

Pour valider **votre propre** projet : `python -m validator /chemin/vers/mon_projet` (n'importe où sur le disque).

---

## 4. Lire le résultat

### Les 4 verdicts

| Verdict | Signification | Code de sortie |
|---|---|---|
| `ACCEPTED` | Conforme | `0` |
| `ACCEPTED_WITH_WARNINGS` | Conforme, avec des points à surveiller | `0` |
| `NEEDS_CONFIRMATION` | Un humain doit confirmer quelque chose (ex. `mlops.yaml` absent) | `10` |
| `REJECTED` | Non conforme : au moins un bloquant | `20` |

### Les 4 niveaux de gravité d'un constat

`✖ BLOCKER` rejette · `? CONFIRM` demande une confirmation · `! WARNING` à surveiller · `i INFO` simple information.

### Exemple de sortie

```
Verdict : ACCEPTED   (0 bloquant(s), 0 confirmation(s), 0 avertissement(s))   [4.1s]
Niveaux : structure ✔ | manifest ✔ | static ✔ | data ✔ | dynamic ✔
Passage de relais : READY (peut continuer : oui)
Métriques observées : accuracy=0.7150, ..., roc_auc=0.7773
```

Chaque constat indique un **code stable** (ex. `C6.UNPINNED`), le fichier et la ligne, une **preuve** et une **piste de correction** (`piste`). Exemple :

```
[✖ BLOCKER] C6.UNPINNED requirements.txt:3 — Dépendance non épinglée : scikit-learn>=1.3
        piste  : Épinglez la version exacte, ex. `scikit-learn==<version testée>`.
```

### Les 5 niveaux de contrôle

| # | Niveau | Exécute le code du projet ? |
|---|---|---|
| 1 | Structure et dépendances | Non |
| 2 | Manifest `mlops.yaml` | Non |
| 3 | Analyse du code (AST, secrets, Bandit) | Non |
| 4 | Données (CSV) | Non |
| 5 | Exécution de contrôle en environnement isolé | **Oui, seulement si 1 à 4 n'ont aucun bloquant** |

---

## 5. Préparer un projet ML pour qu'il soit accepté

### Ce que doit contenir le dossier du projet

```
mon_projet/
├── main.py            # lance tout, sans argument ni saisie
├── train.py           # def train(data_path, output_dir) -> dict de métriques
├── predict.py         # def load_model(model_path) et def predict(model, data)
├── requirements.txt   # versions épinglées : paquet==version
├── data.csv           # données (UTF-8, avec la colonne cible)
└── mlops.yaml         # tâche, cible, métrique
```

### Le contrat en clair

| Règle | En clair |
|---|---|
| C1 | Les 4 fichiers `main.py`, `train.py`, `predict.py`, `requirements.txt` sont **à la racine** |
| C2 | Python + **scikit-learn** + pandas. Pas de PyTorch, TensorFlow, XGBoost… (modifiable, voir §7) |
| C3 | `train(data_path, output_dir)` retourne un **dict** contenant la métrique du manifest et écrit `model.joblib` dans `output_dir` |
| C4 | `load_model(model_path)` ; `predict(model, data)` accepte un **dict** *et* un **DataFrame**, et retourne une **liste** |
| C5 | `main.py` tourne sans argument, **sans `input()`**, et écrit dans `$OUTPUT_DIR` |
| C6 | Toutes les dépendances sont épinglées avec `==` |
| C7 | CSV en UTF-8, au moins 50 lignes, colonne cible présente, au moins 2 classes |
| C8 | `mlops.yaml` présent et valide (absent = confirmation humaine demandée) |
| C9 | Graines fixées (`random_state=...`) : deux entraînements donnent le même résultat |
| C10 | Aucun chemin absolu ; **rien n'est écrit en dehors de `output_dir`** |
| C11 | Pas de `eval`/`exec`/`os.system`, pas de réseau, pas de secret dans le code, pas de `.pkl` commité |
| C12 | Limites : 500 fichiers, 100 Mo, 50 Mo de données, 300 s d'exécution |

### Exemple de `mlops.yaml`

```yaml
project:
  name: credit-risk
  task: classification          # classification | regression
data:
  path: data.csv
  target: target                # nom de la colonne à prédire
evaluation:
  primary_metric: roc_auc       # classification : accuracy, precision, recall, f1, roc_auc
  min_threshold: 0.70           # régression : rmse, mae, r2
```

### Exemple minimal des fonctions attendues

```python
# train.py
def train(data_path, output_dir):
    ...                                       # entraîne, écrit output_dir/model.joblib
    return {"roc_auc": 0.78, "f1": 0.79}      # doit contenir la métrique du manifest

# predict.py
def load_model(model_path): ...
def predict(model, data): ...                 # data : dict, liste de dict ou DataFrame -> liste
```

Un projet complet et conforme est fourni : `eval/projects/credit_risk/`.

---

## 6. Le rapport et le passage de relais (« handoff »)

Le rapport `compatibility_report.json` contient le verdict, les constats détaillés, et surtout une section **`handoff`** : le résumé destiné à l'agent suivant.

| Élément | Rôle |
|---|---|
| `handoff.can_proceed` | L'agent suivant a-t-il le droit de démarrer ? |
| `handoff` (tâche, cible, métrique, colonnes d'entrée, dépendances, métriques observées) | Ce que le validateur a déjà établi : l'agent n'a pas à le refaire |
| `handoff.carry_forward` | Mises en garde à transmettre (classes déséquilibrées, petit jeu de données…) |
| `project_hash` | Empreinte du projet validé : une seule modification la change |
| `snapshot_path` | Copie figée en lecture seule du projet validé (avec `--snapshot-dir`) |
| `report_sha256` | Sceau : détecte toute modification du rapport |

### Les 5 statuts du handoff

| Statut | Signification | L'agent suivant peut démarrer ? |
|---|---|---|
| `READY` / `READY_WITH_WARNINGS` | Conforme et exécuté | Oui |
| `AWAITING_CONFIRMATION` | Confirmation humaine requise | Non |
| `INCOMPLETE` | Pas d'erreur, mais le projet n'a **pas été exécuté** (`--no-dynamic`) | Non |
| `BLOCKED` | Rejeté | Non |

> L'orchestrateur doit lire **`can_proceed`** (ou appeler `verify_handoff`), **pas seulement le verdict** : avec `--no-dynamic`, le verdict peut être `ACCEPTED` alors que le passage est refusé.

### Côté agent / orchestrateur

```python
from validator.handoff import verify_handoff

check = verify_handoff("compatibility_report.json", project_dir)   # project_dir facultatif si snapshot_path
if not check.ok:
    raise SystemExit(check.reasons)        # liste lisible des raisons du refus
```

`verify_handoff` contrôle : la version du format, l'intégrité du rapport, `can_proceed`, et que le projet n'a pas changé depuis la validation. Schéma JSON du rapport : `docs/compatibility_report.schema.json`.

---

## 7. Options utiles

| Option | Effet |
|---|---|
| `--json FICHIER` | Écrit le rapport JSON |
| `--snapshot-dir DOSSIER` | Copie figée du projet validé (créée seulement si le passage est autorisé) |
| `--no-dynamic` | N'exécute jamais le code du projet (analyses 1 à 4 seulement) |
| `--allow-framework xgboost` | Élargit le contrat à un framework (répétable) |
| `--env venv` (défaut) / `--env current` | `venv` : environnement isolé construit depuis le `requirements.txt` du projet. `current` : Python courant (déconseillé) |
| `--cache-dir DOSSIER` | Emplacement du cache des environnements (défaut `~/.cache/contract-validator`, ou variable `CV_CACHE_DIR`) |
| `--timeout N` | Limite d'exécution en secondes (défaut 300) |
| `--no-determinism` | Saute le second entraînement de contrôle |
| `--no-bandit` | Désactive Bandit |
| `--confirmed-by "Nom"` | Enregistre qui a confirmé le manifest (simple trace : ne change **jamais** le verdict) |
| `-q` | Mode silencieux (utile avec `--json` ou dans la CI) |

---

## 8. Organisation du dépôt

```
agentic-mlops/
├── README.md
├── pyproject.toml
├── validator/                  # le code du validateur
│   ├── __main__.py             # ligne de commande
│   ├── engine.py               # enchaîne les niveaux et calcule le verdict
│   ├── contract.py             # le contrat C1–C12 (à modifier pour changer une règle)
│   ├── models.py               # schémas du rapport et du manifest
│   ├── check_structure.py      # niveau 1 : fichiers, dépendances
│   ├── check_manifest.py       # niveau 2 : mlops.yaml
│   ├── check_static.py         # niveau 3 : analyse du code sans l'exécuter
│   ├── check_data.py           # niveau 4 : CSV
│   ├── check_dynamic.py        # niveau 5 : exécution de contrôle
│   ├── sandbox.py              # environnement isolé (venv + Docker)
│   ├── harness.py              # script exécuté dans le sandbox
│   ├── handoff.py              # passage de relais, empreinte, sceau
│   ├── context.py  fs.py       # état partagé, lecture sûre des fichiers
├── llm_validator/              # validateur piloté par LLM (Groq) : voir §13
│   ├── agent.py  tools.py  llm_client.py
│   └── prompts/system.md       # le contrat en langage naturel = le prompt
├── tests/                      # 62 tests
├── eval/
│   ├── make_variants.py        # génère les projets cassés pour les tests
│   ├── make_adversarial.py     # génère les cas pièges
│   ├── compare.py              # compare baseline et LLM (faux acceptés, stabilité, tokens)
│   └── projects/
│       ├── credit_risk_raw/    # projet original, non conforme (copie intacte)
│       ├── credit_risk/        # version conforme (référence)
│       ├── variants/           # 14 variantes cassées (générées)
│       └── adversarial/        # 3 cas pièges (générés) : injection de prompt, appel obfusqué, métriques truquées
└── docs/
    └── compatibility_report.schema.json
```

---

## 9. Tests et projets d'essai

```bash
python -m pytest -q                         # tout
python -m pytest tests/test_handoff.py -q   # le passage de relais seulement
python eval/make_variants.py                # régénère les 14 variantes cassées
```

Les tests valident **16 projets dont le verdict est connu à l'avance** : la référence doit être acceptée, le projet brut rejeté, et chaque variante cassée doit recevoir exactement le verdict prévu (par exemple `02_unpinned_deps` → `C6.UNPINNED`). Ils vérifient aussi le sandbox : boucle infinie arrêtée, réseau bloqué, variables secrètes non transmises.

**Après avoir modifié `models.py`**, régénérez le schéma (un test échoue sinon) :

```bash
python -m validator.handoff schema docs/compatibility_report.schema.json
```

---

## 10. Problèmes fréquents

| Symptôme | Cause et solution |
|---|---|
| `externally-managed-environment` | L'environnement virtuel n'est pas activé. Faites `source .venv/bin/activate`. N'utilisez pas `--break-system-packages` |
| `No module named validator` | Environnement non activé, ou `pip install -e ".[dev]"` non fait, ou commande lancée hors de la racine du dépôt |
| `Multiple top-level packages discovered` à l'installation | `pyproject.toml` doit contenir `[tool.setuptools]` et `packages = ["validator"]` |
| `ensurepip is not available` / création du venv impossible | `sudo apt install -y python3-venv` |
| `C6.INSTALL_FAILED` | Une version épinglée n'existe pas pour Python 3.12, ou pas de réseau. Lisez le champ `preuve` |
| Premier lancement très long | Normal : création et remplissage du cache. Pour repartir de zéro : `rm -rf ~/.cache/contract-validator` |
| Téléchargements ou TLS qui échouent (WSL2) | `sudo ip link set dev eth0 mtu 1400` |
| `Réseau non isolé dans ce sandbox` (INFO) | Ubuntu bloque `unshare`. Ce n'est pas bloquant, mais le validateur est alors moins étanche. Testez : `unshare -rn true` |
| Tests qui échouent sans Internet | Le premier lancement a besoin du réseau pour construire l'environnement du projet d'essai |
| `INCOMPLETE` alors que le verdict est `ACCEPTED` | Vous avez utilisé `--no-dynamic` : le projet n'a pas été exécuté, donc pas de passage de relais |

---

## 11. Limites connues

- **Le backend Docker est écrit mais n'a pas été testé.** Seul le repli `subprocess` est couvert par les tests.
- Ce repli n'est **pas** une barrière de sécurité complète : réseau coupé et ressources limitées, mais ni système de fichiers en lecture seule ni utilisateur non-root. Utilisez Docker en production.
- L'analyse statique ne voit pas le code volontairement obfusqué : c'est pourquoi le code n'est exécuté que dans le sandbox.
- Les secrets sont détectés par une liste de motifs (pas de gitleaks).
- Le **sceau** du rapport détecte les modifications accidentelles, il n'authentifie pas son auteur (pas de signature).
- `--confirmed-by` n'authentifie pas la personne : c'est une déclaration, enregistrée comme trace.
- Linux et macOS uniquement (sous Windows : WSL2).

---

## 12. Travailler en équipe

```bash
git checkout -b feature/ma-modification
# ... modifications ...
python -m pytest -q                 # doit rester vert
git add .
git commit -m "feat(validator): description courte"
git push -u origin feature/ma-modification
```

Puis ouvrez une Pull Request vers `main`. Pour changer une règle du contrat, modifiez `validator/contract.py` et ajoutez une variante cassée dans `eval/make_variants.py` : elle sert de test de non-régression.

---

## 13. Mode LLM (Groq)

`python -m llm_validator` remplace les règles codées en Python par un **LLM qui lit le projet, appelle des outils et dépose un verdict prouvé**. Le contrat n'est plus du code : c'est un texte (`llm_validator/prompts/system.md`) que l'on modifie sans toucher au programme. Le validateur déterministe est conservé comme **baseline** de comparaison.

```bash
export GROQ_API_KEY="..."              # ne jamais la committer ni la coller dans un message
export GROQ_MODEL="..."                # facultatif ; défaut openai/gpt-oss-120b (vérifiez la liste sur console.groq.com/docs/models)

python -m llm_validator eval/projects/credit_risk
python -m llm_validator eval/projects/credit_risk --json compatibility_report.json --snapshot-dir ~/validated
python -m llm_validator PROJET --cache-dir ~/.cache/llm-validator/cassettes          # enregistre les réponses du LLM
python -m llm_validator PROJET --cache-dir ~/.cache/llm-validator/cassettes --replay # rejoue sans appeler Groq
```

Le rapport est **le même format** que celui du mode déterministe : `verify_handoff`, l'empreinte et le sceau fonctionnent à l'identique.

### Comment ça marche

1. Le LLM reçoit le prompt (contrat, procédure, règles de preuve, consignes de sécurité, exemples) et **5 outils** : `list_files`, `read_file`, `profile_csv`, `run_contract_check` (exécution en sandbox), `submit_verdict`.
2. Il lit, raisonne, appelle les outils, puis dépose son verdict avec un `project_spec` et des constats.
3. Le code vérifie le **protocole**, pas le projet : chaque constat doit citer un extrait réellement présent dans une sortie d'outil (sinon le LLM doit corriger) ; le verdict doit correspondre à la gravité de ses constats ; pas d'`ACCEPTED` sans exécution réussie.

### Garde-fous qui ne dépendent pas du LLM

| Garde-fou | Pourquoi |
|---|---|
| Exécution **uniquement** dans le sandbox, commande fixe | Un projet piégé ne doit pas pouvoir convaincre le LLM de le lancer sans isolation |
| Lecture confinée au projet, secrets masqués, jamais de ligne de données envoyée | Le code et les données partent vers un service externe |
| Contenu du projet balisé `<file … untrusted="true">` + consigne d'ignorer ses instructions | Injection de prompt |
| **Échec fermé** : erreur Groq, quota, budget de tokens, trop d'étapes → `NEEDS_CONFIRMATION` | Jamais `ACCEPTED` par défaut |
| Budgets : étapes (14), tokens (60 000), appels d'outils par étape | Coût et boucles infinies |

### Techniques utilisées

| Technique | Où |
|---|---|
| Prompt système structuré (rôle, contrat, procédure, format, sécurité) | `prompts/system.md` |
| Contrat en langage naturel avec cas conforme / non conforme | `prompts/system.md` §Contrat |
| Few-shot (3 exemples inventés, séparés des projets d'évaluation) | `prompts/system.md` §Exemples |
| Appel d'outils (function calling) en boucle observer → décider → agir | `agent.py`, `tools.py` |
| Sortie structurée validée par schéma (Pydantic) | `submit_verdict`, `agent.py` |
| Ancrage des preuves (extraits vérifiés) et auto-correction sur refus | `tools.is_grounded`, `agent._check_submission` |
| Délimitation des données non fiables (anti-injection) | `tools.read_file` |
| Température 0, nouvelles tentatives, respect du quota de tokens | `llm_client.py` |
| Versionnage du prompt (hash enregistré dans chaque rapport) | `facts.llm.prompt_sha256` |
| Enregistrement / rejeu des réponses | `CachedClient` |
| Évaluation contre une vérité de référence, stabilité sur N lancements, cas adverses | `eval/compare.py` |

Non utilisés : fine-tuning, RAG, plusieurs agents qui débattent, LLM qui juge un autre LLM.

### Évaluation

```bash
python eval/compare.py --mode baseline                          # validateur déterministe
python eval/compare.py --mode llm --runs 3 --out llm.json       # LLM, 3 lancements par projet
```

Résultat **mesuré** pour la baseline sur 19 projets (14 variantes, 2 de référence, 3 cas pièges) : **17/19 exacts, 2 faux acceptés** — l'appel système obfusqué (`A2`) et les métriques truquées (`A3`). Ce sont les cas où un LLM peut apporter quelque chose. **Le résultat du mode LLM n'est pas encore mesuré** : il faut une clé Groq.

### Limites à connaître

- **Le client Groq n'a pas été testé contre l'API réelle** (réseau simulé uniquement). Premier essai : un projet simple, en lisant le message d'erreur s'il y en a un.
- **Plan gratuit Groq** : quotas serrés (tokens par minute). Le prompt fait environ 2 000 tokens et est renvoyé à chaque étape : attendez-vous à des pauses, ou prenez un plan payant pour les évaluations.
- **Non-déterminisme** : même à température 0, le verdict peut varier. Mesurez-le avec `--runs`.
- **Confidentialité** : le code (secrets masqués) et des statistiques sur les données sont envoyés à Groq. Vérifiez que c'est acceptable pour vos données.
- Le sandbox local n'est pas une barrière complète (voir §11) : le cas `A2` est **exécuté** par la baseline (inoffensif ici) ; utilisez Docker en production.
