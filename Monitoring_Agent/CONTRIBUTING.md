# Contribuer à l'Agent 4

## Avant de commencer

```bash
cd Monitoring_Agent
source agent_env/bin/activate        # ou ton environnement Python 3.11
pip install -r requirements.txt
cp .env.example .env                 # tes clés restent dans .env, jamais ailleurs
```

## Branches

- Le travail de l'Agent 4 se fait sur `feature/monitoring-agent`.
- L'intégration avec les autres agents passe par l'orchestrateur, avec une pull request relue par au moins un membre du groupe.
- Une branche courte par changement (`fix/...`, `feat/...`) si plusieurs personnes travaillent sur l'agent en même temps.

## Messages de commit

Format court, en français : `type(portée): ce qui change`.

| Type | Quand |
|---|---|
| `feat` | Nouvelle fonctionnalité |
| `fix` | Correction de bug |
| `docs` | Documentation seulement |
| `test` | Tests seulement |
| `refactor` | Réorganisation sans changement de comportement |

Exemple : `fix(monitoring): délai du juge réduit à 60 s`.

## Tests

```bash
python -m pytest -q                  # tous les tests
python -m ruff check .               # style et erreurs courantes
```

Règles pour les tests :

- jamais d'appel à un vrai LLM, à un vrai webhook ou à une API payante : on utilise de faux modèles qui répondent selon un script (voir `tests/test_agent.py`) ;
- un test qui a besoin des jeux de données doit être sauté proprement s'ils sont absents ;
- une règle de la policy modifiée = un test qui le prouve.

## Ce qui ne doit jamais être commité

- `.env` et toute clé d'API (Groq, Gemini, Langfuse, webhook) ;
- `data/` (environ 26 Go) ;
- `runs/` (état local : bases SQLite, décisions, mémoire, adaptateurs LoRA) ;
- `agent_env/` ou tout autre environnement virtuel.

Le `.gitignore` à la racine du dépôt couvre déjà tout ça. Vérifie quand même avec `git status` avant chaque commit.

## Modifier un prompt

Les prompts sont dans `monitoring_agent/agent/prompts.py`. Toute modification doit changer `PROMPT_VERSION` (format `AAAA-MM-JJ.n`). Ensuite, relance l'évaluation pour vérifier qu'on n'a rien cassé :

```bash
python -m monitoring_agent evaluate --method both --per-dataset 2
python -m monitoring_agent evaluate --redteam
```

Si le modèle local est utilisé, garde à l'esprit qu'il a été entraîné sur le prompt du diagnostiqueur. Changer ce prompt peut demander de refaire le fine-tuning.

## Modifier un seuil

Les seuils du moteur d'analyse sont dans `monitoring_agent/config.py`, ceux de la policy aussi (`PolicyConfig`). Ils ne sont jamais choisis par un LLM. Un changement de seuil se fait par commit, avec une phrase qui explique pourquoi.
