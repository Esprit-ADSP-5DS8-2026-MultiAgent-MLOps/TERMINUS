# Multi-Agent MLOps Pipeline


You give this platform a machine learning project. It checks the project, trains and compares models,
deploys the best one behind an API, then watches it in production and asks for a retraining when the model
starts to go wrong. Each step is handled by its own agent. An orchestrator chains them, and a human signs off
every sensitive decision: putting a model in production, publishing an image, retraining.

The web interface is called **Terminus**. It shows each run as a train line: every agent is a station, and
every human decision is a signal the train waits at.

[English](#english) · [Français](#français)

---

## English

### How a run goes

```
ML project
   │
   ▼
Contract validator (v1) ── blocked ──► Validator v2 + Repair + Judge ──► fixed copy, checked again by v1
   │ accepted
   ▼
Agent 1   reads the code and the data: task, target, metric, naive baseline
   │
   ▼
Agent 2   trains several models (Optuna), logs everything in MLflow, runs the quality gate
   │
   ▼
[human]   put this model in production? publish the image on Docker Hub?
   │
   ▼
Agent 3   generates the FastAPI service and its Dockerfile, tests them, runs the container on port 8000
   │
   ▼
Agent 4   watches the predictions through its gateway, diagnoses drift, asks for a retraining
   │
   ▼
[human]   retrain on the recent data?  ──► back to Agent 2, then Agent 3 redeploys
```

The original project is never modified. When the repair agent fixes something, it works on a copy, a second
model judges the fix, and the copy goes through the validator again.

Decisions can be taken in three places, and they all write to the same run: the Terminus web page, Slack
(buttons in a channel), or the command line (`python -m orchestrator approve <run_id>`).

### The integration branch

Each member first built their agent on their own branch. The `integration` branch is where everything was
brought together. Only the agent folders were taken from each branch, never a full merge, because several
branches carried their own copy of the validator.

What holds the agents together:

- **one contract**, `common/schemas.py`: what each agent receives and what it must write. Agents never read
  another agent's internal files, only the contract files (`project_spec`, `data_profile`, `baseline`,
  `mlflow_metadata`, `deployment_result`, `retraining_request`...).
- **one adapter per agent** (`orchestrator_adapter.py` in the agent's package). It translates the contract into
  the agent's own format, so the agent code itself did not have to be rewritten.
- **one registry**, `orchestrator/registry.py`, which tells the orchestrator which function to call for each step.
- **one `.env` per agent**, in the agent's folder. The orchestrator only applies it while that agent is running.

`main` holds the finished version. New work starts from `integration` and comes back through a pull request.
The full integration log (what was taken from which branch, and the choices made with each member) is in
[docs/integration.md](docs/integration.md).

### What is in the repository

| Folder | Content |
|---|---|
| `validator/` | Contract validator v1: structure, manifest, static analysis, data, sandboxed run. No LLM. |
| `agents/validator/` | Validator v2, repair agent and judge (LLM based), used when v1 blocks a project |
| `agents/agent1/` | Agent 1: code and data analysis |
| `agents/agent2/` | Agent 2: experiments, MLflow registry, quality gate |
| `agents/agent3/` | Agent 3: API generation, Docker image, deployment, Docker Hub |
| `agents/agent4/` | Agent 4: monitoring gateway, diagnosis, Prometheus, Grafana, Slack alerts |
| `orchestrator/` | LangGraph pipeline, supervisor, human gates, Terminus API, Slack service |
| `common/` | The shared contract (`schemas.py`) and the per-agent `.env` handling |
| `web/` | Terminus, the React interface |
| `eval/projects/` | Sample projects: `credit_risk` (classification) and `house_prices` (regression) |
| `infra/` | PostgreSQL for the run history, Slack app manifest |
| `docs/` | Integration log, Terminus API, Slack setup |
| `tests/` | Orchestrator and validator tests (no real LLM is ever called) |

Every agent folder has its own README that explains how the agent works inside.

### What you need

| | macOS | Windows |
|---|---|---|
| System | macOS 13 or later (Apple Silicon or Intel) | Windows 10 or 11 with **WSL2** (Ubuntu) |
| Python | 3.11, installed by `uv` | 3.11, installed by `uv` inside WSL |
| Docker | Docker Desktop | Docker Desktop with WSL integration turned on |
| Node.js | 20 or later | 20 or later, inside WSL |
| Git | yes | yes, inside WSL |

On Windows, everything runs inside WSL. The validator uses Linux process limits to sandbox the projects it
runs, so it does not work in plain Windows Python.

You also need your own API keys. Each member uses their own: the free quotas do not survive a whole team on
one key.

- Groq: https://console.groq.com/keys (free)
- Google AI Studio (Gemini): https://aistudio.google.com/apikey (free)
- Optional: a Docker Hub access token, a Slack app, a Langfuse account

### Install on macOS

```bash
# 1. Tools (Homebrew: https://brew.sh)
brew install git uv node
# then install Docker Desktop from https://www.docker.com/products/docker-desktop/ and start it

# 2. Code
git clone https://github.com/Esprit-ADSP-5DS8-2026-MultiAgent-MLOps/Multi-Agent-MLOps-Pipeline.git
cd Multi-Agent-MLOps-Pipeline

# 3. One Python environment for the whole platform
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
uv pip install -r agents/agent1/requirements.txt
uv pip install -e agents/agent2 -r agents/agent2/requirements.txt
uv pip install -e agents/agent3 -r agents/agent3/requirements.txt
uv pip install -e agents/agent4 -r agents/agent4/requirements.txt
uv pip install -e agents/validator

# 4. Web interface
cd web && npm install && cd ..
```

### Install on Windows (WSL2)

Open PowerShell as administrator:

```powershell
wsl --install -d Ubuntu-24.04
```

Restart if asked, open "Ubuntu" from the Start menu and create your Linux user. Then install Docker Desktop on
Windows, and in Docker Desktop go to Settings > Resources > WSL integration and turn it on for Ubuntu.

Everything below is typed in the Ubuntu terminal:

```bash
# 1. Tools
sudo apt update && sudo apt install -y git build-essential curl
curl -LsSf https://astral.sh/uv/install.sh | sh
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs
exec $SHELL                      # reload the shell so that uv is found

# 2. Code: clone in your Linux home, not in /mnt/c (much slower, and file permissions break)
cd ~
git clone https://github.com/Esprit-ADSP-5DS8-2026-MultiAgent-MLOps/Multi-Agent-MLOps-Pipeline.git
cd Multi-Agent-MLOps-Pipeline

# 3. and 4.: same commands as on macOS, from "uv venv" to "npm install"
```

Check that Docker answers from WSL with `docker ps`. The pages (Terminus, Grafana...) open in your usual
Windows browser on `localhost`.

### Set your keys

Copy the examples, then fill in your keys. These files are ignored by git: never commit them, never paste a key
in Slack or in an issue.

```bash
cp .env.example .env
cp agents/agent1/.env.example agents/agent1/.env
cp agents/agent3/.env.example agents/agent3/.env
cp agents/agent4/.env.example agents/agent4/.env
cp agents/validator/.env.example agents/validator/.env
```

| File | What to put in it | Without it |
|---|---|---|
| `.env` | `GEMINI_API_KEY` (supervisor, Agent 2 experiment plan) | the supervisor uses its rules, Agent 2 its default plan |
| `agents/agent1/.env` | `GROQ_API_KEY` | Agent 1 skips its LLM suggestions |
| `agents/validator/.env` | `GROQ_API_KEY` | no automatic repair when v1 blocks |
| `agents/agent3/.env` | Docker Hub account and token, only to publish images | images stay local |
| `agents/agent4/.env` | `GROQ_API_KEY`, `GEMINI_API_KEY` | Agent 4 decides with its rules only |

Agent 4 decider: the example file uses `mlx:default_model`, our fine-tuned Qwen3-4B, which runs only on an
Apple Silicon Mac with `mlx_lm.server` and the LoRA adapter. The adapter is not in git. If you do not have it,
or if you are on Windows, write this instead in `agents/agent4/.env`:

```
MONITORING_DECIDER=groq:openai/gpt-oss-120b
```

If you leave `mlx` without the server running, Agent 4 still works: it falls back on Groq at every decision,
it is just slower.

### Run it

Start Docker Desktop first. You will use several terminals. In **each** of them, start with:

```bash
cd ~/path/to/Multi-Agent-MLOps-Pipeline       # your clone
source .venv/bin/activate
set -a; source .env; set +a
```

Then one command per terminal:

```bash
# Terminal 1: monitoring stack (Prometheus, Alertmanager, Grafana), once
docker compose -f agents/agent4/infra/docker-compose.yml up -d

# Terminal 2: Terminus API
python -m orchestrator api                     # http://127.0.0.1:8090/docs

# Terminal 3: Terminus web page
cd web && npm run dev                          # http://localhost:5173

# Terminal 4 (optional): Slack buttons, see docs/slack.md
python -m orchestrator slack
```

Open http://localhost:5173, pick `credit_risk` and launch a line. The validator, Agents 1 and 2 run, then the
line stops at the signal "put in production?". Approve it in Terminus (Decisions page). Agent 3 builds and
starts the API, Agent 4 prepares the monitoring, and the line ends as DONE.

The first run takes longer: the validator builds an isolated Python environment for the project and downloads
its dependencies. It is cached afterwards.

Run history is kept in SQLite (`runs/checkpoints.sqlite`) by default. To use PostgreSQL instead, run
`docker compose -f infra/docker-compose.yml up -d` and uncomment `ORCH_DATABASE_URL` in `.env`.

### See the monitoring and the retraining loop

Once a line is DONE, its model answers on port 8000. Start the Agent 4 gateway for that project, then send
it simulated traffic:

```bash
# Terminal 5
cd agents/agent4
python -m monitoring_agent gateway credit_risk --host 0.0.0.0     # console: http://localhost:8100/console/

# Terminal 6
cd agents/agent4
python -m monitoring_agent simulate credit_risk --scenario concept
```

Scenarios: `normal`, `covariate` (the inputs shift), `quality` (broken data), `concept` (same inputs, but the
link to the target changes). Every 200 predictions, the gateway analyses a window. With `concept`, Agent 4
should find the drop in performance and ask for a retraining. The request shows up in Terminus (and in Slack
if the service runs). Approve it: Agent 2 retrains on the recent data and compares the new model with the
current one on the same test set, you approve the new champion, Agent 3 redeploys.

After a redeployment, restart the gateway (Ctrl+C, same command): it keeps the old reference in memory.

To start a project again from zero (runs, MLflow model, container, monitoring history):

```bash
python -m orchestrator reset credit_risk          # shows what will be removed
python -m orchestrator reset credit_risk --yes    # does it (stop the gateway first)
```

Without the web page, the same line runs in the terminal, which asks the questions itself:

```bash
python -m orchestrator run eval/projects/house_prices
python -m orchestrator status
```

### Addresses

| Service | Address |
|---|---|
| Terminus | http://localhost:5173 |
| Terminus API | http://127.0.0.1:8090/docs |
| Deployed model (Agent 3) | http://127.0.0.1:8000/docs |
| Agent 4 gateway and console | http://localhost:8100/console/ |
| Grafana (admin / admin) | http://localhost:3000 |
| Prometheus alerts | http://localhost:9090/alerts |
| MLflow | `mlflow ui --backend-store-uri sqlite:///runs/mlflow/mlflow.db`, then http://127.0.0.1:5000 |
| PostgreSQL (optional) | localhost:5432, user and password `mlops` |

### Tests

```bash
python -m pytest tests -q                         # orchestrator and validator v1
cd agents/agent4 && python -m pytest -q           # each agent has its own tests in its folder
```

Tests never call a real LLM, Slack or Docker Hub.

### When something goes wrong

- **After a `git pull`, the old behaviour is still there.** Restart the API, the Slack service and the gateway.
  Python processes keep the code they started with. The web page reloads by itself.
- **The gateway says `monitoring.json` is missing.** The line has not reached Agent 4 yet. Wait for DONE.
- **The Santé page shows a service as stopped.** It also shows the command to start it.
- **Agent 3 fails at the Docker step.** Check that Docker Desktop is running (`docker ps`).
- **WSL: downloads fail with TLS errors.** On some networks you need a lower MTU:
  `sudo ip link set dev eth0 mtu 1400` (lost when WSL restarts).
- **A port is already taken.** `lsof -i :8090` (macOS or WSL) shows which process holds it.

---

## Français

### Ce que fait la plateforme

Vous lui donnez un projet de machine learning. Elle le vérifie, entraîne et compare des modèles, déploie le
meilleur derrière une API, puis le surveille en production et demande un réentraînement quand il commence à se
tromper. Chaque étape est confiée à un agent. Un orchestrateur les enchaîne, et un humain valide chaque décision
sensible : mise en production, publication de l'image, réentraînement.

L'interface web s'appelle **Terminus**. Elle montre chaque exécution comme une ligne de train : chaque agent est
une gare, chaque décision humaine est un signal où le train attend.

### Le déroulement d'une ligne

```
Projet ML
   │
   ▼
Validateur de contrat (v1) ── bloqué ──► Validateur v2 + Repair + Juge ──► copie corrigée, revalidée par le v1
   │ accepté
   ▼
Agent 1   lit le code et les données : tâche, cible, métrique, modèle naïf de référence
   │
   ▼
Agent 2   entraîne plusieurs modèles (Optuna), trace tout dans MLflow, applique le quality gate
   │
   ▼
[humain]  mettre ce modèle en production ? publier l'image sur Docker Hub ?
   │
   ▼
Agent 3   génère l'API FastAPI et son Dockerfile, les teste, lance le conteneur sur le port 8000
   │
   ▼
Agent 4   observe les prédictions via sa gateway, diagnostique le drift, demande un réentraînement
   │
   ▼
[humain]  réentraîner sur les données récentes ?  ──► retour à l'Agent 2, puis l'Agent 3 redéploie
```

Le projet d'origine n'est jamais modifié. Quand l'agent de réparation corrige quelque chose, il travaille sur
une copie, un second modèle juge la correction, et la copie repasse par le validateur.

Les décisions se prennent à trois endroits, qui écrivent tous dans la même ligne : la page Terminus, Slack
(boutons dans un canal) ou la ligne de commande (`python -m orchestrator approve <run_id>`).

### La branche d'intégration

Chaque membre a d'abord développé son agent sur sa propre branche. La branche `integration` est celle où tout a
été réuni. On a repris uniquement les dossiers des agents, jamais une fusion complète, parce que plusieurs
branches portaient leur propre copie du validateur.

Ce qui fait tenir les agents ensemble :

- **un contrat**, `common/schemas.py` : ce que chaque agent reçoit et ce qu'il doit écrire. Un agent ne lit
  jamais les fichiers internes d'un autre, seulement les fichiers du contrat (`project_spec`, `data_profile`,
  `baseline`, `mlflow_metadata`, `deployment_result`, `retraining_request`...).
- **un adaptateur par agent** (`orchestrator_adapter.py` dans son paquet). Il traduit le contrat dans le format
  propre à l'agent : le code de l'agent n'a pas eu à être réécrit.
- **un registre**, `orchestrator/registry.py`, qui indique à l'orchestrateur quelle fonction appeler à chaque étape.
- **un `.env` par agent**, dans son dossier. L'orchestrateur ne l'applique que pendant que cet agent travaille.

`main` contient la version terminée. Les nouveaux travaux partent d'`integration` et y reviennent par une pull
request. Le journal complet de l'intégration (ce qui a été pris dans quelle branche, et les choix faits avec
chaque membre) est dans [docs/integration.md](docs/integration.md).

### Le contenu du dépôt

| Dossier | Contenu |
|---|---|
| `validator/` | Validateur de contrat v1 : structure, manifeste, analyse statique, données, exécution isolée. Sans LLM. |
| `agents/validator/` | Validateur v2, agent de réparation et juge (avec LLM), appelés quand le v1 bloque un projet |
| `agents/agent1/` | Agent 1 : analyse du code et des données |
| `agents/agent2/` | Agent 2 : expériences, registre MLflow, quality gate |
| `agents/agent3/` | Agent 3 : génération de l'API, image Docker, déploiement, Docker Hub |
| `agents/agent4/` | Agent 4 : gateway de monitoring, diagnostic, Prometheus, Grafana, alertes Slack |
| `orchestrator/` | Pipeline LangGraph, superviseur, validations humaines, API de Terminus, service Slack |
| `common/` | Le contrat commun (`schemas.py`) et la gestion du `.env` de chaque agent |
| `web/` | Terminus, l'interface React |
| `eval/projects/` | Projets d'exemple : `credit_risk` (classification) et `house_prices` (régression) |
| `infra/` | PostgreSQL pour l'historique des lignes, manifeste de l'app Slack |
| `docs/` | Journal d'intégration, API de Terminus, configuration de Slack |
| `tests/` | Tests de l'orchestrateur et du validateur (aucun vrai LLM n'est appelé) |

Chaque dossier d'agent a son propre README, qui explique le fonctionnement interne de l'agent.

### Ce qu'il faut

| | macOS | Windows |
|---|---|---|
| Système | macOS 13 ou plus récent (Apple Silicon ou Intel) | Windows 10 ou 11 avec **WSL2** (Ubuntu) |
| Python | 3.11, installé par `uv` | 3.11, installé par `uv` dans WSL |
| Docker | Docker Desktop | Docker Desktop, intégration WSL activée |
| Node.js | 20 ou plus | 20 ou plus, dans WSL |
| Git | oui | oui, dans WSL |

Sous Windows, tout tourne dans WSL. Le validateur utilise les limites de processus de Linux pour isoler les
projets qu'il exécute : il ne fonctionne pas avec un Python Windows classique.

Il faut aussi vos propres clés d'API. Chaque membre utilise les siennes : les quotas gratuits ne tiennent pas
avec une seule clé pour toute l'équipe.

- Groq : https://console.groq.com/keys (gratuit)
- Google AI Studio (Gemini) : https://aistudio.google.com/apikey (gratuit)
- Facultatif : un jeton d'accès Docker Hub, une app Slack, un compte Langfuse

### Installation sur macOS

```bash
# 1. Outils (Homebrew : https://brew.sh)
brew install git uv node
# puis installer Docker Desktop depuis https://www.docker.com/products/docker-desktop/ et le lancer

# 2. Code
git clone https://github.com/Esprit-ADSP-5DS8-2026-MultiAgent-MLOps/Multi-Agent-MLOps-Pipeline.git
cd Multi-Agent-MLOps-Pipeline

# 3. Un seul environnement Python pour toute la plateforme
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
uv pip install -r agents/agent1/requirements.txt
uv pip install -e agents/agent2 -r agents/agent2/requirements.txt
uv pip install -e agents/agent3 -r agents/agent3/requirements.txt
uv pip install -e agents/agent4 -r agents/agent4/requirements.txt
uv pip install -e agents/validator

# 4. Interface web
cd web && npm install && cd ..
```

### Installation sur Windows (WSL2)

Ouvrir PowerShell en administrateur :

```powershell
wsl --install -d Ubuntu-24.04
```

Redémarrer si c'est demandé, ouvrir « Ubuntu » depuis le menu Démarrer et créer votre utilisateur Linux.
Installer ensuite Docker Desktop sous Windows, puis dans Docker Desktop : Settings > Resources > WSL integration,
et l'activer pour Ubuntu.

Tout ce qui suit se tape dans le terminal Ubuntu :

```bash
# 1. Outils
sudo apt update && sudo apt install -y git build-essential curl
curl -LsSf https://astral.sh/uv/install.sh | sh
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs
exec $SHELL                      # recharger le shell pour que uv soit trouvé

# 2. Code : cloner dans votre dossier Linux, pas dans /mnt/c (beaucoup plus lent, et les droits des fichiers cassent)
cd ~
git clone https://github.com/Esprit-ADSP-5DS8-2026-MultiAgent-MLOps/Multi-Agent-MLOps-Pipeline.git
cd Multi-Agent-MLOps-Pipeline

# 3. et 4. : mêmes commandes que sur macOS, de « uv venv » à « npm install »
```

Vérifiez que Docker répond depuis WSL avec `docker ps`. Les pages (Terminus, Grafana...) s'ouvrent dans votre
navigateur Windows habituel, sur `localhost`.

### Mettre vos clés

Copiez les exemples, puis remplissez vos clés. Ces fichiers sont ignorés par git : ne les committez jamais, et ne
collez jamais une clé dans Slack ou dans une issue.

```bash
cp .env.example .env
cp agents/agent1/.env.example agents/agent1/.env
cp agents/agent3/.env.example agents/agent3/.env
cp agents/agent4/.env.example agents/agent4/.env
cp agents/validator/.env.example agents/validator/.env
```

| Fichier | Ce qu'on y met | Sans lui |
|---|---|---|
| `.env` | `GEMINI_API_KEY` (superviseur, plan d'expériences de l'Agent 2) | le superviseur applique ses règles, l'Agent 2 son plan par défaut |
| `agents/agent1/.env` | `GROQ_API_KEY` | l'Agent 1 saute ses suggestions par LLM |
| `agents/validator/.env` | `GROQ_API_KEY` | pas de réparation automatique quand le v1 bloque |
| `agents/agent3/.env` | compte et jeton Docker Hub, seulement pour publier les images | les images restent en local |
| `agents/agent4/.env` | `GROQ_API_KEY`, `GEMINI_API_KEY` | l'Agent 4 décide avec ses règles seulement |

Décideur de l'Agent 4 : le fichier d'exemple utilise `mlx:default_model`, notre Qwen3-4B fine-tuné, qui ne
tourne que sur un Mac Apple Silicon avec `mlx_lm.server` et l'adaptateur LoRA. L'adaptateur n'est pas dans git.
Si vous ne l'avez pas, ou si vous êtes sous Windows, mettez plutôt dans `agents/agent4/.env` :

```
MONITORING_DECIDER=groq:openai/gpt-oss-120b
```

Si vous laissez `mlx` sans que le serveur tourne, l'Agent 4 fonctionne quand même : il bascule sur Groq à
chaque décision, il est seulement plus lent.

### Lancer

Démarrez d'abord Docker Desktop. Il faut plusieurs terminaux. Dans **chacun**, commencez par :

```bash
cd ~/chemin/vers/Multi-Agent-MLOps-Pipeline   # votre clone
source .venv/bin/activate
set -a; source .env; set +a
```

Puis une commande par terminal :

```bash
# Terminal 1 : pile de monitoring (Prometheus, Alertmanager, Grafana), une seule fois
docker compose -f agents/agent4/infra/docker-compose.yml up -d

# Terminal 2 : API de Terminus
python -m orchestrator api                     # http://127.0.0.1:8090/docs

# Terminal 3 : page web de Terminus
cd web && npm run dev                          # http://localhost:5173

# Terminal 4 (facultatif) : boutons Slack, voir docs/slack.md
python -m orchestrator slack
```

Ouvrez http://localhost:5173, choisissez `credit_risk` et lancez une ligne. Le validateur et les Agents 1 et 2
s'exécutent, puis la ligne s'arrête au signal « mettre en production ? ». Approuvez dans Terminus (page
Décisions). L'Agent 3 construit et démarre l'API, l'Agent 4 prépare le monitoring, et la ligne se termine en DONE.

La première exécution est plus longue : le validateur construit un environnement Python isolé pour le projet et
télécharge ses dépendances. Elles restent en cache ensuite.

L'historique des lignes est gardé dans SQLite (`runs/checkpoints.sqlite`) par défaut. Pour utiliser PostgreSQL,
lancez `docker compose -f infra/docker-compose.yml up -d` et décommentez `ORCH_DATABASE_URL` dans `.env`.

### Voir le monitoring et la boucle de réentraînement

Une fois la ligne en DONE, son modèle répond sur le port 8000. Lancez la gateway de l'Agent 4 pour ce projet,
puis envoyez-lui du trafic simulé :

```bash
# Terminal 5
cd agents/agent4
python -m monitoring_agent gateway credit_risk --host 0.0.0.0     # console : http://localhost:8100/console/

# Terminal 6
cd agents/agent4
python -m monitoring_agent simulate credit_risk --scenario concept
```

Scénarios : `normal`, `covariate` (les entrées changent), `quality` (données cassées), `concept` (mêmes entrées,
mais le lien avec la cible change). Toutes les 200 prédictions, la gateway analyse une fenêtre. Avec `concept`,
l'Agent 4 doit trouver la baisse de performance et demander un réentraînement. La demande apparaît dans Terminus
(et dans Slack si le service tourne). Approuvez-la : l'Agent 2 réentraîne sur les données récentes et compare le
nouveau modèle à l'actuel sur le même jeu de test, vous approuvez le nouveau champion, l'Agent 3 redéploie.

Après un redéploiement, relancez la gateway (Ctrl+C, même commande) : elle garde l'ancienne référence en mémoire.

Pour repartir de zéro sur un projet (lignes, modèle MLflow, conteneur, historique du monitoring) :

```bash
python -m orchestrator reset credit_risk          # montre ce qui sera supprimé
python -m orchestrator reset credit_risk --yes    # le fait (arrêter la gateway avant)
```

Sans la page web, la même ligne tourne dans le terminal, qui pose lui-même les questions :

```bash
python -m orchestrator run eval/projects/house_prices
python -m orchestrator status
```

### Adresses

| Service | Adresse |
|---|---|
| Terminus | http://localhost:5173 |
| API de Terminus | http://127.0.0.1:8090/docs |
| Modèle déployé (Agent 3) | http://127.0.0.1:8000/docs |
| Gateway et console de l'Agent 4 | http://localhost:8100/console/ |
| Grafana (admin / admin) | http://localhost:3000 |
| Alertes Prometheus | http://localhost:9090/alerts |
| MLflow | `mlflow ui --backend-store-uri sqlite:///runs/mlflow/mlflow.db`, puis http://127.0.0.1:5000 |
| PostgreSQL (facultatif) | localhost:5432, utilisateur et mot de passe `mlops` |

### Tests

```bash
python -m pytest tests -q                         # orchestrateur et validateur v1
cd agents/agent4 && python -m pytest -q           # chaque agent a ses propres tests dans son dossier
```

Les tests n'appellent jamais un vrai LLM, Slack ou Docker Hub.

### Quand quelque chose ne marche pas

- **Après un `git pull`, l'ancien comportement est toujours là.** Relancez l'API, le service Slack et la gateway :
  un processus Python garde le code avec lequel il a démarré. La page web se recharge toute seule.
- **La gateway dit que `monitoring.json` est absent.** La ligne n'est pas encore passée par l'Agent 4. Attendez DONE.
- **La page Santé montre un service arrêté.** Elle affiche aussi la commande pour le relancer.
- **L'Agent 3 échoue à l'étape Docker.** Vérifiez que Docker Desktop tourne (`docker ps`).
- **WSL : les téléchargements échouent avec des erreurs TLS.** Sur certains réseaux, il faut baisser le MTU :
  `sudo ip link set dev eth0 mtu 1400` (perdu au redémarrage de WSL).
- **Un port est déjà pris.** `lsof -i :8090` (macOS ou WSL) montre quel processus l'occupe.
