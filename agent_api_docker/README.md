# agent_api_docker

Genere `api.py`, `Dockerfile`, `.dockerignore` (et `docker-compose.yml` si services externes)
a partir du rapport d'analyse.

## Ou le rapport est lu (ordre de priorite)
1. `--report <chemin>` (CLI)
2. variable d'environnement `AGENT_REPORT_PATH`
3. `workspaces/<project_id>/analysis_report.json`   <- mode reel (ecrit par l'agent d'analyse)
4. `eval/expected_reports/<project_id>.analysis_report.json`   <- mode dev / tests
5. `eval/projects/<project_id>/expected_reports/<project_id>.analysis_report.json` (ou `.../analysis_report.json`)
6. sinon : **l'agent LLM de recherche** explore le depot (voir ci-dessous)

## Agent de recherche du rapport (LLM)
Si aucun chemin habituel ne marche, un LLM explore le depot avec des outils en lecture seule
(`list_dir`, `find_files`, `inspect_json`), puis propose son choix via `finish(report_path, project_dir)`.
Le LLM decide ou chercher ; le CODE valide son choix (`discovery_agent.py`) :
- le fichier doit respecter le contrat `AnalysisReport` (un rapport de STRUCTURE est refuse) ;
- le rapport doit concerner le bon projet (`project_name` ~ `project_id`) ;
- le dossier du code doit contenir le fichier d'inference du rapport (`inference.entrypoint.file`) ;
- les outils ne sortent jamais du depot / de `workspaces/`.
Un choix refuse est renvoye au LLM avec la raison, dans la limite de `AGENT_DISCOVERY_MAX_STEPS` (8).
La trajectoire est ecrite dans `workspaces/<id>/traces.jsonl` (`step: "discover"`).

    python -m agent_api_docker credit_risk              # chemins habituels, puis LLM si rien trouve
    python -m agent_api_docker credit_risk --agent      # le LLM cherche toujours
    python -m agent_api_docker credit_risk --no-agent   # jamais de LLM

Configuration (.env) : `LLM_PROVIDER` (openai | groq | gemini | ollama) ou `LLM_BASE_URL` (tout serveur
compatible OpenAI : vLLM...), `LLM_MODEL`, `LLM_API_KEY_DOCKER`. Le modele doit gerer les appels d'outils.

## Demo (une commande)
    python -m agent_api_docker credit_risk            # genere, valide, teste (etapes affichees)
    python -m agent_api_docker credit_risk --show     # + affiche api.py et Dockerfile
    python -m agent_api_docker credit_risk --serve    # puis lance l'API : http://127.0.0.1:8000/docs
    python -m agent_api_docker credit_risk --generate-only
Resultats : workspaces/<id>/generated (dossier UNIQUE et complet : projet + fichiers generes + corrections, pret pour
`docker build`), result.json, traces.jsonl.

## Utilisation
    python -m agent_api_docker credit_risk
    python -m agent_api_docker credit_risk --report chemin/rapport.json
    pytest tests/test_agent_api_docker.py

## Correction automatique par le LLM (fixer.py)
Si la validation ou les tests echouent, le LLM recoit : la categorie d'erreur, les logs utiles, les fichiers generes,
le code d'inference (lecture seule), un extrait du rapport et la liste des corrections deja essayees. Il repond en JSON
`{diagnosis, target, patches:[{path, old, new}]}`. Le CODE controle puis applique, puis on revalide et on reteste :
- seuls `api.py`, `requirements.api.txt`, `Dockerfile`, `.dockerignore`, `docker-compose.yml` sont modifiables
  (jamais le code du projet, jamais les tests : ils sont regeneres depuis le rapport a chaque passage) ;
- `old` doit apparaitre exactement une fois ; un patch deja essaye est refuse ; JSON invalide = nouvelle demande ;
- budget : `AGENT_MAX_ITERATIONS` (6) corrections, ou `--max-fixes N` ; arret si aucun patch n'est applicable ;
- panne ou absence de LLM : le pipeline s'arrete proprement avec la cause (jamais de plantage) ;
- chaque tentative est ecrite dans `workspaces/<id>/traces.jsonl` (`step: "fix"`) et dans `result.json` (`fixes`) :
  c'est le dataset du futur fine-tuning.

    python -m agent_api_docker credit_risk                 # correction automatique activee par defaut
    python -m agent_api_docker credit_risk --no-fix        # desactive le fixer
    python -m agent_api_docker credit_risk --max-fixes 3
    pytest tests/test_fixer.py                             # faux LLM, aucun appel reseau

## Corrections durables (fix_store.py)
Les patchs du LLM s'appliquent sur `workspaces/<id>/generated/`, que chaque execution efface avant de regenerer depuis
les templates. Sans memoire, la meme erreur revenait a chaque run. Maintenant :
- une execution qui **reussit** ecrit ses patchs appliques dans `fixes/<project_id>.json` (dossier `AGENT_FIXES_DIR`,
  hors de `workspaces/` : **a versionner dans git**) ;
- l'execution suivante les **rejoue** juste apres la generation, dans l'ordre, sans LLM ;
- un patch devenu inapplicable (`old` introuvable : template ou rapport modifie) est ignore, signale, puis retire a la
  prochaine reussite ; la boucle de correction reprend alors la main ;
- un echec n'ecrit rien ; les patchs de `Dockerfile` / `requirements.api.txt` / `.dockerignore` ne sont memorises
  qu'apres un build + des tests reussis dans le conteneur (`--docker`) ; ceux d'`api.py` apres les tests locaux ;
- le code du projet n'est jamais modifie. Un bug du projet reste signale, pas patche.

    python -m agent_api_docker credit_risk --docker      # corrige, valide, memorise
    python -m agent_api_docker credit_risk               # rejoue les correctifs memorises
    python -m agent_api_docker credit_risk --no-memory   # generation nue (debogage des templates)
Un correctif qui revient pour plusieurs projets doit etre promu dans le template `.j2` (revue humaine).

## Contenu de l'image
Le Dockerfile copie le projet **entier** (`COPY . .` : fichiers de base + fichiers generes). Le contexte de build est
`workspaces/<id>/generated/` : ce dossier contient le projet complet, les fichiers generes (`api.py`, `Dockerfile`,
`requirements.api.txt`, `.dockerignore`...), les correctifs (rejoues ou appliques par le LLM) et le modele. Il sert aussi
de dossier d'execution pour la validation et les tests : un seul emplacement, que l'on peut construire a la main :

    docker build -t mon-image workspaces/<id>/generated

Les artefacts des tests locaux (`test_api_generated.py`, `junit.xml`, `server.log`) restent dans ce dossier pour le
debogage mais sont exclus de l'image par `.dockerignore`. `--generate-only` n'ecrit, lui, que les fichiers generes. Les exclusions se font dans `.dockerignore` (`project.ignorable_paths` du rapport + caches,
`.env*`, notebooks, `.git`), plus par une liste `runtime_files` a tenir a jour. Si le modele est entraine au build, la
version entrainee dans l'image remplace celle du contexte.

## Lancer le conteneur (Ctrl+C = arret + suppression)
    python -m agent_api_docker credit_risk --serve-docker   # pipeline + build + tests, puis lance le conteneur
    python -m agent_api_docker credit_risk --run-only       # relance la derniere image VALIDEE, sans rien regenerer
    python -m agent_api_docker credit_risk --run-only --port 8001
L'API est sur http://127.0.0.1:<port>/docs (127.0.0.1 uniquement) ; les logs du conteneur s'affichent en direct.
Ctrl+C (ou SIGTERM / fermeture du terminal) lance `docker stop` puis `docker rm -f` : aucun conteneur ne reste en
arriere-plan. Un 2e Ctrl+C pendant l'arret force la suppression. Port deja pris : message avec `--port <autre>`.
`--serve-docker` implique `--docker` ; `--serve` (uvicorn local), `--serve-docker` et `--run-only` sont exclusifs.

## Phase 2-3 : validation et tests de l'API
    from agent_api_docker.validate_api import validate_api
    from agent_api_docker.api_tests import prepare_run_dir, run_api_tests
    run = prepare_run_dir(projet, dossier_genere, report, dossier_run)  # copie + entraine le modele
    v = validate_api(report, run)       # syntaxe, ruff, import, routes, contrat OpenAPI
    t = run_api_tests(report, run)      # serveur local + pytest genere
    v.summary() / t.summary()           # texte a renvoyer au LLM en cas d'echec
Les tests generes lisent BASE_URL : ils serviront tels quels contre le conteneur (phase 4).
Prerequis : l'interpreteur utilise (python_exe) doit avoir fastapi, uvicorn, httpx, pytest, ruff et les deps du projet.

## Etape Docker : image, conteneur, tests, push (docker_stage.py)
    python -m agent_api_docker credit_risk --docker              # build + conteneur + tests contre le conteneur
    python -m agent_api_docker credit_risk --push                # + publication Docker Hub apres approbation
    python -m agent_api_docker credit_risk --push --yes          # approbation automatique (liee au sha256 teste)
    python -m agent_api_docker credit_risk --docker --max-docker-fixes 3
Deroule (apres une API locale valide) :
1. pre-controle : Docker installe et daemon lance (sinon message clair, pas de correction inutile) ;
2. hadolint sur le Dockerfile (ignore s'il n'est pas installe) ;
3. `docker build` (progression affichee ; erreurs reseau passageres reessayees sans correction LLM) ;
4. `docker run` (port publie sur 127.0.0.1 uniquement), attente de `/health` ;
5. les MEMES tests generes que pour l'API locale, avec BASE_URL = le conteneur (le test « API == modele direct »
   est ignore : il ne vaut qu'en local) ; le conteneur est toujours supprime a la fin ;
6. push : le CODE demande l'approbation (invite `o/N`, ou `--yes`), relit le sha256 de l'image locale et refuse
   si elle a change ; login par token sur stdin ; tags `AAAAMMJJ-<sha8>` et `latest` ; deconnexion ensuite.
Echec a l'etape 2-5 : meme boucle de correction que pour l'API (fixer.py), budget `AGENT_MAX_DOCKER_ITERATIONS` (6),
avec en plus la liste des fichiers du contexte de build. Les problemes d'environnement (daemon arrete, quota Docker Hub,
disque plein, timeout de build) ne sont PAS envoyes au LLM : arret avec le conseil adapte.
Un patch d'`api.py` pendant cette phase declenche d'abord une revalidation locale.
Traces : `traces.jsonl` (`step: "docker"`, `"fix"` avec `stage`, `"push"`) et `result.json` (`docker`, `push`).

Identifiants Docker Hub : avant le push, l'agent les **demande dans le terminal** (nom d'utilisateur ; token saisi
sans echo, cree sur hub.docker.com > Account settings > Personal access tokens, droits Read & Write). Ils restent en
memoire le temps du push : ni fichier, ni log, ni trace. Token refuse : 3 essais (token seulement), puis arret propre.
L'espace de nom du depot est `--namespace`, sinon `DOCKER_IMAGE_NAMESPACE`, sinon le compte saisi.
Sans terminal interactif (CI, redirection), ils sont lus dans `.env` comme avant, et verifies AVANT le build.

Configuration (.env), toutes optionnelles en mode interactif :
    DOCKERHUB_USERNAME=...        # propose comme valeur par defaut a la saisie ; requis (avec le token) hors terminal
    DOCKERHUB_TOKEN=...           # uniquement pour le mode non interactif (jamais lu quand le terminal demande)
    DOCKER_IMAGE_NAMESPACE=...    # optionnel : organisation (defaut = compte saisi)
    AGENT_BUILD_TIMEOUT_S=600     # optionnel
Attention : un depot Docker Hub cree par un push est PUBLIC par defaut et l'image contient le modele.
Pour un modele confidentiel, creer d'abord le depot en prive sur hub.docker.com.
    pytest tests/test_docker_stage.py     # faux Docker (vrai uvicorn + vrais tests), aucun reseau

## Etat
- Fait : etape Docker complete (tools/docker.py, tools/checks.py, tools/publish.py, docker_stage.py) + CLI --docker/--push.
- Fait : validation du rapport, generation par templates, classification d'erreurs, garde d'approbation du push.
- Fait aussi : validate_api.py, api_tests.py (generation de tests + runner).
- Fait : agent LLM de recherche du rapport (llm.py, discovery_agent.py, tools/discovery.py).
- Fait : boucle de correction LLM (fixer.py + loop.py) sur la validation et les tests locaux.
- A tester : un vrai build Docker sur credit_risk (les tests utilisent un faux Docker).
