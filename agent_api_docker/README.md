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
Resultats : workspaces/<id>/generated (fichiers), run (projet assemble), result.json, traces.jsonl.

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

Configuration (.env) :
    DOCKERHUB_USERNAME=...        # compte Docker Hub
    DOCKERHUB_TOKEN=...           # Personal access token, droits Read & Write
    DOCKER_IMAGE_NAMESPACE=...    # optionnel : organisation (defaut = DOCKERHUB_USERNAME)
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
