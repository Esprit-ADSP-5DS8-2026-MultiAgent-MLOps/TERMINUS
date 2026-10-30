# Historique des versions

Les changements notables de l'Agent 4. Format inspiré de [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/).

## [0.4.0] - 2026-10-30

### Ajouté
- « Demande à ton monitoring » : un chat dans la console (et la commande `ask`) qui répond aux questions avec les outils MCP en lecture seule.
- Avis sur les décisions (bonne décision / à corriger avec la bonne cause et la bonne action) et sur les réponses du chat, gardés dans `ratings.jsonl`. Commande `ratings --export` pour en faire des exemples de fine-tuning.
- Notifications réelles : webhook Slack, Teams, Discord ou Mattermost (`MONITORING_NOTIFY_WEBHOOK`), plus un journal `notifications.jsonl`. Le message contient un lien direct vers la décision.
- La console ouvre directement une décision depuis un lien `/console/#<id>`.
- Documentation : `docs/architecture.md`, `CONTRIBUTING.md` et ce fichier.

### Modifié
- Les cas similaires trouvés par la mémoire RAG sont aussi donnés à l'enquêteur et au juge quand le diagnostiqueur est le modèle local (version des prompts `2026-10-30.1`).
- Le juge a un seul essai de 60 s maximum (réglable avec `MONITORING_JUDGE_TIMEOUT_S`) au lieu de près de 4 minutes avant de passer au second avis des règles.
- Le chat connaît la procédure et les règles de l'agent, et `previous_decisions` donne la cause et l'explication de chaque décision : il explique les décisions sans les deviner.
- README réécrit pour quelqu'un qui découvre le projet.

### Corrigé
- `service_health` renvoyait `NaN` sans trafic récent, ce qui n'est pas du JSON valide. Il renvoie maintenant `null`.
- Les libellés de preuve inversés par le LLM (« mesurée sans labels ») sont corrigés automatiquement à la vérification.
- Dans le rapport affiché par la console, les noms d'action avec des « _ » ne sont plus transformés en italique.
- La trace Langfuse de chaque décision a une sortie (action finale, cause, statut) au lieu de « undefined ».
- Le panneau Grafana « Réveils de l'agent » affichait « No data » avant le premier réveil. Il affiche 0.
- Les journaux de requêtes HTTP de `httpx` ne polluent plus la sortie des commandes.

## [0.3.0] - 2026-10-27

### Ajouté
- Serveur MCP (8 outils en lecture seule, ressources, prompts), monté sur la gateway et disponible en stdio.
- Mémoire d'incidents : RAG hybride (texte + signature numérique) avec Qdrant, fusion RRF calibrée.
- Fine-tuning QLoRA de Qwen3-4B avec MLX : génération de scénarios, distillation vérifiée, évaluation sur des bases jamais vues.
- Suivi des appels LLM avec Langfuse.
- Red team : injection de consignes par les données.

## [0.2.0] - 2026-10-25

### Ajouté
- Agent LangGraph : enquête avec outils, trois diagnostics avec vote, vérification des preuves, juge, règles R1 à R9, actions.
- Console web de l'agent.
- Évaluation chiffrée : règles, LLM seul, agent complet.

## [0.1.0] - 2026-10-24

### Ajouté
- Moteur d'analyse : drift, qualité des données, performance mesurée et estimée (CBPE, DLE).
- Gateway devant l'API de l'Agent 3, métriques Prometheus, règles d'alerte et dashboard Grafana générés par projet.
