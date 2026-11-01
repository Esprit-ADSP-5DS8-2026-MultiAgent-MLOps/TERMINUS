# Données de fine-tuning du Repair Agent

Génère des trajectoires de réparation **vérifiées** : projets ML sains (générés) → défaut injecté → Validator → Repair (chaque appel LLM enregistré) → revalidation → Judge → filtres → JSONL.

## Idée directrice

Un exemple n'est retenu que s'il réussit **toutes** les vérifications indépendantes du modèle. Sans cela, le modèle apprendrait à « satisfaire les règles » : lors de nos essais, le Repair a remplacé un chemin absolu par `data/credit.csv`, fichier inexistant, dans une constante jamais utilisée ; le Validator a accepté, seul le Judge a protesté.

| Filtre | Ce qu'il empêche |
|---|---|
| tests de l'infrastructure, revalidation du Validator (aucun nouveau problème) | réparation qui ne répare pas |
| aucun signalement de périmètre, métriques inchangées | touche aux tests, données, choix scientifiques |
| fichiers touchés ⊆ fichiers de l'injection, diff ≤ max(10 lignes, 4× l'injection) | réécritures hors sujet |
| chemin introduit = fichier existant | chemin inventé |
| Judge `ACCEPT` avec modèle **indépendant** | réparation cosmétique |
| ≤ 12 étapes, trajectoire non compactée, secrets masqués | exemples coûteux ou tronqués |

## Utilisation

```bash
# 0. sans LLM : vérifie que chaque défaut est bien détecté ET envoyé au Repair (aucun token)
python -m finetune.collect --out data/check --bases 2 --check-faults

# 1. pilote (quelques tâches), puis collecte réelle
GROQ_API_KEY=... python -m finetune.collect --out data/pilot --bases 2 --faults unpinned_dep abs_path_used syntax_error --judge
GROQ_API_KEY=... python -m finetune.collect --out data/run1 --bases 20 --fault-seeds 2 --samples 3 --judge --temperature 0.7

# 2. export (découpage par projet de base ; défauts « holdout » réservés au test)
python -m finetune.export --runs data/run1/runs.jsonl --out data/run1/export --per-task 1 --holdout-faults typo_import,no_return
```

La collecte est **reprenable** (relancez la même commande). `runs.jsonl` garde aussi les exécutions écartées avec leurs raisons (`rejected_by`) : elles servent de contre-exemples (préférences) et de diagnostic.

Sorties de l'export : `sft_{train,val,test,test_unseen}.jsonl` (messages + outils, format chat), `dpo_*.jsonl` (`prompt`, `chosen`, `rejected` pour la même tâche), `stats.json`.

## Ce qui est volontairement exclu

Le triage par nature est fait **en Python avant** l'appel au LLM (`repair_agent/agent.py`). Les problèmes de sécurité, de logique ou de méthode ne lui sont jamais montrés, et les défauts qui s'arrêtent au Validator (appel dangereux, secret, réseau → `REJECTED` ; graine absente, artefact commité → revue humaine) n'atteignent pas le Repair. Ils ne produiraient donc aucun exemple utile : seuls les 16 défauts réparables de `faults.py` sont injectés (`--check-faults` le vérifie).

## Précautions

- `--temperature 0` donne des échantillons identiques ; le cache LLM (`--llm-cache`) aussi. Utilisez une température > 0 pour obtenir plusieurs tentatives différentes.
- `--env venv` (défaut) : les `ENV.SANDBOX_ISOLATION` du mode `current` n'apparaissent pas ; les problèmes présents sur le projet sain sont de toute façon retirés de la comparaison.
- Le Judge doit rester d'une autre famille de modèle que le Repair (`--require-distinct-model` est imposé), et le jeu d'évaluation final doit être jugé par un modèle différent de celui qui a servi au filtrage.
- Ordres de grandeur observés : ~13 000 tokens par exécution du Repair, ~17 000 à 20 000 par Judge. Avec 16 défauts × N projets, commencez petit.
- Les projets générés sont synthétiques et tous de la même forme (classification tabulaire sklearn) : un modèle entraîné dessus peut très bien réparer ces projets sans être meilleur sur des projets réels. Ajoutez des cas réels annotés au jeu de test.
