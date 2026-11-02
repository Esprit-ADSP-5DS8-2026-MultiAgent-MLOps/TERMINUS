# Terminus — interface web

Interface React (Vite + TypeScript) de la plateforme. Elle ne parle qu'à l'API de
l'orchestrateur (`orchestrator/api.py`, port 8090) ; Vite relaie `/api` vers elle.

## Lancer

```bash
# terminal 1, à la racine du dépôt (.venv activé, .env chargé)
set -a; source .env; set +a
python -m orchestrator api                 # http://127.0.0.1:8090/docs

# terminal 2
cd web
npm install                                # une seule fois
npm run dev                                # http://localhost:5173
```

API sur une autre machine ou un autre port : `TERMINUS_API=http://hote:port npm run dev`.

## Écrans

| Route | Écran |
|---|---|
| `/` | Lignes : lancer un projet, liste des runs |
| `/lignes/:id` | Pipeline d'un run : plan de ligne animé, annonce, journal du superviseur, signal à décider, fichiers |
| `/pipeline` | Redirige vers la ligne la plus récente |
| `/decisions` | Signaux en attente (dont réparation à valider et désaccord v1/v2, uniquement ici) |
| `/agents` | Les six gares et un essai `POST /predict` sur le modèle déployé |
| `/registre` | Versions MLflow par modèle, alias `@champion` |
| `/sante` | État des services et commandes pour les relancer avant la démo |

## Vérifier

```bash
npm run build      # typecheck + build de production dans web/dist/
```
