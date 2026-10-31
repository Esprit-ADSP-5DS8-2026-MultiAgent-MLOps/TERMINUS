"""
Export et hébergement du diagnostiqueur fine-tuné (Qwen3-4B + adaptateur LoRA).

    adaptateur LoRA (runs/finetune/data/adapters)
        │  mlx_lm.fuse
        ├──► models/mlx-4bit/      modèle fusionné, quantifié 4 bits : pour mlx_lm.server sur Mac
        └──► models/hf-f16/        modèle fusionné, poids f16 (format Hugging Face)
                 │  llama.cpp : convert_hf_to_gguf.py puis llama-quantize
                 └──► models/gguf/agent4-qwen3-4b-Q4_K_M.gguf  : pour Ollama et llama.cpp (Linux, Docker, CPU)

Publication (finetune publish) :
- un dépôt de modèle privé sur Hugging Face : adaptateur, GGUF, MLX 4 bits, Modelfile Ollama, fiche du modèle ;
- un Space privé (Docker) qui sert le GGUF avec llama-server : une API compatible OpenAI, appelée par l'agent
  avec MONITORING_DECIDER=hf:agent4 et MONITORING_HF_BASE_URL=https://<user>-<space>.hf.space/v1.

On publie l'adaptateur final (adapters.safetensors), celui qui a été évalué (eval_slm.md).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from .build import BASE_MODEL

GGUF_NAME = "agent4-qwen3-4b-Q4_K_M.gguf"
OLLAMA_NAME = "agent4-qwen"

MODELFILE = f"""\
# Modèle Ollama du diagnostiqueur de l'Agent 4 (Qwen3-4B-Instruct-2507 + LoRA, quantifié Q4_K_M)
# ollama create {OLLAMA_NAME} -f Modelfile
FROM ./{GGUF_NAME}

TEMPLATE \"\"\"{{{{- range .Messages }}}}<|im_start|>{{{{ .Role }}}}
{{{{ .Content }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
\"\"\"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0
PARAMETER num_ctx 8192
"""

SPACE_DOCKERFILE = """\
# Sert le GGUF du diagnostiqueur avec llama-server (API compatible OpenAI) sur le port des Spaces.
# Le modèle est téléchargé au démarrage depuis le dépôt privé, avec le secret HF_TOKEN du Space.
FROM ghcr.io/ggml-org/llama.cpp:server
ENV LLAMA_CACHE=/tmp/llama-cache
CMD ["--hf-repo", "{repo}", "--hf-file", "{gguf}", "--host", "0.0.0.0", "--port", "7860", \\
     "--ctx-size", "12288", "--parallel", "3", "--jinja", "--alias", "agent4"]
"""

SPACE_README = """\
---
title: Agent 4 diagnostiqueur
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

Serveur du diagnostiqueur fine-tuné de l'Agent 4 (monitoring MLOps) : llama-server, API compatible OpenAI
sur `/v1/chat/completions`. Modèle : `{repo}` ({gguf}).
"""


def run(cmd: list[str], progress: Callable[[str], None] = print) -> None:
    progress("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def fuse(adapters: Path, out: Path, progress: Callable[[str], None] = print) -> dict[str, Path]:
    """Fusionne l'adaptateur dans le modèle de base (deux versions : MLX 4 bits et f16). Mac Apple Silicon."""
    mlx_dir, hf_dir = out / "mlx-4bit", out / "hf-f16"
    base = [sys.executable, "-m", "mlx_lm.fuse", "--model", BASE_MODEL, "--adapter-path", str(adapters)]
    if not mlx_dir.exists():
        run(base + ["--save-path", str(mlx_dir)], progress)
    if not hf_dir.exists():
        try:
            run(base + ["--save-path", str(hf_dir), "--dequantize"], progress)
        except subprocess.CalledProcessError:   # anciennes versions de mlx-lm
            run(base + ["--save-path", str(hf_dir), "--de-quantize"], progress)
    return {"mlx": mlx_dir, "hf": hf_dir}


def to_gguf(hf_dir: Path, out: Path, llama_cpp: Path, quant: str = "Q4_K_M", python: str | None = None,
            progress: Callable[[str], None] = print) -> Path:
    """f16 Hugging Face -> GGUF f16 -> GGUF quantifié.

    `llama_cpp` : un clone du dépôt llama.cpp ; `python` : l'environnement qui a ses dépendances de conversion
    (torch, transformers...), séparé de celui de l'agent pour ne pas mélanger les versions."""
    out.mkdir(parents=True, exist_ok=True)
    f16, final = out / "agent4-qwen3-4b-f16.gguf", out / GGUF_NAME.replace("Q4_K_M", quant)
    if not f16.exists():
        run([python or sys.executable, str(llama_cpp / "convert_hf_to_gguf.py"), str(hf_dir), "--outfile", str(f16),
             "--outtype", "f16"], progress)
    if not final.exists():
        quantize = shutil.which("llama-quantize") or str(llama_cpp / "build" / "bin" / "llama-quantize")
        run([quantize, str(f16), str(final), quant], progress)
    # Le Modelfile à côté du GGUF : `ollama create agent4-qwen -f <dossier>/Modelfile` marche directement
    (out / "Modelfile").write_text(MODELFILE.replace(GGUF_NAME, final.name), encoding="utf-8")
    return final


def model_card(repo: str, eval_md: str) -> str:
    return f"""---
base_model: {BASE_MODEL}
library_name: mlx
tags: [mlops, monitoring, lora, qwen3, gguf]
language: [fr]
---

# Agent 4 : diagnostiqueur de monitoring (Qwen3-4B + LoRA)

Petit modèle fine-tuné pour diagnostiquer un problème de modèle en production (drift, concept drift, données
cassées) à partir des faits calculés par le moteur d'analyse, et proposer une action. Il répond en JSON
(cause, preuves citées, action, confiance). Il est utilisé par l'Agent 4 de la plateforme Agentic MLOps,
derrière un vérificateur de preuves, un juge et des règles en code.

## Fichiers

| Fichier | Usage |
|---|---|
| `adapters/` | Adaptateur LoRA MLX (rang 16), à appliquer sur `{BASE_MODEL}` |
| `mlx-4bit/` | Modèle fusionné pour `mlx_lm.server` (Mac Apple Silicon) |
| `{GGUF_NAME}` | Modèle fusionné pour Ollama ou llama.cpp (Linux, Windows, Docker, CPU) |
| `Modelfile` | `ollama create {OLLAMA_NAME} -f Modelfile` |

## Entraînement

QLoRA avec MLX : 694 exemples (distillation vérifiée d'un grand modèle + règles), découpage par jeu de
données (les jeux de test n'ont jamais été vus), rang 16, alpha 32, 16 couches, 200 itérations.

## Évaluation (40 cas de test, jeux jamais vus)

{eval_md.strip() or "Voir runs/finetune/eval_slm.md dans le dépôt de l'Agent 4."}

## Utilisation

```bash
# Mac
mlx_lm.server --model {repo} --port 8080            # avec le sous-dossier mlx-4bit
# Ollama
huggingface-cli download {repo} {GGUF_NAME} Modelfile --local-dir .
ollama create {OLLAMA_NAME} -f Modelfile
```
"""


def prepare_repo(out: Path, adapters: Path, repo: str, eval_md: str) -> Path:
    """Dossier prêt à publier (sans copier les gros fichiers : ils sont envoyés depuis leur place)."""
    stage = out / "hub"
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "Modelfile").write_text(MODELFILE, encoding="utf-8")
    (stage / "README.md").write_text(model_card(repo, eval_md), encoding="utf-8")
    keep = ["adapters.safetensors", "adapter_config.json"]
    (stage / "adapters").mkdir(exist_ok=True)
    for name in keep:
        if (adapters / name).exists():
            shutil.copy2(adapters / name, stage / "adapters" / name)
    return stage


def prepare_space(out: Path, repo: str) -> Path:
    space = out / "space"
    space.mkdir(parents=True, exist_ok=True)
    (space / "Dockerfile").write_text(SPACE_DOCKERFILE.format(repo=repo, gguf=GGUF_NAME), encoding="utf-8")
    (space / "README.md").write_text(SPACE_README.format(repo=repo, gguf=GGUF_NAME), encoding="utf-8")
    return space


def publish(out: Path, adapters: Path, repo: str, space: str | None, token: str, eval_md: str = "",
            progress: Callable[[str], None] = print) -> dict:
    """Envoie le modèle (dépôt privé) et, si demandé, crée le Space privé qui le sert."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    stage = prepare_repo(out, adapters, repo, eval_md)
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    progress(f"Dépôt du modèle : https://huggingface.co/{repo}")
    api.upload_folder(repo_id=repo, folder_path=str(stage), commit_message="fiche, Modelfile, adaptateur")
    gguf = out / "gguf" / GGUF_NAME
    if gguf.exists():
        progress(f"Envoi de {gguf.name} ({gguf.stat().st_size / 1e9:.1f} Go)...")
        api.upload_file(path_or_fileobj=str(gguf), path_in_repo=GGUF_NAME, repo_id=repo,
                        commit_message="GGUF Q4_K_M")
    if (out / "mlx-4bit").exists():
        progress("Envoi du modèle MLX 4 bits...")
        api.upload_folder(repo_id=repo, folder_path=str(out / "mlx-4bit"), path_in_repo="mlx-4bit",
                          commit_message="modèle fusionné MLX 4 bits")
    result = {"model": f"https://huggingface.co/{repo}"}
    if space:
        if not gguf.exists():
            raise RuntimeError("pas de GGUF : lance d'abord `finetune export` avec --llama-cpp")
        folder = prepare_space(out, repo)
        api.create_repo(space, repo_type="space", space_sdk="docker", private=True, exist_ok=True)
        api.add_space_secret(space, "HF_TOKEN", token)      # pour télécharger le modèle privé
        api.upload_folder(repo_id=space, repo_type="space", folder_path=str(folder),
                          commit_message="serveur llama.cpp")
        owner, name = space.split("/")
        url = f"https://{owner.lower()}-{name.lower().replace('_', '-').replace('.', '-')}.hf.space/v1"
        result |= {"space": f"https://huggingface.co/spaces/{space}", "api": url}
    (out / "published.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
