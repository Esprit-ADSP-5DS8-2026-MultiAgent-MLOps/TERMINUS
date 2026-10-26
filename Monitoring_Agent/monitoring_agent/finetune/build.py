"""
Assemblage du dataset MLX et de la configuration LoRA.

- découpage PAR BASE (scenarios.py l'a déjà fixé) : train = 6 bases, valid = mc_prepaid (jamais vue
  à l'entraînement), test = car_loan / car_price (jamais vues, réservées à l'évaluation) ;
- cible = diagnostic du professeur VÉRIFIÉ, sinon (option) celui des règles ;
- équilibrage : aucune action ne dépasse `cap_ratio` × la classe la plus rare (sinon le modèle apprend
  « notify_data_team » par défaut) ;
- format chat MLX : {"messages": [system, user, assistant]} ; mask_prompt -> la perte ne porte que sur
  la réponse ;
- lora.yaml : hyperparamètres justifiés ci-dessous, nombre d'itérations calculé depuis la taille du train.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import yaml

BASE_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def example(sc: dict, target: dict) -> dict:
    return {"messages": sc["messages"] + [{"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}]}


def build(scenarios_path: Path, teacher_path: Path, out_dir: Path, templates: bool = True, cap_ratio: float = 2.0,
          epochs: int = 2, batch_size: int = 4, grad_accum: int = 2, seed: int = 42,
          model: str = BASE_MODEL) -> dict:
    rng = random.Random(seed)
    scenarios = _jsonl(scenarios_path)
    taught = {r["id"]: r for r in _jsonl(teacher_path) if r.get("diagnosis")}
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: dict[str, list[tuple[dict, dict, str]]] = defaultdict(list)
    for sc in scenarios:
        if sc["split"] == "test":
            rows["test"].append((sc, sc["rules"], "rules"))
            continue
        if sc["id"] in taught:
            rows[sc["split"]].append((sc, taught[sc["id"]]["diagnosis"], taught[sc["id"]]["source"]))
        elif templates:
            rows[sc["split"]].append((sc, sc["rules"], "rules"))

    # Équilibrage du train par action attendue (on garde en priorité les exemples du professeur)
    by_action: dict[str, list] = defaultdict(list)
    for r in rows["train"]:
        by_action[r[0]["oracle"]["action"]].append(r)
    if by_action:
        smallest = min(len(v) for v in by_action.values())
        cap = max(int(cap_ratio * smallest), 30)
        balanced = []
        for items in by_action.values():
            items.sort(key=lambda r: (r[2] == "rules", rng.random()))
            balanced += items[:cap]
        rng.shuffle(balanced)
        rows["train"] = balanced

    stats: dict = {"model": model}
    for split in ("train", "valid", "test"):
        items = rows.get(split, [])
        with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for sc, target, _src in items:
                f.write(json.dumps(example(sc, target), ensure_ascii=False) + "\n")
        stats[split] = {"n": len(items),
                        "actions": dict(Counter(sc["oracle"]["action"] for sc, _, _ in items)),
                        "sources": dict(Counter(src for _, _, src in items)),
                        "bases": dict(Counter(sc["base"] for sc, _, _ in items))}
    # Cas de test avec leurs métadonnées (évaluation à 4 voies, finetune eval)
    with (out_dir / "test_cases.jsonl").open("w", encoding="utf-8") as f:
        for sc, _, _ in rows.get("test", []):
            f.write(json.dumps({k: sc[k] for k in ("id", "base", "injection", "oracle", "facts", "features",
                                                   "signals", "messages")}, ensure_ascii=False) + "\n")

    n_train = stats["train"]["n"]
    iters = max(50, epochs * n_train // batch_size)
    updates = max(1, iters // grad_accum)
    config = {
        "model": model,               # Qwen3-4B-Instruct-2507 quantifié 4 bits : QLoRA (adaptateurs en 16 bits)
        "train": True,
        "data": str(out_dir),
        "fine_tune_type": "lora",
        "num_layers": -1,             # adaptateurs sur TOUTES les couches (pas seulement les 16 dernières)
        "lora_parameters": {"rank": 16, "scale": 2.0, "dropout": 0.05},   # alpha = 32 = 2 × rank ; toutes les Linear
        "batch_size": batch_size,
        "grad_accumulation_steps": grad_accum,   # batch effectif = 8 sans exploser la mémoire
        "iters": iters,               # ≈ `epochs` passes sur le train
        "learning_rate": 1e-4,
        "lr_schedule": {"name": "cosine_decay", "warmup": max(5, updates // 10), "warmup_init": 1e-6,
                        "arguments": [1e-4, updates, 1e-6]},
        "optimizer": "adamw",
        "optimizer_config": {"adamw": {"weight_decay": 0.01}},
        "mask_prompt": True,          # la perte ne porte que sur la réponse JSON
        "max_seq_length": 3072,
        "grad_checkpoint": True,      # moins de mémoire (24 Go partagés avec macOS)
        "steps_per_report": 10,
        "steps_per_eval": max(25, iters // 8),
        "val_batches": -1,
        "save_every": max(50, iters // 4),
        "adapter_path": str(out_dir / "adapters"),
        "seed": seed,
    }
    (out_dir / "lora.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    # Variante « ~1 h » mesurée sur un M4 Pro 24 Go : 16 dernières couches, sans checkpointing, ~1,15 époque.
    fast_iters = max(50, int(1.15 * n_train / batch_size))
    fast_updates = max(1, fast_iters // grad_accum)
    fast = {**config, "num_layers": 16, "grad_checkpoint": False, "iters": fast_iters, "val_batches": 8,
            "steps_per_eval": 50, "save_every": 50,
            "lr_schedule": {"name": "cosine_decay", "warmup": max(5, fast_updates // 10), "warmup_init": 1e-6,
                            "arguments": [1e-4, fast_updates, 1e-6]}}
    (out_dir / "lora_fast.yaml").write_text(yaml.safe_dump(fast, sort_keys=False, allow_unicode=True), encoding="utf-8")
    stats["lora"] = {"iters": iters, "updates": updates, "effective_batch": batch_size * grad_accum}
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats
