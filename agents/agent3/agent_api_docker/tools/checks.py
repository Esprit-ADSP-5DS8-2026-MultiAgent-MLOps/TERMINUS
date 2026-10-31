"""Validations cote conteneur : hadolint (optionnel), attente de /health, smoke test de /predict."""
from __future__ import annotations

import re
import shutil
import subprocess
import time
from typing import Callable, Optional

import httpx


def lint_dockerfile(path: str) -> dict:
    """hadolint (niveau erreur seulement). Ignore proprement si hadolint n'est pas installe."""
    exe = shutil.which("hadolint")
    if not exe:
        return {"skipped": True, "ok": True, "output": "hadolint non installe : lint du Dockerfile ignore"}
    proc = subprocess.run([exe, "--failure-threshold", "error", path], capture_output=True, text=True, timeout=60)
    return {"skipped": False, "ok": proc.returncode == 0, "output": (proc.stdout + proc.stderr).strip()}


def healthcheck(url: str, timeout_s: int = 60, interval_s: float = 1.0,
                alive: Optional[Callable[[], bool]] = None) -> dict:
    """GET url jusqu'a 200. `alive()` -> False arrete l'attente des que le conteneur est mort (inutile d'attendre)."""
    t0 = time.time()
    detail = "pas de reponse"
    while time.time() - t0 < timeout_s:
        if alive is not None and not alive():
            return {"ok": False, "status": None, "elapsed_s": round(time.time() - t0, 1),
                    "detail": "le conteneur s'est arrete avant de repondre"}
        try:
            r = httpx.get(url, timeout=3)
            if r.status_code == 200:
                return {"ok": True, "status": 200, "elapsed_s": round(time.time() - t0, 1), "detail": r.text[:200]}
            detail = f"HTTP {r.status_code} : {r.text[:150]}"
        except httpx.HTTPError as e:
            detail = f"{type(e).__name__}: {e}"
        time.sleep(interval_s)
    return {"ok": False, "status": None, "elapsed_s": round(time.time() - t0, 1),
            "detail": f"/health ne repond pas en {timeout_s} s ({detail})"}


def smoke_test_predict(base_url: str, example_input: dict, example_output: Optional[dict]) -> dict:
    """POST /predict avec l'exemple du rapport ; compare la prediction a l'exemple s'il y en a un."""
    try:
        r = httpx.post(base_url.rstrip("/") + "/predict", json=example_input, timeout=30)
    except httpx.HTTPError as e:
        return {"ok": False, "detail": f"{type(e).__name__}: {e}"}
    if r.status_code != 200:
        return {"ok": False, "detail": f"HTTP {r.status_code} : {r.text[:300]}"}
    body = r.json()
    if example_output and "prediction" in example_output and body.get("prediction") != example_output["prediction"]:
        return {"ok": False, "detail": f"prediction {body.get('prediction')!r} != attendue {example_output['prediction']!r}",
                "body": body}
    return {"ok": True, "detail": "", "body": body}


def missing_copy_sources(run_dir) -> list[str]:
    """Sources des instructions COPY (hors `--from=`) absentes du contexte de build : erreur detectee avant `docker build`."""
    from pathlib import Path
    root = Path(run_dir)
    text = (root / "Dockerfile").read_text(encoding="utf-8", errors="replace").replace("\\\n", " ")
    missing = []
    for line in text.splitlines():
        parts = line.split()
        if not parts or parts[0].upper() != "COPY" or "[" in line:
            continue
        flags = [p for p in parts[1:] if p.startswith("--")]
        if any(f.startswith("--from") for f in flags):
            continue
        args = [p for p in parts[1:] if not p.startswith("--")]
        for src in args[:-1]:
            if src not in (".", "./") and not re.search(r"[*?\[]", src) and not (root / src).exists():
                missing.append(src)
    return missing
