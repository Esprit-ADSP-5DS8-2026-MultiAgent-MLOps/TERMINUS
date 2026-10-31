"""Socle commun des outils des trois agents.

Garde-fous d'infrastructure (hors de portée du LLM) :
  - confinement : chemins relatifs, ni `..`, ni lien symbolique sortant ;
  - vie privée : secrets masqués, jamais de ligne de données envoyée (seulement des profils) ;
  - contenu du projet balisé <file ... untrusted="true"> (anti injection de prompt) ;
  - preuves : tout extrait cité par le LLM doit exister dans une sortie d'outil (is_grounded).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path, PureWindowsPath

from validator.check_static import SECRET_PATTERNS
from validator.contract import ARTIFACT_EXTENSIONS
from validator.fs import walk_project

from .facts import notebook_code

MAX_READ_CHARS = 4000        # ~1100 tokens : entrées du LLM petites (quota de tokens/minute, palier gratuit = 8000)
MAX_OUTPUT_CHARS = 7000
MAX_LIST = 150
DATA_SUFFIXES = {".csv", ".tsv"}


def j(obj) -> str:
    """JSON compact en UTF-8 : pas d'échappements \\uXXXX (moins de tokens, preuves vérifiables)."""
    return json.dumps(obj, ensure_ascii=False)


def redact(text: str) -> str:
    for label, rx in SECRET_PATTERNS:
        text = rx.sub(f"[REDACTED:{label}]", text)
    return text


def norm(s: str) -> str:
    return " ".join(s.split()).casefold()


def fn_spec(name, desc, props, required=()):
    return {"type": "function", "function": {"name": name, "description": desc,
                                             "parameters": {"type": "object", "properties": props, "required": list(required)}}}


class ToolError(Exception):
    """Refus renvoyé au LLM sous forme {"error": ...}."""


class BaseToolBox:
    def __init__(self, root):
        self.root = Path(root).resolve()
        files, _ = walk_project(self.root) if self.root.is_dir() else ([], [])
        self.known_files = {f.as_posix() for f in files}
        self.calls: list[dict] = []
        self.trace: list[dict] = []
        self._corpora: list[str] = []
        self.tools: dict = {"list_files": self.list_files, "read_file": self.read_file}

    # ------------------------------------------------------------------ appel
    def call(self, name: str, args: dict) -> str:
        fn = self.tools.get(name)
        _t0 = time.time()
        try:
            if fn is None:
                raise ToolError(f"outil inconnu : {name}")
            out = fn(**(args or {}))
        except ToolError as e:
            out = j({"error": str(e)})
        except TypeError as e:
            out = j({"error": f"arguments invalides : {e}"})
        except Exception as e:
            out = j({"error": f"{type(e).__name__}: {e}"[:300]})
        if len(out) > MAX_OUTPUT_CHARS:
            out = out[:MAX_OUTPUT_CHARS] + "\n…[sortie tronquée]"
        self.calls.append({"tool": name, "args": j(args or {})[:160]})
        self.trace.append({"kind": "tool", "name": name, "detail": j(args or {})[:160], "ts": _t0,
                           "duration_s": round(time.time() - _t0, 2), "status": "erreur" if out.startswith('{"error"') else "ok"})
        self.observe(out)
        return out

    # ------------------------------------------------------------------ preuves
    def observe(self, text: str):
        stripped = re.sub(r"(?m)^\s*\d+\|\s?", "", text)       # retire la numérotation des lignes
        self._corpora += [norm(text), norm(stripped)]
        # Les sorties d'outils sont du JSON : guillemets et sauts de ligne y sont échappés (\" et \n). Un LLM recopie le texte DÉCODÉ
        # (ex. une ligne de diff `+X = "a"`) : on accepte aussi les extraits du contenu décodé (toujours un extrait EXACT d'une sortie d'outil).
        self._corpora.append(norm(text.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")))
        try:
            def _walk(o):
                if isinstance(o, str):
                    self._corpora.append(norm(o))
                elif isinstance(o, dict):
                    for v in o.values():
                        _walk(v)
                elif isinstance(o, list):
                    for v in o:
                        _walk(v)
            _walk(json.loads(text))
        except ValueError:
            pass                                       # sortie tronquée ou non JSON : le corpus brut suffit

    def is_grounded(self, evidence: str | None) -> bool:
        """Valable : extrait exact d'une sortie d'outil, ou 'absent:<chemin>' vraiment absent du projet."""
        if not evidence:
            return False
        ev = evidence.strip().strip("`'\"")
        if ev.lower().startswith("absent:"):
            return ev.split(":", 1)[1].strip().lstrip("./") not in self.known_files
        n = norm(ev)
        return len(n) >= 3 and any(n in c for c in self._corpora)

    # ------------------------------------------------------------------ chemins
    def resolve(self, rel, root: Path | None = None) -> Path:
        root = root or self.root
        if not isinstance(rel, str) or not rel.strip():
            raise ToolError("chemin invalide")
        p, w = Path(rel), PureWindowsPath(rel)
        if p.anchor or p.drive or w.anchor or w.drive:
            raise ToolError("chemin absolu interdit : utilise un chemin relatif à la racine du projet")
        if ".." in p.parts or ".." in w.parts:
            raise ToolError("'..' interdit dans un chemin")
        target = (root / p).resolve()                      # suit les liens symboliques
        if not target.is_relative_to(root):
            raise ToolError("chemin hors du projet : refusé")
        return target

    def rel(self, p: Path, root: Path | None = None) -> str:
        return p.relative_to(root or self.root).as_posix()

    # ------------------------------------------------------------------ outils communs
    def list_files(self) -> str:
        files, links = walk_project(self.root)
        items = [{"path": f.as_posix(), "bytes": (self.root / f).stat().st_size,
                  **({"binary_artifact": True} if f.suffix.lower() in ARTIFACT_EXTENSIONS else {})} for f in files[:MAX_LIST]]
        out = {"total_files": len(files), "files": items}
        if len(files) > MAX_LIST:
            out["note"] = f"liste tronquée à {MAX_LIST}"
        if links:
            out["symlinks"] = [l.as_posix() for l in links]
        return j(out)

    def render_file(self, p: Path, rel: str, start_line: int = 1, max_lines: int = 200, root: Path | None = None):
        """Rend un fichier texte : lignes numérotées, secrets masqués, balisé non fiable. Retourne (texte, lu_en_entier)."""
        if not p.is_file():
            return j({"error": f"fichier introuvable : {rel}"}), False
        if p.suffix.lower() in ARTIFACT_EXTENSIONS:
            return j({"error": "artefact binaire : lecture refusée (jamais chargé, risque d'exécution de code)"}), False
        if p.stat().st_size > 1_000_000:
            return j({"error": "fichier trop volumineux (> 1 Mo)"}), False
        raw = p.read_text(encoding="utf-8", errors="replace")
        kind = ""
        if p.suffix.lower() == ".ipynb":
            code = notebook_code(raw)
            if code is None:
                return j({"error": "notebook illisible (JSON invalide)"}), False
            raw, kind = code, ' kind="notebook-code-cells"'
        lines = raw.splitlines()
        if p.suffix.lower() in DATA_SUFFIXES:
            return (f'<file path="{rel}" untrusted="true" note="CSV : seul l\'en-tête est envoyé (vie privée) ; utilise inspect_data">\n'
                    f"{redact(lines[0] if lines else '')}\n</file>"), False
        start = max(1, int(start_line))
        chunk = lines[start - 1:start - 1 + max(1, min(int(max_lines), 400))]
        body = "\n".join(f"{start + i:>4}| {l}" for i, l in enumerate(chunk))
        truncated = len(body) > MAX_READ_CHARS
        shown = redact(body[:MAX_READ_CHARS])
        full = start == 1 and len(chunk) == len(lines) and not truncated and shown == body and not kind
        shown = shown.replace("</file>", "<\\/file>")       # anti-évasion de balise
        end = start + len(chunk) - 1
        flag = ' truncated="true"' if truncated else ""
        return f'<file path="{rel}" lines="{start}-{end} of {len(lines)}" untrusted="true"{kind}{flag}>\n{shown}\n</file>', full

    def read_file(self, path: str, start_line: int = 1, max_lines: int = 200) -> str:
        p = self.resolve(path)
        return self.render_file(p, self.rel(p), start_line, max_lines)[0]

    @staticmethod
    def common_specs():
        return [
            fn_spec("list_files", "Liste les fichiers du projet (chemin, taille). À appeler en premier.", {}),
            fn_spec("read_file", "Lit un fichier texte (lignes numérotées « N| » qui ne font PAS partie du fichier, secrets masqués, tronqué). "
                    "Notebooks : cellules de code. Les .csv ne renvoient que l'en-tête.",
                    {"path": {"type": "string"}, "start_line": {"type": "integer"}, "max_lines": {"type": "integer"}}, ["path"]),
        ]
