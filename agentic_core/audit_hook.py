"""Hook d'audit (PEP 578) exécuté DANS le sandbox, installé avant tout code du candidat.
Il note les tentatives de réseau et de lancement de processus, quelle que soit la façon dont le code les dissimule
(getattr, concaténation de noms, exec…) : le noyau Python émet l'évènement, pas le texte du source.

Chaque évènement est attribué : « candidate » (le code du projet est l'appelant direct) ou « library » (un paquet installé,
p. ex. joblib/loky qui lance `lscpu`). Python, côté Validator, ne sanctionne que « candidate » ; « library » reste une limite signalée.
Usage module : install(log_path, project_dir).   Usage script : python _audit.py LOG PROJET script.py  (comme `python script.py`).
Bibliothèque standard uniquement. L'état du hook vit dans une fermeture : le code du candidat ne peut pas le modifier.
"""
import json
import os
import sys
import sysconfig

NETWORK = frozenset({"socket.connect", "socket.getaddrinfo", "socket.gethostbyname", "urllib.Request"})
PROCESS = frozenset({"subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn"})
WATCHED = NETWORK | PROCESS
LOOPBACK = {"127.0.0.1", "::1", "localhost", "0.0.0.0", ""}


def install(log_path, project_dir):
    project = os.path.realpath(project_dir)
    stdlib = os.path.realpath(sysconfig.get_paths()["stdlib"])
    here = os.path.realpath(__file__)
    seen = set()

    def caller_origin():
        f = sys._getframe(2)
        while f is not None:
            fn = f.f_code.co_filename
            real = fn if fn.startswith("<") else os.path.realpath(fn)      # chemin relatif : résolu contre le cwd (= projet)
            internal = real == here or fn.startswith("<") or ("site-packages" not in real and real.startswith(stdlib + os.sep))
            if not internal:
                if real.startswith(project + os.sep) and "site-packages" not in real:
                    return "candidate", f"{os.path.relpath(real, project)}:{f.f_lineno}"
                return "library", f"{os.path.basename(os.path.dirname(real))}/{os.path.basename(real)}:{f.f_lineno}"
            f = f.f_back
        return "library", "?"

    def detail(event, args):
        try:
            if event == "socket.connect":
                addr = args[1]
                host = addr[0] if isinstance(addr, tuple) else str(addr)
                if not isinstance(addr, tuple) or str(host) in LOOPBACK:
                    return None                                   # AF_UNIX et boucle locale : pas du réseau externe
                return f"{host}:{addr[1]}"
            if event in ("socket.getaddrinfo", "socket.gethostbyname"):
                host = args[0]
                return None if str(host) in LOOPBACK or host is None else str(host)
            if event == "urllib.Request":
                return str(args[0])[:120]
            if event == "subprocess.Popen":
                a = args[1] if len(args) > 1 else args[0]
                return (" ".join(map(str, a)) if isinstance(a, (list, tuple)) else str(a))[:120]
            if event == "os.system":
                return str(args[0])[:120]
            return " ".join(map(str, args[1:2]))[:120] or str(args[0])[:120]
        except Exception:
            return "?"

    def hook(event, args):
        if event not in WATCHED:
            return
        d = detail(event, args)
        if d is None:
            return
        origin, where = caller_origin()
        key = (event, d, where)
        if key in seen:
            return
        seen.add(key)
        try:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"event": event, "kind": "network" if event in NETWORK else "process",
                                     "origin": origin, "where": where, "detail": d}) + "\n")
        except Exception:
            pass

    sys.addaudithook(hook)


if __name__ == "__main__":
    import runpy
    log, proj, script = sys.argv[1], sys.argv[2], sys.argv[3]
    os.chdir(proj)
    install(log, proj)
    sys.argv = [script]
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    runpy.run_path(script, run_name="__main__")
