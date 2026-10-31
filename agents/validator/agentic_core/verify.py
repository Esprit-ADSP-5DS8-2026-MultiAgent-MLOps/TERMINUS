"""Vérifie un rapport (version, sceau, empreinte du projet) et affiche son lignage.
    python -m agentic_core.verify rapport.json [dossier_du_projet]
"""
import sys

from .handoff import verify_report
from .schemas import load_report, report_id_of


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print(__doc__)
        return 2
    rep = load_report(argv[0])
    chk = verify_report(rep, argv[1] if len(argv) > 1 else None)
    print(f"{rep.report_type} {report_id_of(rep)}")
    for ref in rep.lineage:
        print(f"  ← {ref.report_type} {ref.report_id}")
    print("OK" if chk.ok else "REFUSÉ : " + " ; ".join(chk.reasons))
    return 0 if chk.ok else 1


if __name__ == "__main__":
    sys.exit(main())
