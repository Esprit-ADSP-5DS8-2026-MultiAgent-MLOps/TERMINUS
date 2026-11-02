import type { ReactNode } from "react";

/** Le tableau d'annonces de la gare : c'est ainsi que parle Terminus. */
export function Annonce({ children }: { children: ReactNode }) {
  return (
    <div className="annonce" role="status">
      <span className="label">ANNONCE</span>
      <span className="small">{children}</span>
    </div>
  );
}

export function ErrorLine({ error }: { error: string | null }) {
  return error ? <div className="error" role="alert">{error}</div> : null;
}

export function Cmd({ children, opt }: { children: ReactNode; opt?: ReactNode }) {
  return (
    <span className="cmd">
      $ {children} {opt && <span className="opt">{opt}</span>}
    </span>
  );
}

// Paysage ASCII : décor seulement (lu par personne, ignoré par les lecteurs d'écran).
const TERRAIN = [
  "               _    ' \\       /                \\          _                                                         -                -  -          /   '                            ",
  "       ../      ^                     /             /               /        - \\         '    .                     -       .           \\                  / /      .           -   ",
  " \\/    _      _  \\  \\   _  -  \\ \\//    _ \\           X   \\\\   /_       _\\=    X/         /           # #  #            -\\=   ##       -           =  _#  X/   # _ \\   -X     - X-#  ",
  "/  ▒=  #/  /   / =▒  #/X=#  = \\/▒# X  ▒=▒  ▒ ▒#/#X▒#=   ▒=▒ #\\    /=▒ / ▒\\  \\X \\ ##X#  XX## ▒ ▒▒X\\/ =X X ▒#     ▒#\\ = =\\/#▒▒=  \\ #/= X= /    \\ X ==  X   =   =  = X▒ XX = / \\ ▒  =▒\\",
  "▒#  ▒X ▓#▒▓▓ ▒▓▓  #▒ X▓#▓XX▒###▒▒▓▓▓▓▒ #▒▒▒X ▓#▒▒X▓#XX## #▓##▒XX▓#▓XX▓ XX ▓▒#▒▓▒#▓##X▓#▒▒▒▒ ▒ ▓ ▒▓▓▒XX  ▒▒ ▓#▒   X #▒▓#▒#▓▓▒ X#▓X▒▒ ▓X ##▓XX ▒#▒#X##XX ▓#  #▓▓▓▓X▒ X #▓# ▒ ▓ #X▒X##▒",
  "▒▓▓▓▒▓█▓█▒▓█▓▓██▓▒▓▓▒▒▓█▓▒█▒▓▓█ ▓█▓█▓▓▒█▓▒▓█▓▓▒▒▒▓▓▒▓▓▓▒▓▓▓▓█▓▒▒█▓▓▓▓▒▓▓▒▓▓▒▒▓▓▓█▒▓█▓▓▓▓▒▒▒▓▓▒▓▓▓▓▓▓███▓▒▓▓█▓▓▓▓▒▓▓█▒█▒▒ ▓▓▓█▓▓▓▒▒█▓█▓▓▓ ▓▒▓▓▒▓▓▓█▓▒▒▒█▒▒▓▓▓▓▒▓▓▓▓▓ ▓▓▒▓█▓██▒█▓▒▒▓▓█",
].join("\n");

export function Terrain({ moving }: { moving: boolean }) {
  return (
    <pre aria-hidden="true" className="terrain">
      <span className={`inner${moving ? " moving" : ""}`}>{TERRAIN}</span>
    </pre>
  );
}
