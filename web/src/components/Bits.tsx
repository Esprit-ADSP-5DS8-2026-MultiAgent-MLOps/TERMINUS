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

// Chaque ligne est doublée : le décor défile en boucle sans couture (translateX -50 %).
const WIDTH = Math.max(...TERRAIN.split("\n").map((l) => l.length));
const ROWS = TERRAIN.split("\n").map((l) => l.padEnd(WIDTH, " ").repeat(2));
const SWAYING = 3;   // les 3 lignes du haut (herbes, débris) bougent un peu ; le sol reste fixe

const GRASS = new Set(["/", "\\", "|", "'", "^", ",", "\""]);   // brins qui se penchent ; le reste : débris qui sautent

/** Herbes et débris : les brins se penchent dans le vent, les débris sautillent, chacun à son rythme
 * (délai et durée tirés de sa position, donc stables d'un rendu à l'autre). */
function swaying(row: string, r: number) {
  const out: ReactNode[] = [];
  let plain = "";
  [...row].forEach((c, i) => {
    if (c === " ") {
      plain += c;
      return;
    }
    if (plain) out.push(plain);
    plain = "";
    const seed = (i * 7919 + r * 104729) % 997;
    const kind = GRASS.has(c) ? "herbe" : seed % 4 === 0 ? "debris saute" : "debris";
    out.push(<span key={i} className={kind} style={{ animationDelay: `${-(seed % 40) / 10}s`,
      animationDuration: `${1.4 + (seed % 9) / 6}s` }}>{c}</span>);
  });
  if (plain) out.push(plain);
  return out;
}

const SCENE = ROWS.map((row, r) => (
  <span key={r}>{r < SWAYING ? swaying(row, r) : row}{r < ROWS.length - 1 ? "\n" : ""}</span>
));

/** Le paysage défile tant que la loco roule, et s'arrête avec elle (au signal, en gare). */
export function Terrain({ moving }: { moving: boolean }) {
  return (
    <pre aria-hidden="true" className="terrain">
      <span className={`inner${moving ? " travel" : ""}`}>{SCENE}</span>
    </pre>
  );
}
