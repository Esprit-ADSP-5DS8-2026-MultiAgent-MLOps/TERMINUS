// La locomotive de Terminus, en pixel art (dessin original). Un seul élément, les pixels sont des ombres.
const SPRITE = [
  "..........KK..",
  "KKKKK.....KK..",
  "KWWSK.SSSSSSS.",
  "KWWSKSSSSSSSSK",
  "KSSSKSSSSSSSWK",
  "KSSSSSSSSSSSSK",
  "KKKKKKKKKKKKKK",
  ".KK...KK...KK.",
  ".KK...KK...KK.",
];
const COLORS: Record<string, string> = { K: "var(--ink)", S: "var(--signal)", W: "var(--panel)" };

function shadows(px: number): string {
  const out: string[] = [];
  SPRITE.forEach((row, y) =>
    [...row].forEach((c, x) => {
      if (c !== ".") out.push(`${x * px}px ${y * px}px 0 0 ${COLORS[c]}`);
    }),
  );
  return out.join(", ");
}

const CACHE: Record<number, string> = {};

export function Loco({ px = 6, rolling = false, label }: { px?: number; rolling?: boolean; label?: string }) {
  const shadow = (CACHE[px] ??= shadows(px));
  return (
    <div
      role={label ? "img" : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
      className={rolling ? "rolling" : undefined}
      style={{ position: "relative", width: 14 * px, height: 9 * px, flex: "none" }}
    >
      <span style={{ position: "absolute", left: 0, top: 0, width: px, height: px, boxShadow: shadow }} />
    </div>
  );
}

/** La petite loco qui traverse la barre du haut, de temps en temps. */
export function PassingLoco({ delay = 4 }: { delay?: number }) {
  return (
    <div className="passage" aria-hidden="true" style={{ animationDelay: `${delay}s` }}>
      <Loco px={3} />
    </div>
  );
}
