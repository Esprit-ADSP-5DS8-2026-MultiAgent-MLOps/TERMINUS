"""Génère eval/projects/house_prices : projet de RÉGRESSION de test (prix de logements, données synthétiques).

    python eval/make_house_prices.py

Les prix suivent une règle connue (surface, quartier, âge, distance au centre, parking) plus un bruit :
le modèle a quelque chose à apprendre, et l'Agent 4 peut simuler un drift ou un concept drift dessus.
Graine fixe : le fichier est identique à chaque génération.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROWS = 1500
SEED = 7
DISTRICTS = {"centre": 1.35, "nord": 1.10, "est": 0.95, "ouest": 1.00, "sud": 0.85}


def make(rows: int = ROWS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    surface = rng.gamma(shape=6.0, scale=14.0, size=rows).clip(25, 300).round(0)
    rooms = np.clip(np.round(surface / 28 + rng.normal(0, 0.6, rows)), 1, 8).astype(int)
    age = rng.integers(0, 70, rows)
    district = rng.choice(list(DISTRICTS), size=rows, p=[0.18, 0.22, 0.2, 0.22, 0.18])
    distance = np.where(district == "centre", rng.uniform(0.3, 4, rows), rng.uniform(2, 25, rows)).round(1)
    floor = rng.integers(0, 12, rows)
    parking = (rng.random(rows) < 0.45).astype(int)

    mult = np.array([DISTRICTS[d] for d in district])
    price = (2.1 * surface * mult * (1 - 0.0045 * age) - 1.8 * distance + 14 * parking
             + 1.5 * np.minimum(floor, 6) + 6 * rooms)
    price = price * rng.lognormal(0, 0.08, rows) + rng.normal(0, 8, rows)     # bruit : on ne prédit pas tout
    return pd.DataFrame({
        "surface_m2": surface, "rooms": rooms, "age_years": age, "district": district,
        "distance_center_km": distance, "floor": floor, "has_parking": parking,
        "price_k": price.clip(20).round(1),                                     # prix en milliers
    })


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "projects" / "house_prices" / "data.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    make().to_csv(out, index=False)
    print(f"{out} : {ROWS} lignes")
