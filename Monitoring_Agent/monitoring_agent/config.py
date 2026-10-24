"""
Seuils du moteur d'analyse.

Ils sont regroupés ici, versionnés, et jamais choisis par le LLM :
un seuil qui change doit laisser une trace dans git.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DriftConfig:
    # Au-delà de cette taille d'échantillon, un test statistique détecte des écarts minuscules
    # (p-value ~ 0) : on passe alors à des DISTANCES, qui mesurent l'ampleur de l'écart.
    # C'est la règle par défaut d'Evidently.
    large_sample: int = 1000
    wasserstein: float = 0.1      # distance de Wasserstein / écart-type de la référence
    jensen_shannon: float = 0.1   # distance de Jensen-Shannon (base 2, entre 0 et 1)
    p_value: float = 0.05         # petits échantillons : KS (numérique), chi² (catégoriel)
    psi_moderate: float = 0.1     # seuils usuels du PSI (secteur bancaire)
    psi_major: float = 0.25
    psi_bins: int = 10
    dataset_share: float = 0.5    # dataset en drift si >= 50 % des features dérivent


@dataclass(frozen=True)
class QualityConfig:
    missing_increase: float = 0.05     # +5 points de valeurs manquantes par rapport à la référence
    missing_critical: float = 0.20
    unseen_rate: float = 0.01          # >1 % de catégories jamais vues
    unseen_critical: float = 0.10
    out_of_range_rate: float = 0.01    # >1 % hors de [min, max] de la référence
    type_mismatch_rate: float = 0.0    # une seule valeur non numérique suffit


@dataclass(frozen=True)
class PerformanceConfig:
    # Une fenêtre est dégradée si la métrique sort de moyenne ± n_std * écart-type,
    # calculés sur des fenêtres de la référence : le seuil vient de la variabilité
    # naturelle du modèle, pas d'un chiffre choisi au hasard.
    n_std: float = 3.0
    min_reference_windows: int = 4    # en dessous : bootstrap pour estimer la variabilité
    bootstrap_samples: int = 30
    random_state: int = 42


@dataclass(frozen=True)
class AnalyzerConfig:
    window_size: int = 5000
    drift: DriftConfig = field(default_factory=DriftConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
