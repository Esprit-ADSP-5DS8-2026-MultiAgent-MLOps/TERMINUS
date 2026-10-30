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
    # (p-value ~ 0) : on passe alors à des distances, qui mesurent l'ampleur de l'écart.
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
    out_of_range_rate: float = 0.01    # >1 % hors de [min, max] de la référence : information
    out_of_range_warning: float = 0.30  # >=30 % hors de tout ce qui a été vu : changement d'unité, pas un drift
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


@dataclass(frozen=True)
class PolicyConfig:
    """Ce que l'agent a le droit de faire seul (agent/guards.py)."""

    min_confidence: float = 0.6        # confiance déclarée du diagnostic
    min_agreement: float = 0.66        # au moins 2 diagnostics sur 3 d'accord (2/3 = 0.666…)
    retrain_cooldown_h: float = 24.0   # pas deux demandes de réentraînement en 24 h
    max_auto_actions_24h: int = 10     # disjoncteur : au-delà, un humain reprend la main


@dataclass(frozen=True)
class AgentConfig:
    samples: int = 3                   # self-consistency : 3 diagnostics indépendants
    sample_temperature: float = 0.7    # diversité entre les 3 (le vote mesure l'incertitude)
    max_tool_calls: int = 6            # budget de l'enquête
    max_revisions: int = 1             # le juge peut demander une révision
    use_mcp: bool = True               # l'enquêteur découvre et appelle les outils via MCP (sinon : appel direct)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
