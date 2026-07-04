"""Public surface for the naval_salvo package used by the Streamlit app."""

from .domains import (
    CYBER_SUBTYPES,
    DOMAIN_ORDER,
    KINETIC_DOMAINS,
    N_DOMAINS,
    CyberSubtype,
    Domain,
    parse_domain,
)
from .admissibility import (
    DEFAULT_CHI,
    Admissibility,
    canonical_matrix,
    degenerate_johns_pilnick_hughes,
)
from .state import BattleState, Force, UnitType, UnitTypeState
from .parameters import DirectionalParameters, EngagementParameters, PairParameters
from .targeting import Manual, StrengthProportional, TargetingPolicy, ThreatWeighted, Uniform
from .coefficients import EngagementBuilder, ThroughputGrid, apply_targeting_policy
from .cyber import (
    ChannelPhi,
    CyberModulator,
    DecomposedPhi,
    HauskenPhi,
    SimplePhi,
    phi_logistic,
    phi_sigmoid,
)
from .dynamics import SalvoResult, salvo_step
from .solver import CampaignTrajectory, run_campaign
from .stochastic import (
    EngagementOrder,
    HomogeneousForce,
    MultiDomainForce,
    StochasticResult,
    UnitGroup,
    run_homogeneous_battle,
    run_multidomain_battle,
)
from .scenarios import BACIA_CAMPOS_PARAMETERS, BaciaCamposConfig, build_bacia_campos
from .validation import (
    BRITISH_GROUPS,
    GERMAN_GROUPS,
    CoronelGroup,
    CoronelTargetingMinute,
    HughesScenario,
    build_coronel_engagement,
    build_hughes_homogeneous_engagement,
    coronel_forces,
    coronel_minute_one_targeting,
    hughes_analytical,
    jph_minute_one_delta_good_hope,
)

__all__ = [name for name in globals() if not name.startswith("_")]
