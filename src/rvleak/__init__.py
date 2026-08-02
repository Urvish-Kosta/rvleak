"""rvleak -- microarchitectural leakage analysis for RV32IM software.

A cycle-level RV32IM simulator that emits a per-cycle activity trace alongside
an exact cycle-to-instruction mapping, plus the statistical machinery (TVLA,
CPA) to turn those traces into a verdict about whether a piece of software
leaks, and which instruction is responsible.
"""

from .analysis import cpa, hd_hypotheses, hw_hypotheses, tvla
from .asm import Program, assemble
from .campaign import cpa_campaign, full_key_campaign, null_campaign, tvla_campaign
from .uarch import CacheConfig, Machine, ModelConfig, Trace

__version__ = "0.1.0"
__all__ = [
    "assemble", "Program", "Machine", "ModelConfig", "CacheConfig", "Trace",
    "tvla", "cpa", "hw_hypotheses", "hd_hypotheses",
    "tvla_campaign", "cpa_campaign", "full_key_campaign", "null_campaign",
]
