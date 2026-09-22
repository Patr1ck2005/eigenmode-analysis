"""Raw-observation grouping for ordinary real-frequency band surfaces.

This opt-in API integrates the frozen Stable v1 policy. It does not replace
legacy path/graph tracking or certify a unique physical mode identity.
"""
from .api import (
    ALGORITHM_VERSION,
    BandCandidates,
    BandGroupingResult,
    GroupingConfig,
    ParameterGrid,
    group_bands,
)

__all__ = ['ALGORITHM_VERSION', 'BandCandidates', 'BandGroupingResult',
           'GroupingConfig', 'ParameterGrid', 'group_bands']
