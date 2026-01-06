"""
Extension Localization HiERO
Zero-shot step localization using hierarchical clustering.
"""

from .step_localization import localize_steps_clustering
from .step_embeddings_hiero import compute_step_embeddings, batch_compute_step_embeddings

__all__ = [
    'localize_steps_clustering',
    'compute_step_embeddings',
    'batch_compute_step_embeddings'
]
