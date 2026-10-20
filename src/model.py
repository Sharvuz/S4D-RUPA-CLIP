"""
Model definitions for S4D-RUPA-CLIP.
Re-exports S4DRUPACLIP from rupa_clip_model for modularity.
"""

from rupa_clip_model import (
    S4DRUPACLIP,
    NormalPrototypeBank,
    ResidualSemanticProjector,
    DynamicEvidentialSafeGate
)
from s4d_block import BiS4DBlock, BiS4DEncoder, DiagonalSSM

__all__ = [
    'S4DRUPACLIP',
    'NormalPrototypeBank',
    'ResidualSemanticProjector',
    'DynamicEvidentialSafeGate',
    'BiS4DBlock',
    'BiS4DEncoder',
    'DiagonalSSM'
]
