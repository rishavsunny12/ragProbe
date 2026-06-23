"""Adversarial question generators, one per structural failure mode."""

from .base import BaseGenerator
from .buried_fact import BuriedFactGenerator
from .distractor import DistractorGenerator
from .multihop import MultiHopGenerator
from .near_miss import NearMissGenerator

GENERATORS = {
    "multi_hop": MultiHopGenerator,
    "buried_fact": BuriedFactGenerator,
    "distractor": DistractorGenerator,
    "near_miss": NearMissGenerator,
}

__all__ = [
    "BaseGenerator",
    "MultiHopGenerator",
    "BuriedFactGenerator",
    "DistractorGenerator",
    "NearMissGenerator",
    "GENERATORS",
]
