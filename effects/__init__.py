"""
Effects package for Latrix Engine.

Provides animation effects for layers.
"""

from effects.base import Effect, CompositeEffect
from effects.breathing import BreathingEffect
from effects.camera import CameraEffect

__all__ = [
    "Effect",
    "CompositeEffect",
    "BreathingEffect",
    "CameraEffect",
]
