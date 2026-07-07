"""
Geometry package for Latrix Engine.

Provides vector math, easing functions, and spring physics.
"""

# Explicit imports from files
from geometry.vec2 import Vec2
from geometry.easing import *
from geometry.spring import ScalarSpring, Vec2Spring

__all__ = [
    "Vec2",
    "ScalarSpring",
    "Vec2Spring",
    # easing functions will be included via *
]
