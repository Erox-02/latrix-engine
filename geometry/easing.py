"""
Easing functions for smooth, cinematic animation.

All functions take a float t in [0, 1] and return a float in [0, 1].
Pure functions - no state, no side effects.

Mathematical principle: Easing functions map linear time progression
to non-linear value progression, creating natural-looking motion.
"""

import math
from collections.abc import Callable


def linear(t: float) -> float:
    """
    Linear interpolation: f(t) = t
    
    Constant speed. No acceleration or deceleration.
    Use when you want mechanical, uniform motion.
    """
    return t


def smoothstep(t: float) -> float:
    """
    Smoothstep: f(t) = 3t² - 2t³
    
    Hermite interpolation with zero derivatives at endpoints.
    Creates smooth acceleration and deceleration.
    
    Great default easing for most animations.
    """
    return t * t * (3.0 - 2.0 * t)


def smootherstep(t: float) -> float:
    """
    Smootherstep: f(t) = 6t⁵ - 15t⁴ + 10t³
    
    Hermite interpolation with zero first AND second derivatives at endpoints.
    Even smoother than smoothstep.
    
    Excellent for camera motion where jerk should be minimized.
    """
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def ease_in_quad(t: float) -> float:
    """
    Ease-in quadratic: f(t) = t²
    
    Starts slow, ends fast.
    Formula: t²
    
    Use for: Objects accelerating from rest.
    """
    return t * t


def ease_out_quad(t: float) -> float:
    """
    Ease-out quadratic: f(t) = t(2 - t)
    
    Starts fast, ends slow.
    Formula: t(2 - t)
    
    Use for: Objects decelerating to rest.
    """
    return t * (2.0 - t)


def ease_in_out_quad(t: float) -> float:
    """
    Ease-in-out quadratic.
    
    Formula: 2t² for t < 0.5, else 1 - 2(1-t)²
    
    Smooth acceleration and deceleration.
    Classic animation curve - widely used.
    """
    if t < 0.5:
        return 2.0 * t * t
    return 1.0 - 2.0 * (1.0 - t) * (1.0 - t)


def ease_in_cubic(t: float) -> float:
    """
    Ease-in cubic: f(t) = t³
    
    More dramatic than quadratic.
    Formula: t³
    
    Use for: Strong acceleration, dramatic entrances.
    """
    return t * t * t


def ease_out_cubic(t: float) -> float:
    """
    Ease-out cubic: f(t) = 1 - (1-t)³
    
    More dramatic deceleration.
    Formula: 1 - (1-t)³
    
    Use for: Strong deceleration, dramatic exits.
    """
    return 1.0 - (1.0 - t) * (1.0 - t) * (1.0 - t)


def ease_in_out_cubic(t: float) -> float:
    """
    Ease-in-out cubic.
    
    Formula: 4t³ for t < 0.5, else 1 - 4(1-t)³
    
    Very smooth, natural motion.
    Probably your most-used easing function.
    """
    if t < 0.5:
        return 4.0 * t * t * t
    return 1.0 - 4.0 * (1.0 - t) * (1.0 - t) * (1.0 - t)


def ease_in_sine(t: float) -> float:
    """
    Ease-in sine: f(t) = 1 - cos(t * π/2)
    
    Very soft start.
    Use for: Breathing, opacity, subtle motion.
    """
    return 1.0 - math.cos(t * math.pi * 0.5)


def ease_out_sine(t: float) -> float:
    """
    Ease-out sine: f(t) = sin(t * π/2)
    
    Very soft deceleration.
    Use for: Gentle settling, natural decay.
    """
    return math.sin(t * math.pi * 0.5)


def ease_in_out_sine(t: float) -> float:
    """
    Ease-in-out sine: f(t) = -(cos(t * π) - 1) / 2
    
    Extremely smooth, organic motion.
    Perfect for natural phenomena like breathing, drifting, floating.
    """
    return -(math.cos(t * math.pi) - 1.0) * 0.5


def clamp(t: float) -> float:
    """Clamp t to [0, 1] range. Useful for safety."""
    return max(0.0, min(1.0, t))


# ============================================================
# Public API
# ============================================================

__all__ = [
    "linear",
    "smoothstep",
    "smootherstep",
    "ease_in_quad",
    "ease_out_quad",
    "ease_in_out_quad",
    "ease_in_cubic",
    "ease_out_cubic",
    "ease_in_out_cubic",
    "ease_in_sine",
    "ease_out_sine",
    "ease_in_out_sine",
    "clamp",
    "EASING_FUNCTIONS",
]

# Dictionary for runtime lookup from config files
EASING_FUNCTIONS: dict[str, Callable[[float], float]] = {
    "linear": linear,
    "smoothstep": smoothstep,
    "smootherstep": smootherstep,
    "ease_in_quad": ease_in_quad,
    "ease_out_quad": ease_out_quad,
    "ease_in_out_quad": ease_in_out_quad,
    "ease_in_cubic": ease_in_cubic,
    "ease_out_cubic": ease_out_cubic,
    "ease_in_out_cubic": ease_in_out_cubic,
    "ease_in_sine": ease_in_sine,
    "ease_out_sine": ease_out_sine,
    "ease_in_out_sine": ease_in_out_sine,
}
