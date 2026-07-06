"""
Base Effect: Abstract base class for all effects in Latrix.

An Effect is a self-contained animation module that operates on a layer.
Effects are like Unity Components - they attach to layers and modify them over time.

Each effect:
    - Attaches to a layer
    - Applies mathematical animation over time
    - Can be configured via parameters
    - Can be composed with other effects

Mathematical principle: Effects transform layer properties (position, scale, rotation, opacity)
using mathematical functions over time. This creates procedural animation without
hardcoded keyframes.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Tuple

from ..graphics.layer import Layer
from ..geometry.vec2 import Vec2
from ..geometry.easing import EASING_FUNCTIONS


class Effect(ABC):
    """
    Abstract base class for all effects.

    An effect attaches to a layer and modifies it over time.

    Attributes:
        layer: The layer this effect is attached to (None if not attached)
        enabled: Whether the effect is active
        duration: Total duration in seconds (0 = infinite)
        elapsed: Time elapsed since effect started
        params: Additional parameters for the effect
    """

    __slots__ = ('layer', 'enabled', 'duration', 'elapsed', 'params')

    def __init__(
        self,
        duration: float = 0.0,
        enabled: bool = True,
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize an effect.

        Args:
            duration: Total duration in seconds (0 = infinite)
            enabled: Whether the effect starts enabled
            params: Additional parameters
        """
        self.layer = None
        self.enabled = enabled
        self.duration = float(duration)
        self.elapsed = 0.0
        self.params = params if params is not None else {}

    def attach(self, layer: Layer) -> None:
        """
        Attach the effect to a layer.

        Called when the effect is added to a layer.

        Args:
            layer: The layer to attach to
        """
        self.layer = layer

    def detach(self) -> None:
        """Detach the effect from its layer."""
        self.layer = None
        self.reset()

    @abstractmethod
    def update(self, dt: float) -> None:
        """
        Update the effect by dt seconds.

        This is where the animation logic lives.
        Subclasses must implement this.

        Args:
            dt: Time step in seconds
        """
        pass

    def reset(self) -> None:
        """Reset the effect to its initial state."""
        self.elapsed = 0.0

    def start(self) -> None:
        """Enable and reset the effect."""
        self.enabled = True
        self.reset()

    def stop(self) -> None:
        """Disable the effect."""
        self.enabled = False

    def is_finished(self) -> bool:
        """
        Check if the effect has completed.

        Returns:
            True if duration is finite and elapsed >= duration
        """
        if self.duration <= 0:
            return False
        return self.elapsed >= self.duration

    def progress(self) -> float:
        """
        Get the progress through the effect.

        Returns:
            Float in [0, 1] representing completion, or 0 if infinite
        """
        if self.duration <= 0:
            return 0.0
        return min(1.0, self.elapsed / self.duration)

    def apply_easing(self, t: float, easing: str = 'smoothstep') -> float:
        """
        Apply an easing function to a progress value.

        Args:
            t: Input value (typically progress)
            easing: Name of easing function

        Returns:
            Eased value
        """
        easing_func = EASING_FUNCTIONS.get(easing, EASING_FUNCTIONS['smoothstep'])
        return easing_func(t)

    def lerp_float(self, start: float, end: float, t: float, easing: str = 'linear') -> float:
        """
        Linearly interpolate a float value with easing.

        Args:
            start: Start value
            end: End value
            t: Progress [0, 1]
            easing: Easing function name

        Returns:
            Interpolated value
        """
        eased = self.apply_easing(t, easing)
        return start + (end - start) * eased

    def lerp_vec2(self, start: Vec2, end: Vec2, t: float, easing: str = 'linear') -> Vec2:
        """
        Linearly interpolate a Vec2 with easing.

        Args:
            start: Start vector
            end: End vector
            t: Progress [0, 1]
            easing: Easing function name

        Returns:
            Interpolated Vec2
        """
        eased = self.apply_easing(t, easing)
        return start.lerp(end, eased)

    def lerp_angle(self, start: float, end: float, t: float, easing: str = 'linear') -> float:
        """
        Interpolate an angle with easing.

        Handles angle wrapping to take the shortest path.

        Args:
            start: Start angle in radians
            end: End angle in radians
            t: Progress [0, 1]
            easing: Easing function name

        Returns:
            Interpolated angle in radians
        """
        # Calculate shortest angle difference
        diff = end - start
        diff = (diff + math.pi) % (2 * math.pi) - math.pi

        eased = self.apply_easing(t, easing)
        return start + diff * eased

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize effect to dictionary.

        Useful for saving configurations.

        Returns:
            Dictionary representation
        """
        return {
            'type': self.__class__.__name__,
            'duration': self.duration,
            'params': self.params,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Effect:
        """
        Create effect from dictionary.

        Subclasses should override this.

        Args:
            data: Dictionary with effect data

        Returns:
            Effect instance
        """
        raise NotImplementedError("Subclasses must implement from_dict")


class CompositeEffect(Effect):
    """
    Effect that runs multiple effects in parallel.

    Useful for combining animations.
    Composites can be nested.

    Attributes:
        children: List of child effects
    """

    __slots__ = ('children',)

    def __init__(
        self,
        children: Optional[List[Effect]] = None,
        **kwargs
    ):
        """
        Initialize a composite effect.

        Args:
            children: List of child effects
            **kwargs: Passed to Effect
        """
        super().__init__(**kwargs)
        self.children = children if children is not None else []

        # Attach all children to the same layer
        if self.layer is not None:
            for child in self.children:
                child.attach(self.layer)

    def attach(self, layer: Layer) -> None:
        """Attach to a layer and all children."""
        super().attach(layer)
        for child in self.children:
            child.attach(layer)

    def detach(self) -> None:
        """Detach from layer and all children."""
        for child in self.children:
            child.detach()
        super().detach()

    def add_effect(self, effect: Effect) -> None:
        """Add a child effect."""
        if self.layer is not None:
            effect.attach(self.layer)
        self.children.append(effect)

    def remove_effect(self, effect: Effect) -> None:
        """Remove a child effect."""
        if effect in self.children:
            effect.detach()
            self.children.remove(effect)

    def clear_effects(self) -> None:
        """Remove all child effects."""
        for effect in self.children:
            effect.detach()
        self.children.clear()

    def update(self, dt: float) -> None:
        """Update all child effects."""
        if not self.enabled:
            return

        self.elapsed += dt

        for effect in self.children:
            if effect.enabled:
                effect.update(dt)


# ============================================================
# Convenience Functions
# ============================================================

def ease_float(
    start: float,
    end: float,
    t: float,
    easing: str = 'smoothstep'
) -> float:
    """
    Convenience function for easing a float.

    Args:
        start: Start value
        end: End value
        t: Progress [0, 1]
        easing: Easing function name

    Returns:
        Eased value
    """
    func = EASING_FUNCTIONS.get(easing, EASING_FUNCTIONS['smoothstep'])
    return start + (end - start) * func(t)


def ease_vec2(
    start: Vec2,
    end: Vec2,
    t: float,
    easing: str = 'smoothstep'
) -> Vec2:
    """
    Convenience function for easing a Vec2.

    Args:
        start: Start vector
        end: End vector
        t: Progress [0, 1]
        easing: Easing function name

    Returns:
        Eased Vec2
    """
    func = EASING_FUNCTIONS.get(easing, EASING_FUNCTIONS['smoothstep'])
    return start.lerp(end, func(t))


# ============================================================
# Public API
# ============================================================

__all__ = [
    "Effect",
    "CompositeEffect",
    "ease_float",
    "ease_vec2",
]
