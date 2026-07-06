"""
Breathing Effect: Subtle idle animation for characters.

Creates a gentle, organic breathing motion by oscillating scale and position.
Perfect for characters, creatures, and anything that should feel alive.

Mathematical principle: Breathing is modeled as an asymmetric wave with
slower inhale and faster exhale, with subtle delays between different body parts.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from .base import Effect
from ..geometry.vec2 import Vec2
from ..geometry.easing import ease_in_cubic, ease_out_cubic


class BreathingEffect(Effect):
    """
    Subtle breathing animation for characters.

    Attributes:
        amplitude_scale: Maximum scale change (0.01 = 1%)
        amplitude_x: Maximum horizontal movement in pixels
        amplitude_y: Maximum vertical movement in pixels
        speed: Breathing speed in cycles per second (0.5 = slow, 1.0 = normal)
        asymmetry: Inhale/exhale ratio (0.4 = 40% inhale, 60% exhale)
        pause_duration: Pause at peak/trough (0.0-0.2)
        lag: Delay between scale and position (seconds)
        initial_scale: Saved initial scale when attached
        initial_position: Saved initial position when attached
    """

    __slots__ = (
        'amplitude_scale', 'amplitude_x', 'amplitude_y',
        'speed', 'asymmetry', 'pause_duration', 'lag',
        '_initial_scale', '_initial_position'
    )

    def __init__(
        self,
        amplitude_scale: float = 0.01,
        amplitude_x: float = 2.0,
        amplitude_y: float = 1.0,
        speed: float = 0.8,
        asymmetry: float = 0.4,
        pause_duration: float = 0.08,
        lag: float = 0.02,
        **kwargs
    ):
        """
        Initialize a breathing effect.

        Args:
            amplitude_scale: Maximum scale change (0.01 = 1%)
            amplitude_x: Maximum horizontal movement in pixels
            amplitude_y: Maximum vertical movement in pixels
            speed: Breathing speed in cycles per second (0.5-1.5 typical)
            asymmetry: Inhale/exhale ratio (0.3-0.5 typical)
            pause_duration: Pause at peak and trough (0.0-0.2)
            lag: Delay between scale and position (seconds)
            **kwargs: Passed to Effect
        """
        super().__init__(**kwargs)
        self.amplitude_scale = float(amplitude_scale)
        self.amplitude_x = float(amplitude_x)
        self.amplitude_y = float(amplitude_y)
        self.speed = float(speed)
        self.asymmetry = float(asymmetry)
        self.pause_duration = float(pause_duration)
        self.lag = float(lag)

        self._initial_scale = None
        self._initial_position = None

    def attach(self, layer) -> None:
        """Save initial state when attached."""
        super().attach(layer)
        if layer is not None:
            self._initial_scale = layer.scale.copy()
            self._initial_position = layer.position.copy()

    def detach(self) -> None:
        """Clear saved state when detached."""
        self._initial_scale = None
        self._initial_position = None
        super().detach()

    def _breath_value(self, phase: float) -> float:
        """
        Calculate breath value at a given phase [0, 1].

        Returns value in [0, 1] where:
            - 0 = fully exhaled
            - 1 = fully inhaled

        Uses asymmetric breathing with pause at peaks.
        """
        # Apply pause at peak and trough
        pause = self.pause_duration

        if phase < self.asymmetry:
            # Inhale: 0 -> 1
            t = phase / self.asymmetry
            # Apply pause at start of inhale (trough)
            if t < pause:
                return 0.0
            t = (t - pause) / (1.0 - pause)
            return ease_in_cubic(t)
        else:
            # Exhale: 1 -> 0
            t = (phase - self.asymmetry) / (1.0 - self.asymmetry)
            # Apply pause at start of exhale (peak)
            if t < pause:
                return 1.0
            t = (t - pause) / (1.0 - pause)
            return 1.0 - ease_out_cubic(t)

    def update(self, dt: float) -> None:
        """
        Update breathing animation.

        Scale and position are phase-shifted for natural lag.
        """
        if not self.enabled:
            return

        if self.layer is None:
            return

        if self._initial_scale is None or self._initial_position is None:
            return

        self.elapsed += dt

        # Breathing phase [0, 1]
        phase = (self.elapsed * self.speed) % 1.0

        # Scale phase (no lag)
        scale_value = self._breath_value(phase)

        # Position phase (with lag)
        lag_phase = (self.elapsed * self.speed - self.lag * self.speed) % 1.0
        pos_value = self._breath_value(lag_phase)

        # Apply scale
        scale_factor = 1.0 + self.amplitude_scale * scale_value
        self.layer.scale = self._initial_scale.scale(scale_factor)

        # Apply position sway (phase-shifted from breathing)
        # Sway lags slightly behind breathing
        sway_x = self.amplitude_x * math.sin(pos_value * 2 * math.pi + 0.3)
        sway_y = self.amplitude_y * math.cos(pos_value * 2 * math.pi + 0.5)

        self.layer.position = Vec2(
            self._initial_position.x + sway_x,
            self._initial_position.y + sway_y
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        data = super().to_dict()
        data['params'].update({
            'amplitude_scale': self.amplitude_scale,
            'amplitude_x': self.amplitude_x,
            'amplitude_y': self.amplitude_y,
            'speed': self.speed,
            'asymmetry': self.asymmetry,
            'pause_duration': self.pause_duration,
            'lag': self.lag,
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BreathingEffect:
        """Create from dictionary."""
        params = data.get('params', {})
        return cls(
            amplitude_scale=params.get('amplitude_scale', 0.01),
            amplitude_x=params.get('amplitude_x', 2.0),
            amplitude_y=params.get('amplitude_y', 1.0),
            speed=params.get('speed', 0.8),
            asymmetry=params.get('asymmetry', 0.4),
            pause_duration=params.get('pause_duration', 0.08),
            lag=params.get('lag', 0.02),
            duration=data.get('duration', 0.0),
            enabled=data.get('enabled', True),
        )


# ============================================================
# Public API
# ============================================================

__all__ = [
    "BreathingEffect",
]
