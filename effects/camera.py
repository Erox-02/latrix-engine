"""
Camera Effect: Smooth camera motion with shake, drift, and target following.

Creates cinematic camera movement including:
    - Smooth target following (using spring physics)
    - Camera shake (impulse-based)
    - Gentle drift (floating motion)
    - Zoom control
    - Viewport control

Mathematical principle: Camera motion is driven by critically damped springs
for smooth, natural movement. Shake is modeled as a decaying sine wave
with random impulses for organic feel.

The camera transforms the entire scene by applying a view matrix that
maps world coordinates to screen coordinates.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, Optional, Tuple

from .base import Effect
from ..geometry.vec2 import Vec2
from ..geometry.spring import Vec2Spring, ScalarSpring


class CameraEffect(Effect):
    """
    Cinematic camera with smooth motion, shake, and drift.

    Attributes:
        position: Current camera position (Vec2)
        target: Desired camera position (Vec2)
        zoom: Current zoom level (1.0 = normal)
        target_zoom: Desired zoom level
        shake_intensity: Current shake intensity (pixels)
        shake_decay: How quickly shake dies down (0.9-0.99)
        drift_amplitude: Drift movement amplitude (pixels)
        drift_speed: Drift movement speed (cycles/second)
        damping: Spring damping for smooth following (higher = faster)
        position_spring: Spring for position tracking
        zoom_spring: Spring for zoom tracking
    """

    __slots__ = (
        'position', 'target', 'zoom', 'target_zoom',
        'shake_intensity', 'shake_decay', 'shake_offset',
        'drift_amplitude', 'drift_speed', 'drift_phase',
        'damping', '_position_spring', '_zoom_spring',
        '_shake_time'
    )

    def __init__(
        self,
        position: Optional[Vec2] = None,
        target: Optional[Vec2] = None,
        zoom: float = 1.0,
        damping: float = 6.0,
        shake_decay: float = 0.95,
        drift_amplitude: float = 5.0,
        drift_speed: float = 0.3,
        **kwargs
    ):
        """
        Initialize a camera effect.

        Args:
            position: Initial camera position (default: (0, 0))
            target: Initial target position (default: same as position)
            zoom: Initial zoom level (1.0 = normal)
            damping: Spring damping for smooth following (2-20)
            shake_decay: Shake decay rate (0.9-0.99, higher = longer shake)
            drift_amplitude: Drift movement amplitude in pixels
            drift_speed: Drift movement speed in cycles/second
            **kwargs: Passed to Effect
        """
        super().__init__(**kwargs)

        self.position = position if position is not None else Vec2.zero()
        self.target = target if target is not None else self.position.copy()
        self.zoom = float(zoom)
        self.target_zoom = float(zoom)
        self.damping = float(damping)
        self.shake_decay = float(shake_decay)
        self.drift_amplitude = float(drift_amplitude)
        self.drift_speed = float(drift_speed)

        # Springs for smooth motion
        self._position_spring = Vec2Spring(
            initial_value=self.position,
            damping=damping
        )
        self._position_spring.target = self.target.copy()

        self._zoom_spring = ScalarSpring(
            initial_value=self.zoom,
            damping=damping
        )
        self._zoom_spring.target = self.target_zoom

        # Shake state
        self.shake_intensity = 0.0
        self.shake_offset = Vec2.zero()
        self._shake_time = 0.0

        # Drift phase
        self.drift_phase = random.random() * 2 * math.pi

    def attach(self, layer) -> None:
        """Attach to layer."""
        super().attach(layer)
        # Camera typically modifies the layer's parent or viewport
        # For now, we store the layer reference

    def update(self, dt: float) -> None:
        """
        Update camera motion.

        Handles:
            1. Smooth position tracking via spring
            2. Zoom smoothing
            3. Shake decay
            4. Drift generation
        """
        if not self.enabled:
            return

        self.elapsed += dt

        # Update position spring
        self._position_spring.update(dt)
        self.position = self._position_spring.x

        # Update zoom spring
        self._zoom_spring.update(dt)
        self.zoom = self._zoom_spring.x

        # Update shake
        self._update_shake(dt)

        # Update drift
        self._update_drift(dt)

    def _update_shake(self, dt: float) -> None:
        """
        Update camera shake.

        Shake decays exponentially and oscillates randomly.
        """
        if self.shake_intensity <= 0.01:
            self.shake_intensity = 0.0
            self.shake_offset = Vec2.zero()
            return

        self._shake_time += dt

        # Decay intensity
        self.shake_intensity *= (1.0 - (1.0 - self.shake_decay) * dt * 60)

        # Generate shake offset as decaying sine with random phase
        if self.shake_intensity > 0.01:
            freq = 20.0 + 10.0 * math.sin(self._shake_time * 0.5)
            phase_x = self._shake_time * freq
            phase_y = self._shake_time * (freq + 1.7)

            # Randomize direction slightly
            x = math.sin(phase_x) + 0.5 * math.sin(phase_x * 2.3 + 1.2)
            y = math.cos(phase_y) + 0.5 * math.cos(phase_y * 1.7 + 0.8)

            # Normalize to intensity
            magnitude = self.shake_intensity
            self.shake_offset = Vec2(x, y).normalize().scale(magnitude)
        else:
            self.shake_offset = Vec2.zero()

    def _update_drift(self, dt: float) -> None:
        """Update gentle camera drift."""
        self.drift_phase += dt * self.drift_speed * 2 * math.pi

        # Drift is independent of target following
        # It adds subtle movement even when camera is still
        # This is applied as an offset to the spring position

    def shake(self, intensity: float) -> None:
        """
        Trigger a camera shake.

        Args:
            intensity: Shake magnitude in pixels
        """
        self.shake_intensity = max(self.shake_intensity, intensity)
        self._shake_time = 0.0

    def set_target(self, target: Vec2) -> None:
        """
        Set the camera target position.

        The camera will smoothly follow this target.

        Args:
            target: New target position
        """
        self.target = target.copy()
        self._position_spring.set_target(target)

    def set_position(self, position: Vec2) -> None:
        """
        Set camera position instantly (no smoothing).

        Args:
            position: New position
        """
        self.position = position.copy()
        self.target = position.copy()
        self._position_spring.set_value(position)
        self._position_spring.set_target(position)

    def set_zoom(self, zoom: float) -> None:
        """
        Set target zoom level.

        Args:
            zoom: New zoom level (1.0 = normal)
        """
        self.target_zoom = float(zoom)
        self._zoom_spring.set_target(zoom)

    def set_zoom_instant(self, zoom: float) -> None:
        """
        Set zoom instantly (no smoothing).

        Args:
            zoom: New zoom level (1.0 = normal)
        """
        self.zoom = float(zoom)
        self.target_zoom = float(zoom)
        self._zoom_spring.set_value(zoom)
        self._zoom_spring.set_target(zoom)

    def get_view_matrix(self) -> Tuple[float, float, float]:
        """
        Get the view transform for rendering.

        Returns a tuple of (x_offset, y_offset, zoom) that can be
        applied to layer transforms.

        Returns:
            (offset_x, offset_y, zoom_factor)
        """
        # Apply drift as subtle offset
        drift_x = self.drift_amplitude * math.sin(self.drift_phase)
        drift_y = self.drift_amplitude * 0.7 * math.cos(self.drift_phase * 0.7 + 0.5)

        # Combine position and shake
        offset_x = self.position.x + self.shake_offset.x + drift_x
        offset_y = self.position.y + self.shake_offset.y + drift_y

        return (offset_x, offset_y, self.zoom)

    def apply_to_layer(self, layer) -> None:
        """
        Apply camera transform directly to a layer.

        Modifies layer's position and scale to simulate camera movement.
        Useful when not using a viewport.

        Args:
            layer: Layer to apply camera transform to
        """
        offset_x, offset_y, zoom = self.get_view_matrix()

        # Invert position (camera moves opposite to layer)
        layer.position = Vec2(-offset_x, -offset_y)
        layer.scale = Vec2(zoom, zoom)

    def is_settled(self, tolerance: float = 1e-3) -> bool:
        """
        Check if camera has settled at its target.

        Args:
            tolerance: Position tolerance in pixels

        Returns:
            True if camera is settled
        """
        pos_settled = self._position_spring.is_settled()
        zoom_settled = self._zoom_spring.is_settled()
        shake_settled = self.shake_intensity < 0.01

        return pos_settled and zoom_settled and shake_settled

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        data = super().to_dict()
        data['params'].update({
            'position': self.position.as_tuple(),
            'target': self.target.as_tuple(),
            'zoom': self.zoom,
            'damping': self.damping,
            'shake_decay': self.shake_decay,
            'drift_amplitude': self.drift_amplitude,
            'drift_speed': self.drift_speed,
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CameraEffect:
        """Create from dictionary."""
        params = data.get('params', {})
        pos = params.get('position', (0, 0))
        tgt = params.get('target', pos)

        return cls(
            position=Vec2(pos[0], pos[1]),
            target=Vec2(tgt[0], tgt[1]),
            zoom=params.get('zoom', 1.0),
            damping=params.get('damping', 6.0),
            shake_decay=params.get('shake_decay', 0.95),
            drift_amplitude=params.get('drift_amplitude', 5.0),
            drift_speed=params.get('drift_speed', 0.3),
            duration=data.get('duration', 0.0),
            enabled=data.get('enabled', True),
        )


# ============================================================
# Public API
# ============================================================

__all__ = [
    "CameraEffect",
]
