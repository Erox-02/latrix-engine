"""
Critically damped spring for smooth, natural motion.

Tracks a target value smoothly without overshoot or oscillation.
"""

import math
from typing import Optional

from .vec2 import Vec2


class ScalarSpring:
    """
    Critically damped spring for scalar values.

    Attributes:
        x: Current position
        v: Current velocity
        target: Desired equilibrium position
        damping: Response speed (higher = faster)
        eps: Tolerance for settling detection
    """

    __slots__ = ('x', 'v', 'target', 'damping', 'eps')

    def __init__(self, initial_value: float = 0.0, damping: float = 5.0, eps: float = 1e-6):
        """
        Initialize a critically damped spring.

        Args:
            initial_value: Starting position
            damping: Response speed (typical: 2-20). Must be > 0.
            eps: Tolerance for settling. Must be > 0.
        """
        if damping <= 0:
            raise ValueError(f"damping must be > 0, got {damping}")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")

        self.x = float(initial_value)
        self.v = 0.0
        self.target = float(initial_value)
        self.damping = float(damping)
        self.eps = float(eps)

    def update(self, dt: float) -> float:
        """
        Advance the spring by dt seconds.

        Uses exact analytical solution for critical damping.
        """
        if dt <= 0:
            return self.x

        diff = self.x - self.target
        d = self.damping
        exp_factor = math.exp(-d * dt)

        vel_contrib = (self.v + d * diff) * dt

        self.x = self.target + (diff + vel_contrib) * exp_factor
        self.v = (self.v + d * diff) * exp_factor - d * (diff + vel_contrib) * exp_factor

        return self.x

    def update_to_target(self, dt: float, target: float) -> float:
        """Update with a new target."""
        self.target = float(target)
        return self.update(dt)

    def set_value(self, value: float) -> None:
        """Set position instantly. Resets velocity to 0."""
        self.x = float(value)
        self.target = float(value)
        self.v = 0.0

    def set_target(self, target: float) -> None:
        """Set a new target without changing position or velocity."""
        self.target = float(target)

    def set_velocity(self, velocity: float) -> None:
        """Set velocity directly."""
        self.v = float(velocity)

    def snap_to_target(self) -> None:
        """Snap instantly to target. Resets velocity to 0."""
        self.x = self.target
        self.v = 0.0

    def is_settled(self) -> bool:
        """Check if the spring has settled at its target."""
        return abs(self.x - self.target) < self.eps and abs(self.v) < self.eps * 10

    def error(self) -> float:
        """Get the current error (target - position)."""
        return self.target - self.x

    def impulse(self, velocity: float) -> None:
        """Apply a velocity impulse."""
        self.v += velocity

    def set_damping(self, damping: float) -> None:
        """Change the damping value."""
        if damping <= 0:
            raise ValueError(f"damping must be > 0, got {damping}")
        self.damping = float(damping)

    def reset(self, value: float = 0.0) -> None:
        """Reset to a value with zero velocity."""
        self.x = float(value)
        self.v = 0.0
        self.target = float(value)

    def copy(self) -> 'ScalarSpring':
        """Create a copy."""
        spring = ScalarSpring(
            initial_value=self.x,
            damping=self.damping,
            eps=self.eps
        )
        spring.v = self.v
        spring.target = self.target
        return spring


class Vec2Spring:
    """
    Critically damped spring for 2D vectors.

    Uses Vec2 for position, velocity, and target.
    All vector operations work naturally.

    Attributes:
        x: Current position (Vec2)
        v: Current velocity (Vec2)
        target: Desired equilibrium position (Vec2)
        damping: Response speed (higher = faster)
        eps: Tolerance for settling detection
    """

    __slots__ = ('x', 'v', 'target', 'damping', 'eps')

    def __init__(
        self,
        initial_value: Optional[Vec2] = None,
        damping: float = 5.0,
        eps: float = 1e-6
    ):
        """
        Initialize a 2D critically damped spring.

        Args:
            initial_value: Starting position (Vec2). Defaults to zero.
            damping: Response speed (typical: 2-20). Must be > 0.
            eps: Tolerance for settling. Must be > 0.
        """
        if damping <= 0:
            raise ValueError(f"damping must be > 0, got {damping}")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")

        if initial_value is None:
            initial_value = Vec2.zero()

        self.x = initial_value.copy()
        self.v = Vec2.zero()
        self.target = initial_value.copy()
        self.damping = float(damping)
        self.eps = float(eps)

    def update(self, dt: float) -> Vec2:
        """
        Advance the spring by dt seconds.

        Uses exact analytical solution for critical damping.
        """
        if dt <= 0:
            return self.x

        diff = self.x.sub(self.target)
        d = self.damping
        exp_factor = math.exp(-d * dt)

        vel_contrib = self.v.add(diff.scale(d)).scale(dt)

        # Position: target + (diff + vel_contrib) * e^(-d*dt)
        self.x = self.target.add(diff.add(vel_contrib).scale(exp_factor))

        # Velocity: (v + d*diff) * e^(-d*dt) - d * (diff + vel_contrib) * e^(-d*dt)
        self.v = (
            self.v.add(diff.scale(d)).scale(exp_factor)
            .sub(diff.add(vel_contrib).scale(d).scale(exp_factor))
        )

        return self.x

    def update_to_target(self, dt: float, target: Vec2) -> Vec2:
        """Update with a new target."""
        self.target = target.copy()
        return self.update(dt)

    def set_value(self, value: Vec2) -> None:
        """Set position instantly. Resets velocity to 0."""
        self.x = value.copy()
        self.target = value.copy()
        self.v = Vec2.zero()

    def set_target(self, target: Vec2) -> None:
        """Set a new target without changing position or velocity."""
        self.target = target.copy()

    def set_velocity(self, velocity: Vec2) -> None:
        """Set velocity directly."""
        self.v = velocity.copy()

    def snap_to_target(self) -> None:
        """Snap instantly to target. Resets velocity to 0."""
        self.x = self.target.copy()
        self.v = Vec2.zero()

    def is_settled(self) -> bool:
        """Check if the spring has settled at its target."""
        return (
            self.x.distance_to(self.target) < self.eps and
            self.v.length() < self.eps * 10
        )

    def error(self) -> Vec2:
        """Get the current error (target - position)."""
        return self.target.sub(self.x)

    def impulse(self, velocity: Vec2) -> None:
        """Apply a velocity impulse."""
        self.v = self.v.add(velocity)

    def set_damping(self, damping: float) -> None:
        """Change the damping value."""
        if damping <= 0:
            raise ValueError(f"damping must be > 0, got {damping}")
        self.damping = float(damping)

    def reset(self, value: Optional[Vec2] = None) -> None:
        """Reset to a value with zero velocity."""
        if value is None:
            value = Vec2.zero()
        self.x = value.copy()
        self.v = Vec2.zero()
        self.target = value.copy()

    def copy(self) -> 'Vec2Spring':
        """Create a copy."""
        spring = Vec2Spring(
            initial_value=self.x,
            damping=self.damping,
            eps=self.eps
        )
        spring.v = self.v.copy()
        spring.target = self.target.copy()
        return spring


# ============================================================
# Public API
# ============================================================

__all__ = [
    "ScalarSpring",
    "Vec2Spring",
]
