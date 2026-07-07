"""
Vec2: A 2D vector implementation with comprehensive mathematical operations.

The fundamental building block for all spatial calculations in Latrix.
Every position, direction, velocity, and acceleration is a Vec2 instance.
"""

from __future__ import annotations
import math
from typing import Tuple, Optional, Union

Number = Union[int, float]


class Vec2:
    """
    A 2D vector with comprehensive mathematical operations.

    Attributes:
        x: X-coordinate (float)
        y: Y-coordinate (float)
    """

    __slots__ = ('x', 'y')

    def __init__(self, x: Number, y: Number):
        """Initialize a 2D vector from Cartesian coordinates."""
        self.x = float(x)
        self.y = float(y)

    # ============================================================
    # Factory Methods
    # ============================================================

    @classmethod
    def zero(cls) -> Vec2:
        """Return the zero vector (0, 0)."""
        return cls(0.0, 0.0)

    @classmethod
    def one(cls) -> Vec2:
        """Return the vector (1, 1)."""
        return cls(1.0, 1.0)

    @classmethod
    def unit_x(cls) -> Vec2:
        """Return the unit vector along the X-axis (1, 0)."""
        return cls(1.0, 0.0)

    @classmethod
    def unit_y(cls) -> Vec2:
        """Return the unit vector along the Y-axis (0, 1)."""
        return cls(0.0, 1.0)

    @classmethod
    def from_polar(cls, angle: Number, magnitude: Number) -> Vec2:
        """
        Create a vector from polar coordinates (angle in radians, magnitude).

        A vector in polar form: (r*cos(θ), r*sin(θ)).
        Useful for specifying direction and speed independently.
        """
        return cls(
            magnitude * math.cos(angle),
            magnitude * math.sin(angle)
        )

    @classmethod
    def from_angle(cls, angle: Number) -> Vec2:
        """Create a unit vector at the given angle (radians)."""
        return cls.from_polar(angle, 1.0)

    # ============================================================
    # Core Operations
    # ============================================================

    def add(self, other: Vec2) -> Vec2:
        """Vector addition: (x1+x2, y1+y2)."""
        return Vec2(self.x + other.x, self.y + other.y)

    def sub(self, other: Vec2) -> Vec2:
        """Vector subtraction: (x1-x2, y1-y2)."""
        return Vec2(self.x - other.x, self.y - other.y)

    def scale(self, scalar: Number) -> Vec2:
        """Scalar multiplication: (x*s, y*s)."""
        return Vec2(self.x * scalar, self.y * scalar)

    def div(self, scalar: Number) -> Vec2:
        """Scalar division: (x/s, y/s)."""
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide vector by zero")
        return Vec2(self.x / scalar, self.y / scalar)

    def neg(self) -> Vec2:
        """Negate the vector: (-x, -y)."""
        return Vec2(-self.x, -self.y)

    def inverse(self) -> Vec2:
        """
        Component-wise inverse: (1/x, 1/y).

        Raises ZeroDivisionError if either component is zero.
        """
        if self.x == 0.0 or self.y == 0.0:
            raise ZeroDivisionError(
                f"Cannot compute inverse of vector with zero component: {self}"
            )
        return Vec2(1.0 / self.x, 1.0 / self.y)

    def safe_inverse(self) -> Vec2:
        """
        Component-wise inverse, returning zero for zero components.

        Useful for handling edge cases without exceptions.
        """
        return Vec2(
            1.0 / self.x if self.x != 0.0 else 0.0,
            1.0 / self.y if self.y != 0.0 else 0.0
        )

    # ============================================================
    # Product Operations
    # ============================================================

    def dot(self, other: Vec2) -> float:
        """
        Dot product: x1*x2 + y1*y2.

        Useful for projections, angles, and directional similarity.
        """
        return self.x * other.x + self.y * other.y

    def cross(self, other: Vec2) -> float:
        """
        Cross product (scalar): x1*y2 - y1*x2.

        Returns the signed area of the parallelogram formed by two vectors.
        Positive = counter-clockwise, negative = clockwise.
        Useful for orientation and collision detection.
        """
        return self.x * other.y - self.y * other.x

    def project(self, onto: Vec2) -> Vec2:
        """
        Project this vector onto another vector.

        Formula: (A·B / |B|²) * B
        Raises ZeroDivisionError if the target vector has zero length.
        """
        denom = onto.length_squared()
        if denom == 0:
            raise ZeroDivisionError(f"Cannot project onto zero vector: {onto}")
        scalar = self.dot(onto) / denom
        return onto.scale(scalar)

    def project_or_zero(self, onto: Vec2) -> Vec2:
        """Project onto another vector, returning zero if target is zero."""
        denom = onto.length_squared()
        if denom == 0:
            return Vec2.zero()
        scalar = self.dot(onto) / denom
        return onto.scale(scalar)

    def reflect(self, normal: Vec2) -> Vec2:
        """
        Reflect this vector across a surface with the given normal.

        Formula: A - 2*(A·N)*N

        Note: Normal must be normalized for correct reflection.
        For safety, pass through safe_normalize() first.
        """
        # Use safe_normalize to handle zero normals gracefully
        n = normal.safe_normalize()
        return self.sub(n.scale(2.0 * self.dot(n)))

    # ============================================================
    # Magnitude and Normalization
    # ============================================================

    def length(self) -> float:
        """Euclidean norm: sqrt(x² + y²). The straight-line distance from origin."""
        return math.hypot(self.x, self.y)

    def length_squared(self) -> float:
        """Squared length: x² + y². More efficient for comparisons."""
        return self.x * self.x + self.y * self.y

    def normalize(self) -> Vec2:
        """Return the unit vector in the same direction."""
        length = self.length()
        if length == 0:
            raise ZeroDivisionError("Cannot normalize zero vector")
        inv_length = 1.0 / length
        return Vec2(self.x * inv_length, self.y * inv_length)

    def safe_normalize(self) -> Vec2:
        """Return normalized vector, or zero vector if length is zero."""
        length = self.length()
        if length == 0:
            return Vec2.zero()
        inv_length = 1.0 / length
        return Vec2(self.x * inv_length, self.y * inv_length)

    def is_zero(self, tolerance: float = 1e-10) -> bool:
        """Check if the vector is (approximately) zero."""
        return self.length_squared() < tolerance * tolerance

    def is_normalized(self, tolerance: float = 1e-10) -> bool:
        """Check if the vector has unit length (approximately)."""
        return abs(self.length_squared() - 1.0) < tolerance

    def clamp_length(self, max_length: Number) -> Vec2:
        """Clamp the vector's magnitude to max_length."""
        length = self.length()
        if length > max_length:
            inv_length = max_length / length
            return Vec2(self.x * inv_length, self.y * inv_length)
        return self.copy()

    def limit(self, max_length: Number) -> Vec2:
        """Alias for clamp_length(). Used frequently in particle systems."""
        return self.clamp_length(max_length)

    def set_length(self, new_length: Number) -> Vec2:
        """Set the vector's magnitude to new_length while preserving direction."""
        length = self.length()
        if length == 0:
            return Vec2.zero()
        scale_factor = new_length / length
        return Vec2(self.x * scale_factor, self.y * scale_factor)

    # ============================================================
    # Interpolation
    # ============================================================

    def lerp(self, target: Vec2, t: Number) -> Vec2:
        """
        Linear interpolation: (1-t)*self + t*target.

        The foundation of all smooth transitions in Latrix.
        """
        t = float(t)
        return Vec2(
            self.x + (target.x - self.x) * t,
            self.y + (target.y - self.y) * t
        )

    def slerp(self, target: Vec2, t: Number) -> Vec2:
        """
        Spherical interpolation along the shortest arc on the unit circle.

        Preserves directions while interpolating magnitudes independently.
        Useful for smooth angular transitions.
        """
        t = float(t)
        mag_self = self.length()
        mag_target = target.length()

        if mag_self < 1e-10 or mag_target < 1e-10:
            return self.lerp(target, t)

        dir_self = self.div(mag_self)
        dir_target = target.div(mag_target)

        dot = max(-1.0, min(1.0, dir_self.dot(dir_target)))
        angle = math.acos(dot)

        if angle < 1e-10:
            return self.lerp(target, t)

        sin_angle = math.sin(angle)
        a = math.sin((1.0 - t) * angle) / sin_angle
        b = math.sin(t * angle) / sin_angle

        dir_result = dir_self.scale(a).add(dir_target.scale(b))
        magnitude = mag_self + (mag_target - mag_self) * t

        return dir_result.scale(magnitude)

    def move_towards(self, target: Vec2, distance: Number) -> Vec2:
        """
        Move towards a target by a fixed distance.

        Unlike lerp, this moves by a constant step distance.
        Useful for smooth, consistent movement in game loops.
        """
        diff = target.sub(self)
        diff_length = diff.length()
        if diff_length <= distance:
            return target.copy()
        return self.add(diff.safe_normalize().scale(distance))

    # ============================================================
    # Transformations
    # ============================================================

    def rotate(self, angle: Number) -> Vec2:
        """
        Rotate by angle (radians) around the origin.

        Uses the rotation matrix:
        [cos(θ)  -sin(θ)] [x]
        [sin(θ)   cos(θ)] [y]
        """
        c = math.cos(angle)
        s = math.sin(angle)
        return Vec2(
            self.x * c - self.y * s,
            self.x * s + self.y * c
        )

    def rotate_around(self, center: Vec2, angle: Number) -> Vec2:
        """Rotate around a specified center point."""
        relative = self.sub(center)
        rotated = relative.rotate(angle)
        return rotated.add(center)

    def rotate90(self) -> Vec2:
        """Rotate 90 degrees counter-clockwise: (-y, x)."""
        return Vec2(-self.y, self.x)

    def rotate180(self) -> Vec2:
        """Rotate 180 degrees: (-x, -y)."""
        return Vec2(-self.x, -self.y)

    def rotate270(self) -> Vec2:
        """Rotate 270 degrees counter-clockwise (90° clockwise): (y, -x)."""
        return Vec2(self.y, -self.x)

    def perpendicular(self) -> Vec2:
        """Alias for rotate90(). Returns a perpendicular vector (-y, x)."""
        return Vec2(-self.y, self.x)

    def perpendicular_right(self) -> Vec2:
        """Return the right-hand perpendicular: (y, -x)."""
        return Vec2(self.y, -self.x)

    def snap(self, grid_size: Number) -> Vec2:
        """Snap components to the nearest grid multiple."""
        g = float(grid_size)
        return Vec2(
            round(self.x / g) * g,
            round(self.y / g) * g
        )

    # ============================================================
    # Distance Functions
    # ============================================================

    def distance_to(self, other: Vec2) -> float:
        """Euclidean distance between two points."""
        return self.sub(other).length()

    def distance_squared_to(self, other: Vec2) -> float:
        """Squared distance between two points (faster for comparisons)."""
        return self.sub(other).length_squared()

    def manhattan_distance_to(self, other: Vec2) -> float:
        """Manhattan distance (L1 norm): |x1-x2| + |y1-y2|."""
        return abs(self.x - other.x) + abs(self.y - other.y)

    def midpoint(self, other: Vec2) -> Vec2:
        """Return the midpoint between two points."""
        return Vec2((self.x + other.x) * 0.5, (self.y + other.y) * 0.5)

    # ============================================================
    # Component Operations
    # ============================================================

    def component_min(self, other: Vec2) -> Vec2:
        """Component-wise minimum values."""
        return Vec2(min(self.x, other.x), min(self.y, other.y))

    def component_max(self, other: Vec2) -> Vec2:
        """Component-wise maximum values."""
        return Vec2(max(self.x, other.x), max(self.y, other.y))

    def clamp(self, min_val: Vec2, max_val: Vec2) -> Vec2:
        """Clamp both components between min and max vectors."""
        return Vec2(
            min(max(self.x, min_val.x), max_val.x),
            min(max(self.y, min_val.y), max_val.y)
        )

    def clamp_scalar(self, min_val: Number, max_val: Number) -> Vec2:
        """Clamp both components between the same min and max values."""
        return Vec2(
            min(max(self.x, float(min_val)), float(max_val)),
            min(max(self.y, float(min_val)), float(max_val))
        )

    def abs(self) -> Vec2:
        """Absolute value of each component."""
        return Vec2(abs(self.x), abs(self.y))

    def round(self) -> Vec2:
        """Round each component to the nearest integer."""
        return Vec2(round(self.x), round(self.y))

    def floor(self) -> Vec2:
        """Floor each component."""
        return Vec2(math.floor(self.x), math.floor(self.y))

    def ceil(self) -> Vec2:
        """Ceiling of each component."""
        return Vec2(math.ceil(self.x), math.ceil(self.y))

    def sign(self) -> Vec2:
        """Component-wise sign: -1, 0, or 1."""
        return Vec2(
            1.0 if self.x > 0 else -1.0 if self.x < 0 else 0.0,
            1.0 if self.y > 0 else -1.0 if self.y < 0 else 0.0
        )

    def component_sum(self) -> float:
        """Sum of components: x + y."""
        return self.x + self.y

    def component_product(self) -> float:
        """Product of components: x * y."""
        return self.x * self.y

    def max_component(self) -> float:
        """Maximum of the two components."""
        return max(self.x, self.y)

    def min_component(self) -> float:
        """Minimum of the two components."""
        return min(self.x, self.y)

    # ============================================================
    # Angle Operations
    # ============================================================

    def angle(self) -> float:
        """Angle in radians from the positive X-axis (atan2)."""
        return math.atan2(self.y, self.x)

    def angle_to(self, other: Vec2) -> float:
        """Signed angle from this vector to another in radians [-π, π]."""
        if self.length_squared() == 0 or other.length_squared() == 0:
            return 0.0
        return math.atan2(self.cross(other), self.dot(other))

    # ============================================================
    # Copy and Utilities
    # ============================================================

    def copy(self) -> Vec2:
        """Create a copy of this vector."""
        return Vec2(self.x, self.y)

    def as_tuple(self) -> Tuple[float, float]:
        """Return as (x, y) tuple. Useful for OpenCV/Numpy interfaces."""
        return (self.x, self.y)

    def as_int_tuple(self) -> Tuple[int, int]:
        """Return as integer tuple (int(x), int(y)). Used for pixel coordinates."""
        return (int(round(self.x)), int(round(self.y)))

    # ============================================================
    # Magic Methods
    # ============================================================

    def __add__(self, other: object) -> Vec2:
        if not isinstance(other, Vec2):
            return NotImplemented
        return self.add(other)

    def __sub__(self, other: object) -> Vec2:
        if not isinstance(other, Vec2):
            return NotImplemented
        return self.sub(other)

    def __mul__(self, scalar: Number) -> Vec2:
        return self.scale(scalar)

    def __rmul__(self, scalar: Number) -> Vec2:
        return self.scale(scalar)

    def __truediv__(self, scalar: Number) -> Vec2:
        return self.div(scalar)

    def __neg__(self) -> Vec2:
        return self.neg()

    def __abs__(self) -> float:
        """Return the length (magnitude) of the vector."""
        return self.length()

    def __bool__(self) -> bool:
        """Zero vector is False, all others are True."""
        return not self.is_zero()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vec2):
            return False
        return (math.isclose(self.x, other.x, rel_tol=1e-9, abs_tol=1e-12) and
                math.isclose(self.y, other.y, rel_tol=1e-9, abs_tol=1e-12))

    # No __hash__ implementation - approximate equality breaks hash invariants.
    # Vec2 instances should not be used as dictionary keys or set elements.

    def __iter__(self):
        """Allow unpacking: x, y = vec"""
        yield self.x
        yield self.y

    def __getitem__(self, index: int) -> float:
        """Allow indexing: vec[0] = x, vec[1] = y."""
        if index == 0:
            return self.x
        if index == 1:
            return self.y
        raise IndexError("Vec2 index out of range")

    def __repr__(self) -> str:
        return f"Vec2({self.x:.6f}, {self.y:.6f})"

    def __str__(self) -> str:
        return f"({self.x:.3f}, {self.y:.3f})"
