"""
Layer: A compositing layer with image, mask, and transform.

A Layer represents a single visual element in the scene.
Each layer has:
    - A texture (the image)
    - An optional mask (alpha channel override)
    - Position, scale, rotation (in radians)
    - Opacity
    - Pivot point for rotation/scale
    - Blend mode

Layers are the building blocks of the scene graph.
"""

from __future__ import annotations

import math
import numpy as np
from typing import Optional, Tuple, List, Any

# Use absolute imports
from geometry.vec2 import Vec2
from graphics.texture import Texture


class Layer:
    """
    A compositing layer with image, mask, and transform.

    Attributes:
        texture: The image data
        mask: Optional mask texture (grayscale)
        position: Position (Vec2)
        scale: Scale factor (Vec2, 1.0 = original)
        rotation: Rotation angle in radians
        pivot: Pivot point for rotation/scale (default: center)
        opacity: Opacity (0.0 = transparent, 1.0 = opaque)
        blend_mode: Blend mode ('alpha', 'add', 'multiply', 'screen', 'overlay')
        visible: Whether the layer is rendered
        name: Optional identifier for debugging
        _effects: List of attached effects
    """

    __slots__ = (
        'texture', 'mask', 'position', 'scale',
        'rotation', 'pivot', 'opacity', 'blend_mode',
        'visible', 'name', '_effects'
    )

    def __init__(
        self,
        texture: Texture,
        mask: Optional[Texture] = None,
        position: Optional[Vec2] = None,
        scale: Optional[Vec2] = None,
        rotation: float = 0.0,
        pivot: Optional[Vec2] = None,
        opacity: float = 1.0,
        blend_mode: str = 'alpha',
        visible: bool = True,
        name: str = ""
    ):
        """
        Initialize a layer.

        Args:
            texture: The image to display
            mask: Optional mask (grayscale, same size as texture)
            position: Position (default: Vec2.zero())
            scale: Scale factor (default: Vec2.one())
            rotation: Rotation in radians
            pivot: Pivot point for rotation/scale (default: center of texture)
            opacity: Opacity (0-1)
            blend_mode: Blend mode ('alpha', 'add', 'multiply', 'screen', 'overlay')
            visible: Whether to render this layer
            name: Optional identifier

        Raises:
            ValueError: If mask dimensions don't match texture
        """
        self.texture = texture
        self.mask = None

        if mask is not None:
            if mask.width != texture.width or mask.height != texture.height:
                raise ValueError(
                    f"Mask size ({mask.width}x{mask.height}) must match texture "
                    f"({texture.width}x{texture.height})"
                )
            self.mask = mask

        self.position = position if position is not None else Vec2.zero()
        self.scale = scale if scale is not None else Vec2.one()
        self.rotation = float(rotation)
        
        # Default pivot is center of texture
        if pivot is None:
            self.pivot = Vec2(texture.width / 2.0, texture.height / 2.0)
        else:
            self.pivot = pivot.copy()
            
        self.opacity = float(opacity)
        self.blend_mode = str(blend_mode)
        self.visible = bool(visible)
        self.name = str(name)
        self._effects = []

    # ============================================================
    # Transform Operations
    # ============================================================

    def set_position(self, position: Vec2) -> None:
        """Set position."""
        self.position = position.copy()

    def set_scale(self, scale: Vec2) -> None:
        """Set scale."""
        self.scale = scale.copy()

    def set_rotation(self, rotation: float) -> None:
        """Set rotation in radians."""
        self.rotation = float(rotation)

    def set_pivot(self, pivot: Vec2) -> None:
        """Set pivot point in texture coordinates."""
        self.pivot = pivot.copy()

    def set_opacity(self, opacity: float) -> None:
        """Set opacity (0-1)."""
        self.opacity = max(0.0, min(1.0, float(opacity)))

    def set_blend_mode(self, blend_mode: str) -> None:
        """Set blend mode."""
        self.blend_mode = str(blend_mode)

    def set_mask(self, mask: Optional[Texture]) -> None:
        """
        Set or remove mask.

        Args:
            mask: Mask texture (grayscale, same size) or None to remove
        """
        if mask is not None:
            if mask.width != self.texture.width or mask.height != self.texture.height:
                raise ValueError(
                    f"Mask size ({mask.width}x{mask.height}) must match texture "
                    f"({self.texture.width}x{self.texture.height})"
                )
        self.mask = mask

    def get_transform_matrix(self) -> np.ndarray:
        """
        Get the 2D affine transform matrix.

        Returns a 2x3 OpenCV affine matrix.
        Handles pivot, scale, rotation, and position in the correct order.

        Transformation order:
            1. Translate to pivot (so pivot is at origin)
            2. Scale
            3. Rotate
            4. Translate to position
        """
        c = math.cos(self.rotation)
        s = math.sin(self.rotation)

        sx = self.scale.x
        sy = self.scale.y
        px = self.pivot.x
        py = self.pivot.y
        
        matrix = np.array([
            [c * sx, -s * sy, self.position.x - px * c * sx + py * s * sy],
            [s * sx,  c * sy, self.position.y - px * s * sx - py * c * sy]
        ], dtype=np.float32)

        return matrix

    # ============================================================
    # Mask Helpers
    # ============================================================

    def has_mask(self) -> bool:
        """Check if layer has a mask."""
        return self.mask is not None

    def get_effective_alpha(self) -> Optional[np.ndarray]:
        """
        Get the effective alpha channel.

        Combines:
            - Texture's alpha channel (if present)
            - Mask (if present)
            - Opacity

        Returns:
            2D grayscale array (HxW) with values 0-255, or None if fully opaque
        """
        # Start with texture alpha or full white
        if self.texture.has_alpha():
            alpha = self.texture.data[:, :, 3].astype(np.float32)
        else:
            # No alpha at all - return None for full opacity
            if self.mask is None and self.opacity >= 1.0:
                return None
            alpha = np.full((self.texture.height, self.texture.width), 255.0, dtype=np.float32)

        # Apply mask
        if self.mask is not None:
            mask_data = self.mask.data
            if mask_data.ndim == 3:
                mask_data = mask_data[:, :, 0]  # Use first channel
            alpha = alpha * (mask_data.astype(np.float32) / 255.0)

        # Apply opacity
        if self.opacity < 1.0:
            alpha = alpha * self.opacity

        # If fully opaque, return None to save work
        if np.all(alpha >= 254.5):
            return None

        return np.clip(alpha, 0, 255).astype(np.uint8)

    # ============================================================
    # Size & Position Helpers
    # ============================================================

    def bounds(self) -> Tuple[float, float, float, float]:
        """
        Get the layer's bounding box in screen space.

        Returns:
            (x, y, width, height) of the bounding box
        """
        w = self.texture.width * abs(self.scale.x)
        h = self.texture.height * abs(self.scale.y)
        return (self.position.x - w/2, self.position.y - h/2, w, h)

    def center(self) -> Vec2:
        """Get the layer's center position."""
        return self.position.copy()

    # ============================================================
    # Effect Management
    # ============================================================

    def add_effect(self, effect) -> None:
        """
        Add an effect to this layer.

        Args:
            effect: The effect to add
        """
        effect.attach(self)
        self._effects.append(effect)

    def remove_effect(self, effect) -> None:
        """Remove an effect from this layer."""
        if effect in self._effects:
            effect.detach()
            self._effects.remove(effect)

    def clear_effects(self) -> None:
        """Remove all effects from this layer."""
        for effect in self._effects:
            effect.detach()
        self._effects.clear()

    def update_effects(self, dt: float) -> None:
        """Update all attached effects."""
        for effect in self._effects:
            if effect.enabled:
                effect.update(dt)

    # ============================================================
    # Copy
    # ============================================================

    def copy(self) -> 'Layer':
        """Create a copy of this layer."""
        layer = Layer(
            texture=self.texture.copy(),
            mask=self.mask.copy() if self.mask else None,
            position=self.position.copy(),
            scale=self.scale.copy(),
            rotation=self.rotation,
            pivot=self.pivot.copy(),
            opacity=self.opacity,
            blend_mode=self.blend_mode,
            visible=self.visible,
            name=self.name
        )
        # Copy effects
        for effect in self._effects:
            # This is a shallow copy - effects are not deep copied
            layer._effects.append(effect)
        return layer

    # ============================================================
    # Magic Methods
    # ============================================================

    def __repr__(self) -> str:
        name = f" '{self.name}'" if self.name else ""
        return (
            f"Layer{name}({self.texture.width}x{self.texture.height}, "
            f"pos=({self.position.x:.1f},{self.position.y:.1f}), "
            f"scale=({self.scale.x:.2f},{self.scale.y:.2f}), "
            f"rotation={self.rotation:.2f}rad, "
            f"blend={self.blend_mode}, "
            f"opacity={self.opacity:.2f})"
        )


# ============================================================
# Public API
# ============================================================

__all__ = [
    "Layer",
]
