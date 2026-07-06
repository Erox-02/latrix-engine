"""
Renderer: Renders layers to the screen.

The renderer takes layers and composites them into a final image.
It doesn't know about blend modes or alpha generation - layers handle that.

Renderer is dumb. That's intentional.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import List, Optional, Tuple

from .texture import Texture
from .layer import Layer
from .blend import BLEND_MODES


class Renderer:
    """
    Renders layers to a canvas.

    Attributes:
        width: Canvas width in pixels
        height: Canvas height in pixels
        background: Background color (BGR tuple)
        canvas: Current render buffer
    """

    __slots__ = ('width', 'height', 'background', 'canvas')

    def __init__(
        self,
        width: int,
        height: int,
        background: Tuple[int, int, int] = (0, 0, 0)
    ):
        """
        Initialize a renderer.

        Args:
            width: Canvas width in pixels
            height: Canvas height in pixels
            background: Background color (B, G, R)
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"Width and height must be positive, got {width}x{height}")

        self.width = width
        self.height = height
        self.background = tuple(background)
        self.canvas = None

        self.clear()

    # ============================================================
    # Canvas Management
    # ============================================================

    def clear(self) -> None:
        """Clear the canvas to the background color."""
        self.canvas = np.full(
            (self.height, self.width, 3),
            self.background,
            dtype=np.uint8
        )

    def resize(self, width: int, height: int) -> None:
        """
        Resize the renderer canvas.

        Args:
            width: New width
            height: New height
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"Width and height must be positive, got {width}x{height}")

        self.width = width
        self.height = height
        self.clear()

    def get_canvas(self) -> np.ndarray:
        """Get the current canvas (BGR)."""
        return self.canvas.copy()

    # ============================================================
    # Layer Rendering
    # ============================================================

    def render_layer(self, layer: Layer) -> None:
        """
        Render a single layer onto the canvas.

        Layer provides:
            - Texture data
            - Transform matrix
            - Effective alpha
            - Blend mode

        Renderer just applies them.

        Args:
            layer: The layer to render
        """
        if not layer.visible:
            return

        if layer.texture is None:
            return

        tex = layer.texture
        tex_data = tex.data

        # Apply transform
        matrix = layer.get_transform_matrix()
        warped = cv2.warpAffine(
            tex_data,
            matrix,
            (self.width, self.height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0) if tex.has_alpha() else (0, 0, 0)
        )

        # Get effective alpha from layer
        alpha = layer.get_effective_alpha()

        # Warp the alpha if it exists
        if alpha is not None:
            alpha_warped = cv2.warpAffine(
                alpha,
                matrix,
                (self.width, self.height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
        else:
            alpha_warped = None

        # Get blend function from layer
        blend_func = BLEND_MODES.get(layer.blend_mode, blend_alpha)

        # Blend onto canvas
        self.canvas = blend_func(
            src=warped,
            dst=self.canvas,
            alpha=alpha_warped,
            opacity=1.0  # Already applied in get_effective_alpha
        )

    def render_layers(self, layers: List[Layer]) -> None:
        """
        Render multiple layers in order.

        Layers are rendered back-to-front (order of the list).

        Args:
            layers: List of layers to render
        """
        self.clear()
        for layer in layers:
            self.render_layer(layer)

    def render_scene(self, scene) -> None:
        """
        Render a scene.

        Scene provides its layers and the renderer renders them.

        Args:
            scene: Scene object with get_layers() method
        """
        self.render_layers(scene.get_layers())

    # ============================================================
    # Viewport
    # ============================================================

    def render_layer_with_viewport(
        self,
        layer: Layer,
        viewport: Tuple[int, int, int, int]
    ) -> None:
        """
        Render a layer to a specific viewport region.

        Args:
            layer: The layer to render
            viewport: (x, y, width, height) in canvas coordinates
        """
        # TODO: Implement viewport clipping
        # For now, just render normally
        self.render_layer(layer)

    # ============================================================
    # Output
    # ============================================================

    def to_texture(self) -> Texture:
        """
        Convert the current canvas to a Texture.

        Returns:
            Texture of the current canvas
        """
        if self.canvas is None:
            raise RuntimeError("Canvas is empty. Render something first.")

        return Texture(self.canvas.copy())


# Need to import blend_alpha for default
from .blend import blend_alpha


# ============================================================
# Public API
# ============================================================

__all__ = [
    "Renderer",
]
