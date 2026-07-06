"""
Blend: Alpha blending and compositing operations.

Handles combining layers with various blend modes.
All operations work with BGRA/BGR numpy arrays.

Mathematical principle: Alpha blending computes the final color as
a weighted average of source and destination based on alpha.

    output = src * alpha + dst * (1 - alpha)

This is the foundation of all compositing in Latrix.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple


def _prepare_blend(
    src: np.ndarray,
    dst: np.ndarray,
    alpha: Optional[np.ndarray] = None,
    opacity: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare source and destination for blending.

    Extracts color channels, computes effective alpha from:
        - Source alpha channel (if present)
        - Mask (if provided)
        - Opacity multiplier

    Returns:
        Tuple of (src_color, dst_color, alpha_expanded)
        All are float32 arrays in [0, 1] range.
        alpha_expanded is 3-channel for broadcasting.

    Raises:
        ValueError: If src and dst have different sizes
    """
    if src.shape[:2] != dst.shape[:2]:
        raise ValueError(
            f"Source size {src.shape[:2]} must match destination size {dst.shape[:2]}"
        )

    # Get source color (float, normalized to [0, 1])
    if src.shape[2] == 4:
        src_color = src[:, :, :3].astype(np.float32) / 255.0
        src_alpha = src[:, :, 3].astype(np.float32) / 255.0
    else:
        src_color = src.astype(np.float32) / 255.0
        src_alpha = np.ones((src.shape[0], src.shape[1]), dtype=np.float32)

    # Destination color (float, normalized to [0, 1])
    dst_color = dst.astype(np.float32) / 255.0

    # Apply mask if provided
    if alpha is not None:
        if alpha.shape[:2] != src.shape[:2]:
            raise ValueError(
                f"Mask size {alpha.shape[:2]} must match source size {src.shape[:2]}"
            )
        if alpha.dtype == np.uint8:
            mask_alpha = alpha.astype(np.float32) / 255.0
        else:
            mask_alpha = alpha.astype(np.float32)
        src_alpha = src_alpha * mask_alpha

    # Apply opacity
    src_alpha = src_alpha * float(opacity)
    src_alpha = np.clip(src_alpha, 0.0, 1.0)

    # Expand alpha to 3 channels for broadcasting
    alpha_expanded = src_alpha[:, :, np.newaxis]

    return src_color, dst_color, alpha_expanded


def blend_alpha(
    src: np.ndarray,
    dst: np.ndarray,
    alpha: Optional[np.ndarray] = None,
    opacity: float = 1.0
) -> np.ndarray:
    """
    Alpha blending.

    Formula: output = src * alpha + dst * (1 - alpha)

    Standard compositing. Use for most layer overlays.
    """
    src_color, dst_color, a = _prepare_blend(src, dst, alpha, opacity)
    output = src_color * a + dst_color * (1.0 - a)
    return (np.clip(output, 0, 1) * 255).astype(np.uint8)


def blend_add(
    src: np.ndarray,
    dst: np.ndarray,
    alpha: Optional[np.ndarray] = None,
    opacity: float = 1.0
) -> np.ndarray:
    """
    Additive blending.

    Formula: output = src * alpha + dst

    Brightens the result. Use for particles, glows, and lights.
    """
    src_color, dst_color, a = _prepare_blend(src, dst, alpha, opacity)
    output = dst_color + src_color * a
    return (np.clip(output, 0, 1) * 255).astype(np.uint8)


def blend_multiply(
    src: np.ndarray,
    dst: np.ndarray,
    alpha: Optional[np.ndarray] = None,
    opacity: float = 1.0
) -> np.ndarray:
    """
    Multiply blending.

    Formula: output = (src * dst) * alpha + dst * (1 - alpha)

    Darkens the result. Use for shadows and darkening effects.
    """
    src_color, dst_color, a = _prepare_blend(src, dst, alpha, opacity)
    multiplied = src_color * dst_color
    output = dst_color * (1.0 - a) + multiplied * a
    return (np.clip(output, 0, 1) * 255).astype(np.uint8)


def blend_screen(
    src: np.ndarray,
    dst: np.ndarray,
    alpha: Optional[np.ndarray] = None,
    opacity: float = 1.0
) -> np.ndarray:
    """
    Screen blending.

    Formula: output = (1 - (1-src)*(1-dst)) * alpha + dst * (1 - alpha)

    Brightens the result. Inverse of multiply. Use for glows and highlights.
    """
    src_color, dst_color, a = _prepare_blend(src, dst, alpha, opacity)
    screened = 1.0 - (1.0 - src_color) * (1.0 - dst_color)
    output = dst_color * (1.0 - a) + screened * a
    return (np.clip(output, 0, 1) * 255).astype(np.uint8)


def blend_overlay(
    src: np.ndarray,
    dst: np.ndarray,
    alpha: Optional[np.ndarray] = None,
    opacity: float = 1.0
) -> np.ndarray:
    """
    Overlay blending.

    Combines multiply and screen based on destination brightness.
    Dark areas get darker, light areas get lighter.

    Use for dramatic lighting and contrast enhancement.
    """
    src_color, dst_color, a = _prepare_blend(src, dst, alpha, opacity)

    # Overlay: if dst < 0.5, multiply; else screen
    mask = dst_color < 0.5
    overlay = np.where(
        mask,
        2.0 * src_color * dst_color,
        1.0 - 2.0 * (1.0 - src_color) * (1.0 - dst_color)
    )

    output = dst_color * (1.0 - a) + overlay * a
    return (np.clip(output, 0, 1) * 255).astype(np.uint8)


# ============================================================
# Blend Mode Registry
# ============================================================

BLEND_MODES = {
    'alpha': blend_alpha,
    'add': blend_add,
    'multiply': blend_multiply,
    'screen': blend_screen,
    'overlay': blend_overlay,
}


def get_blend_mode(name: str):
    """Get a blend function by name."""
    return BLEND_MODES.get(name, blend_alpha)


# ============================================================
# Public API
# ============================================================

__all__ = [
    "blend_alpha",
    "blend_add",
    "blend_multiply",
    "blend_screen",
    "blend_overlay",
    "BLEND_MODES",
    "get_blend_mode",
]
