"""
Warp: Mesh-based image deformation using OpenCV remap.

Mathematical principle: Image warping maps each pixel from the source image
to a new position in the output image using a displacement map or grid.

The warp is defined by a mesh of control points. Each control point has a
displacement vector. Points between control points are interpolated using
bilinear interpolation.

This creates smooth, organic deformations useful for:
    - Cloth simulation
    - Hair movement
    - Liquid/fluid effects
    - Character stretching
    - Face morphing
"""

from __future__ import annotations

import numpy as np
import cv2
from typing import Optional, Tuple, List

from geometry.vec2 import Vec2
from graphics.texture import Texture


class WarpMesh:
    """
    A mesh for image warping.

    Defines a grid of control points. Each point can be displaced to create
    smooth deformations.

    Attributes:
        cols: Number of columns in the mesh
        rows: Number of rows in the mesh
        points: 2D array of control points (rows x cols x 2)
        displacements: 2D array of displacements (rows x cols x 2)
        width: Source image width
        height: Source image height
    """

    __slots__ = ('cols', 'rows', 'points', 'displacements', 'width', 'height')

    def __init__(
        self,
        cols: int,
        rows: int,
        width: int,
        height: int
    ):
        """
        Create a warp mesh.

        Args:
            cols: Number of columns (horizontal control points)
            rows: Number of rows (vertical control points)
            width: Source image width
            height: Source image height

        Raises:
            ValueError: If cols or rows < 2
        """
        if cols < 2 or rows < 2:
            raise ValueError(f"Mesh must have at least 2x2 points, got {cols}x{rows}")

        self.cols = cols
        self.rows = rows
        self.width = width
        self.height = height

        # Create grid points
        x_coords = np.linspace(0, width, cols, dtype=np.float32)
        y_coords = np.linspace(0, height, rows, dtype=np.float32)
        xx, yy = np.meshgrid(x_coords, y_coords)
        self.points = np.stack([xx, yy], axis=-1)

        # Displacements start at zero
        self.displacements = np.zeros((rows, cols, 2), dtype=np.float32)

    def get_displaced_points(self) -> np.ndarray:
        """Get points after applying displacements."""
        return self.points + self.displacements

    def set_displacement(self, col: int, row: int, dx: float, dy: float) -> None:
        """
        Set displacement at a specific control point.

        Args:
            col: Column index
            row: Row index
            dx: Displacement in X
            dy: Displacement in Y
        """
        self.displacements[row, col] = [dx, dy]

    def set_displacement_vec2(self, col: int, row: int, displacement: Vec2) -> None:
        """Set displacement using Vec2."""
        self.displacements[row, col] = [displacement.x, displacement.y]

    def get_displacement(self, col: int, row: int) -> Tuple[float, float]:
        """Get displacement at a control point."""
        return (self.displacements[row, col, 0], self.displacements[row, col, 1])

    def reset(self) -> None:
        """Reset all displacements to zero."""
        self.displacements.fill(0.0)

    def apply_to_point(self, x: float, y: float) -> Tuple[float, float]:
        """
        Apply warp to a single point using bilinear interpolation.

        Args:
            x: X coordinate in source
            y: Y coordinate in source

        Returns:
            (new_x, new_y) in source space
        """
        # Find which cell this point is in
        col = (x / self.width) * (self.cols - 1)
        row = (y / self.height) * (self.rows - 1)

        col0 = int(np.floor(col))
        col1 = min(col0 + 1, self.cols - 1)
        row0 = int(np.floor(row))
        row1 = min(row0 + 1, self.rows - 1)

        # Fractional position within cell
        frac_col = col - col0
        frac_row = row - row0

        # Get displacements at corners
        d00 = self.displacements[row0, col0]
        d10 = self.displacements[row0, col1]
        d01 = self.displacements[row1, col0]
        d11 = self.displacements[row1, col1]

        # Bilinear interpolation of displacement
        # Interpolate along columns first
        d_top = d00 * (1 - frac_col) + d10 * frac_col
        d_bottom = d01 * (1 - frac_col) + d11 * frac_col

        # Interpolate along rows
        d = d_top * (1 - frac_row) + d_bottom * frac_row

        return (x + d[0], y + d[1])


def warp_texture(
    texture: Texture,
    mesh: WarpMesh,
    output_size: Optional[Tuple[int, int]] = None
) -> Texture:
    """
    Warp a texture using a mesh.

    Uses OpenCV remap for fast, high-quality warping.

    Args:
        texture: The texture to warp
        mesh: The warp mesh (must match texture size)
        output_size: Optional (width, height) for output. Defaults to texture size.

    Returns:
        Warped texture

    Raises:
        ValueError: If mesh dimensions don't match texture
    """
    if mesh.width != texture.width or mesh.height != texture.height:
        raise ValueError(
            f"Mesh size ({mesh.width}x{mesh.height}) must match texture "
            f"({texture.width}x{texture.height})"
        )

    if output_size is None:
        output_size = (texture.width, texture.height)

    out_w, out_h = output_size

    # Get displaced points
    pts = mesh.get_displaced_points()

    # Create map_x and map_y for remap
    # remap expects maps from output pixel to source pixel
    # We need to invert: for each output pixel, where does it come from in source?

    # Generate output grid
    x_coords = np.linspace(0, texture.width, out_w, dtype=np.float32)
    y_coords = np.linspace(0, texture.height, out_h, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)

    # For each output pixel, find corresponding source position
    # This is a simplified version - we use the mesh to map each pixel
    map_x = np.zeros((out_h, out_w), dtype=np.float32)
    map_y = np.zeros((out_h, out_w), dtype=np.float32)

    for y in range(out_h):
        for x in range(out_w):
            src_x = grid_x[y, x]
            src_y = grid_y[y, x]
            new_x, new_y = mesh.apply_to_point(src_x, src_y)
            map_x[y, x] = new_x
            map_y[y, x] = new_y

    # Apply warp using OpenCV remap
    warped = cv2.remap(
        texture.data,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0) if texture.has_alpha() else (0, 0, 0)
    )

    return Texture(warped, texture.path)


def warp_texture_fast(
    texture: Texture,
    mesh: WarpMesh,
    output_size: Optional[Tuple[int, int]] = None
) -> Texture:
    """
    Fast version of warp_texture using grid sampling.

    Similar to warp_texture but more efficient for large meshes.

    TODO: Implement using cv2.remap with precomputed maps for speed.
    Currently falls back to warp_texture.
    """
    return warp_texture(texture, mesh, output_size)


def create_wave_mesh(
    width: int,
    height: int,
    cols: int = 20,
    rows: int = 20,
    amplitude: float = 20.0,
    frequency: float = 0.02,
    phase: float = 0.0,
    direction: str = 'vertical'
) -> WarpMesh:
    """
    Create a mesh with a wave deformation.

    Useful for cloth, water, and flag effects.

    Args:
        width: Image width
        height: Image height
        cols: Mesh columns
        rows: Mesh rows
        amplitude: Wave height in pixels
        frequency: Wave frequency (radians per pixel)
        phase: Wave phase offset
        direction: 'vertical', 'horizontal', or 'diagonal'

    Returns:
        WarpMesh with wave displacement
    """
    mesh = WarpMesh(cols, rows, width, height)

    for row in range(rows):
        for col in range(cols):
            x = (col / (cols - 1)) * width
            y = (row / (rows - 1)) * height

            if direction == 'vertical':
                dx = amplitude * np.sin(frequency * x + phase)
                dy = 0
            elif direction == 'horizontal':
                dx = 0
                dy = amplitude * np.sin(frequency * y + phase)
            elif direction == 'diagonal':
                dx = amplitude * np.sin(frequency * (x + y) + phase)
                dy = amplitude * np.sin(frequency * (x - y) + phase)
            else:
                dx = 0
                dy = 0

            mesh.set_displacement(col, row, dx, dy)

    return mesh


def create_sine_wave_mesh(
    width: int,
    height: int,
    cols: int = 20,
    rows: int = 20,
    amplitude_x: float = 10.0,
    amplitude_y: float = 10.0,
    frequency_x: float = 0.02,
    frequency_y: float = 0.02,
    phase_x: float = 0.0,
    phase_y: float = 0.0
) -> WarpMesh:
    """
    Create a mesh with independent sine waves in X and Y.

    More flexible than create_wave_mesh.

    Returns:
        WarpMesh with sinusoidal displacement
    """
    mesh = WarpMesh(cols, rows, width, height)

    for row in range(rows):
        for col in range(cols):
            x = (col / (cols - 1)) * width
            y = (row / (rows - 1)) * height

            dx = amplitude_x * np.sin(frequency_x * x + phase_x)
            dy = amplitude_y * np.sin(frequency_y * y + phase_y)

            mesh.set_displacement(col, row, dx, dy)

    return mesh


# ============================================================
# Public API
# ============================================================

__all__ = [
    "WarpMesh",
    "warp_texture",
    "warp_texture_fast",
    "create_wave_mesh",
    "create_sine_wave_mesh",
]
