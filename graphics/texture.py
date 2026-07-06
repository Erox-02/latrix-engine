"""
Texture: Wrapper around OpenCV images with memory management and basic operations.

A Texture represents an image loaded into memory, ready for rendering.
Handles loading from disk, color space conversions, and basic image manipulation.

All operations are immutable - they return new Textures.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]


class Texture:
    """
    Wrapper around an OpenCV image (numpy array).

    Stores image in BGR format (OpenCV's native format).
    Supports 3-channel (BGR) and 4-channel (BGRA) images.

    Attributes:
        data: The image data as a numpy array (HxWxC)
        width: Image width in pixels
        height: Image height in pixels
        channels: Number of color channels (3 or 4)
        path: Source file path (if loaded from disk)
    """

    __slots__ = ('data', 'width', 'height', 'channels', 'path')

    def __init__(
        self,
        data: np.ndarray,
        path: Optional[PathLike] = None
    ):
        """
        Initialize a Texture from numpy array data.

        Args:
            data: Image data (HxW) or (HxWxC). Grayscale is auto-converted to BGR.
            path: Source file path (optional, for reference)

        Raises:
            ValueError: If data is empty or has unsupported shape
        """
        if data is None or data.size == 0:
            raise ValueError("Texture data cannot be empty")

        # Handle grayscale (HxW) -> BGR (HxWx3)
        if len(data.shape) == 2:
            data = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)

        if len(data.shape) != 3:
            raise ValueError(f"Expected 2D (HxW) or 3D (HxWxC), got shape {data.shape}")

        channels = data.shape[2]
        if channels not in (3, 4):
            raise ValueError(f"Expected 3 or 4 channels, got {channels}")

        self.data = data.copy()
        self.height, self.width = data.shape[:2]
        self.channels = channels
        self.path = str(path) if path else None

    # ============================================================
    # Factory Methods
    # ============================================================

    @classmethod
    def from_file(cls, path: PathLike) -> Texture:
        """
        Load an image from disk.

        Supports all formats OpenCV supports (PNG, JPG, WEBP, etc).
        Preserves alpha channel if present.

        Args:
            path: Path to image file

        Returns:
            Texture containing the loaded image

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If image fails to load
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        # Load with alpha channel preserved
        data = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

        if data is None:
            raise ValueError(f"Failed to load image: {path}")

        return cls(data, path)

    @classmethod
    def blank(cls, width: int, height: int, color: tuple = (0, 0, 0)) -> Texture:
        """
        Create a blank texture.

        Args:
            width: Width in pixels
            height: Height in pixels
            color: BGR color tuple (b, g, r), default black

        Returns:
            Texture filled with the given color
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"Width and height must be positive, got {width}x{height}")

        data = np.full((height, width, 3), color, dtype=np.uint8)
        return cls(data)

    @classmethod
    def from_numpy(cls, data: np.ndarray) -> Texture:
        """
        Create a Texture from a numpy array.

        Args:
            data: Image data (HxWxC) in BGR/BGRA format

        Returns:
            Texture wrapping the data
        """
        return cls(data)

    # ============================================================
    # Color Space Conversions
    # ============================================================

    def to_rgb(self) -> np.ndarray:
        """
        Convert to RGB format for display/interaction.

        Returns a numpy array in RGB format.
        Use this when you need to display with libraries that expect RGB.
        """
        if self.channels == 3:
            return cv2.cvtColor(self.data, cv2.COLOR_BGR2RGB)
        else:
            return cv2.cvtColor(self.data, cv2.COLOR_BGRA2RGBA)

    def to_bgr(self) -> np.ndarray:
        """
        Convert from RGB back to BGR.

        Returns a numpy array in BGR format.
        Use this when you need to pass to OpenCV functions.
        """
        if self.channels == 3:
            return cv2.cvtColor(self.data, cv2.COLOR_RGB2BGR)
        else:
            return cv2.cvtColor(self.data, cv2.COLOR_RGBA2BGRA)

    # ============================================================
    # Basic Operations
    # ============================================================

    def resize(self, width: int, height: int) -> Texture:
        """
        Resize the texture.

        Uses bilinear interpolation by default.

        Args:
            width: New width
            height: New height

        Returns:
            Resized Texture
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"Width and height must be positive, got {width}x{height}")

        resized = cv2.resize(
            self.data,
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )
        return Texture(resized, self.path)

    def resize_aspect(self, target_width: int, target_height: int) -> Texture:
        """
        Resize while maintaining aspect ratio.

        Fits the image within target dimensions, centering with padding.

        Args:
            target_width: Maximum width
            target_height: Maximum height

        Returns:
            Resized Texture centered in target dimensions
        """
        scale = min(target_width / self.width, target_height / self.height)
        new_w = int(self.width * scale)
        new_h = int(self.height * scale)

        resized = cv2.resize(
            self.data,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR
        )

        canvas = np.zeros((target_height, target_width, self.channels), dtype=np.uint8)

        y_offset = (target_height - new_h) // 2
        x_offset = (target_width - new_w) // 2

        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        return Texture(canvas, self.path)

    def crop(self, x: int, y: int, width: int, height: int) -> Texture:
        """
        Crop the texture.

        Args:
            x: Left coordinate
            y: Top coordinate
            width: Crop width
            height: Crop height

        Returns:
            Cropped Texture
        """
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError("Invalid crop parameters")

        if x + width > self.width or y + height > self.height:
            raise ValueError("Crop region out of bounds")

        cropped = self.data[y:y + height, x:x + width].copy()
        return Texture(cropped, self.path)

    def flip(self, horizontally: bool = False, vertically: bool = False) -> Texture:
        """
        Flip the texture.

        Args:
            horizontally: Flip left-right
            vertically: Flip top-bottom

        Returns:
            Flipped Texture
        """
        if horizontally and vertically:
            flipped = cv2.flip(self.data, -1)
        elif horizontally:
            flipped = cv2.flip(self.data, 1)
        elif vertically:
            flipped = cv2.flip(self.data, 0)
        else:
            flipped = self.data.copy()

        return Texture(flipped, self.path)

    def rotate(self, angle: float) -> Texture:
        """
        Rotate by angle (degrees).

        Args:
            angle: Rotation angle in degrees

        Returns:
            Rotated Texture
        """
        center = (self.width // 2, self.height // 2)
        matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)

        cos = abs(matrix[0, 0])
        sin = abs(matrix[0, 1])
        new_w = int(self.height * sin + self.width * cos)
        new_h = int(self.height * cos + self.width * sin)

        matrix[0, 2] += (new_w / 2) - center[0]
        matrix[1, 2] += (new_h / 2) - center[1]

        border = (0, 0, 0, 0) if self.channels == 4 else (0, 0, 0)
        rotated = cv2.warpAffine(
            self.data,
            matrix,
            (new_w, new_h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border
        )

        return Texture(rotated, self.path)

    # ============================================================
    # Mask Operations
    # ============================================================

    def create_mask_from_alpha(self, threshold: int = 127) -> Texture:
        """
        Create a binary mask from the alpha channel.

        Args:
            threshold: Alpha value threshold (0-255)

        Returns:
            Grayscale texture mask (0 or 255)

        Raises:
            ValueError: If texture has no alpha channel
        """
        if self.channels != 4:
            raise ValueError("Texture has no alpha channel")

        alpha = self.data[:, :, 3]
        mask = np.where(alpha > threshold, 255, 0).astype(np.uint8)
        return Texture(mask, self.path)

    def create_mask_from_color(self, color: tuple, tolerance: int = 10) -> Texture:
        """
        Create a mask from a specific color.

        Uses HSV color space for better color matching.

        Args:
            color: BGR color tuple (b, g, r)
            tolerance: Color tolerance (0-255)

        Returns:
            Grayscale texture mask (0 or 255)
        """
        if self.channels < 3:
            raise ValueError("Texture must have color channels")

        hsv = cv2.cvtColor(self.data, cv2.COLOR_BGR2HSV)
        color_hsv = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0][0]

        lower = np.array([max(0, color_hsv[0] - tolerance), 0, 0])
        upper = np.array([min(179, color_hsv[0] + tolerance), 255, 255])

        mask = cv2.inRange(hsv, lower, upper)
        return Texture(mask, self.path)

    # ============================================================
    # Properties & Utilities
    # ============================================================

    def size(self) -> tuple[int, int]:
        """Get texture size as (width, height)."""
        return (self.width, self.height)

    def has_alpha(self) -> bool:
        """Check if texture has alpha channel."""
        return self.channels == 4

    def copy(self) -> Texture:
        """Create a copy of the texture."""
        return Texture(self.data.copy(), self.path)

    def save(self, path: PathLike) -> None:
        """
        Save the texture to disk.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        success = cv2.imwrite(str(path), self.data)
        if not success:
            raise RuntimeError(f"Failed to save image: {path}")

    # ============================================================
    # Magic Methods
    # ============================================================

    def __repr__(self) -> str:
        return f"Texture({self.width}x{self.height}, {self.channels}ch, path={self.path})"


# ============================================================
# Public API
# ============================================================

__all__ = [
    "Texture",
]
