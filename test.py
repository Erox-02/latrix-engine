"""
Test script for Latrix Engine with sample image.
Loads an image, applies breathing and camera effects, and renders.
"""

import sys
import os
import math
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from geometry.vec2 import Vec2
from graphics.texture import Texture
from graphics.layer import Layer
from graphics.renderer import Renderer
from graphics.warp import WarpMesh, warp_texture, create_wave_mesh
from effects.breathing import BreathingEffect
from effects.camera import CameraEffect


def main():
    """Run the test wallpaper."""
    
    # ============================================================
    # 1. Load Image
    # ============================================================
    
    image_path = Path(__file__).parent.parent / "assets" / "02.jpg"
    
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        print("Please place 02.jpg in the assets/ folder")
        return
    
    print(f"Loading image: {image_path}")
    texture = Texture.from_file(str(image_path))
    print(f"Image loaded: {texture.width}x{texture.height}")
    
    # ============================================================
    # 2. Create Layers
    # ============================================================
    
    # Main layer with the image
    main_layer = Layer(
        texture=texture,
        position=Vec2(960, 540),  # Center of 1920x1080
        scale=Vec2(0.8, 0.8),      # Slightly smaller
        name="main"
    )
    
    # ============================================================
    # 3. Add Effects
    # ============================================================
    
    # Breathing effect - subtle idle motion
    breath = BreathingEffect(
        amplitude_scale=0.008,     # 0.8% scale change
        amplitude_x=5.0,          # 5px horizontal sway
        amplitude_y=3.0,          # 3px vertical sway
        speed=0.6,                # Slow, calm breathing
        asymmetry=0.4,            # 40% inhale, 60% exhale
        pause_duration=0.06,      # Brief pause at peak
        lag=0.02                  # Sway lags behind scale
    )
    main_layer.add_effect(breath)
    
    # Camera effect - gentle drift
    camera = CameraEffect(
        position=Vec2(960, 540),
        damping=8.0,               # Fast response
        drift_amplitude=15.0,      # 15px drift
        drift_speed=0.15,          # Very slow drift
        shake_decay=0.98
    )
    main_layer.add_effect(camera)
    
    # ============================================================
    # 4. Create Renderer
    # ============================================================
    
    renderer = Renderer(1920, 1080, background=(20, 20, 30))
    
    # ============================================================
    # 5. Animation Loop
    # ============================================================
    
    print("\nStarting animation loop...")
    print("Press Ctrl+C to stop\n")
    
    frame = 0
    fps = 60
    dt = 1.0 / fps
    start_time = time.time()
    
    try:
        while True:
            # Update all effects
            main_layer.update_effects(dt)
            
            # Apply camera transform to layer
            # Get camera transform and apply it
            offset_x, offset_y, zoom = camera.get_view_matrix()
            
            # Camera moves opposite to layer position
            # We offset the layer's position by camera position
            # Since camera position is center of screen, we negate it
            
            # The layer's position is already set in the breathing effect
            # We add camera offset on top
            base_pos = Vec2(960, 540)
            
            # Apply camera offset (inverse)
            main_layer.position = Vec2(
                base_pos.x - (offset_x - 960),
                base_pos.y - (offset_y - 540)
            )
            
            # Apply zoom
            main_layer.scale = Vec2(0.8 * zoom, 0.8 * zoom)
            
            # Render
            renderer.clear()
            renderer.render_layer(main_layer)
            
            # Save frame every 30 frames
            if frame % 30 == 0:
                output_path = f"output/frame_{frame:04d}.png"
                renderer.save(output_path)
                print(f"Saved: {output_path}")
            
            # Display (optional)
            # cv2.imshow("Latrix", renderer.get_canvas())
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            
            frame += 1
            
            # Real-time control
            time.sleep(dt)
            
    except KeyboardInterrupt:
        print(f"\nStopped after {frame} frames")
    
    # ============================================================
    # 6. Final Output
    # ============================================================
    
    print(f"\nRendered {frame} frames")
    print(f"FPS: {frame / (time.time() - start_time):.1f}")
    
    # Save final frame
    renderer.save("output/final.png")
    print("Final frame saved: output/final.png")


if __name__ == "__main__":
    # Create output directory
    Path("output").mkdir(exist_ok=True)
    
    main()
