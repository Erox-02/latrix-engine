"""
Generate a video from the animation.
"""

import sys
import cv2
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from geometry.vec2 import Vec2
from graphics.texture import Texture
from graphics.layer import Layer
from graphics.renderer import Renderer
from effects.breathing import BreathingEffect
from effects.camera import CameraEffect


def main():
    """Generate a video file."""
    
    print("Latrix Engine - Video Generation")
    print("=" * 50)
    
    # Load image
    image_path = Path("assets/02.jpg")
    if not image_path.exists():
        print(f"ERROR: Image not found at {image_path}")
        return
    
    print(f"Loading: {image_path}")
    texture = Texture.from_file(str(image_path))
    print(f"Loaded: {texture.width}x{texture.height}")
    
    # Create layer
    layer = Layer(
        texture=texture,
        position=Vec2(960, 540),
        scale=Vec2(0.9, 0.9),
        name="main"
    )
    
    # Add effects
    breath = BreathingEffect(
        amplitude_scale=0.01,
        amplitude_x=4.0,
        amplitude_y=2.0,
        speed=0.7,
        asymmetry=0.4,
        pause_duration=0.08,
        lag=0.02
    )
    layer.add_effect(breath)
    
    camera = CameraEffect(
        position=Vec2(960, 540),
        damping=8.0,
        drift_amplitude=10.0,
        drift_speed=0.15,
        shake_decay=0.98
    )
    layer.add_effect(camera)
    
    # Create renderer
    renderer = Renderer(1920, 1080, background=(20, 20, 30))
    
    # Create output directory
    Path("output").mkdir(exist_ok=True)
    
    # Video settings
    fps = 30
    duration = 5  # seconds
    total_frames = fps * duration
    dt = 1.0 / fps
    
    print(f"\nGenerating {total_frames} frames at {fps}fps")
    print(f"Duration: {duration} seconds")
    print(f"Output: output/animation.mp4")
    print("-" * 50)
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        "output/animation.mp4",
        fourcc,
        fps,
        (1920, 1080)
    )
    
    start_time = time.time()
    
    for frame in range(total_frames):
        # Update effects
        layer.update_effects(dt)
        
        # Apply camera transform
        offset_x, offset_y, zoom = camera.get_view_matrix()
        base_pos = Vec2(960, 540)
        layer.position = Vec2(
            base_pos.x - (offset_x - 960),
            base_pos.y - (offset_y - 540)
        )
        layer.scale = Vec2(0.9 * zoom, 0.9 * zoom)
        
        # Render
        renderer.clear()
        renderer.render_layer(layer)
        canvas = renderer.get_canvas()
        
        # Write frame
        video_writer.write(canvas)
        
        # Progress
        if frame % 30 == 0:
            progress = (frame / total_frames) * 100
            print(f"  {progress:.1f}% complete")
    
    video_writer.release()
    
    elapsed = time.time() - start_time
    print(f"\nDone! Video saved to output/animation.mp4")
    print(f"Generated {total_frames} frames in {elapsed:.1f} seconds")
    print(f"Average FPS: {total_frames / elapsed:.1f}")


if __name__ == "__main__":
    main()
