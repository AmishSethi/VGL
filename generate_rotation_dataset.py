#!/usr/bin/env python
"""
Generate synthetic dataset with rotation angle conditioning.
Creates asymmetric arrows at specific rotation angles for training diffusion models.
"""
import os
import numpy as np
from PIL import Image, ImageDraw
import argparse
from tqdm import tqdm
import json
import random
import math


def draw_rotated_arrow(angle_degrees, image_size=64, arrow_size=20,
                       arrow_color=(255, 100, 100), background_color=(255, 255, 255),
                       antialiasing=4):
    """
    Draw an asymmetric arrow rotated by the specified angle.
    Uses compass convention: 0° = up, increases clockwise.
    
    Args:
        angle_degrees: Rotation angle in degrees (0° = up, clockwise)
        image_size: Square image size
        arrow_size: Size of the arrow (length from tail to tip)
        arrow_color: RGB color of the arrow
        background_color: RGB background color
        antialiasing: Factor for smoother rendering
    """
    # Create larger image for antialiasing
    large_size = image_size * antialiasing
    large_arrow_size = arrow_size * antialiasing
    
    img = Image.new('RGB', (large_size, large_size), background_color)
    draw = ImageDraw.Draw(img)
    
    # Calculate arrow vertices (asymmetric arrow pointing up initially)
    center_x = large_size // 2
    center_y = large_size // 2
    
    # Convert angle to radians
    # Use compass convention: 0° = up, increases clockwise
    # Subtract 90° to convert from compass to math convention, then negate for clockwise
    angle_rad = math.radians(-angle_degrees + 90)
    
    # Define arrow shape vertices (pointing up initially for 0°)
    # Arrow with pointed tip, wide body, and narrow tail
    arrow_points = [
        # Tip (pointing up)
        (0, -large_arrow_size * 0.5),
        # Left part of arrowhead
        (-large_arrow_size * 0.25, -large_arrow_size * 0.2),
        # Left body
        (-large_arrow_size * 0.15, -large_arrow_size * 0.2),
        # Left tail
        (-large_arrow_size * 0.1, large_arrow_size * 0.4),
        # Tail tip
        (0, large_arrow_size * 0.5),
        # Right tail
        (large_arrow_size * 0.1, large_arrow_size * 0.4),
        # Right body
        (large_arrow_size * 0.15, -large_arrow_size * 0.2),
        # Right part of arrowhead
        (large_arrow_size * 0.25, -large_arrow_size * 0.2),
    ]
    
    # Rotate and translate vertices
    rotated_points = []
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    for x, y in arrow_points:
        # Rotate around origin
        rotated_x = x * cos_a - y * sin_a
        rotated_y = x * sin_a + y * cos_a
        # Translate to center
        final_x = center_x + rotated_x
        final_y = center_y + rotated_y
        rotated_points.append((final_x, final_y))
    
    # Draw the arrow
    draw.polygon(rotated_points, fill=arrow_color, outline=arrow_color)
    
    # Resize to final size
    img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
    
    return img


def draw_rotated_right_triangle(angle_degrees, image_size=64, triangle_size=20,
                                triangle_color=(255, 100, 100), background_color=(255, 255, 255),
                                antialiasing=4):
    """
    Alternative: Draw a right triangle rotated by the specified angle.
    Uses compass convention: 0° = up, increases clockwise.
    
    Args:
        angle_degrees: Rotation angle in degrees (0° = up, clockwise)
        image_size: Square image size
        triangle_size: Size of the triangle
        triangle_color: RGB color of the triangle
        background_color: RGB background color
        antialiasing: Factor for smoother rendering
    """
    # Create larger image for antialiasing
    large_size = image_size * antialiasing
    large_triangle_size = triangle_size * antialiasing
    
    img = Image.new('RGB', (large_size, large_size), background_color)
    draw = ImageDraw.Draw(img)
    
    # Calculate triangle vertices (right triangle)
    center_x = large_size // 2
    center_y = large_size // 2
    
    # Convert angle to radians
    # Use compass convention: 0° = up, increases clockwise
    angle_rad = math.radians(-angle_degrees + 90)
    
    # Define right triangle vertices (pointing up initially)
    # Make it clearly asymmetric
    triangle_points = [
        (0, -large_triangle_size * 0.5),  # Top vertex (pointing up)
        (large_triangle_size * 0.433, large_triangle_size * 0.25),  # Bottom right
        (-large_triangle_size * 0.2, large_triangle_size * 0.25),  # Bottom left (shorter side)
    ]
    
    # Rotate and translate vertices
    rotated_points = []
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    for x, y in triangle_points:
        # Rotate around origin
        rotated_x = x * cos_a - y * sin_a
        rotated_y = x * sin_a + y * cos_a
        # Translate to center
        final_x = center_x + rotated_x
        final_y = center_y + rotated_y
        rotated_points.append((final_x, final_y))
    
    # Draw the triangle
    draw.polygon(rotated_points, fill=triangle_color, outline=triangle_color)
    
    # Resize to final size
    img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
    
    return img


def add_background_variations(img, vary_background=True, noise_level=0.01):
    """
    Add variations to background only, keeping shape exact.
    
    Args:
        img: PIL Image
        vary_background: Add small background variations
        noise_level: Amount of noise to add to background
    """
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # Create mask for the shape (non-white pixels)
    gray = np.mean(img_array, axis=2)
    shape_mask = gray < 0.95  # Assuming white or near-white background
    
    if vary_background:
        # Only add noise to background
        background_mask = ~shape_mask
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, img_array.shape)
            for c in range(3):
                img_array[:, :, c][background_mask] += noise[:, :, c][background_mask]
    
    img_array = np.clip(img_array, 0, 1)
    img_array = (img_array * 255).astype(np.uint8)
    return Image.fromarray(img_array)


def generate_background_color():
    """Generate a light background color with some variation."""
    base_colors = [
        (255, 255, 255),  # White
        (250, 250, 250),  # Very light gray
        (255, 250, 245),  # Light cream
        (245, 255, 250),  # Light mint
        (245, 245, 255),  # Light lavender
    ]
    
    base = random.choice(base_colors)
    # Add small variation
    var = np.random.randint(-5, 5, 3)
    bg_color = tuple(np.clip(np.array(base) + var, 240, 255))
    return bg_color


def generate_shape_color():
    """Generate a vibrant shape color."""
    colors = [
        (255, 100, 100),  # Red
        (100, 255, 100),  # Green
        (100, 100, 255),  # Blue
        (255, 200, 100),  # Orange
        (255, 100, 255),  # Magenta
        (100, 255, 255),  # Cyan
        (200, 100, 255),  # Purple
    ]
    
    base = random.choice(colors)
    # Add small variation
    var = np.random.randint(-20, 20, 3)
    color = tuple(np.clip(np.array(base) + var, 50, 255))
    return color


def generate_rotation_angles(min_angle, max_angle, num_angles):
    """
    Generate evenly spaced rotation angles within a range.
    
    Args:
        min_angle: Minimum rotation angle (e.g., 45)
        max_angle: Maximum rotation angle (e.g., 315)
        num_angles: Number of discrete angles to generate
    
    Returns:
        List of rotation angles in degrees
    """
    angles = np.linspace(min_angle, max_angle, num_angles, endpoint=False, dtype=float)
    return angles.tolist()


def generate_dataset(output_dir, image_size=64, shape_size=20,
                    min_angle=45, max_angle=315, coverage_percent=50,
                    samples_per_class=100, shape_type='arrow'):
    """
    Generate the complete rotation dataset.
    
    Args:
        output_dir: Root directory for the dataset
        image_size: Size of generated images
        shape_size: Size of the shape
        min_angle: Minimum rotation angle for training range
        max_angle: Maximum rotation angle for training range
        coverage_percent: Coverage percentage (25, 50, or 75)
        samples_per_class: Number of samples per rotation angle
        shape_type: Type of shape ('arrow' or 'triangle')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate angle range
    angle_range = max_angle - min_angle
    
    # Determine number of angles based on coverage
    # Base the spacing on typical reasonable intervals
    if coverage_percent == 25:
        # Sparse coverage: ~30° spacing
        num_angles = max(3, int(angle_range / 30))
    elif coverage_percent == 50:
        # Medium coverage: ~15° spacing
        num_angles = max(6, int(angle_range / 15))
    else:  # 75%
        # Dense coverage: ~10° spacing
        num_angles = max(9, int(angle_range / 1))
    
    # Generate rotation angles
    rotation_angles = generate_rotation_angles(min_angle, max_angle, num_angles)
    
    # Calculate actual spacing
    actual_spacing = angle_range / num_angles if num_angles > 0 else 0
    
    print(f"Generated {num_angles} rotation angles")
    print(f"Rotation range: [{min_angle}°, {max_angle}°] (span: {angle_range}°)")
    print(f"Coverage: {coverage_percent}%")
    print(f"Shape type: {shape_type}")
    print(f"Angle spacing: {actual_spacing:.1f}°")
    print(f"First few angles: {[f'{a:.1f}°' for a in rotation_angles[:5]]}")
    print(f"Samples per angle: {samples_per_class}")
    print(f"Total images to generate: {num_angles * samples_per_class}")
    print(f"Extrapolation ranges: [0°, {min_angle}°) and ({max_angle}°, 360°)")
    
    # Choose drawing function based on shape type
    if shape_type == 'arrow':
        draw_function = draw_rotated_arrow
    else:
        draw_function = draw_rotated_right_triangle
    
    # Create directories and generate images
    total_generated = 0
    for angle in tqdm(rotation_angles, desc=f"Generating {shape_type} rotations"):
        # Create directory name for this angle
        # Format angle to avoid decimal points in folder names
        angle_str = f"{angle:.1f}".replace('.', 'p').replace('-', 'neg')
        angle_dir = os.path.join(output_dir, angle_str)
        os.makedirs(angle_dir, exist_ok=True)
        
        for i in range(samples_per_class):
            # Generate colors
            shape_color = generate_shape_color()
            bg_color = generate_background_color()
            
            # Generate image with exact rotation angle (no variation)
            img = draw_function(
                angle_degrees=angle,  # Exact angle
                image_size=image_size,
                **{f'{shape_type}_size': shape_size},
                **{f'{shape_type}_color': shape_color},
                background_color=bg_color
            )
            
            # Add background variations only
            img = add_background_variations(img, vary_background=True, noise_level=0.01)
            
            # Save image
            img_path = os.path.join(angle_dir, f"{shape_type}_{i:05d}.png")
            img.save(img_path)
            total_generated += 1
    
    # Save dataset metadata
    metadata = {
        'shape_type': shape_type,
        'min_angle': min_angle,
        'max_angle': max_angle,
        'angle_range': angle_range,
        'coverage_percent': coverage_percent,
        'num_angles': num_angles,
        'rotation_angles': rotation_angles,
        'samples_per_angle': samples_per_class,
        'total_images': total_generated,
        'image_size': image_size,
        'shape_size': shape_size,
        'angle_step': actual_spacing,
        'extrapolation_ranges': [
            {'range': [0, min_angle], 'description': 'Lower extrapolation'},
            {'range': [max_angle, 360], 'description': 'Upper extrapolation'}
        ],
        'angle_convention': 'compass (0° = up, clockwise)',
        'asymmetric_shape': True
    }
    
    with open(os.path.join(output_dir, 'dataset_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nDataset generated successfully at {output_dir}")
    print(f"Total images: {total_generated}")
    print(f"Total rotation classes: {num_angles}")
    print(f"Rotation step size: {actual_spacing:.1f}°")
    print(f"For extrapolation testing:")
    print(f"  - Test angles 0°-{min_angle-1}° (lower extrapolation)")
    print(f"  - Test angles {max_angle+1}°-359° (upper extrapolation)")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic dataset with asymmetric shape rotation conditioning')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for the dataset')
    parser.add_argument('--image-size', type=int, default=64,
                       help='Size of generated images')
    parser.add_argument('--shape-size', type=int, default=20,
                       help='Size of shapes in pixels')
    parser.add_argument('--min-angle', type=float, default=45,
                       help='Minimum rotation angle for training range')
    parser.add_argument('--max-angle', type=float, default=315,
                       help='Maximum rotation angle for training range')
    parser.add_argument('--coverage-percent', type=int, choices=[25, 50, 75], required=True,
                       help='Coverage percentage of the rotation range')
    parser.add_argument('--samples-per-class', type=int, default=35,
                       help='Number of samples per rotation angle')
    parser.add_argument('--shape-type', type=str, choices=['arrow', 'triangle'], default='arrow',
                       help='Type of asymmetric shape to generate')
    
    args = parser.parse_args()
    
    generate_dataset(
        output_dir=args.output_dir,
        image_size=args.image_size,
        shape_size=args.shape_size,
        min_angle=args.min_angle,
        max_angle=args.max_angle,
        coverage_percent=args.coverage_percent,
        samples_per_class=args.samples_per_class,
        shape_type=args.shape_type
    )


if __name__ == "__main__":
    main()