import os
import numpy as np
from PIL import Image, ImageDraw
import argparse
from tqdm import tqdm
import random

def generate_circle_image_at_position(x_pos, y_pos, image_size=64, circle_radius=8, 
                                    circle_color=(255, 128, 100), background_color=(255, 255, 255), 
                                    antialiasing=4):
    """
    Generate an image with a circle at the specified position.
    
    Args:
        x_pos: X position relative to center (in pixels, can be negative)
        y_pos: Y position relative to center (in pixels, can be negative)
        image_size: Square image size
        circle_radius: Fixed radius of the circle
        circle_color: RGB tuple for circle color
        background_color: RGB tuple for background
        antialiasing: Factor for smoother circles (higher = smoother)
    """
    # Create larger image for antialiasing
    large_size = image_size * antialiasing
    large_radius = circle_radius * antialiasing
    large_x = x_pos * antialiasing
    large_y = y_pos * antialiasing
    
    img = Image.new('RGB', (large_size, large_size), background_color)
    draw = ImageDraw.Draw(img)
    
    # Calculate circle center (offset from image center)
    center_x = large_size // 2 + large_x
    center_y = large_size // 2 + large_y
    
    # Circle bounds
    left = center_x - large_radius
    top = center_y - large_radius
    right = center_x + large_radius
    bottom = center_y + large_radius
    
    # Only draw if circle is at least partially within the image
    if (right > 0 and left < large_size and bottom > 0 and top < large_size):
        draw.ellipse([left, top, right, bottom], fill=circle_color, outline=circle_color)
    
    # Resize to final size
    img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
    
    return img

def add_variations(img, vary_color=True, noise_level=0.02):
    """
    Add variations to make the dataset more realistic.
    
    Args:
        img: PIL Image
        vary_color: Add small color variations
        noise_level: Amount of noise to add
    """
    img_array = np.array(img).astype(np.float32) / 255.0
    
    if vary_color:
        color_shift = np.random.normal(0, 0.05, (1, 1, 3))
        img_array = np.clip(img_array + color_shift, 0, 1)
    
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 1)
    
    img_array = (img_array * 255).astype(np.uint8)
    return Image.fromarray(img_array)

def generate_contrasting_colors():
    """
    Generate a circle color and background color with good contrast.
    """
    # Define vibrant circle colors
    circle_colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Purple
    ]
    
    # Define light background colors
    background_colors = [
        (255, 255, 255),  # White
        (240, 240, 240),  # Light gray
        (255, 250, 240),  # Cream
        (240, 255, 240),  # Light green
        (240, 240, 255),  # Light blue
        (255, 240, 240),  # Light red
        (255, 255, 240),  # Light yellow
        (255, 240, 255),  # Light magenta
    ]
    
    # Choose random circle color
    circle_color = random.choice(circle_colors)
    
    # Add some variation to circle color
    color_var = np.random.randint(-20, 20, 3)
    circle_color = tuple(np.clip(np.array(circle_color) + color_var, 0, 255))
    
    # Choose background color with good contrast
    best_bg_color = random.choice(background_colors)
    bg_var = np.random.randint(-15, 15, 3)
    best_bg_color = tuple(np.clip(np.array(best_bg_color) + bg_var, 200, 255))
    
    return circle_color, best_bg_color

def generate_position_coordinates(square_size, num_positions_per_axis):
    """
    Generate evenly spaced position coordinates within a square.
    
    Args:
        square_size: Size of the square (positions range from -square_size/2 to +square_size/2)
        num_positions_per_axis: Number of discrete positions per axis
    
    Returns:
        List of (x, y) coordinate tuples
    """
    # Generate evenly spaced coordinates
    half_size = square_size / 2
    coords = np.linspace(-half_size, half_size, num_positions_per_axis)
    
    positions = []
    for x in coords:
        for y in coords:
            positions.append((x, y))
    
    return positions

def generate_dataset(output_dir, total_images=15000, image_size=64, circle_radius=8,
                    square_size=20, num_positions_per_axis=5):
    """
    Generate the complete position dataset.
    
    Args:
        output_dir: Root directory for the dataset
        total_images: Total number of images to generate
        image_size: Size of generated images
        circle_radius: Fixed radius of circles
        square_size: Size of the square to sample positions from
        num_positions_per_axis: Number of discrete positions per axis
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate position coordinates
    positions = generate_position_coordinates(square_size, num_positions_per_axis)
    print(f"Generated {len(positions)} position classes")
    print(f"Position range: [{-square_size/2:.1f}, {square_size/2:.1f}] in both x and y")
    print(f"Positions: {positions[:10]}..." if len(positions) > 10 else f"Positions: {positions}")
    
    # Calculate samples per position
    samples_per_position = total_images // len(positions)
    print(f"Samples per position: {samples_per_position}")
    
    # Create directories and generate images
    for pos_idx, (x, y) in enumerate(positions):
        # Create directory name as "x_y" format
        pos_name = f"{x:.1f}_{y:.1f}".replace('-', 'neg').replace('.', 'p')
        pos_dir = os.path.join(output_dir, pos_name)
        os.makedirs(pos_dir, exist_ok=True)
        
        print(f"\nGenerating position class ({x:.1f}, {y:.1f}) -> {pos_name}")
        
        for i in tqdm(range(samples_per_position), desc=f"Position ({x:.1f}, {y:.1f})"):
            # Generate contrasting colors
            circle_color, bg_color = generate_contrasting_colors()
            
            # Add small random variation to position (±0.5 pixels)
            x_var = x
            y_var = y
            
            img = generate_circle_image_at_position(
                x_var, y_var,
                image_size=image_size,
                circle_radius=circle_radius,
                circle_color=circle_color,
                background_color=bg_color
            )
            
            img = add_variations(img, vary_color=True)
            
            # Save image
            img_path = os.path.join(pos_dir, f"circle_{i:05d}.png")
            img.save(img_path)
    
    # Save dataset metadata
    metadata = {
        'square_size': square_size,
        'num_positions_per_axis': num_positions_per_axis,
        'total_positions': len(positions),
        'samples_per_position': samples_per_position,
        'total_images': len(positions) * samples_per_position,
        'image_size': image_size,
        'circle_radius': circle_radius,
        'positions': positions
    }
    
    import json
    with open(os.path.join(output_dir, 'dataset_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nDataset generated successfully at {output_dir}")
    print(f"Total images: {len(positions) * samples_per_position}")
    print(f"Total position classes: {len(positions)}")

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic circle dataset with position conditioning')
    parser.add_argument('--output-dir', type=str, default='circle_position_dataset',
                       help='Output directory for the dataset')
    parser.add_argument('--total-images', type=int, default=10000,
                       help='Total number of images to generate')
    parser.add_argument('--image-size', type=int, default=64,
                       help='Size of generated images')
    parser.add_argument('--circle-radius', type=int, default=4,
                       help='Fixed radius of circles in pixels')
    parser.add_argument('--square-size', type=float, default=36,
                       help='Size of square to sample positions from (range: [-size/2, +size/2])')
    parser.add_argument('--num-positions-per-axis', type=int, default=16,
                       help='Number of discrete positions per axis (total positions = this^2)')
    
    args = parser.parse_args()
    
    generate_dataset(
        output_dir=args.output_dir,
        total_images=args.total_images,
        image_size=args.image_size,
        circle_radius=args.circle_radius,
        square_size=args.square_size,
        num_positions_per_axis=args.num_positions_per_axis
    )

if __name__ == "__main__":
    main() 