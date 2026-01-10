import os
import numpy as np
from PIL import Image, ImageDraw
import argparse
from tqdm import tqdm
import random

def generate_circle_image(radius, image_size=256, circle_color=(255, 128, 100), 
                         background_color=(255, 255, 255), antialiasing=4):
    """
    Generate an image with a centered circle of given radius.
    
    Args:
        radius: Circle radius in pixels
        image_size: Square image size
        circle_color: RGB tuple for circle color
        background_color: RGB tuple for background
        antialiasing: Factor for smoother circles (higher = smoother)
    """
    # Create larger image for antialiasing
    large_size = image_size * antialiasing
    large_radius = radius * antialiasing
    
    img = Image.new('RGB', (large_size, large_size), background_color)
    draw = ImageDraw.Draw(img)
    
    center = large_size // 2
    left = center - large_radius
    top = center - large_radius
    right = center + large_radius
    bottom = center + large_radius
    
    draw.ellipse([left, top, right, bottom], fill=circle_color, outline=circle_color)
    
    img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
    
    return img

def add_variations(img, vary_position=True, vary_color=True, noise_level=0.02):
    """
    Add variations to make the dataset more realistic.
    
    Args:
        img: PIL Image
        vary_position: Add small position variations
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

def color_distance(color1, color2):
    """
    Calculate Euclidean distance between two RGB colors.
    """
    return np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)))

def generate_contrasting_colors():
    """
    Generate a circle color and background color with good contrast.
    """
    # Define vibrant circle colors
    circle_colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        # (255, 255, 0),  # Yellow
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
    best_bg_color = None
    max_distance = 0
    
    for bg_color in background_colors:
        # Add some variation to background color
        bg_var = np.random.randint(-15, 15, 3)
        bg_color_var = tuple(np.clip(np.array(bg_color) + bg_var, 200, 255))  # Keep light
        
        distance = color_distance(circle_color, bg_color_var)
        if distance > max_distance:
            max_distance = distance
            best_bg_color = bg_color_var
    
    # Ensure minimum contrast threshold
    if max_distance < 100:  # If contrast is too low
        # Force a high-contrast background
        if np.mean(circle_color) > 128:  # Dark circle
            best_bg_color = (255, 255, 255)  # White background
        else:  # Light circle
            best_bg_color = (50, 50, 50)  # Dark background
    
    return circle_color, best_bg_color

def generate_gaussian_samples_per_class(radii, total_samples, center=None, std_factor=0.25):
    """
    Generate number of samples per radius class following a Gaussian distribution.
    
    Args:
        radii: List of radius values
        total_samples: Total number of samples across all classes
        center: Center of Gaussian distribution (default: middle of range)
        std_factor: Standard deviation as fraction of range
    
    Returns:
        Dictionary mapping radius to number of samples
    """
    if center is None:
        center = (min(radii) + max(radii)) / 2
    
    radius_range = max(radii) - min(radii)
    std = radius_range * std_factor
    
    # Calculate Gaussian weights for each radius
    weights = []
    for r in radii:
        weight = np.exp(-0.5 * ((r - center) / std) ** 2)
        weights.append(weight)
    
    # Normalize weights to sum to total_samples
    weights = np.array(weights)
    weights = weights / weights.sum() * total_samples
    
    # Convert to integer samples, ensuring minimum of 50 per class
    samples_per_class = {}
    remaining_samples = total_samples
    
    for i, r in enumerate(radii):
        if i == len(radii) - 1:  # Last class gets remaining samples
            samples_per_class[r] = max(50, remaining_samples)
        else:
            samples = max(50, int(round(weights[i])))
            samples_per_class[r] = samples
            remaining_samples -= samples
    
    return samples_per_class

def generate_dataset(output_dir, total_samples=15000, image_size=64, 
                    train_radii=None, gaussian_distribution=True, center=None, std_factor=0.25,
                    bias=0, bias_description=""):
    """
    Generate the complete dataset.
    
    Args:
        output_dir: Root directory for the dataset
        total_samples: Total number of samples across all classes
        image_size: Size of generated images
        train_radii: List of radii to generate (if None, generates 16 radii from 5 to 20)
        gaussian_distribution: Whether to use Gaussian distribution of samples
        center: Center of Gaussian distribution (default: middle of range)
        std_factor: Standard deviation as fraction of range for Gaussian distribution
        bias: Value to add to all radius labels (e.g., bias=4 means radius 1 gets label 5)
        bias_description: Description of the bias for logging
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Default to 16 radii from 5 to 20
    if train_radii is None:
        train_radii = np.linspace(5, 20, 16).tolist()
    
    # Apply bias to create shifted labels
    if bias != 0:
        # Create mapping from actual radius to label
        radius_to_label = {r: r + bias for r in train_radii}
        label_to_radius = {r + bias: r for r in train_radii}
        
        # Use the biased labels for folder names and sampling
        biased_radii = [r + bias for r in train_radii]
        
        print(f"Applying bias: {bias}")
        print(f"Bias description: {bias_description}")
        print(f"Actual radius -> Label mapping:")
        for actual_r, label_r in radius_to_label.items():
            print(f"  Radius {actual_r} pixels -> Label {label_r}")
    else:
        biased_radii = train_radii
        radius_to_label = {r: r for r in train_radii}
        label_to_radius = {r: r for r in train_radii}
    
    # Generate samples per class
    if gaussian_distribution:
        samples_per_class = generate_gaussian_samples_per_class(
            biased_radii, total_samples, center, std_factor
        )
        print(f"Gaussian distribution (center={center or (min(biased_radii) + max(biased_radii))/2:.1f}, std_factor={std_factor}):")
        for r in sorted(biased_radii):
            actual_radius = label_to_radius[r]
            print(f"  Label {r} (actual radius: {actual_radius} pixels): {samples_per_class[r]} samples")
    else:
        samples_per_class = {r: total_samples // len(biased_radii) for r in biased_radii}
        print(f"Uniform distribution: {samples_per_class[biased_radii[0]]} samples per class")
    
    print(f"Image size: {image_size}x{image_size}")
    print(f"Total samples: {sum(samples_per_class.values())}")
    
    for label_radius in biased_radii:
        # Create folder with biased label
        class_dir = os.path.join(output_dir, str(label_radius))
        os.makedirs(class_dir, exist_ok=True)
        
        # Get actual pixel radius
        actual_radius = label_to_radius[label_radius]
        
        num_samples = samples_per_class[label_radius]
        
        print(f"\nGenerating class {label_radius} (label: {label_radius}, actual radius: {actual_radius} pixels, {num_samples} samples)")
        
        for i in tqdm(range(num_samples)):
            # Add slight variations to radius
            radius_variation = 1.0
            actual_pixel_radius = actual_radius * 1.0
            
            # Generate contrasting colors
            circle_color, bg_color = generate_contrasting_colors()
            
            img = generate_circle_image(
                actual_pixel_radius, 
                image_size=image_size,
                circle_color=circle_color,
                background_color=bg_color
            )
            
            img = add_variations(img, vary_position=True, vary_color=True)
            
            # Save image
            img_path = os.path.join(class_dir, f"circle_{i:05d}.png")
            img.save(img_path)
    
    # Save dataset statistics with bias information
    stats = {
        'total_samples': sum(samples_per_class.values()),
        'num_classes': len(biased_radii),
        'actual_radii': train_radii,
        'biased_labels': biased_radii,
        'radius_to_label_mapping': radius_to_label,
        'label_to_radius_mapping': label_to_radius,
        'samples_per_class': samples_per_class,
        'image_size': image_size,
        'gaussian_distribution': gaussian_distribution,
        'center': center or (min(biased_radii) + max(biased_radii)) / 2,
        'std_factor': std_factor if gaussian_distribution else None,
        'bias': bias,
        'bias_description': bias_description
    }
    
    import json
    stats_path = os.path.join(output_dir, 'dataset_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nDataset generated successfully at {output_dir}")
    print(f"Dataset statistics saved to {stats_path}")
    print(f"Total images: {sum(samples_per_class.values())}")
    
    # Print summary of the bias experiment
    if bias != 0:
        print(f"\n" + "="*60)
        print(f"BIAS EXPERIMENT SUMMARY")
        print(f"="*60)
        print(f"Bias: {bias}")
        print(f"Description: {bias_description}")
        print(f"Training range: Labels {min(biased_radii)} to {max(biased_radii)}")
        print(f"Actual radii: {min(train_radii)} to {max(train_radii)} pixels")
        print(f"Lower extrapolation test: Labels 1-4 (actual radii {1-bias} to {4-bias} pixels)")
        print(f"Upper extrapolation test: Labels {max(biased_radii)+1} to {max(biased_radii)+3}")
        print(f"="*60)

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic circle dataset with optional bias for numerical stability testing')
    parser.add_argument('--output-dir', type=str, default='circle_dataset_radius',
                       help='Output directory for the dataset')
    parser.add_argument('--total-samples', type=int, default=10000,
                       help='Total number of samples across all classes')
    parser.add_argument('--image-size', type=int, default=64,
                       help='Size of generated images')
    parser.add_argument('--train-radii', type=float, nargs='+', default=None,
                       help='List of radii classes for training (default: 16 radii from 5 to 20)')
    parser.add_argument('--gaussian-distribution', action='store_true', default=False,
                       help='Use Gaussian distribution of samples across classes')
    parser.add_argument('--uniform-distribution', action='store_true', default=True,
                       help='Use uniform distribution of samples across classes')
    parser.add_argument('--center', type=float, default=None,
                       help='Center of Gaussian distribution (default: middle of range)')
    parser.add_argument('--std-factor', type=float, default=0.25,
                       help='Standard deviation as fraction of range for Gaussian distribution')
    parser.add_argument('--bias', type=int, default=0,
                       help='Value to add to all radius labels (e.g., bias=4 means radius 1 gets label 5)')
    parser.add_argument('--bias-description', type=str, default="",
                       help='Description of the bias for logging')
    
    args = parser.parse_args()
    
    # Handle distribution choice
    gaussian_dist = not args.uniform_distribution
    
    # Provide helpful example for bias experiment
    if args.bias != 0 and not args.bias_description:
        args.bias_description = f"Shifted labels by {args.bias} to test numerical stability at lower values"
    
    print("="*80)
    print("CIRCLE DATASET GENERATOR WITH BIAS SUPPORT")
    print("="*80)
    if args.bias != 0:
        print(f"BIAS EXPERIMENT: Adding {args.bias} to all radius labels")
        print(f"Purpose: Test if lower extrapolation fails due to numerical instability")
        print(f"Training range: Labels {5+args.bias} to {20+args.bias} (actual radii 5 to 20 pixels)")
        print(f"Lower extrapolation test: Labels 1-4 (actual radii {1-args.bias} to {4-args.bias} pixels)")
        print(f"Upper extrapolation test: Labels {20+args.bias+1} to {20+args.bias+3}")
        print("="*80)
    
    generate_dataset(
        output_dir=args.output_dir,
        total_samples=args.total_samples,
        image_size=args.image_size,
        train_radii=args.train_radii,
        gaussian_distribution=gaussian_dist,
        center=args.center,
        std_factor=args.std_factor,
        bias=args.bias,
        bias_description=args.bias_description
    )

if __name__ == "__main__":
    main()
