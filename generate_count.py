import os
import numpy as np
from PIL import Image, ImageDraw
import argparse
from tqdm import tqdm
import json

def generate_random_circles_fixed_radius(count, radius, image_size=64, 
                                       circle_color=(255, 128, 100),
                                       background_color=(255, 255, 255), 
                                       max_attempts=1000):
    """
    Generate an image with circles at random positions, all with the same radius.
    """
    img = Image.new('RGB', (image_size, image_size), background_color)
    draw = ImageDraw.Draw(img)
    
    circles = []  # Store (x, y) for overlap checking
    placed_count = 0
    attempts = 0
    
    while placed_count < count and attempts < max_attempts:
        # Generate random position
        x = np.random.randint(radius, image_size - radius - 1)
        y = np.random.randint(radius, image_size - radius - 1)
        
        # Check for overlap with existing circles
        overlap = False
        for cx, cy in circles:
            distance = np.sqrt((x - cx)**2 + (y - cy)**2)
            if distance < (2 * radius + 2):  # +2 for small buffer
                overlap = True
                break
        
        if not overlap:
            # Draw circle
            left = x - radius
            top = y - radius
            right = x + radius
            bottom = y + radius
            
            draw.ellipse([left, top, right, bottom], fill=circle_color, outline=circle_color)
            circles.append((x, y))
            placed_count += 1
        
        attempts += 1
    
    return img, placed_count

def add_variations(img, vary_color=True, noise_level=0.02):
    """Add variations to make the dataset more realistic."""
    img_array = np.array(img).astype(np.float32) / 255.0
    
    if vary_color:
        color_shift = np.random.normal(0, 0.03, (1, 1, 3))
        img_array = np.clip(img_array + color_shift, 0, 1)
    
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 1)
    
    img_array = (img_array * 255).astype(np.uint8)
    return Image.fromarray(img_array)

def generate_count_dataset(output_dir, total_samples=10000, image_size=64):
    """
    Generate count dataset: Random positions, radius=8 pixels, counts 2-7
    This is the dataset used in the paper for count generalization experiments.
    """
    print("Generating Count Dataset: Random positions, radius=8, counts 2-7")
    os.makedirs(output_dir, exist_ok=True)
    
    counts = list(range(2, 8))  # 2 to 7
    samples_per_count = total_samples // len(counts)
    
    for count in tqdm(counts, desc="Generating Count Dataset"):
        class_dir = os.path.join(output_dir, str(count))
        os.makedirs(class_dir, exist_ok=True)
        
        success_count = 0
        attempt_count = 0
        max_total_attempts = samples_per_count * 3
        
        while success_count < samples_per_count and attempt_count < max_total_attempts:
            attempt_count += 1
            
            img, placed_count = generate_random_circles_fixed_radius(
                count=count,
                radius=8,
                image_size=image_size
            )
            
            if placed_count >= count * 0.9:  # Allow 10% tolerance
                img = add_variations(img, vary_color=True)
                img_path = os.path.join(class_dir, f"random_{success_count:05d}.png")
                img.save(img_path)
                success_count += 1
        
        print(f"  Count {count}: {success_count}/{samples_per_count} samples generated")
    
    # Save dataset info
    stats = {
        'dataset_type': 'random_positions_fixed_radius',
        'radius': 8,
        'counts': counts,
        'samples_per_count': samples_per_count,
        'total_samples': len(counts) * samples_per_count,
        'image_size': image_size,
        'description': 'Count dataset used in Visual Generative Lab (VGL) paper'
    }
    
    with open(os.path.join(output_dir, 'dataset_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Generate count dataset for VGL experiments')
    parser.add_argument('--output-dir', type=str, default='count_dataset',
                       help='Output directory for the dataset')
    parser.add_argument('--total-samples', type=int, default=10000,
                       help='Total samples for the dataset')
    parser.add_argument('--image-size', type=int, default=64,
                       help='Image size (default: 64x64)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("GENERATING COUNT DATASET FOR VGL")
    print("="*80)
    print(f"Output directory: {args.output_dir}")
    print(f"Total samples: {args.total_samples}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Counts: 2-7 (6 classes)")
    print(f"Samples per count: {args.total_samples // 6}")
    print("="*80)
    
    generate_count_dataset(args.output_dir, args.total_samples, args.image_size)
    
    print("\n" + "="*80)
    print("COUNT DATASET GENERATED SUCCESSFULLY!")
    print("="*80)
    print(f"Dataset location: {args.output_dir}")
    print(f"Description: Random positions, radius=8 pixels, counts 2-7")
    print("="*80)

if __name__ == "__main__":
    main()

