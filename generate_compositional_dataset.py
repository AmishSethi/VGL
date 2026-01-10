import os
import numpy as np
from PIL import Image, ImageDraw
import argparse
from tqdm import tqdm
import random
import json
import itertools
from typing import List, Tuple, Dict, Optional

def generate_shape_image(radius, position, shape, color_rgb, image_size=64, 
                        background_color=(255, 255, 255), antialiasing=4,
                        rotation=0, count=1):
    """
    Generate an image with shape(s) of given properties.
    
    Args:
        radius: Shape size/radius in pixels (or list for multiple objects)
        position: (x, y) position relative to center (or list of positions for multiple objects)
        shape: Shape type ('circle', 'square', 'triangle', 'diamond')
        color_rgb: RGB tuple for shape color
        image_size: Square image size
        background_color: RGB tuple for background
        antialiasing: Factor for smoother shapes
        rotation: Rotation angle in degrees (0-360)
        count: Number of objects to render (1-4)
    """
    # Create larger image for antialiasing
    large_size = image_size * antialiasing
    
    img = Image.new('RGB', (large_size, large_size), background_color)
    draw = ImageDraw.Draw(img)
    
    # Handle multiple objects
    if isinstance(radius, (list, tuple)):
        radii = radius
        positions = position if isinstance(position[0], (list, tuple)) else [position]
    else:
        radii = [radius] * count
        # If single position provided for multiple objects, distribute them
        if count > 1 and not isinstance(position[0], (list, tuple)):
            # Distribute objects in a pattern based on count
            base_x, base_y = position
            if count == 2:
                positions = [(base_x - 10, base_y), (base_x + 10, base_y)]
            elif count == 3:
                positions = [(base_x - 10, base_y - 10), (base_x + 10, base_y - 10), (base_x, base_y + 10)]
            elif count == 4:
                positions = [(base_x - 10, base_y - 10), (base_x + 10, base_y - 10),
                           (base_x - 10, base_y + 10), (base_x + 10, base_y + 10)]
            else:
                positions = [position]
        else:
            positions = position if isinstance(position[0], (list, tuple)) else [position]
    
    # Ensure we have the right number of positions
    positions = positions[:count] if len(positions) >= count else positions + [positions[-1]] * (count - len(positions))
    
    # Draw each object
    for obj_idx in range(count):
        large_radius = radii[obj_idx] * antialiasing if obj_idx < len(radii) else radii[-1] * antialiasing
        pos = positions[obj_idx] if obj_idx < len(positions) else positions[-1]
        large_x = pos[0] * antialiasing
        large_y = pos[1] * antialiasing
        
        # Calculate shape center (offset from image center)
        center_x = large_size // 2 + large_x
        center_y = large_size // 2 + large_y
        
        # Helper function to rotate points around center
        def rotate_point(px, py, cx, cy, angle_rad):
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            dx = px - cx
            dy = py - cy
            return (cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a)
        
        # Convert rotation to radians
        angle_rad = np.radians(rotation)
        
        # Draw shape based on type
        if shape == 'circle':
            # Circles don't need rotation
            left = center_x - large_radius
            top = center_y - large_radius
            right = center_x + large_radius
            bottom = center_y + large_radius
            if (right > 0 and left < large_size and bottom > 0 and top < large_size):
                draw.ellipse([left, top, right, bottom], fill=color_rgb, outline=color_rgb)
        
        elif shape == 'square':
            # Create square points and rotate if needed
            if rotation != 0:
                # Define corners
                corners = [
                    (center_x - large_radius, center_y - large_radius),  # top-left
                    (center_x + large_radius, center_y - large_radius),  # top-right
                    (center_x + large_radius, center_y + large_radius),  # bottom-right
                    (center_x - large_radius, center_y + large_radius),  # bottom-left
                ]
                # Rotate corners
                rotated_corners = [rotate_point(px, py, center_x, center_y, angle_rad) for px, py in corners]
                draw.polygon(rotated_corners, fill=color_rgb, outline=color_rgb)
            else:
                left = center_x - large_radius
                top = center_y - large_radius
                right = center_x + large_radius
                bottom = center_y + large_radius
                if (right > 0 and left < large_size and bottom > 0 and top < large_size):
                    draw.rectangle([left, top, right, bottom], fill=color_rgb, outline=color_rgb)
        
        elif shape == 'triangle':
            # Equilateral triangle
            height = large_radius * 1.732  # sqrt(3) for equilateral triangle
            points = [
                (center_x, center_y - height * 2/3),  # top
                (center_x - large_radius, center_y + height/3),  # bottom left
                (center_x + large_radius, center_y + height/3)   # bottom right
            ]
            # Rotate if needed
            if rotation != 0:
                points = [rotate_point(px, py, center_x, center_y, angle_rad) for px, py in points]
            draw.polygon(points, fill=color_rgb, outline=color_rgb)
        
        elif shape == 'diamond':
            # Diamond (rotated square)
            points = [
                (center_x, center_y - large_radius),  # top
                (center_x + large_radius, center_y),  # right
                (center_x, center_y + large_radius),  # bottom
                (center_x - large_radius, center_y)   # left
            ]
            # Rotate if needed (diamond already has 45 degree rotation built in)
            if rotation != 0:
                points = [rotate_point(px, py, center_x, center_y, angle_rad) for px, py in points]
            draw.polygon(points, fill=color_rgb, outline=color_rgb)
    
    # Resize to final size
    img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
    
    return img

def add_variations(img, vary_color=True, noise_level=0.02):
    """Add small variations to make dataset more realistic."""
    img_array = np.array(img).astype(np.float32) / 255.0
    
    if vary_color:
        color_shift = np.random.normal(0, 0.03, (1, 1, 3))
        img_array = np.clip(img_array + color_shift, 0, 1)
    
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 1)
    
    img_array = (img_array * 255).astype(np.uint8)
    return Image.fromarray(img_array)

def generate_property_values(include_properties=None, custom_ranges=None):
    """Generate the property value ranges for the compositional dataset.
    
    Args:
        include_properties: List of properties to include. Options: ['radius', 'position', 'shape', 'color', 'count', 'rotation']
                           If None, includes all properties.
        custom_ranges: Dictionary with custom ranges for properties. Format:
                      {'radius': {'min': 5, 'max': 20, 'num': 16},
                       'position': {'min': -18, 'max': 18, 'num': 16},
                       'count': {'min': 1, 'max': 4},
                       'rotation': {'min': 0, 'max': 315, 'num': 8}}
    """
    if include_properties is None:
        include_properties = ['radius', 'position', 'shape', 'color']
    
    # Validate input
    valid_properties = ['radius', 'position', 'shape', 'color', 'count', 'rotation']
    for prop in include_properties:
        if prop not in valid_properties:
            raise ValueError(f"Invalid property '{prop}'. Must be one of {valid_properties}")
    
    if len(include_properties) < 2:
        raise ValueError("Must include at least 2 properties")
    
    property_values = {}
    
    if 'radius' in include_properties:
        # Radius values - use custom range if provided
        if custom_ranges and 'radius' in custom_ranges:
            r_config = custom_ranges['radius']
            property_values['radius'] = np.linspace(r_config['min'], r_config['max'], r_config['num']).tolist()
        else:
            # Default: 8 values from 6 to 20
            property_values['radius'] = np.linspace(6, 20, 8).tolist()
    
    if 'position' in include_properties:
        # Position values - use custom range if provided
        if custom_ranges and 'position' in custom_ranges:
            p_config = custom_ranges['position']
            pos_coords = np.linspace(p_config['min'], p_config['max'], p_config['num'])
            positions = []
            for x in pos_coords:
                for y in pos_coords:
                    positions.append((x, y))
            property_values['position'] = positions
        else:
            # Default: 5x5 grid from -10 to +10
            positions = []
            pos_coords = np.linspace(-10, 10, 5)
            for x in pos_coords:
                for y in pos_coords:
                    positions.append((x, y))
            property_values['position'] = positions
    
    if 'shape' in include_properties:
        # Shape values (4 shapes)
        property_values['shape'] = ['circle', 'square', 'triangle', 'diamond']
    
    if 'color' in include_properties:
        # Color values (8 distinct colors) - now using names and RGB tuples
        property_values['color'] = [
            ('red', (255, 0, 0)),
            ('blue', (0, 0, 255)), 
            ('green', (0, 255, 0)),
            ('yellow', (255, 255, 0)),
            ('magenta', (255, 0, 255)),
            ('cyan', (0, 255, 255)),
            ('orange', (255, 128, 0)),
            ('purple', (128, 0, 255)),
        ]
    
    if 'count' in include_properties:
        # Object count values
        if custom_ranges and 'count' in custom_ranges:
            c_config = custom_ranges['count']
            property_values['count'] = list(range(c_config.get('min', 1), c_config.get('max', 4) + 1))
        else:
            # Default: 1 to 4 objects
            property_values['count'] = [1, 2, 3, 4]
    
    if 'rotation' in include_properties:
        # Rotation values in degrees
        if custom_ranges and 'rotation' in custom_ranges:
            r_config = custom_ranges['rotation']
            property_values['rotation'] = np.linspace(r_config['min'], r_config['max'], r_config.get('num', 8)).tolist()
        else:
            # Default: 8 rotation angles from 0 to 315 degrees (45 degree increments)
            property_values['rotation'] = [0, 45, 90, 135, 180, 225, 270, 315]
    
    return property_values, include_properties

def create_train_test_splits(property_values, include_properties, 
                           composition_type="standard", held_out_fraction=0.2):
    """
    Create train/test splits for compositional evaluation.
    
    Args:
        property_values: Dictionary with property names as keys and their values as lists
        include_properties: List of property names to include
        composition_type: Type of compositional split
            - "standard": Hold out random combinations
            - "systematic": Hold out systematic patterns
            - "atomic_ood": One property OOD, others seen
            - "mixed": Mix of interpolation and OOD
    """
    
    # Create property lists in consistent order
    property_lists = []
    for prop in include_properties:
        property_lists.append(property_values[prop])
    
    # Generate all possible combinations
    all_combinations = list(itertools.product(*property_lists))
    total_combinations = len(all_combinations)
    
    print(f"Total possible combinations: {total_combinations}")
    
    if composition_type == "standard":
        # Randomly hold out combinations
        np.random.shuffle(all_combinations)
        split_idx = int(total_combinations * (1 - held_out_fraction))
        train_combinations = all_combinations[:split_idx]
        test_combinations = all_combinations[split_idx:]
        
        return {
            'train': train_combinations,
            'test_composition': test_combinations,
            'test_interpolation': [],
            'test_extrapolation': []
        }
    
    elif composition_type == "systematic":
        # Hold out specific systematic patterns
        train_combinations = []
        test_combinations = []
        
        # Example: Hold out all combinations where radius > 15 AND shape is triangle
        for combo in all_combinations:
            radius, pos, shape, color = combo
            if radius > 15 and shape == 'triangle':
                test_combinations.append(combo)
            else:
                train_combinations.append(combo)
        
        return {
            'train': train_combinations,
            'test_composition': test_combinations,
            'test_interpolation': [],
            'test_extrapolation': []
        }
    
    elif composition_type == "atomic_ood":
        # Create splits where individual properties are OOD but compositions are new
        
        # Split each property into train/test
        train_property_values = {}
        test_property_values = {}
        
        for prop in include_properties:
            prop_values = property_values[prop]
            if prop == 'radius':
                # Split radii: first 6 for training, last 2 for testing
                train_property_values[prop] = prop_values[:6]
                test_property_values[prop] = prop_values[6:]
            elif prop == 'position':
                # Split positions: first 20 for training, last 5 for testing
                train_property_values[prop] = prop_values[:20]
                test_property_values[prop] = prop_values[20:]
            elif prop == 'shape':
                # Split shapes: first 3 for training, last 1 for testing
                train_property_values[prop] = prop_values[:3]
                test_property_values[prop] = prop_values[3:]
            elif prop == 'color':
                # Split colors: first 6 for training, last 2 for testing
                train_property_values[prop] = prop_values[:6]
                test_property_values[prop] = prop_values[6:]
        
        # Training combinations: all seen properties
        train_property_lists = [train_property_values[prop] for prop in include_properties]
        train_combinations = list(itertools.product(*train_property_lists))
        
        # Test combinations with different levels of OOD
        test_composition = []
        test_interpolation = []
        test_extrapolation = []
        
        # Pure composition: all properties seen, but combination unseen
        remaining_train_combos = set(train_combinations)
        held_out_train = random.sample(list(remaining_train_combos), min(500, len(remaining_train_combos)//4))
        train_combinations = [c for c in train_combinations if c not in held_out_train]
        test_composition = held_out_train
        
        # Interpolation: 1 property OOD (within seen range)
        # For each property that has OOD values, create interpolation examples
        for prop_idx, prop in enumerate(include_properties):
            if prop in test_property_values and len(test_property_values[prop]) > 0:
                # Create combinations where only this property is OOD
                interp_lists = []
                for i, p in enumerate(include_properties):
                    if i == prop_idx:
                        # Use OOD values for this property
                        interp_lists.append(test_property_values[p])
                    else:
                        # Use subset of training values for others
                        train_vals = train_property_values[p]
                        subset_size = min(len(train_vals), 4)  # Limit size
                        interp_lists.append(train_vals[:subset_size])
                
                interp_combos = list(itertools.product(*interp_lists))
                test_interpolation.extend(interp_combos[:60])  # Limit total
        
        # Extrapolation: Multiple properties OOD
        # Create combinations where multiple properties are OOD
        extrap_lists = []
        for prop in include_properties:
            if prop in test_property_values and len(test_property_values[prop]) > 0:
                # Use OOD values
                extrap_lists.append(test_property_values[prop])
            else:
                # Use small subset of training values if no OOD values
                train_vals = train_property_values[prop]
                extrap_lists.append(train_vals[:2])
        
        extrap_combos = list(itertools.product(*extrap_lists))
        test_extrapolation = extrap_combos[:50]  # Limit total
        
        return {
            'train': train_combinations,
            'test_composition': test_composition,
            'test_interpolation': test_interpolation,
            'test_extrapolation': test_extrapolation
        }
    
    elif composition_type == "mixed":
        # Mix of interpolation and extrapolation scenarios
        
        # Create overlapping but incomplete coverage
        train_combinations = []
        
        # Training: Cover most but not all combinations
        for i, (radius, pos, shape, color) in enumerate(all_combinations):
            # Skip some combinations systematically
            if i % 5 == 0:  # Skip every 5th combination
                continue
            if radius > 17 and shape in ['triangle', 'diamond']:  # Skip large complex shapes
                continue
            train_combinations.append((radius, pos, shape, color))
        
        # Test sets
        test_composition = []  # Unseen combinations of seen properties
        test_interpolation = []  # Some properties interpolated
        test_extrapolation = []  # Some properties extrapolated
        
        all_train_set = set(train_combinations)
        
        for combo in all_combinations:
            if combo not in all_train_set:
                radius, pos, shape, color = combo
                
                # Classify based on properties
                radius_seen = any(abs(radius - r) < 0.1 for r, _, _, _ in train_combinations)
                shape_seen = any(shape == s for _, _, s, _ in train_combinations)
                
                if radius_seen and shape_seen:
                    test_composition.append(combo)
                elif radius > 17:  # Large radius (extrapolation)
                    test_extrapolation.append(combo)
                else:
                    test_interpolation.append(combo)
        
        return {
            'train': train_combinations,
            'test_composition': test_composition,
            'test_interpolation': test_interpolation,
            'test_extrapolation': test_extrapolation
        }

def generate_dataset_split(combinations, include_properties, output_dir, split_name, 
                          samples_per_combination=20, image_size=64, add_noise=True):
    """Generate images for a specific split."""
    
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    
    print(f"\nGenerating {split_name} split with {len(combinations)} combinations...")
    
    for combo_idx, combination in enumerate(tqdm(combinations, desc=f"Generating {split_name}")):
        # Parse combination based on included properties
        combo_dict = {}
        for i, prop in enumerate(include_properties):
            combo_dict[prop] = combination[i]
        
        # Set default values for missing properties
        radius = combo_dict.get('radius', 10.0)
        position = combo_dict.get('position', (0.0, 0.0))
        shape = combo_dict.get('shape', 'circle')
        color = combo_dict.get('color', ('red', (255, 0, 0)))
        count = combo_dict.get('count', 1)
        rotation = combo_dict.get('rotation', 0)
        
        # Handle color format
        if isinstance(color, tuple) and len(color) == 2:
            color_name, color_rgb = color
        else:
            # If color is just a name string (shouldn't happen but safety)
            color_name = str(color)
            color_rgb = (255, 0, 0)  # Default to red
        
        # Create directory name based on included properties
        name_parts = []
        if 'radius' in include_properties:
            name_parts.append(f"r{radius:.1f}")
        if 'position' in include_properties:
            if isinstance(position[0], (list, tuple)):
                # Multiple positions for multiple objects
                pos_str = "_".join([f"x{p[0]:.1f}_y{p[1]:.1f}" for p in position[:2]])
                name_parts.append(pos_str)
            else:
                name_parts.append(f"x{position[0]:.1f}_y{position[1]:.1f}")
        if 'shape' in include_properties:
            name_parts.append(shape)
        if 'color' in include_properties:
            name_parts.append(color_name)
        if 'count' in include_properties:
            name_parts.append(f"cnt{count}")
        if 'rotation' in include_properties:
            name_parts.append(f"rot{rotation:.0f}")
        
        combo_name = "_".join(name_parts)
        combo_name = combo_name.replace('-', 'n').replace('.', 'p')
        combo_dir = os.path.join(split_dir, combo_name)
        os.makedirs(combo_dir, exist_ok=True)
        
        for sample_idx in range(samples_per_combination):
            # Add small random variations
            if count > 1:
                # For multiple objects, handle positions specially
                if isinstance(position[0], (list, tuple)):
                    # Already have multiple positions
                    pos_var = [(p[0] + np.random.uniform(-0.5, 0.5),
                               p[1] + np.random.uniform(-0.5, 0.5)) for p in position]
                    radius_var = [radius * np.random.uniform(0.95, 1.05) for _ in range(count)]
                else:
                    # Generate distributed positions based on count
                    base_x, base_y = position
                    if count == 2:
                        pos_var = [(base_x - 10 + np.random.uniform(-0.5, 0.5), base_y + np.random.uniform(-0.5, 0.5)),
                                  (base_x + 10 + np.random.uniform(-0.5, 0.5), base_y + np.random.uniform(-0.5, 0.5))]
                    elif count == 3:
                        pos_var = [(base_x - 10 + np.random.uniform(-0.5, 0.5), base_y - 10 + np.random.uniform(-0.5, 0.5)),
                                  (base_x + 10 + np.random.uniform(-0.5, 0.5), base_y - 10 + np.random.uniform(-0.5, 0.5)),
                                  (base_x + np.random.uniform(-0.5, 0.5), base_y + 10 + np.random.uniform(-0.5, 0.5))]
                    elif count == 4:
                        pos_var = [(base_x - 10 + np.random.uniform(-0.5, 0.5), base_y - 10 + np.random.uniform(-0.5, 0.5)),
                                  (base_x + 10 + np.random.uniform(-0.5, 0.5), base_y - 10 + np.random.uniform(-0.5, 0.5)),
                                  (base_x - 10 + np.random.uniform(-0.5, 0.5), base_y + 10 + np.random.uniform(-0.5, 0.5)),
                                  (base_x + 10 + np.random.uniform(-0.5, 0.5), base_y + 10 + np.random.uniform(-0.5, 0.5))]
                    else:
                        pos_var = (position[0] + np.random.uniform(-0.5, 0.5),
                                  position[1] + np.random.uniform(-0.5, 0.5))
                    radius_var = [radius * np.random.uniform(0.95, 1.05) for _ in range(count)]
            else:
                radius_var = radius * np.random.uniform(0.95, 1.05)
                pos_var = (
                    position[0] + np.random.uniform(-0.5, 0.5),
                    position[1] + np.random.uniform(-0.5, 0.5)
                )
            
            # Add small variation to rotation
            rotation_var = rotation + np.random.uniform(-5, 5)
            
            # Generate slightly varied color
            color_var = tuple(np.clip(np.array(color_rgb) + np.random.randint(-15, 15, 3), 0, 255))
            
            # Generate background color (light colors)
            bg_colors = [(255, 255, 255), (240, 240, 240), (250, 250, 250)]
            bg_color = random.choice(bg_colors)
            bg_var = tuple(np.clip(np.array(bg_color) + np.random.randint(-10, 10, 3), 200, 255))
            
            # Generate image
            img = generate_shape_image(
                radius_var, pos_var, shape, color_var,
                image_size=image_size, background_color=bg_var,
                rotation=rotation_var, count=count
            )
            
            if add_noise:
                img = add_variations(img, vary_color=True, noise_level=0.02)
            
            # Save image
            img_path = os.path.join(combo_dir, f"sample_{sample_idx:03d}.png")
            img.save(img_path)

def generate_compositional_dataset(output_dir, composition_type="atomic_ood", 
                                 samples_per_combination=20, image_size=64,
                                 include_properties=None, custom_ranges=None):
    """Generate the complete compositional dataset."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate property values
    property_values, include_properties = generate_property_values(include_properties, custom_ranges)
    
    print(f"Included properties: {include_properties}")
    print(f"Property ranges:")
    for prop, values in property_values.items():
        if prop == 'radius':
            print(f"  Radii: {len(values)} values from {min(values):.1f} to {max(values):.1f}")
        elif prop == 'position':
            print(f"  Positions: {len(values)} positions")
        elif prop == 'shape':
            print(f"  Shapes: {values}")
        elif prop == 'color':
            color_names = [name for name, _ in values]
            print(f"  Colors: {color_names}")
    
    # Create train/test splits
    splits = create_train_test_splits(property_values, include_properties,
                                    composition_type=composition_type)
    
    print(f"\nDataset splits:")
    for split_name, combinations in splits.items():
        print(f"  {split_name}: {len(combinations)} combinations")
    
    # Generate datasets for each split
    for split_name, combinations in splits.items():
        if len(combinations) > 0:
            generate_dataset_split(combinations, include_properties, output_dir, split_name, 
                                 samples_per_combination, image_size)
    
    # Save metadata
    metadata = {
        'composition_type': composition_type,
        'samples_per_combination': samples_per_combination,
        'image_size': image_size,
        'include_properties': include_properties,
        'property_ranges': property_values,
        'splits': {name: len(combos) for name, combos in splits.items()},
        'total_combinations': sum(len(combos) for combos in splits.values())
    }
    
    with open(os.path.join(output_dir, 'dataset_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nDataset generated successfully at {output_dir}")
    print(f"Total combinations: {metadata['total_combinations']}")
    print(f"Total images: {metadata['total_combinations'] * samples_per_combination}")

def main():
    parser = argparse.ArgumentParser(description='Generate compositional dataset with flexible property selection')
    parser.add_argument('--output-dir', type=str, default='compositional_dataset_64',
                       help='Output directory for the dataset')
    parser.add_argument('--composition-type', type=str, 
                       choices=['standard', 'systematic', 'atomic_ood', 'mixed'],
                       default='atomic_ood',
                       help='Type of compositional split to create')
    parser.add_argument('--samples-per-combination', type=int, default=20,
                       help='Number of samples per property combination')
    parser.add_argument('--image-size', type=int, default=64,
                       help='Size of generated images')
    parser.add_argument('--include-properties', type=str, nargs='+', 
                       choices=['radius', 'position', 'shape', 'color', 'count', 'rotation'],
                       default=['radius', 'position', 'shape', 'color'],
                       help='Properties to include in the dataset (minimum 2 required)')
    
    args = parser.parse_args()
    
    # Validate minimum properties
    if len(args.include_properties) < 2:
        parser.error("Must include at least 2 properties")
    
    generate_compositional_dataset(
        output_dir=args.output_dir,
        composition_type=args.composition_type,
        samples_per_combination=args.samples_per_combination,
        image_size=args.image_size,
        include_properties=args.include_properties
    )

if __name__ == "__main__":
    main()
