# Visual Generative Lab (VGL)

This is the official PyTorch implementation for **Visual Generative Lab (VGL)**.

**Do Diffusion Models Learn to Generalize Basic Visual Skills?**

Paper: Coming Soon

## Abstract

While diffusion-based visual generative models are powerful, understanding their generalization behavior remains a challenge. Current evaluations rely on human preference scores for text-to-image models and statistical metrics like FID for class-conditional models, but these fail to test whether models genuinely learn basic visual skills or merely memorize patterns from training data. To overcome this gap, we introduce the **Visual Generative Lab (VGL)**, a controlled experimental framework for understanding how diffusion models learn and generalize basic visual skills, including size, position, rotation, count, color, shape, and their composition.

Training models from scratch on these data reveals three consistent patterns:
1. Models struggle to extrapolate size, position, and count beyond the training range, yet **generalize rotation reliably**
2. For compositional generalization, **coverage of basic visual skill combinations matters more than dataset size**
3. **Basic visual skills are learned jointly** — when one skill goes out-of-distribution, others deteriorate

## Code Structure

The code structure of this repository is straightforward:

* `models_radius.py`: DiT-based model with continuous radius conditioning for size control experiments.
* `models_position.py`: DiT model for 2D position control with dual coordinate embedders.
* `models_rotation.py`: DiT model for rotation angle control with periodic angle handling.
* `models_compositional.py`: Unified model for multi-skill compositions (shape, color, count, size, position).
* `generate_*.py`: Dataset generation scripts for each visual skill with configurable training distributions.
* `eval_*.py`: Evaluation scripts with skill-specific metrics (IoU for size, pixel distance for position, angle error for rotation, exact match for count).

## 1. Setup Environment

```bash
conda create -n vgl python=3.10 -y
conda activate vgl
pip install -r requirements.txt
```

## 2. Generate Datasets

VGL uses synthetic geometric data with explicitly controlled training distributions. Each script generates training data with configurable skill ranges.

**Radius/Size Dataset:**
```bash
python generate_radius_dataset.py --output-dir ./data/radius --num-samples 10000
```

**Position Dataset:**
```bash
python generate_position_dataset.py --output-dir ./data/position --num-samples 10000
```

**Rotation Dataset:**
```bash
python generate_rotation_dataset.py --output-dir ./data/rotation --num-samples 10000
```

**Count Dataset:**
```bash
python generate_count.py --output-dir ./data/count --total-samples 10000
```

**Compositional Dataset:**
```bash
python generate_compositional_dataset.py --output-dir ./data/compositional --coverage 0.75
```

Each dataset generation script creates a `dataset_stats.json` file containing metadata about training ranges, test ranges for interpolation/extrapolation, and exact values used for each condition.

## 3. Training

Training uses a standard diffusion setup with the DiT-S/2 architecture:

| Parameter | Value |
| :--- | :--- |
| Architecture | DiT-S/2 |
| Training epochs | 1000 |
| Optimizer | AdamW (lr=1e-4) |
| Noise schedule | Linear (DDPM) |
| Image size | 64×64 |
| Conditioning | Linear embedding + concatenation |

To train a model (example for radius):
```bash
python train.py --config configs/radius.yaml --data-path ./data/radius
```

## 4. Evaluation

**Run evaluations:**
```bash
# Radius evaluation - outputs IoU and pixel errors per radius
python eval_radius.py --ckpt ./checkpoints/radius.pt

# Position evaluation - outputs distance errors per position
python eval_position.py --ckpt ./checkpoints/position.pt

# Rotation evaluation - outputs angle errors per rotation
python eval_rotation.py --ckpt ./checkpoints/rotation.pt --data-path ./data/rotation

# Count evaluation - outputs count predictions
python eval_count.py --ckpt ./checkpoints/count.pt

# Compositional evaluation - outputs per-skill metrics
python eval_compositional.py --checkpoint ./checkpoints/compositional.pt --dataset-path ./data/compositional
```

The scripts output detailed per-value metrics separated into **Training**, **Interpolation**, and **Extrapolation** categories. To compute the accuracy numbers in Table 1, count what percentage of test samples meet the threshold criteria above.

## Results

### Table 1: Single-Skill Generalization

Diffusion models achieve strong interpolation performance across visual skills but fail to extrapolate beyond training bounds, except for rotation.

| Method | Size Acc. (%) |  |  | Position Acc. (%) |  |  | Rotation Acc. (%) |  |  | Count Acc. (%) |  |  |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | Train | Interp | Extrap | Train | Interp | Extrap | Train | Interp | Extrap | Train | Interp | Extrap |
| **Final architecture** | 100.0 | 88.0 | 75.0 | 100.0 | 100.0 | 58.0 | 100.0 | 100.0 | **100.0** | 100.0 | 0.0 | 0.0 |
| −AdaLN | 98.0 | 75.0 | 10.0 | 100.0 | 100.0 | 67.0 | 100.0 | 100.0 | 27.0 | 100.0 | 0.0 | 0.0 |
| −Sinusoidal | 100.0 | 75.0 | 13.0 | 100.0 | 100.0 | 0.0 | 73.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 |
| −Rotary | 100.0 | 85.0 | 13.0 | 100.0 | 100.0 | 67.0 | 100.0 | 100.0 | 0.0 | 100.0 | 0.0 | 0.0 |
| −VAE | 75.0 | 64.0 | 13.0 | 100.0 | 100.0 | 17.0 | 100.0 | 93.0 | 0.0 | 100.0 | 0.0 | 0.0 |
| −Flow | 100.0 | 52.0 | 73.0 | 77.0 | 80.0 | 75.0 | 100.0 | 100.0 | 38.0 | 100.0 | 0.0 | 0.0 |
| −UNet | 100.0 | 52.0 | 13.0 | 0.0 | 0.0 | 0.0 | 53.0 | 40.0 | 3.0 | 100.0 | 0.0 | 0.0 |

*Accuracies threshold the reported metrics: size counts as correct when mean IoU ≥ 0.90, position uses a 2px tolerance, rotation uses a 2° tolerance, and count requires an exact match. Final architecture represents the default DiT-S/2 configuration with linear embeddings, concatenation conditioning, and standard DDPM sampling (250 steps, CFG = 1). Each row swaps a single component relative to this baseline.*

### Key Findings

1. **Rotation generalizes reliably**: Unlike other skills, rotation achieves 100% extrapolation accuracy, suggesting alignment with architectural inductive biases.

2. **Coverage > Dataset size**: Models trained with 75% coverage of skill combinations (fewer samples each) outperform models with 25% coverage (more samples per combination).

3. **Skills are interdependent**: When one skill goes out-of-distribution, all skills deteriorate together.

## Design Choices

The final architecture uses:
- ✅ Linear embedding for conditions (not sinusoidal or rotary)
- ✅ Concatenation conditioning (not AdaLN for rotation generalization)
- ✅ Pixel-space training (not VAE-based)
- ✅ DiT-S/2 transformer architecture (not UNet)
- ✅ DDPM sampling (not flow matching)

## BibTeX

```bibtex
@article{sethi2026vgl,
  title={Do Diffusion Models Learn to Generalize Basic Visual Skills?},
  author={Sethi, Amish and Zeng, Boya and Chai, Wenhao and Liu, Zhuang},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

This project is released under the MIT License.
