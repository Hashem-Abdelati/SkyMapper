# Building Footprint Segmentation from Satellite Imagery

This project explores automated **building footprint segmentation** from high-resolution satellite imagery using deep learning. We compare a **convolutional baseline (ResNet-50 FCN)** against a **transformer-based model (SegFormer)** on the SpaceNet-2 Las Vegas dataset to evaluate how local versus global context modeling impacts segmentation quality.

---

## Project Overview

- **Task**: Binary semantic segmentation (building vs. background)
- **Dataset**: SpaceNet-2 (AOI 2 – Las Vegas)
- **Models**:
  - ResNet-50 Fully Convolutional Network (CNN baseline)
  - SegFormer (Transformer-based segmentation)
- **Framework**: PyTorch
- **Metrics**: Precision, Recall, F1 Score, Intersection-over-Union (IoU)

---

## Dataset

The project uses the **SpaceNet-2 Building Footprint Dataset**, focusing on **Area of Interest 2 (Las Vegas)**.

- High-resolution RGB satellite image tiles
- Polygon-based building footprint annotations (GeoJSON)
- Building polygons rasterized into binary segmentation masks
- Images and masks resized to **256 × 256**
- Dataset split:
  - 80% Training
  - 10% Validation
  - 10% Test

---

## Methods

### ResNet-50 FCN (Baseline)

- Encoder: ResNet-50 with residual connections
- Decoder: Fully convolutional upsampling head
- Outputs per-pixel building probabilities
- Strengths:
  - Strong local feature extraction
  - High recall
- Limitations:
  - Over-segmentation
  - Less precise building boundaries in dense areas

---

### SegFormer (Transformer-Based)

- Hierarchical transformer encoder (MiT backbone)
- Lightweight segmentation decoder
- Initialized with ImageNet pre-trained weights
- Strengths:
  - Captures long-range spatial dependencies
  - Cleaner and more coherent building boundaries
- Limitations:
  - Slightly lower recall on small or irregular buildings

---

## Training Details

- Image size: `256 × 256`
- Epochs: `5`
- Loss: Binary Cross-Entropy with logits
- Optimizers:
  - ResNet-FCN: SGD
  - SegFormer: Adam
- Output threshold: `0.5`
- SegFormer outputs upsampled to match input resolution before evaluation

---

## Results

| Model        | Precision | Recall | F1 Score | IoU   |
|--------------|----------:|-------:|---------:|------:|
| ResNet-50 FCN | 0.7864 | **0.8667** | 0.8246 | 0.7016 |
| SegFormer     | **0.8571** | 0.8137 | **0.8348** | **0.7164** |

### Key Observations

- ResNet-50 achieves higher recall but produces more false positives
- SegFormer yields sharper boundaries and higher overall segmentation quality
- SegFormer consistently outperforms the CNN baseline in F1 and IoU
- Statistical testing confirms the performance difference is significant

---

## Conclusion

Both models successfully learn building footprint segmentation from satellite imagery, demonstrating that supervised semantic segmentation is a scalable alternative to manual mapping. While the ResNet-50 FCN provides a strong baseline, **SegFormer achieves superior overall performance** by leveraging global context through self-attention.

---

## Future Work

- Extend training with early stopping
- Apply data augmentation and class-imbalance-aware loss functions
- Evaluate generalization across additional SpaceNet AOIs
- Explore hybrid CNN–Transformer architectures

---

## References

Full references and experimental details are provided in the accompanying project report.
