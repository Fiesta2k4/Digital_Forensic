# Standard Test Images

This folder contains 4 standard grayscale test images commonly used in image processing and steganography research.

## Images

| Image | Size | Description | Source |
|-------|------|-------------|--------|
| **lena.pgm** | 512×512 | Woman with hat (Lena Söderberg) | USC-SIPI Database |
| **baboon.pgm** | 512×512 | Mandrill face (high texture) | USC-SIPI Database |
| **airplane.pgm** | 512×512 | F-16 fighter jet over mountains | USC-SIPI Database |
| **peppers.pgm** | 512×512 | Bell peppers | USC-SIPI Database |

## Image Statistics

### lena.pgm
- Min: 0, Max: 246, Mean: 127.7
- Boundary pixels (≤10 or ≥245): **0.28%**
- ✅ Excellent for all steganography methods

### baboon.pgm
- Min: 0, Max: 231, Mean: 129.6
- Boundary pixels: **0.10%**
- ⚠️ High texture - PVD/DE may fail occasionally
- ✅ Best for testing robustness

### airplane.pgm
- Min: 12, Max: 223, Mean: 132.4
- Boundary pixels: **0.00%**
- ✅ Perfect for all methods (smooth + texture balance)

### peppers.pgm
- Min: 0, Max: 228, Mean: 120.2
- Boundary pixels: **1.27%**
- ✅ Good for most methods

## Method Compatibility (Tested)

| Method | lena | baboon | airplane | peppers |
|--------|------|--------|----------|---------|
| LSB Substitution | ✅ | ✅ | ✅ | ✅ |
| LSB Matching | ✅ | ✅ | ✅ | ✅ |
| PVD | ✅ | ⚠️ | ⚠️ | ✅ |
| EMD | ✅ | ✅ | ✅ | ✅ |
| Histogram Shifting | ✅ | ✅ | ✅ | ✅ |
| Interpolation | ✅ | ✅ | ✅ | ✅ |
| Difference Expansion | ✅ | ⚠️ | ✅ | ⚠️ |

**Legend:**
- ✅ Always works
- ⚠️ May fail with complex messages or specific conditions

## Download Script

If images are missing, run:
```bash
python download_standard_images.py
```

This will download the images from USC-SIPI Image Database.

## Citation

These images are from the USC-SIPI Image Database:
- Signal and Image Processing Institute
- University of Southern California
- http://sipi.usc.edu/database/

## Notes

- All images are **512×512 pixels**, grayscale (8-bit)
- Format: **PGM** (Portable Gray Map)
- Much better quality than BOSSbase_256 (256×256) for steganography
- **4× more capacity** than 256×256 images
- Widely used in research papers for comparison
