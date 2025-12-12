# Roman Pottery Classification

Automated classification system for Roman pottery fragments based on profile cross-sections. This tool helps archaeologists identify pottery types by comparing test samples against reference drawings from archaeological literature.

## Overview

The system works in three steps:
1. **Preprocess reference images** - Extract and standardize pottery profiles from literature
2. **Convert test data** - Process SVG files of pottery fragments  
3. **Classify** - Match test samples against references using computer vision features

## Installation
```bash
# Create conda environment
conda create -n keramik python=3.11
conda activate keramik

# Install dependencies
pip install Pillow scikit-image scipy numpy selenium
```

## Usage

### 1. Preprocess Reference Images
```bash
python literatur_preprocessing.py
```
- Extracts left portion (cross-section only)
- Crops top section (rim/lip area)
- Fills contours to create black silhouettes

### 2. Convert SVG Test Images
```bash
python svg_to_png.py
```
- Converts SVG files to PNG using Chrome/Selenium
- Removes legends and labels
- Extracts pottery profile section

### 3. Classify Test Images
```bash
python keramik_classifier.py
```
- Extracts features (HOG, Hu Moments, contours, histograms)
- Compares against all references
- Outputs top-5 matches with similarity scores
- Saves results to CSV

## How It Works

The classifier uses multiple computer vision features:
- **HOG (Histogram of Oriented Gradients)**: Captures edge orientations and shape
- **Hu Moments**: Scale/rotation invariant shape descriptors
- **Contour Properties**: Height, width, aspect ratio
- **Pixel Histograms**: Intensity distribution

Features are weighted and combined to produce a similarity score (0-1) between test and reference images.

## Output

Results are saved as CSV with columns:
- Test image filename
- Top 5 matching classes
- Similarity scores for each match

## Requirements

- Python 3.11+
- Chrome browser (for SVG conversion)
- Conda (recommended)

## Project Structure
```
├── literatur_preprocessing.py  # Reference image preprocessing
├── svg_to_png.py              # SVG to PNG converter
├── keramik_classifier.py      # Main classification system
└── README.md
```

## Notes

- Works with limited training data (one-shot learning)
- Designed for fragmentary pottery with rim sections
- Best results when reference and test images are similarly processed

