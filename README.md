# Data-Challenge-Keramik
Automated classification system for Roman pottery fragments based on profile cross-sections. Matches archaeological test samples against reference drawings from literature using computer vision techniques.

#Overview
The system works in three steps:

1. Preprocess reference images - Extract and standardize pottery profiles from literature
2. Convert test data - Process svg files of pottery fragments into png files
3. Classify - Match test samples against references using computer vision features

#Installation

# Create conda environment
conda create -n env_name python=3.11
conda activate env_name

# Install dependencies
pip install Pillow scikit-image scipy numpy selenium
