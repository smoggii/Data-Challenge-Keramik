Roman Pottery Classification (SVG Vector Version)
Automated classification system for Roman pottery fragments based on vector profile cross-sections. This tool matches archaeological test samples against reference drawings from literature by analyzing SVG path data directly.

Overview
The system has been updated to work directly with vector data, eliminating the need for image preprocessing or browser-based conversion. It operates in two main phases:

Feature Extraction: Parses SVG paths to extract mathematical descriptors including wall thickness, symmetry, and Fourier components.

Classification: Matches test samples against a reference database using weighted similarity scores.

Installation
1. Create Environment

Bash
# Create and activate conda environment
conda create -n keramik python=3.11
conda activate keramik
2. Install Dependencies

The vector-based version requires only mathematical and scientific libraries:

Bash
pip install numpy scipy
Note: Selenium and Chrome are not required for this version.

Usage
The system is centralized in the advanced classifier script. Run it from your terminal:

Bash
python keramik_svg_classifier_newfeatures.py
Input: Provide the paths to your reference folder (e.g., images_typentafel) and test folder (e.g., svg_files) when prompted.

Matching: The system compares geometric profiles and displays the top-K matches with similarity scores.

Export: Results are automatically saved to a CSV file in your output directory.

How It Works
The classifier utilizes high-precision vector features specifically designed for archaeological pottery analysis:

Radial Profile & Symmetry: Analyzes the vessel as a rotational body to determine axial symmetry and radial distribution.

Wall Thickness Analysis: Estimates the thickness of the fragment across different height segments.

Typological Zone Analysis: Specifically examines the rim (top), belly (middle), and base (bottom) of the profile.

Fourier Descriptors: Provides rotation and scale-invariant shape descriptors for global form matching.

Curvature & Complexity: Measures the "smoothness" versus "ornamentation" (tortuosity) of the vessel's contour.

Output
Results are saved as a CSV file with a timestamp (e.g., svg_classification_results_20260201.csv):

Test_SVG: The filename of the analyzed fragment.

Match_X_Class: The name of the matching reference type.

Match_X_Similarity: Confidence score (0 to 1) based on weighted features.
