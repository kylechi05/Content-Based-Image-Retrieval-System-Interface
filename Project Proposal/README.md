# Image retrieval by image appearance using histogram intersection and indexing
#### CS:4980:0005 - Topics in Computer Science II: *Multimedia Information Systems*
#### Kyle Chi - Project Proposal Code

## Overview
The `Project Proposal` directory contains code for preliminary experiments and work for the Project Proposal presentation. Work includes: introduction to color histograms, histogram intersection, and exhaustive search indexing.

## Packages and Environment
- (Optional) Create a virtual environment for python packages with `python -m venv .venv`
- Install packages with `pip install -r requirements.txt`
- Start the virtual environment with `.venv/Scripts/activate` in powershell

## Scripts
- `color_histogram.py`
    - Computes and displays a channel-wise histogram breakdown of the RGB colors from the sample images
    - Introduces histograms
- `histogram_intersection.py`
    - Computes and displays the histogram intersection between the channel-wise histograms of the RGB colors from the sample images
    - Computes and prints the histogram intersection scores for each RGB channel
    - Introduces histogram intersection
- `histogram_intersection_indexing.py`
    - Computes histogram intersection scores between a query image and a few sample data images for color
    - Introduces exhaustive search indexing method


