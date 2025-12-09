# Image retrieval by image appearance using histogram intersection and indexing
#### CS:4980:0005 - Topics in Computer Science II: *Multimedia Information Systems*
#### Kyle Chi - Final Project Code

## Overview
The `Final Presentation` directory contains code for project progress for the Final presentation. Work includes complete data preprocessing of parasite images, clustering for ground truth definitions, histogram intersection methods used on individual parasites instead of images containing multiple parasites, evaluation of histogram intersection methods (Recall, Precision, F1 Scores), and evaluation of VP Tree indexing methods against exhaustive search method.

## Packages and Environment
Install packages with `pip install -r requirements.txt`

## Modules
- `histogram_intersection.py`
    - Contains relevant functions used in histogram intersection-based indexing methods

## Scripts
To evaluate the "correctness" (recall/precision/F1 score) of histogram intersection against different ground truth clusters, run `evaluate_histogram_intersection.py`. To evaluate the speed of the histogram intersection-based search using a VP Tree database for the parasite images, run `evaluate_comparisons.py`.

To replicate the entire workflow:
- Run `segment.py` to segment the raw image files
- Run `cluster.py` to create a ground truth cluster
    - Comment/Uncomment lines to produce different ground truth clusters
    - Pass in different distance measures (euclidean/manhattan/cosine) to produce different ground truth clusters
- Run `distance_matrix.py` to create a matrix containing histogram intersection distances between all paraasites
- Run `find_similarity.py` to analyze different similarity radii `r` against different ground truth clusters
- Run `evaluate_histogram_intersection.py` to analyze the effectiveness/correctness (recall/precision/F1 score) of histogram intersection-based search
- Run `evaluate_comparisons.py` to analyze the comparison and time speedup using a VP Tree database for faster indexing compared to exhaustive searches
- Run `issues.py` to view a small demonstration on a potential issue with the project that I would improve on given more time

*Note: More information regarding scripts can be found inside their files.*

## Interactive Interface
### Frontend
To start frontend locally:
- `cd` into `interface/frontend`
- Install dependencies with `npm install`
- Start server with `npm run dev`
### Backend
To start backend locally:
- `cd` into `interface/backend`
- Install packages with `pip install -r requirements.txt`
- Start server with `uvicorn app:app --reload`
### Using the interface:
- Navigate to http://localhost:3000
- Upload one of the parasite images in `images/segmented` to retrieve relevant/similar parasite images