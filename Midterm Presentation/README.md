# Image retrieval by image appearance using histogram intersection and indexing
#### CS:4980:0005 - Topics in Computer Science II: *Multimedia Information Systems*
#### Kyle Chi - Midterm Presentation Code

## Overview
The `Midterm Presentation` directory contains code for project progress up to the Midterm presentation. Progress includes introduction to Local Binary Pattern (LBP) image features, exploration of efficient indexing techniques (most using triangle inequality), and construction of a querying interface.

## Packages and Environment
- (Optional) Create a virtual environment for python packages with `python -m venv .venv`
- Install packages with `pip install -r requirements.txt`
- Start the virtual environment with `.venv/Scripts/activate` in powershell

## Modules
- `/lib`
    - `histogram_intersection.py`
        - Contains helper functions used in the histogram intersection method

## Scripts
- `/histogram_intersectioning`
    - `texture_histogram.py`
        - Computes and displays histogram intersection and histogram intersection score for the LPB of two images
        - Introduction to LBP as texture
    - `color_texture_intersection.py`
        - Computes histogram intersection scores between a query image and a few sample data images for color and texture
        - Combines both texture and color features into query
- `annoytree_search.py`
    - Implementation if Spotify's ANNOY tree for fast indexing
- `balltree_search.py`
    - Implementation of ball tree for fast indexing
- `cdist_search.py`
    - Implementation of Scipy CDist for fast indexing
- `exhaustive_search.py`
    - Implementation of exhaustive search for baseline indexing
- `vptree_search.py`
    - Implementation of VP tree for fast indexing

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
- Upload one of the parasite images in `interface/backend/images` to retrieve relevant/similar parasite images
