from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os
import cv2
import numpy as np
from lib.histogram_intersection import compute_3d_hist, compute_lbp_hist, get_histogram_distance
from lib.vp_tree import build_vptree, search_vptree
import json


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load image features for VP tree
segmented_images = "images/segmented"
segmented_image_files = [os.path.join(segmented_images, f) for f in os.listdir(segmented_images)]
all_image_names = set(segmented_image_files)
app.mount("/images/segmented", StaticFiles(directory=segmented_images), name="images")

image_features = []
for img_file in segmented_image_files:
    image = cv2.imread(img_file)
    color_hist = compute_3d_hist(image)
    lbp_hist = compute_lbp_hist(image)
    image_features.append((img_file, (color_hist, lbp_hist)))

img_to_idx = {img_file: idx for idx, img_file in enumerate(segmented_image_files)}

# build VP tree
vptree = build_vptree(image_features)

# API endpoints
@app.post("/query")
async def query_image(method: str = Form(...), cluster: str = Form(...), file: UploadFile = File(...)):
    original_img_path = os.path.join("images/segmented", file.filename)
    if original_img_path not in all_image_names:
        return JSONResponse(status_code=400, content={"error": "Invalid image input, please use one from the database"})
    
    temp_path = f"temp_{file.filename}"
    with open (temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    query_image = cv2.imread(temp_path)
    color_hist = compute_3d_hist(query_image)
    lbp_hist = compute_lbp_hist(query_image)
    query_feature = (color_hist, lbp_hist)

    os.remove(temp_path)

    tau = 0.14
    results = []

    relevant_retrieved = 0
    retrieved = 0
    total_comparisons = 0

    # get clusters
    with open(f"clusters/{cluster}.json", "r") as f:
        clusters = json.load(f)["clusters"]

    # mapping of images to corresponding clusters
    img_to_cluster = {}
    for clust in clusters:
        for img in clusters[clust]:
            img_to_cluster[img] = clust
    
    query_cluster = img_to_cluster[original_img_path]
    relevant_images = len(clusters[query_cluster])

    method = method.lower().strip()
    if method == "exhaustive":
        for img_name, img_feature in image_features:
            dist = get_histogram_distance(query_feature[0], img_feature[0], query_feature[1], img_feature[1], a=0.2, b=0.8)
            if dist <= tau:
                results.append({
                    "image_name": img_name,
                    "cluster": img_to_cluster[img_name],
                    "distance": dist,
                })
            total_comparisons += 1

        retrieved = len(results)
        for res in results:
            if img_to_cluster[res["image_name"]] == query_cluster:
                relevant_retrieved += 1

    elif method == "vp_tree":
        fetched_relevant_images, comparisons = search_vptree(vptree, query_feature, tau)
        for img_name, dist in fetched_relevant_images:
            results.append({
                "image_name": img_name,
                "cluster": img_to_cluster[img_name],
                "distance": dist
            })
        total_comparisons = comparisons

        retrieved = len(fetched_relevant_images)
        for img_name, _ in fetched_relevant_images:
            if img_to_cluster[img_name] == query_cluster:
                relevant_retrieved += 1
    
    else:
        return JSONResponse(status_code=400, content={"error": "Invalid method"})
    
    results = sorted(results, key=lambda x: x["distance"])
    return {
        "status": 200,
        "results": results,
        "method": method,
        "precision": relevant_retrieved / retrieved if retrieved > 0 else 0,
        "recall": relevant_retrieved / relevant_images if relevant_images > 0 else 0,
        "comparisons": total_comparisons
    }

    # NOTES FROM MIDTERM:

    # compare individual parasites - not fully image
    # get a lot more images by doing the above
    # harder to get metrics like recall and precision - find similar (typically not exact)
    # how to estimate similarity radius Tau (r) and how sensitive is it to Tau

    # compare individual images - get more indivisual images
    # comment 2 - think about how similarity is defined, using the above
    # comment 3 - how to figure out r and how sensitive f1 is compared to r