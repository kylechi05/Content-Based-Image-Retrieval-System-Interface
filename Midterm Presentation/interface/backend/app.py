from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os
import cv2
import time
import numpy as np
from lib.histogram_intersection import get_hist, histogram_distance
from lib.vp_tree import build_vptree, search_vptree
from sklearn.neighbors import BallTree
from scipy.spatial.distance import cdist
from annoy import AnnoyIndex

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

image_dir = 'images'
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

app.mount("/images", StaticFiles(directory=image_dir), name="images")

print("Loading dataset and computing features...")
image_features = []
image_names = []
for img_path in image_files:
    img = cv2.imread(img_path)
    feature_vector = get_hist(img)
    image_features.append(feature_vector)
    image_names.append(img_path)

image_features = np.array(image_features)
print("Image Features computed.")

print("Building Ball Tree...")
ball_tree = BallTree(image_features, metric='manhattan', leaf_size=1)
print("Ball Tree built.")

print("Building VP Tree...")
vp_tree = build_vptree(list(zip(image_names, image_features)))
print("VP Tree built.")

print("Building Annoy Index...")
feature_dim = image_features.shape[1]
annoy_index = AnnoyIndex(feature_dim, metric="manhattan")
for i, vec in enumerate(image_features):
    annoy_index.add_item(i, vec.tolist())
annoy_index.build(10)
print("Annoy Index built.")

relevant_images = {'images\\1.png', 'images\\2.png', 'images\\3.png', 'images\\4.png', 
                   'images\\5.png', 'images\\6.png', 'images\\7.png', 'images\\8.png',
                   'images\\9.png', 'images\\10.png'}

# API endpoints
@app.post("/query")
async def query_image(method: str = Form(...), file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open (temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    query_image = cv2.imread(temp_path)
    query_feature = get_hist(query_image).reshape(1, -1)

    os.remove(temp_path)

    tau = 0.25
    results = []

    relevant_retrieved = 0
    retrieved = 0
    elapsed_time = 0

    method = method.lower().strip()
    if method == "exhaustive":
        start_time = time.perf_counter()
        for idx, feature in enumerate(image_features):
            dist = histogram_distance(feature, query_feature.flatten())
            if dist <= tau:
                results.append({
                    "image_name": image_names[idx],
                    "distance": dist,
                })
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        retrieved = len(results)
        for res in results:
            if res["image_name"] in relevant_images:
                relevant_retrieved += 1

    elif method == "ball_tree":
        start_time = time.perf_counter()
        ind = ball_tree.query_radius(query_feature, r=tau*2)[0] # *2 because of hist similarity and manhattan dist computation
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        for idx in ind:
            dist = histogram_distance(image_features[idx], query_feature.flatten())
            results.append({
                "image_name": image_names[idx],
                "distance": dist
            })

        retrieved = len(ind)
        for idx in ind:
            if image_names[idx] in relevant_images:
                relevant_retrieved += 1

    elif method == "vp_tree":
        start_time = time.perf_counter()
        fetched_relevant_images, _ = search_vptree(vp_tree, query_feature.flatten(), tau)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        for img_name, dist in fetched_relevant_images:
            results.append({
                "image_name": img_name,
                "distance": dist
            })

        retrieved = len(fetched_relevant_images)
        for img_name, _ in fetched_relevant_images:
            if img_name in relevant_images:
                relevant_retrieved += 1
    elif method == "annoy":
        start_time = time.perf_counter()
        idxs, dists = annoy_index.get_nns_by_vector(query_feature.flatten().tolist(), 10, include_distances=True)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        for idx, dist in zip(idxs, dists):
            results.append({
                "image_name": image_names[idx],
                "distance": float(dist)
            })
            if image_names[idx] in relevant_images:
                relevant_retrieved += 1

        retrieved = len(idxs)
    elif method == "cdist":
        start_time = time.perf_counter()
        distances = cdist(query_feature, image_features, metric="cityblock").flatten()
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        nearest_indices = np.argsort(distances)

        k = 10
        for idx in nearest_indices[:k]:
            dist = distances[idx]
            results.append({
                "image_name": image_names[idx],
                "distance": float(dist),
            })
            if image_names[idx] in relevant_images:
                relevant_retrieved += 1

        retrieved = k

    else:
        return JSONResponse(status_code=400, content={"error": "Invalid method"})
    
    results = sorted(results, key=lambda x: x["distance"])
    return {
        "status": 200,
        "results": results,
        "method": method,
        "precision": relevant_retrieved / retrieved if retrieved > 0 else 0,
        "recall": relevant_retrieved / len(relevant_images) if len(relevant_images) > 0 else 0,
        "time": elapsed_time
    }

    # NOTES FROM MIDTERM:

    # compare individual parasites - not fully image
    # get a lot more images by doing the above
    # harder to get metrics like recall and precision - find similar (typically not exact)
    # how to estimate similarity radius Tau (r) and how sensitive is it to Tau

    # compare individual images - get more indivisual images
    # comment 2 - think about how similarity is defined, using the above
    # comment 3 - how to figure out r and how sensitive f1 is compared to r