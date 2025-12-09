import numpy as np

from .histogram_intersection import get_histogram_distance

class VPNode:
    def __init__(self, image_name=None, pivot_feature=None, mu=None, left=None, right=None, points=None):
        self.image_name = image_name
        self.pivot_feature = pivot_feature
        self.mu = mu
        self.left = left
        self.right = right
        self.points = points

def build_vptree(images_features):
    if len(images_features) == 0:
        return None
    if len(images_features) == 1:
        name, feature = images_features[0]
        return VPNode(image_name=name, pivot_feature=feature, points=[name])
        
    pivot_idx = np.random.randint(len(images_features))
    pivot_name, pivot_feature = images_features[pivot_idx]

    distances = []
    for i, (name, feature) in enumerate(images_features):
        if i != pivot_idx:
            dist = get_histogram_distance(pivot_feature[0], feature[0], pivot_feature[1], feature[1], a=0.2, b=0.8)
            distances.append((dist, name, feature))
    
    dists_only = [d[0] for d in distances]
    mu = np.median(dists_only)
    
    left_points = [(name, feat) for d, name, feat in distances if d <= mu]
    right_points = [(name, feat) for d, name, feat in distances if d > mu]
    
    left_tree = build_vptree(left_points)
    right_tree = build_vptree(right_points)
    
    return VPNode(image_name=pivot_name,
                  pivot_feature=pivot_feature,
                  mu=mu,
                  left=left_tree,
                  right=right_tree,
                  points=[name for name, _ in images_features]
                  )

def search_vptree(root, query_feature, tau):
    if root is None:
        return []

    comparisons = 0
    fetched_relevant_images = []
    stack = [root]

    while stack:
        node = stack.pop()
        if node is None:
            continue

        dist = get_histogram_distance(node.pivot_feature[0], query_feature[0], node.pivot_feature[1], query_feature[1], a=0.2, b=0.8)
        comparisons += 1

        if dist <= tau:
            fetched_relevant_images.append((node.image_name, dist))

        if node.left is None and node.right is None:
            continue

        if dist < node.mu:
            stack.append(node.left)
            if dist + tau >= node.mu:
                stack.append(node.right)
        else:
            stack.append(node.right)
            if dist - tau <= node.mu:
                stack.append(node.left)

    return (fetched_relevant_images, comparisons)