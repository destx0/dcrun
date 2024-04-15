#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.datasets import make_blobs
import json
import math
import random
import time
from tabulate import tabulate

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from kdtree import *
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_circles
from sklearn.mixture import GaussianMixture
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
from utils import *
from sklearn.datasets import (
    make_blobs,
    make_circles,
    make_moons,
    load_iris,
    load_wine,
    load_breast_cancer,
)
from sklearn.decomposition import PCA

savefile = "data.json"
points_count = 30000
to_plot = False
no_centres = 1

from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# Initialize the parameters
noise = 0.05  # Amount of noise
points_range = [10, 100, 1000, 10000, 50000, 100000]
centers_range = [1, 10, 100, 1000, 5000, 10000, 50000, 100000]

results = []
dataset_types = ["blobs", "circles", "moons", "iris", "wine", "breast_cancer"]

dataset_types = [
    "mnist",
    "fashion_mnist",
    "covertype",
    "poker",
    "emnist",
    "blobs",
    "circles",
    "moons",
]
real_dataset_types = ["mnist", "fashion_mnist", "covertype", "poker", "emnist"]
points_range = [10, 100, 1000, 10000, 50000, 100000]
centers_range = [1, 10, 100, 1000, 5000, 10000, 50000, 100000]
for _ in range(1):
    for _ in range(1):
        for dataset_type in dataset_types:
            if dataset_type in real_dataset_types:
                if dataset_type == "mnist":
                    mnist = fetch_openml("mnist_784", version=1)
                    X = mnist.data[:10000]  # Use the entire dataset (70,000 samples)
                    if X.shape[1] > 2:
                        pca = PCA(n_components=2)
                        X = pca.fit_transform(X)
                elif dataset_type == "fashion_mnist":
                    fashion_mnist = fetch_openml("Fashion-MNIST", version=1)
                    X = fashion_mnist.data[
                        :10000
                    ]  # Use the entire dataset (70,000 samples)
                    if X.shape[1] > 2:
                        pca = PCA(n_components=2)
                        X = pca.fit_transform(X)
                elif dataset_type == "covertype":
                    covertype = fetch_openml("covertype", version=1)
                    X = covertype.data[:10000]  # Use a subset of 20,000 samples
                    if X.shape[1] > 2:
                        pca = PCA(n_components=2)
                        X = pca.fit_transform(X)
                elif dataset_type == "poker":
                    poker = fetch_openml("poker", version=1)
                    X = poker.data[:10000]  # Use a subset of 20,000 samples
                    if X.shape[1] > 2:
                        pca = PCA(n_components=2)
                        X = pca.fit_transform(X)
                elif dataset_type == "emnist":
                    emnist = fetch_openml("emnist", version=1)
                    X = emnist.data[:10000]  # Use a subset of 20,000 samples
                    if X.shape[1] > 2:
                        pca = PCA(n_components=2)
                        X = pca.fit_transform(X)

                points = [(x, y) for x, y in X]

                points_count = len(points)
                maxdis = math.ceil(math.log2(points_count))

                dcran_start_time = time.time()

                def build():
                    tree = KDTree()
                    tree.root = tree.build(points)

                    G = nx.Graph()

                    for point in points:
                        G.add_node(point, pos=point)

                    neighbours = {}
                    maxdis = math.ceil(math.log2(points_count))
                    for point in points:
                        neighbours[point] = i_neighbors(tree, point, maxdis)
                    return G, neighbours

                def merge_comps(core1, core2, core_points_map, mst):
                    pivot2 = min(
                        core_points_map[core2],
                        key=lambda node: euclidean_distance(node, core1),
                    )
                    pivot1 = min(
                        core_points_map[core1],
                        key=lambda node: euclidean_distance(node, pivot2),
                    )
                    mst.add_edge(
                        pivot1, pivot2, weight=euclidean_distance(pivot1, pivot2)
                    )
                    print(
                        f"merging {core1} and {core2} with pivot {pivot1} and {pivot2}"
                    )

                def merge_phase(G):
                    core_points_map = {}
                    for component in nx.connected_components(G):
                        centroid = np.mean([node for node in component], axis=0)
                        closest_point = min(
                            component,
                            key=lambda node: euclidean_distance(node, centroid),
                        )
                        core_points_map[closest_point] = component

                    core_points = list(core_points_map.keys())

                    minc = [float("inf")] * len(core_points[0])
                    maxc = [float("-inf")] * len(core_points[0])

                    for point in core_points:
                        for i in range(len(point)):
                            minc[i] = min(minc[i], point[i])
                            maxc[i] = max(maxc[i], point[i])

                    diff = [maxc[i] - minc[i] for i in range(len(minc))]
                    min_diff_axis = diff.index(max(diff))

                    sorted_core_points = sorted(
                        core_points, key=lambda point: point[min_diff_axis]
                    )
                    for core1, core2 in zip(sorted_core_points, sorted_core_points[1:]):
                        merge_comps(core1, core2, core_points_map, G)

                def dcrun():
                    G, neighbours = build()
                    print(G.number_of_nodes(), G.number_of_edges())
                    k = 0
                    while k**k < maxdis:

                        if (
                            len(
                                connected_components := list(nx.connected_components(G))
                            )
                        ) == 1:
                            break
                        for component in connected_components:
                            for node in component:
                                wt, pos = neighbours[node][k]
                                G.add_edge(node, pos, weight=wt)

                                wt, pos = neighbours[node][k**k]
                                G.add_edge(node, pos, weight=wt)
                        k += 1
                    else:
                        print("merge phase")
                        merge_phase(G)
                        print(
                            f"The graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges."
                        )
                        print(
                            "Connected Components : ",
                            len(list(nx.connected_components(G))),
                        )
                    edgecount = G.number_of_edges()
                    mst = nx.minimum_spanning_tree(G)
                    print(mst.number_of_nodes(), mst.number_of_edges())
                    mst_weight = sum(
                        data["weight"] for u, v, data in mst.edges(data=True)
                    )
                    print(
                        f"Minimum Spanning Tree: {mst.number_of_nodes()} nodes, {mst.number_of_edges()} edges, Total Weight: {mst_weight}"
                    )
                    return mst_weight, edgecount

                dc_weight, dc_edgecount = dcrun()

                dcran_end_time = time.time()
                dcran_elapsed_time = dcran_end_time - dcran_start_time

                import math

                eprim_start_time = time.time()

                def calculate_distance(p1, p2):
                    x1, y1 = p1
                    x2, y2 = p2
                    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                def prim_mst(points):
                    n = len(points)
                    visited = [False] * n
                    min_cost = [float("inf")] * n
                    parent = [None] * n

                    min_cost[0] = 0
                    parent[0] = -1

                    for _ in tqdm(
                        range(n), desc="Running Prim's Algorithm", unit="iteration"
                    ):  # Use tqdm for progress bar
                        min_index = -1
                        for i in range(n):
                            if not visited[i] and (
                                min_index == -1 or min_cost[i] < min_cost[min_index]
                            ):
                                min_index = i

                        visited[min_index] = True

                        for i in range(n):
                            if not visited[i]:
                                distance = calculate_distance(
                                    points[min_index], points[i]
                                )
                                if distance < min_cost[i]:
                                    min_cost[i] = distance
                                    parent[i] = min_index

                    mst_edges = []
                    for i in range(1, n):
                        mst_edges.append((parent[i], i))

                    return mst_edges

                mst = prim_mst(points)
                eprim_end_time = time.time()
                eprim_elapsed_time = eprim_end_time - eprim_start_time
                print(
                    "MST Weight: ",
                    eprim_wt := sum(
                        [calculate_distance(points[u], points[v]) for u, v in mst]
                    ),
                )
                print("time taken by dcran : ", eprim_elapsed_time)

                speedup = eprim_elapsed_time / dcran_elapsed_time
                print(f"Speedup: {speedup:.2f}")

                wt_error = abs(dc_weight - eprim_wt) / eprim_wt * 100
                print(f"Weight Error: {wt_error}%")

                # ==================================================================================

                import time
                from sklearn.cluster import KMeans
                from scipy.spatial.distance import pdist, squareform
                from sklearn.datasets import make_blobs
                import numpy as np

                def euclidean_distance(p1, p2):
                    return sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5

                def prims_mst(points):
                    n = len(points)
                    pairwise_distances = squareform(pdist(points))

                    mst_edges = []
                    total_weight = 0
                    visited = [False] * n
                    visited[0] = True

                    for _ in range(n - 1):
                        min_weight = float("inf")
                        min_u, min_v = None, None

                        for u in range(n):
                            if visited[u]:
                                for v in range(n):
                                    if (
                                        not visited[v]
                                        and pairwise_distances[u][v] < min_weight
                                    ):
                                        min_weight = pairwise_distances[u][v]
                                        min_u, min_v = u, v

                        mst_edges.append((points[min_u], points[min_v]))
                        total_weight += min_weight
                        visited[min_v] = True

                    return mst_edges, total_weight

                def fast_mst(points):
                    n = len(points)
                    k = int(n**0.5)

                    # Divide-and-conquer stage
                    kmeans = KMeans(n_clusters=k).fit(points)
                    labels = kmeans.labels_
                    centers = kmeans.cluster_centers_.tolist()

                    mst_edges = []
                    total_weight = 0

                    for i in range(k):
                        cluster_points = [points[j] for j in range(n) if labels[j] == i]
                        cluster_mst_edges, cluster_weight = prims_mst(cluster_points)
                        mst_edges.extend(cluster_mst_edges)
                        total_weight += cluster_weight

                    # Connect the clusters using Prim's algorithm
                    centers_mst_edges, centers_weight = prims_mst(centers)
                    for u, v in centers_mst_edges:
                        cluster1 = [
                            points[m] for m in range(n) if labels[m] == centers.index(u)
                        ]
                        cluster2 = [
                            points[m] for m in range(n) if labels[m] == centers.index(v)
                        ]
                        min_dist = float("inf")
                        min_point1, min_point2 = None, None
                        for p1 in cluster1:
                            for p2 in cluster2:
                                dist = euclidean_distance(p1, p2)
                                if dist < min_dist:
                                    min_dist = dist
                                    min_point1, min_point2 = p1, p2
                        mst_edges.append((min_point1, min_point2))
                        total_weight += min_dist

                    return mst_edges, total_weight

                # Generate blob dataset using scikit-learn
                # n_samples = 1000
                # n_centers = 4
                # X, _ = make_blobs(n_samples=n_samples, centers=n_centers, random_state=42)
                # points = X.tolist()

                # Compute MST using FMST algorithm
                fmst_start_time = time.time()
                fmst_edges, fmst_weight = fast_mst(points)
                fmst_end_time = time.time()
                fmst_elapsed_time = fmst_end_time - fmst_start_time
                print("Time taken by fast MST:", fmst_elapsed_time)
                print("Total weight of the MST (FMST):", fmst_weight)
                fmst_edgecount = len(fmst_edges)

                # Compute MST using Prim's algorithm
                # prims_start_time = time.time()
                # prims_edges, prims_weight = prims_mst(points)
                # prims_end_time = time.time()
                # prims_elapsed_time = prims_end_time - prims_start_time
                # print("Time taken by Prim's MST:", prims_elapsed_time)
                # print("Total weight of the MST (Prim's):", prims_weight)

                # =======================================================================
                # from sklearn.cluster import KMeans
                # from scipy.sparse.csgraph import minimum_spanning_tree
                # from scipy.spatial.distance import pdist, squareform

                # fmst_start_time = time.time()

                # def euclidean_distance(p1, p2):
                #     return sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5

                # def detect_connecting_edge(cluster1, cluster2):
                #     center1 = [sum(x) / len(cluster1) for x in zip(*cluster1)]
                #     center2 = [sum(x) / len(cluster2) for x in zip(*cluster2)]

                #     min_dist1 = float("inf")
                #     min_point1 = None
                #     for point in cluster1:
                #         dist = euclidean_distance(point, center2)
                #         if dist < min_dist1:
                #             min_dist1 = dist
                #             min_point1 = point

                #     min_dist2 = float("inf")
                #     min_point2 = None
                #     for point in cluster2:
                #         dist = euclidean_distance(point, center1)
                #         if dist < min_dist2:
                #             min_dist2 = dist
                #             min_point2 = point

                #     return min_point1, min_point2

                # def fast_mst(points):
                #     n = len(points)
                #     k = int(n**0.5)

                #     # Divide-and-conquer stage
                #     kmeans = KMeans(n_clusters=k).fit(points)
                #     labels = kmeans.labels_
                #     centers = kmeans.cluster_centers_.tolist()

                #     mst_edges = []
                #     total_weight = 0

                #     for i in range(k):
                #         cluster_points = [points[j] for j in range(n) if labels[j] == i]
                #         pairwise_distances = squareform(pdist(cluster_points))
                #         mst = (
                #             minimum_spanning_tree(pairwise_distances).toarray().tolist()
                #         )
                #         edges = [
                #             (u, v)
                #             for u in range(len(cluster_points))
                #             for v in range(u + 1, len(cluster_points))
                #             if mst[u][v] > 0
                #         ]
                #         for u, v in edges:
                #             weight = euclidean_distance(
                #                 cluster_points[u], cluster_points[v]
                #             )
                #             mst_edges.append((cluster_points[u], cluster_points[v]))
                #             total_weight += weight

                #     pairwise_distances_centers = squareform(pdist(centers))
                #     mst_centers = (
                #         minimum_spanning_tree(pairwise_distances_centers)
                #         .toarray()
                #         .tolist()
                #     )
                #     center_edges = [
                #         (i, j)
                #         for i in range(k)
                #         for j in range(i + 1, k)
                #         if mst_centers[i][j] > 0
                #     ]

                #     # Refinement stage
                #     midpoints = []
                #     for i, j in center_edges:
                #         cluster1 = [points[m] for m in range(n) if labels[m] == i]
                #         cluster2 = [points[m] for m in range(n) if labels[m] == j]
                #         u, v = detect_connecting_edge(cluster1, cluster2)
                #         mst_edges.append((u, v))
                #         midpoint = [(a + b) / 2 for a, b in zip(u, v)]
                #         midpoints.append(midpoint)
                #         weight = euclidean_distance(u, v)
                #         total_weight += weight

                #     if len(midpoints) > 0:
                #         kmeans_refine = KMeans(
                #             n_clusters=len(midpoints), init=midpoints, n_init=1
                #         ).fit(points)
                #         labels_refine = kmeans_refine.labels_

                #         for i in range(len(midpoints)):
                #             cluster_points = [
                #                 points[j] for j in range(n) if labels_refine[j] == i
                #             ]
                #             pairwise_distances = squareform(pdist(cluster_points))
                #             mst = (
                #                 minimum_spanning_tree(pairwise_distances)
                #                 .toarray()
                #                 .tolist()
                #             )
                #             edges = [
                #                 (u, v)
                #                 for u in range(len(cluster_points))
                #                 for v in range(u + 1, len(cluster_points))
                #                 if mst[u][v] > 0
                #             ]
                #             for u, v in edges:
                #                 weight = euclidean_distance(
                #                     cluster_points[u], cluster_points[v]
                #                 )
                #                 mst_edges.append((cluster_points[u], cluster_points[v]))
                #                 total_weight += weight

                #     return mst_edges, total_weight

                # mst_edges, fmst_weight = fast_mst(points)
                # fmst_end_time = time.time()
                # fmst_edgecount = len(mst_edges)
                # fmst_elapsed_time = fmst_end_time - fmst_start_time
                # print("time taken by fast mst : ", fmst_elapsed_time)

                # print("Total weight of the MST:", fmst_weight)

                # ==============================================================================================

                with open(savefile, "r") as f:
                    loaded_data = json.load(f)
                results.append(
                    [
                        dataset_type,
                        points_count,
                        no_centres,
                        dc_weight,
                        fmst_weight,
                        eprim_wt,
                        fmst_edgecount,
                        len(mst),
                        abs(fmst_weight - eprim_wt) / eprim_wt * 100,
                        abs(dc_weight - eprim_wt) / eprim_wt * 100,
                        dcran_elapsed_time,
                        fmst_elapsed_time,
                        eprim_elapsed_time,
                        eprim_elapsed_time / fmst_elapsed_time,
                        eprim_elapsed_time / dcran_elapsed_time,
                    ]
                )

                with open(savefile, "w") as f:
                    json.dump(results, f)

                headers = [
                    "Dataset",
                    "Points",
                    "Centres",
                    "DCRAN Wt",
                    "FMST Wt",
                    "Prim's Wt",
                    "FMST Edges",
                    "Prim's Edges",
                    "FMST Acc(%)",
                    "DCRAN Acc(%)",
                    "DCRAN Time (s)",
                    "FMST Time (s)",
                    "Prim's Time (s)",
                    "fmst's Speedup",
                    "DCRAN Speedup",
                ]

                table_str = tabulate(
                    results,
                    headers,
                    tablefmt="pipe",
                    floatfmt=(
                        ".0f",
                        ".0f",
                        ".0f",
                        ".1f",
                        ".1f",
                        ".1f",
                        ".0f",
                        ".0f",
                        ".2f",
                        ".2f",
                        ".2f",
                        ".2f",
                        ".2f",
                        ".2f",
                        ".2f",
                    ),
                )
