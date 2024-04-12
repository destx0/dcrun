#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.datasets import make_blobs
import random
import json
import time
from tabulate import tabulate
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture

from kdtree import *
from utils import *

savefile = "data.json"
points_count = 30000
to_plot = False
no_centres = 1

from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

# Initialize the parameters
noise = 0.05  # Amount of noise

# Generate the data
X, y = make_circles(n_samples=points_count, noise=noise, factor=0.5, random_state=42)

# Convert the data to a list of tuples
points = [(x, y) for x, y in X]
maxdis = math.ceil(math.log2(points_count))


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
        core_points_map[core2], key=lambda node: euclidean_distance(node, core1)
    )
    pivot1 = min(
        core_points_map[core1], key=lambda node: euclidean_distance(node, pivot2)
    )
    mst.add_edge(pivot1, pivot2, weight=euclidean_distance(pivot1, pivot2))
    print(f"merging {core1} and {core2} with pivot {pivot1} and {pivot2}")


def merge_phase(G):
    core_points_map = {}
    for component in nx.connected_components(G):
        centroid = np.mean([node for node in component], axis=0)
        closest_point = min(
            component, key=lambda node: euclidean_distance(node, centroid)
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

    sorted_core_points = sorted(core_points, key=lambda point: point[min_diff_axis])
    for core1, core2 in zip(sorted_core_points, sorted_core_points[1:]):
        merge_comps(core1, core2, core_points_map, G)


def dcrun():
    G, neighbours = build()
    print(G.number_of_nodes(), G.number_of_edges())
    k = 0
    while k**k < maxdis:

        if (len(connected_components := list(nx.connected_components(G)))) == 1:
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
        print("Connected Components : ", len(list(nx.connected_components(G))))
    edgecount = G.number_of_edges()
    mst = nx.minimum_spanning_tree(G)
    print(mst.number_of_nodes(), mst.number_of_edges())
    mst_weight = sum(data["weight"] for u, v, data in mst.edges(data=True))
    print(
        f"Minimum Spanning Tree: {mst.number_of_nodes()} nodes, {mst.number_of_edges()} edges, Total Weight: {mst_weight}"
    )
    return mst_weight, edgecount


dc_weight, dc_edgecount = dcrun()

dcran_end_time = time.time()
dcran_elapsed_time = dcran_end_time - dcran_start_time

prim_end_time = time.time()
prim_elapsed_time = prim_end_time - prim_start_time

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

    for _ in range(n):
        min_index = -1
        for i in range(n):
            if not visited[i] and (
                min_index == -1 or min_cost[i] < min_cost[min_index]
            ):
                min_index = i

        visited[min_index] = True

        for i in range(n):
            if not visited[i]:
                distance = calculate_distance(points[min_index], points[i])
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
    eprim_wt := sum([calculate_distance(points[u], points[v]) for u, v in mst]),
)
print("time taken by dcran : ", eprim_elapsed_time)

speedup = eprim_elapsed_time / dcran_elapsed_time
print(f"Speedup: {speedup:.2f}")

wt_error = abs(dc_weight - eprim_wt) / eprim_wt * 100
print(f"Weight Error: {wt_error}%")

from sklearn.cluster import KMeans
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

fmst_start_time = time.time()


def euclidean_distance(p1, p2):
    return sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5


def detect_connecting_edge(cluster1, cluster2):
    center1 = [sum(x) / len(cluster1) for x in zip(*cluster1)]
    center2 = [sum(x) / len(cluster2) for x in zip(*cluster2)]

    min_dist1 = float("inf")
    min_point1 = None
    for point in cluster1:
        dist = euclidean_distance(point, center2)
        if dist < min_dist1:
            min_dist1 = dist
            min_point1 = point

    min_dist2 = float("inf")
    min_point2 = None
    for point in cluster2:
        dist = euclidean_distance(point, center1)
        if dist < min_dist2:
            min_dist2 = dist
            min_point2 = point

    return min_point1, min_point2


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
        pairwise_distances = squareform(pdist(cluster_points))
        mst = minimum_spanning_tree(pairwise_distances).toarray().tolist()
        edges = [
            (u, v)
            for u in range(len(cluster_points))
            for v in range(u + 1, len(cluster_points))
            if mst[u][v] > 0
        ]
        for u, v in edges:
            weight = euclidean_distance(cluster_points[u], cluster_points[v])
            mst_edges.append((cluster_points[u], cluster_points[v]))
            total_weight += weight

    pairwise_distances_centers = squareform(pdist(centers))
    mst_centers = minimum_spanning_tree(pairwise_distances_centers).toarray().tolist()
    center_edges = [
        (i, j) for i in range(k) for j in range(i + 1, k) if mst_centers[i][j] > 0
    ]

    # Refinement stage
    midpoints = []
    for i, j in center_edges:
        cluster1 = [points[m] for m in range(n) if labels[m] == i]
        cluster2 = [points[m] for m in range(n) if labels[m] == j]
        u, v = detect_connecting_edge(cluster1, cluster2)
        mst_edges.append((u, v))
        midpoint = [(a + b) / 2 for a, b in zip(u, v)]
        midpoints.append(midpoint)
        weight = euclidean_distance(u, v)
        total_weight += weight

    kmeans_refine = KMeans(n_clusters=len(midpoints), init=midpoints, n_init=1).fit(
        points
    )
    labels_refine = kmeans_refine.labels_

    for i in range(len(midpoints)):
        cluster_points = [points[j] for j in range(n) if labels_refine[j] == i]
        pairwise_distances = squareform(pdist(cluster_points))
        mst = minimum_spanning_tree(pairwise_distances).toarray().tolist()
        edges = [
            (u, v)
            for u in range(len(cluster_points))
            for v in range(u + 1, len(cluster_points))
            if mst[u][v] > 0
        ]
        for u, v in edges:
            weight = euclidean_distance(cluster_points[u], cluster_points[v])
            mst_edges.append((cluster_points[u], cluster_points[v]))
            total_weight += weight

    return mst_edges, total_weight


mst_edges, fmst_weight = fast_mst(points)
fmst_end_time = time.time()
fmst_edgecount = len(mst_edges)
fmst_elapsed_time = fmst_end_time - fmst_start_time
print("time taken by fast mst : ", fmst_elapsed_time)

print("Total weight of the MST:", fmst_weight)

with open(savefile, "r") as f:
    loaded_data = json.load(f)
print(loaded_data)
currres = []
loaded_data.append(
    [
        points_count,
        no_centres,
        dc_weight,
        fmst_weight,
        eprim_wt,
        dc_edgecount,
        fmst_edgecount,
        abs(fmst_weight - eprim_wt) / eprim_wt * 100,
        abs(dc_weight - eprim_wt) / eprim_wt * 100,
        dcran_elapsed_time,
        fmst_elapsed_time,
        eprim_elapsed_time,
        eprim_elapsed_time / fmst_elapsed_time,
        eprim_elapsed_time / dcran_elapsed_time,
    ]
)

with open(savefile, "r") as f:
    loaded_data = json.load(f)
print(loaded_data)
currres = []
loaded_data.append(
    [
        points_count,
        no_centres,
        dc_weight,
        fmst_weight,
        eprim_wt,
        len(mst_edges),
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
    json.dump(loaded_data, f)

headers = [
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
    "Prim's Speedup",
    "DCRAN Speedup",
]

table_str = tabulate(
    loaded_data[-15:],
    headers,
    tablefmt="pipe",
    floatfmt=(
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
print(table_str)
