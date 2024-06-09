import math
import networkx as nx
import numpy as np
from kdtree import *
from utils import *
import matplotlib.pyplot as plt


def build(points, points_count):
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


def kmistree(points, to_plot=True):
    points_count = len(points)
    G, neighbours = build(points, points_count)

    if to_plot:
        print(f"Nodes: {len(G.nodes())}, Edges: {len(G.edges())}")

    k = 0
    prev_connected_components = np.inf
    prev_G = G.copy()

    if to_plot:
        graphify(G, to_plot, bottom_text="")

    maxdis = math.ceil(math.log2(points_count))

    while k < maxdis:
        connected_components = list(nx.connected_components(G))
        current_connected_components = len(connected_components)

        if to_plot:
            print(f"Connected Components: {current_connected_components}")
            print(f"The graph has {len(G.nodes())} nodes and {len(G.edges())} edges.")

        if current_connected_components == 1:
            break

        for component in connected_components:
            for node in component:
                wt, pos = neighbours[node][k]
                G.add_edge(node, pos, weight=wt)

        if to_plot:
            graphify(G, to_plot, bottom_text="")

        k += 1

    if to_plot:
        print(f"Minimum Spanning Tree: {len(G.nodes())} nodes, {len(G.edges())} edges")

    connected_components = list(nx.connected_components(G))
    current_connected_components = len(connected_components)

    if to_plot:
        print(f"Connected Components: {current_connected_components}")

    mst = nx.minimum_spanning_tree(G)
    mst_weight = round(sum(data["weight"] for u, v, data in mst.edges(data=True)), 2)

    if to_plot:
        print(
            f"Minimum Spanning Tree: {len(mst.nodes())} nodes, {len(mst.edges())} edges, Total Weight: {mst_weight}"
        )
        graphify(
            mst,
            to_plot,
            bottom_text=f"Minimum Spanning Tree: {len(G.nodes())} nodes, \nTotal Weight: {mst_weight}",
        )

    return mst_weight, len(G.edges()), G
