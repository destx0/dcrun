import math
import numpy as np
import networkx as nx
from kdtree import *
from utils import *
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform


class MST:
    def __init__(self, points):
        self.points = points
        self.points_count = len(points)

    def build(self):
        tree = KDTree()
        tree.root = tree.build(self.points)

        G = nx.Graph()

        for point in self.points:
            G.add_node(point, pos=point)

        neighbours = {}
        maxdis = math.ceil(math.log2(self.points_count))
        for point in self.points:
            neighbours[point] = i_neighbors(tree, point, maxdis)

        return G, neighbours

    def kmistree(self, to_plot=True):
        G, neighbours = self.build()

        if to_plot:
            print(
                f"K-MSTree Initial Graph: {len(G.nodes())} nodes, {len(G.edges())} edges"
            )

        k = 0
        prev_connected_components = np.inf

        if to_plot:
            graphify(G, to_plot, bottom_text="Initial Graph")

        maxdis = math.ceil(math.log2(self.points_count))

        while k < maxdis:
            connected_components = list(nx.connected_components(G))
            current_connected_components = len(connected_components)

            if to_plot:
                print(
                    f"K-MSTree Iteration {k}: {current_connected_components} connected components"
                )
                print(f"K-MSTree Graph: {len(G.nodes())} nodes, {len(G.edges())} edges")

            if current_connected_components == 1:
                break

            for component in connected_components:
                for node in component:
                    wt, pos = neighbours[node][k]
                    G.add_edge(node, pos, weight=wt)

            if to_plot:
                graphify(G, to_plot, bottom_text=f"Iteration {k}")

            k += 1

        connected_components = list(nx.connected_components(G))
        current_connected_components = len(connected_components)

        if current_connected_components > 1:
            self.merge_phase(G)

        if to_plot:
            print(
                f"K-MSTree Final Graph: {len(G.nodes())} nodes, {len(G.edges())} edges"
            )
            print(
                f"K-MSTree Final Connected Components: {current_connected_components}"
            )

        mst = nx.minimum_spanning_tree(G)
        mst_weight = round(
            sum(data["weight"] for u, v, data in mst.edges(data=True)), 2
        )

        if to_plot:
            print(
                f"K-MSTree Minimum Spanning Tree: {len(mst.nodes())} nodes, {len(mst.edges())} edges, Total Weight: {mst_weight}"
            )
            graphify(
                mst,
                to_plot,
                bottom_text=f"Minimum Spanning Tree: {len(G.nodes())} nodes, \nTotal Weight: {mst_weight}",
            )

        return mst_weight, len(G.edges()), G

    def kmist(self, to_plot=True):
        G, neighbours = self.build()

        if to_plot:
            print(
                f"K-MST Initial Graph: {len(G.nodes())} nodes, {len(G.edges())} edges"
            )

        k = 0
        prev_connected_components = np.inf

        if to_plot:
            graphify(G, to_plot, bottom_text="Initial Graph")

        maxdis = math.ceil(math.log2(self.points_count))

        while k**k < maxdis:
            connected_components = list(nx.connected_components(G))
            current_connected_components = len(connected_components)

            if to_plot:
                print(
                    f"K-MST Iteration {k}: {current_connected_components} connected components"
                )
                print(f"K-MST Graph: {len(G.nodes())} nodes, {len(G.edges())} edges")

            if current_connected_components == 1:
                break

            for node in G.nodes():
                k_th_node = (
                    neighbours[node][k][1] if k < len(neighbours[node]) else None
                )
                k_k_th_node = (
                    neighbours[node][k**k][1]
                    if k**k < len(neighbours[node])
                    else None
                )

                if k_th_node:
                    G.add_edge(
                        node, k_th_node, weight=self.euclidean_distance(node, k_th_node)
                    )
                if k_k_th_node:
                    G.add_edge(
                        node,
                        k_k_th_node,
                        weight=self.euclidean_distance(node, k_k_th_node),
                    )

            if to_plot:
                graphify(G, to_plot, bottom_text=f"Iteration {k}")

            k += 1

        connected_components = list(nx.connected_components(G))
        current_connected_components = len(connected_components)

        if current_connected_components > 1:
            self.merge_phase(G)

        if to_plot:
            print(f"K-MST Final Graph: {len(G.nodes())} nodes, {len(G.edges())} edges")
            print(f"K-MST Final Connected Components: {current_connected_components}")

        mst = nx.minimum_spanning_tree(G)
        mst_weight = round(
            sum(data["weight"] for u, v, data in mst.edges(data=True)), 2
        )

        if to_plot:
            print(
                f"K-MST Minimum Spanning Tree: {len(mst.nodes())} nodes, {len(mst.edges())} edges, Total Weight: {mst_weight}"
            )
            graphify(
                mst,
                to_plot,
                bottom_text=f"Minimum Spanning Tree: {len(G.nodes())} nodes, \nTotal Weight: {mst_weight}",
            )

        return mst_weight, len(G.edges()), G

    def prim_mst(self, to_plot=True):
        n = len(self.points)
        visited = [False] * n
        min_cost = [float("inf")] * n
        parent = [None] * n

        min_cost[0] = 0
        parent[0] = -1

        G = nx.Graph()
        for i in range(n):
            G.add_node(i, pos=self.points[i])

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
                    distance = self.euclidean_distance(
                        self.points[min_index], self.points[i]
                    )
                    if distance < min_cost[i]:
                        min_cost[i] = distance
                        parent[i] = min_index

        total_weight = 0
        for i in range(1, n):
            G.add_edge(parent[i], i, weight=min_cost[i])
            total_weight += min_cost[i]

        total_weight = round(total_weight, 2)

        if to_plot:
            print(
                f"Prim's MST: {len(G.nodes())} nodes, {len(G.edges())} edges, Total Weight: {total_weight}"
            )
            graphify(
                G,
                to_plot,
                bottom_text=f"Prim's MST: {len(G.nodes())} nodes, \nTotal Weight: {total_weight}",
            )

        return total_weight

    def fmst(self, to_plot=True):
        n = len(self.points)
        k = int(n**0.5)

        kmeans = KMeans(n_clusters=k).fit(self.points)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_.tolist()

        clusters = [[] for _ in range(k)]
        for point, label in zip(self.points, labels):
            clusters[label].append(tuple(point))

        centroids = []
        for cluster in clusters:
            cluster_points = np.array(cluster)
            centroid = tuple(cluster_points.mean(axis=0))
            centroids.append(centroid)

        if to_plot:
            plt.figure(figsize=(10, 8))

        G = nx.Graph()

        colors = plt.cm.rainbow(np.linspace(0, 1, k))
        for cluster_index, (cluster, color) in enumerate(zip(clusters, colors)):
            if len(cluster) > 1:
                cluster_points = np.array(cluster)
                distances = pdist(cluster_points)
                dist_matrix = squareform(distances)

                cluster_graph = nx.Graph()
                for i in range(len(cluster_points)):
                    for j in range(i + 1, len(cluster_points)):
                        cluster_graph.add_edge(
                            tuple(cluster_points[i]),
                            tuple(cluster_points[j]),
                            weight=dist_matrix[i, j],
                        )

                mst = nx.minimum_spanning_tree(cluster_graph)
                G.add_edges_from(mst.edges(data=True))

                if to_plot:
                    pos = {tuple(i): (i[0], i[1]) for i in cluster_points}
                    nx.draw(
                        mst,
                        pos,
                        with_labels=False,
                        node_color=color,
                        edge_color=color,
                        alpha=0.5,
                    )

            print(
                f"After processing cluster {cluster_index}, number of edges: {G.number_of_edges()}"
            )

        if to_plot:
            for cluster_index, (cluster, color) in enumerate(zip(clusters, colors)):
                cluster_points = np.array(cluster)
                plt.scatter(
                    cluster_points[:, 0],
                    cluster_points[:, 1],
                    color=color,
                    label=f"Cluster {cluster_index}",
                    s=50,
                )
                centroid = centroids[cluster_index]
                plt.scatter(
                    centroid[0],
                    centroid[1],
                    color="black",
                    marker="x",
                    s=100,
                    label=f"Centroid {cluster_index}",
                )

        print(
            f"After adding cluster points and centroids, number of edges: {G.number_of_edges()}"
        )

        centroid_points = np.array(centroids)
        centroid_distances = pdist(centroid_points)
        centroid_dist_matrix = squareform(centroid_distances)

        centroid_graph = nx.Graph()
        for i in range(len(centroid_points)):
            for j in range(i + 1, len(centroid_points)):
                centroid_graph.add_edge(
                    tuple(centroid_points[i]),
                    tuple(centroid_points[j]),
                    weight=centroid_dist_matrix[i, j],
                )

        centroid_mst = nx.minimum_spanning_tree(centroid_graph)

        print(
            f"After creating MST of centroids, number of edges: {centroid_mst.number_of_edges()}"
        )

        core_points_map = {}
        for component in nx.connected_components(G):
            centroid = tuple(np.mean([node for node in component], axis=0))
            closest_point = min(
                component, key=lambda node: self.euclidean_distance(node, centroid)
            )
            core_points_map[closest_point] = component

        for centroid in centroids:
            if centroid not in core_points_map:
                closest_point = min(
                    G.nodes, key=lambda node: self.euclidean_distance(node, centroid)
                )
                core_points_map[centroid] = core_points_map[closest_point]

        for edge in centroid_mst.edges:
            self.merge_comps(edge[0], edge[1], core_points_map, G)
            print(
                f"After merging {edge[0]} and {edge[1]}, number of edges: {G.number_of_edges()}"
            )

        if to_plot:
            plt.figure(figsize=(10, 8))
            pos = {i: (i[0], i[1]) for i in G.nodes}
            nx.draw(
                G,
                pos,
                with_labels=False,
                node_color="lightblue",
                edge_color="gray",
                alpha=0.5,
            )
            plt.title("K-Means Clustering with MSTs and Merged Centroid MST Overlay")
            plt.xlabel("X-coordinate")
            plt.ylabel("Y-coordinate")
            plt.legend()
            plt.show()

        mst_weight = round(sum(data["weight"] for u, v, data in G.edges(data=True)), 2)
        return mst_weight, len(G.edges()), G

    @staticmethod
    def euclidean_distance(point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def merge_phase(self, G):
        core_points_map = {}
        for component in nx.connected_components(G):
            centroid = np.mean([node for node in component], axis=0)
            closest_point = min(
                component, key=lambda node: self.euclidean_distance(node, centroid)
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
            self.merge_comps(core1, core2, core_points_map, G)

    def merge_comps(self, core1, core2, core_points_map, G):
        pivot2 = min(
            core_points_map[core2],
            key=lambda node: self.euclidean_distance(node, core1),
        )
        pivot1 = min(
            core_points_map[core1],
            key=lambda node: self.euclidean_distance(node, pivot2),
        )
        G.add_edge(pivot1, pivot2, weight=self.euclidean_distance(pivot1, pivot2))
        print(f"Merging {core1} and {core2} with pivot {pivot1} and {pivot2}")

    def apply_mst(self, algorithm="kmistree", to_plot=True):
        if algorithm == "kmistree":
            return self.kmistree(to_plot=to_plot)
        elif algorithm == "kmist":
            return self.kmist(to_plot=to_plot)
        elif algorithm == "prim":
            return self.prim_mst(to_plot=to_plot)
        elif algorithm == "fmst":
            return self.fmst(to_plot=to_plot)
        else:
            raise ValueError(
                f"Unsupported algorithm: {algorithm}. Choose 'kmistree', 'kmist', 'prim', or 'fmst'."
            )
