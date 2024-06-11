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
            self.merge_phase(G, to_plot)

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

        return mst_weight, len(G.edges()), mst

    def kmist(self, to_plot=True):
        G, neighbours = self.build()

        if to_plot:
            print(
                f"K-MST Initial Graph: {len(G.nodes())} nodes, {len(G.edges())} edges"
            )

        k = 0

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
            self.merge_phase(G, to_plot)

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

        return mst_weight, len(G.edges()), mst

    def prim_mst(self, to_plot=True):
        n = len(self.points)
        if n == 0:
            return 0, 0, nx.Graph()

        in_mst = [False] * n
        min_edge = [float("inf")] * n
        parent = [-1] * n
        min_edge[0] = 0

        mst_edges = []
        total_weight = 0
        edge_count = 0

        for _ in range(n):
            u = -1
            for i in range(n):
                if not in_mst[i] and (u == -1 or min_edge[i] < min_edge[u]):
                    u = i

            in_mst[u] = True

            if parent[u] != -1:
                mst_edges.append((u, parent[u]))
                total_weight += self.euclidean_distance(
                    self.points[u], self.points[parent[u]]
                )

            for v in range(n):
                if (
                    not in_mst[v]
                    and self.euclidean_distance(self.points[u], self.points[v])
                    < min_edge[v]
                ):
                    min_edge[v] = self.euclidean_distance(
                        self.points[u], self.points[v]
                    )
                    parent[v] = u
                edge_count += 1

        total_weight = round(total_weight, 2)
        if to_plot:
            G = nx.Graph()
            for point in self.points:
                G.add_node(tuple(point), pos=point)

            for u, v in mst_edges:
                G.add_edge(
                    tuple(self.points[u]),
                    tuple(self.points[v]),
                    weight=self.euclidean_distance(self.points[u], self.points[v]),
                )
            print(
                f"Prim's MST: {len(G.nodes())} nodes, {edge_count} edges, Total Weight: {total_weight}"
            )
            graphify(
                G,
                to_plot,
                bottom_text=f"Prim's MST: {len(G.nodes())} nodes, \nTotal Weight: {total_weight}",
            )

        return total_weight, edge_count, G

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

        G = {point: [] for cluster in clusters for point in cluster}
        total_edges = 0

        colors = plt.cm.rainbow(np.linspace(0, 1, k))
        for cluster_index, (cluster, color) in enumerate(zip(clusters, colors)):
            if len(cluster) > 1:
                mst_edges = self.prim_mst_cluster(cluster)
                total_edges += (
                    len(cluster) * (len(cluster) - 1) // 2
                )  # Count all edges in Prim's MST

                for u, v in mst_edges:
                    G[cluster[u]].append(
                        (cluster[v], self.euclidean_distance(cluster[u], cluster[v]))
                    )
                    G[cluster[v]].append(
                        (cluster[u], self.euclidean_distance(cluster[u], cluster[v]))
                    )

                if to_plot:
                    cluster_points = np.array(cluster)
                    pos = {tuple(i): (i[0], i[1]) for i in cluster_points}
                    nx_graph = nx.Graph()
                    for u, v in mst_edges:
                        nx_graph.add_edge(cluster[u], cluster[v])
                    nx.draw(
                        nx_graph,
                        pos,
                        with_labels=False,
                        node_color=color,
                        edge_color=color,
                        alpha=0.5,
                    )

            if to_plot:
                print(
                    f"After processing cluster {cluster_index}, number of edges: {total_edges}"
                )
                print(f"Total weight of edges: {self.calculate_total_weight_fmst(G)}")

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

        if to_plot:
            print(
                f"After adding cluster points and centroids, number of edges: {total_edges}"
            )
            print(f"Total weight of edges: {self.calculate_total_weight_fmst(G)}")

        centroid_points = np.array(centroids)
        centroid_edges = self.prim_mst_cluster(centroid_points)
        total_edges += (
            len(centroid_points) * (len(centroid_points) - 1) // 2
        )  # Count all edges in Prim's MST

        if to_plot:
            print(f"After creating MST of centroids, number of edges: {total_edges}")
            print(
                f"Total weight of edges in centroid MST: {sum(self.euclidean_distance(centroid_points[u], centroid_points[v]) for u, v in centroid_edges)}"
            )

        core_points_map = {}
        for component in clusters:
            if len(component) > 0:
                centroid = tuple(np.mean([node for node in component], axis=0))
                closest_point = min(
                    component, key=lambda node: self.euclidean_distance(node, centroid)
                )
                core_points_map[closest_point] = component

        for centroid in centroids:
            if centroid not in core_points_map:
                closest_point = min(
                    G.keys(), key=lambda node: self.euclidean_distance(node, centroid)
                )
                core_points_map[centroid] = core_points_map[closest_point]

        # Ensure all centroids are mapped
        for centroid in centroids:
            if centroid not in core_points_map:
                closest_point = min(
                    core_points_map.keys(),
                    key=lambda node: self.euclidean_distance(node, centroid),
                )
                core_points_map[centroid] = core_points_map[closest_point]

        centroid_points = [tuple(point) for point in centroid_points]

        for edge in centroid_edges:
            self.merge_comps_fmst(
                centroid_points[edge[0]],
                centroid_points[edge[1]],
                core_points_map,
                G,
                to_plot,
            )
            total_edges += 1
            if to_plot:
                print(
                    f"After merging {centroid_points[edge[0]]} and {centroid_points[edge[1]]}, number of edges: {total_edges}"
                )
                print(f"Total weight of edges: {self.calculate_total_weight_fmst(G)}")

        if to_plot:
            plt.figure(figsize=(10, 8))
            pos = {node: node for node in G.keys()}
            nx_graph = nx.Graph()
            for node, edges in G.items():
                for neighbor, weight in edges:
                    nx_graph.add_edge(node, neighbor, weight=weight)
            nx.draw(
                nx_graph,
                pos,
                with_labels=False,
                node_color="lightblue",
                edge_color="gray",
                alpha=0.5,
            )
            plt.title("K-Means Clustering with MSTs and Merged Centroid MST Overlay")
            plt.xlabel("X-coordinate")
            plt.ylabel("Y-coordinate")
            plt.show()

        mst_weight = self.calculate_total_weight_fmst(G)
        return mst_weight, total_edges, G

    def prim_mst_cluster(self, points):
        n = len(points)
        if n == 0:
            return []

        in_mst = [False] * n
        min_edge = [float("inf")] * n
        parent = [-1] * n
        min_edge[0] = 0

        mst_edges = []

        for _ in range(n):
            u = -1
            for i in range(n):
                if not in_mst[i] and (u == -1 or min_edge[i] < min_edge[u]):
                    u = i

            in_mst[u] = True

            if parent[u] != -1:
                mst_edges.append((u, parent[u]))

            for v in range(n):
                if (
                    not in_mst[v]
                    and self.euclidean_distance(points[u], points[v]) < min_edge[v]
                ):
                    min_edge[v] = self.euclidean_distance(points[u], points[v])
                    parent[v] = u

        return mst_edges

    def merge_comps(self, core1, core2, core_points_map, G, to_plot):
        pivot1 = min(
            core_points_map[core1],
            key=lambda node: self.euclidean_distance(node, core2),
        )
        pivot2 = min(
            core_points_map[core2],
            key=lambda node: self.euclidean_distance(node, pivot1),
        )
        G.add_edge(pivot1, pivot2, weight=self.euclidean_distance(pivot1, pivot2))
        if to_plot:
            print(f"Merging {core1} and {core2} with pivot {pivot1} and {pivot2}")

    def merge_comps_fmst(self, core1, core2, core_points_map, G, to_plot):
        pivot1 = min(
            core_points_map[core1],
            key=lambda node: self.euclidean_distance(node, core2),
        )
        pivot2 = min(
            core_points_map[core2],
            key=lambda node: self.euclidean_distance(node, pivot1),
        )
        G[pivot1].append((pivot2, self.euclidean_distance(pivot1, pivot2)))
        G[pivot2].append((pivot1, self.euclidean_distance(pivot1, pivot2)))
        if to_plot:
            print(f"Merging {core1} and {core2} with pivot {pivot1} and {pivot2}")

    def merge_phase(self, G, to_plot):
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
            self.merge_comps(core1, core2, core_points_map, G, to_plot)

    @staticmethod
    def euclidean_distance(point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    @staticmethod
    def calculate_total_weight(G):
        total_weight = sum(data["weight"] for u, v, data in G.edges(data=True))
        return total_weight

    @staticmethod
    def calculate_total_weight_fmst(G):
        total_weight = sum(weight for edges in G.values() for _, weight in edges) / 2
        return total_weight

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
