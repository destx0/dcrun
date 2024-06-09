import math
import networkx as nx
import numpy as np
from kdtree import *
from utils import *
import matplotlib.pyplot as plt


class MST:
    def __init__(self, points):
        """
        Initialize the MST with a set of points.

        Parameters:
        points (list): List of points to build the KD-Tree and graph.
        """
        self.points = points
        self.points_count = len(points)

    def build(self):
        """
        Build the KD-Tree and graph from the points.

        Returns:
        G (networkx.Graph): Graph with points as nodes.
        neighbours (dict): Dictionary of neighbors for each point.
        """
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
        """
        Create a minimum spanning tree using the k-nearest neighbors approach.

        Parameters:
        to_plot (bool): Whether to plot the graph and MST.

        Returns:
        mst_weight (float): Total weight of the minimum spanning tree.
        edge_count (int): Number of edges in the final graph.
        G (networkx.Graph): The final graph.
        """
        G, neighbours = self.build()

        if to_plot:
            print(f"Initial Graph: {len(G.nodes())} nodes, {len(G.edges())} edges")

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
                    f"Iteration {k}: {current_connected_components} connected components"
                )
                print(f"Graph: {len(G.nodes())} nodes, {len(G.edges())} edges")

            if current_connected_components == 1:
                break

            for component in connected_components:
                for node in component:
                    wt, pos = neighbours[node][k]
                    G.add_edge(node, pos, weight=wt)

            if to_plot:
                graphify(G, to_plot, bottom_text=f"Iteration {k}")

            k += 1

        if to_plot:
            print(f"Final Graph: {len(G.nodes())} nodes, {len(G.edges())} edges")

        connected_components = list(nx.connected_components(G))
        current_connected_components = len(connected_components)

        if to_plot:
            print(f"Final Connected Components: {current_connected_components}")

        mst = nx.minimum_spanning_tree(G)
        mst_weight = round(
            sum(data["weight"] for u, v, data in mst.edges(data=True)), 2
        )

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

    def kmist(self, to_plot=True):
        """
        Create a minimum spanning tree using a custom approach where nodes are connected
        based on the k-th and k^k-th nodes.

        Parameters:
        to_plot (bool): Whether to plot the graph and MST.

        Returns:
        mst_weight (float): Total weight of the minimum spanning tree.
        edge_count (int): Number of edges in the final graph.
        G (networkx.Graph): The final graph.
        """
        G, neighbours = self.build()

        if to_plot:
            print(f"Initial Graph: {len(G.nodes())} nodes, {len(G.edges())} edges")

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
                    f"Iteration {k}: {current_connected_components} connected components"
                )
                print(f"Graph: {len(G.nodes())} nodes, {len(G.edges())} edges")

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
                    G.add_edge(node, k_th_node)
                if k_k_th_node:
                    G.add_edge(node, k_k_th_node)

            if to_plot:
                graphify(G, to_plot, bottom_text=f"Iteration {k}")

            k += 1

        if to_plot:
            print(f"Final Graph: {len(G.nodes())} nodes, {len(G.edges())} edges")

        connected_components = list(nx.connected_components(G))
        current_connected_components = len(connected_components)

        if to_plot:
            print(f"Final Connected Components: {current_connected_components}")

        mst = nx.minimum_spanning_tree(G)
        mst_weight = round(
            sum(data["weight"] for u, v, data in mst.edges(data=True)), 2
        )

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

    def prim_mst(self, to_plot=True):
        """
        Compute the minimum spanning tree using Prim's algorithm.

        Parameters:
        to_plot (bool): Whether to plot the graph and MST.

        Returns:
        total_weight (float): Total weight of the minimum spanning tree.
        """
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
                    distance = self.calculate_distance(
                        self.points[min_index], self.points[i]
                    )
                    if distance < min_cost[i]:
                        min_cost[i] = distance
                        parent[i] = min_index

        total_weight = 0
        for i in range(1, n):
            G.add_edge(parent[i], i, weight=min_cost[i])
            total_weight += min_cost[i]

        total_weight = round(total_weight, 2)  # Rounding to 2 decimal places

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

    @staticmethod
    def calculate_distance(point1, point2):
        """
        Calculate the Euclidean distance between two points.

        Parameters:
        point1 (tuple): The first point.
        point2 (tuple): The second point.

        Returns:
        float: The Euclidean distance between the two points.
        """
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

    def apply_mst(self, algorithm="kmistree", to_plot=True):
        """
        Apply the specified MST algorithm.

        Parameters:
        algorithm (str): The MST algorithm to apply ("kmistree", "kmist", or "prim").
        to_plot (bool): Whether to plot the graph and MST.

        Returns:
        result (varies): The result of the specified MST algorithm.
        """
        if algorithm == "kmistree":
            return self.kmistree(to_plot=to_plot)
        elif algorithm == "kmist":
            return self.kmist(to_plot=to_plot)
        elif algorithm == "prim":
            return self.prim_mst(to_plot=to_plot)
        else:
            raise ValueError(
                f"Unsupported algorithm: {algorithm}. Choose 'kmistree', 'kmist', or 'prim'."
            )


# Example usage:
points = generate_dataset(
    dataset_type="circles", points_count=100, noise=0.1, no_centres=3, to_plot=False
)
mst_builder = MST(points)
mst_weight, edge_count, final_graph = mst_builder.apply_mst(
    algorithm="kmistree", to_plot=True
)
print(f"K-MSTree: Total Weight: {mst_weight}, Edge Count: {edge_count}")

mst_weight, edge_count, final_graph = mst_builder.apply_mst(
    algorithm="kmist", to_plot=True
)
print(f"K-MST: Total Weight: {mst_weight}, Edge Count: {edge_count}")

prim_weight = mst_builder.apply_mst(algorithm="prim", to_plot=True)
print(
    f"Prim's MST: {len(points)} nodes, {len(points) - 1} edges, Total Weight: {prim_weight}"
)
