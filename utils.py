import math
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import itertools


def euclidean_distance(coord1, coord2):
    dis = math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)
    return dis


def count_connected_components(graph):
    return len(list(nx.connected_components(graph)))


def cusrandis(k, maxdis):
    windowsize = max(maxdis - k + 1, 1)
    lpp = np.random.laplace(0, windowsize**0.1)
    lpp = int(k + (abs(lpp)) % windowsize)

    return lpp


# def graphify(graph , to_plot):
#     # pos = {i: coord for coord, i in cordmap.items()}
#     if not to_plot :
#         return

#     pos = nx.get_node_attributes(graph, "pos")
#     nx.draw(
#         graph,
#         pos,
#         with_labels=True,
#         font_weight="bold",
#         node_color="lightgreen",
#         font_size=10,
#         node_size=700,
#     )
#     plt.show()


def graphify(graph, to_plot, bottom_text=""):
    if not to_plot:
        return

    pos = nx.get_node_attributes(graph, "pos")

    # Find all connected components
    connected_components = list(nx.connected_components(graph))

    # Define a color cycle
    colors = itertools.cycle(plt.cm.tab20.colors)

    # Create a color map for each node
    color_map = {}
    for component in connected_components:
        color = next(colors)
        for node in component:
            color_map[node] = color

    # Plot with edges
    plt.figure()
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=100,
        node_color=[color_map[node] for node in graph.nodes()],
        alpha=1,
    )
    nx.draw_networkx_edges(graph, pos, width=3.5, edge_color="gray", alpha=1)
    plt.title(f"")
    plt.show()

    # Plot without edges
    plt.figure()
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=100,
        node_color=[color_map[node] for node in graph.nodes()],
        alpha=1,
    )
    plt.title(f"")
    plt.show()
