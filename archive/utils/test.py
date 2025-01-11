# import random
#
# import networkx as nx
# import numpy as np
# import skfmm
# from matplotlib import pyplot as plt
# from matplotlib import colors
# from tqdm import tqdm
#
#
# def create_grid():
#     grid = np.zeros((6, 9))
#     grid[1:3, 1:4] = 1
#     grid[0, 2] = 1
#     grid[2, 1:-1] = 1
#     grid[3, 4:7] = 1
#     grid[-2:, 5] = 1
#     grid[:3, -2] = 1
#     grid[1, -1] = 1
#     grid[1, 4:7] = 1
#     start, end = (1, 1), (0, 7)
#     return start, end, grid
#
#
# def visualize(data):
#     # create discrete colormap
#     cmap = colors.ListedColormap(['red', 'blue'])
#     bounds = [0, 1]
#     norm = colors.BoundaryNorm(bounds, cmap.N)
#
#     fig, ax = plt.subplots()
#     ax.imshow(data, cmap=cmap, norm=norm)
#
#     # draw gridlines
#     ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
#     ax.set_xticks(np.arange(-.5, grid.shape[1], 1))
#     ax.set_yticks(np.arange(-.5, grid.shape[0], 1))
#
#     plt.show()
#
#
# def calculate_sdf(data):
#     # Compute signed distance
#     return skfmm.distance(data)
#
# def find_path(graph, start, end):
#     possible_pahts = list(nx.shortest_simple_paths(graph, source=start, target=end, weight='weight'))
#     best_score = np.inf
#     best_path = None
#
#     for path in tqdm(possible_pahts):
#         score = np.sum([sdf[y][x] for y, x in path])
#         if score < best_score:
#             best_score = score
#             best_path = path
#     print(best_score)
#     return best_path
#     # return nx.shortest_path(graph, source=start, target=end, weight='weight')
#
#
# def create_weighted_graph(data, sdf):
#     ys, xs = np.where(data)
#     pos = {(ys[i], xs[i]): (ys[i], xs[i]) for i, _ in enumerate(ys)}
#
#     y_max, x_max = data.shape
#     y_max, x_max = y_max - 1, x_max - 1
#
#     G = nx.Graph()
#
#     for (y, x) in pos.keys():
#         current_sdf = sdf[y][x]
#         G.add_node((y, x))
#
#         # Check 4 possible neighbors (no diagonal)
#         # Left
#         if (x-1) >= 0 and data[y][x-1]:
#             G.add_edge((y, x), (y, x-1), weight=(current_sdf + sdf[y][x-1]) / 2)
#
#         # Right
#         if (x+1) <= x_max and data[y][x+1]:
#             G.add_edge((y, x), (y, x+1), weight=(current_sdf + sdf[y][x+1]) / 2)
#
#         # Upper
#         if (y-1) >= 0 and data[y-1][x]:
#             G.add_edge((y, x), (y-1, x), weight=(current_sdf + sdf[y-1][x]) / 2)
#
#         # Lower
#         if (y+1) <= y_max and data[y+1][x]:
#             G.add_edge((y, x), (y+1, x), weight=(current_sdf + sdf[y+1][x]) / 2)
#
#     return G, pos
#
#
# def visualize_graph(G, pos, start, end, path=None):
#     # nodes
#     nx.draw_networkx_nodes(G, pos, node_size=600)
#
#     # edges
#     nx.draw_networkx_edges(G, pos, style='dashed')
#
#     # node labels
#     nx.draw_networkx_labels(G, pos, font_size=8)
#
#     # edge weight labels
#     edge_labels = nx.get_edge_attributes(G, "weight")
#     edge_labels = {k: round(v, 2) for k, v in edge_labels.items()}
#     nx.draw_networkx_edge_labels(G, pos, edge_labels)
#
#     nx.draw_networkx_nodes(G.subgraph(start), pos, node_size=600, node_color='green')
#     nx.draw_networkx_nodes(G.subgraph(end), pos, node_size=600, node_color='red')
#
#     if path is not None:
#         nx.draw_networkx_edges(G.subgraph(path), pos, style='dashed', edge_color='red')
#
#     plt.show()
#
#
# if __name__ == "__main__":
#     start, end, grid = create_grid()
#     # visualize(grid)
#
#     sdf = calculate_sdf(grid)
#
#     # Edge weight is determined by sdf, but ideal path has minimal weight, so high SDF (center of area) = ideal and low SDF (at border) = worst
#     # Therefore 1/sdf s.t. ideal place has lowest SDF, and worst place has highest SDF
#     # sdf = 1/(sdf+1e-5)  # TODO: might be numerically unstable if SDF has 0 values
#     sdf = abs(sdf-sdf.max())
#     plt.imshow(sdf)
#     plt.colorbar()
#     plt.show()
#
#     G, pos = create_weighted_graph(grid, sdf)
#     # visualize_graph(G, pos, start, end)
#
#     path = find_path(G, start, end)
#     visualize_graph(G, pos, start, end, path=path)
#
#     pass
