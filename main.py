import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import sklearn
import random
import heapq
import numpy as np

# select area in Fenway central part
address = "Fenway, Boston, Massachusetts, USA"
graph = ox.graph_from_address(address, dist=200, network_type="all")


# preprocess data
# set category and related color
random.seed(0)
categories = ["Vietnamese", "Chinese", "Thai","American (Traditional)","Vegetarian","Italian","Bars","Mexican",]
colors = ["green", "blue", "yellow", "purple", "orange", "cyan", "magenta", "brown"]
node_colors = dict(zip(categories, colors))

# each restaurant category's restaurant minimum/maximum quantity
m = 2
n = 3

# classify nodes according to category
category_nodes = {category: [] for category in categories}
used_nodes = []
for category in categories:
    num_nodes = random.randint(m, n)
    i = 0
    while i < num_nodes:
        nodes = random.sample(list(graph.nodes),1)
        if nodes not in used_nodes:
            category_nodes[category].append(nodes[0])
            used_nodes.append(nodes[0])
            i += 1

for category, nodes in category_nodes.items():
    print(category + ":",nodes)

id_to_name = {node_id: category for category, node_ids in category_nodes.items() for node_id in node_ids}
#print(id_to_name)

# color restaurant nodes based on restaurant category
nc = [node_colors[id_to_name[node]] if node in id_to_name else 'white' for node in graph.nodes()]

# Heuristic function of A* function
def heuristic(node1, node2, type = 'euclidean'):
    # get the coordinates of the specific node
    x1, y1 = graph.nodes[node1]['x'], graph.nodes[node1]['y']
    x2, y2 = graph.nodes[node2]['x'], graph.nodes[node2]['y']
    # design different heuristic formula
    if type == 'manhattan':
        return abs(x1-x2) + abs(y1-y2)
    elif type == 'euclidean':
        return ((x1 - x2)**2 + (y1 - y2)**2)**0.5
    elif type == 'diagonal':
        dx = abs(x1-x2)
        dy = abs(y1-y2)
        return np.sqrt(2) * min(dx,dy) + abs(x1-x2) + abs(y1-y2)

# A* function
def astar(graph, start, goal, heuristic_type = 'euclidean'):
    # store nodes to be inspected
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}

    # the real cost
    g_score = {node: float('inf') for node in graph.nodes}
    g_score[start] = 0

    # estimate f = g+h
    f_score = {node: float('inf') for node in graph.nodes}
    f_score[start] = heuristic(start, goal, heuristic_type)

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1], g_score[goal]

        for neighbor in graph.neighbors(current):
            tentative_g_score = g_score[current] + graph[current][neighbor][0]['length']
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal, heuristic_type)
                if neighbor not in [i[1] for i in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return [], 0

# based on user's input category, calculate the shortest restaurant and return the shortest path
def shortestpath_to_category(graph, start_node, category):
    end_nodes = category_nodes[category]
    shortestpath = []

    # store the minimum distance of going to different restaurants
    mindistance = float('inf')
    for end_node in end_nodes:
        tmp_path, total_distance = astar(graph, start_node, end_node, heuristic_type = 'euclidean')

        if total_distance < mindistance:
            mindistance = total_distance
            shortestpath = tmp_path

    print("Shortest path:", shortestpath)
    print(f"Total distance {mindistance:.1f} meters")

    fig, ax = ox.plot_graph(graph, figsize=(10, 10), node_size=20, node_color=nc, show=False)

    ox.plot_graph_route(graph, shortestpath, route_linewidth=1, show=False,
                         orig_dest_size=2, close=False, ax=ax)

    legend_elements = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=10) for color in
                       node_colors.values()]
    legend_labels = list(category_nodes.keys())
    ax.legend(legend_elements, legend_labels, loc="lower left", fontsize=10)

    plt.show()

# print(graph.nodes)
#start_node = 7795781152
graph_nodes = list(graph.nodes._nodes.keys())
# set the start node
start_node = graph_nodes[100]
# set the category
category = 'Chinese'
# call the function to get the result
shortestpath_to_category(graph, start_node, category)




