import argparse
import copy
import networkx as nx
import os

from functions.analysis.GPT import query_chatgpt
from functions.analysis.main import adjust_color_by_level, calculate_all_transition_probabilities, analyze_graph
from functions.converters.adj_to_networkx import create_directed_graph_from_adj_matrix
from functions.converters.graph_to_mermaid import export_to_mermaid
from functions.converters.json_to_matrix import json_to_matrix, graph_to_json, generate_node_capacity_template
from functions.inserts.main import insert_node_capacity, truncate_node_capacity_data, fetch_node_capacity
from functions.node_data_manipulation.change import update_node_style_parameter_in_redis
from functions.readers.json_readers import read_json_file
from functions.redis_wrapped.conn import get_redis_connection, connect_to_postgres
from functions.redis_wrapped.json_to_redis import json_to_redis
from simulations.sim import traverse_any_given_graph_and_visualize, simulate_errors

parser = argparse.ArgumentParser()

parser.add_argument('--json_nodes_filename', '-jnf', default="data/nodes.json")
parser.add_argument('--repopulate_node_cap_data', '-rncd', action="store_true")
parser.add_argument('--sim_mode', '-sm', type=str, default="sim", choices=["sim", "intensity", "live"])
parser.add_argument('--global_rps', type=int, required=True, default=1500)
parser.add_argument('--global_rps_fluctuation_percentage', "-grfp", type=float, default=10.0)
parser.add_argument('--number_of_simulations', "-nof", default=1_000_000, type=int)
# parser.add_argument('--include_index', action="store_true")
# parser.add_argument('--include_header', action="store_true")

args = parser.parse_args()

print(f"Global ArchFlow Configuration")
for arg in vars(args):
    print(f"--{arg} {getattr(args, arg)}")

_ = input(f"Double check configuration and press [<any key> + Enter] to continue...")

json_nodes_filename = args.json_nodes_filename
repopulate_node_cap_data = args.repopulate_node_cap_data
sim_mode = args.sim_mode
global_rps = args.global_rps
global_rps_fluctuation_percentage = args.global_rps_fluctuation_percentage
number_of_simulations = args.number_of_simulations

json_data = read_json_file(json_nodes_filename)

adj_matrix, nodes, node_names, edges = json_to_matrix(data=json_data)

node_id_label_map = {key: value for key, value in zip(nodes, node_names)}

graph = create_directed_graph_from_adj_matrix(adj_matrix_in=adj_matrix, node_names=nodes)

"""
============================================================================================ PARAMETERS
"""
start_node = 'n0'  # Assuming you want to start from node FE

REPOPULATE_NODE_CAP_DATA = repopulate_node_cap_data

parameter = "is_highlighted"  # Specify the parameter within style you want to change

increment_amount = 0.1

GLOBAL_RPS = global_rps
GLOBAL_RPS_FLUCTUATION_PERCENTAGE = 10

if GLOBAL_RPS >= 1000:
    print(f"WARNING: Having high RPS resutls in a slower computation time.")

sim = False
intensity = False
live = False

if sim_mode == "intensity":
    intensity = True

if sim_mode == "live":
    live = True

if sim_mode == "sim":
    sim = True

number_of_simulations = 1_000_000

"""
============================================================================================ PARAMETERS
"""

r = get_redis_connection()
pg_conn = connect_to_postgres()

r.flushall()

if REPOPULATE_NODE_CAP_DATA:
    truncate_node_capacity_data(pg_connection=pg_conn)

SIMULATE_ERRORS = False

if SIMULATE_ERRORS:
    graph_with_errors, edges_with_errors = simulate_errors(graph_in=copy.deepcopy(graph), error_rate=1,
                                                           error_weight_in=0.1)

    graph = graph_with_errors
    edges.extend(edges_with_errors)

export_to_mermaid(graph_in=graph)

# graph_cp = copy.deepcopy(graph)
#
# adjusted_graph, new_edges = replicate_nodes_in_graph_and_track_edges(graph_cp, start_node, replication_percentile=99,
#                                                                      min_probability_threshold=0.2)
# edges.extend(new_edges)  # Combine original edges with new edges
#
# graph = adjusted_graph
#
# updated_json_data = graph_to_json(graph)
#
# json_to_redis(json_data=updated_json_data, redis_conn=r)

# Move Json Data to Redis for faster traversal and change
json_to_redis(json_data=graph_to_json(graph), redis_conn=r)

edge_betweenness = nx.edge_betweenness_centrality(graph)

critical_link = max(edge_betweenness, key=edge_betweenness.get)
print("Critical Link by Betweenness:", critical_link)
r.rpush('critical_links', *critical_link)

# degree_centrality = nx.in_degree_centrality(graph)
# betweenness_centrality = nx.betweenness_centrality(graph)
# closeness_centrality = nx.closeness_centrality(graph)
# pagerank = nx.pagerank(graph)
# clustering = nx.clustering(graph)
#
# # Print results
# print("Degree Centrality:", degree_centrality)
# print("Betweenness Centrality:", betweenness_centrality)
# print("Closeness Centrality:", closeness_centrality)
# print("PageRank:", pagerank)

node_capacity_data = generate_node_capacity_template(graph)

if REPOPULATE_NODE_CAP_DATA:
    insert_node_capacity(node_capacity_data=node_capacity_data, pg_connection=pg_conn)
    _ = input(f"You have 10 minutes to fill-in Node Capacity data in PostgreSQL. (Press ANY key to skip)...")

node_capacity_data = fetch_node_capacity(pg_connection=pg_conn)

full_probabilities = calculate_all_transition_probabilities(graph, start_node)

# Pre-Checking Capacity threshold before actually seeing it on Grafana
for node, hit_probability in full_probabilities.items():
    pessimistic_hit_for_node = GLOBAL_RPS + (
            GLOBAL_RPS * GLOBAL_RPS_FLUCTUATION_PERCENTAGE / 100)  # Individual Node number of requests coming in
    approx_n_of_hits_node = pessimistic_hit_for_node * hit_probability

    node_cpu_cap = node_capacity_data[node]['cpu_io_capacity']

    if approx_n_of_hits_node >= node_cpu_cap:
        print(f"WARNING: Node - {node} is beyond a threshold. Needs to be replicated")

    # print(f"NODE: {node}    =>>>>>>>> H:{approx_n_of_hits_node} C:{}")

analysis_results = analyze_graph(G=graph, redis_connection=r, print_or_not=False)

prompt = analysis_results
response = query_chatgpt(prompt)
r.set("chatgpt_recommendation", response)

# analyze_and_recommend(G=graph)

traverse_any_given_graph_and_visualize(global_rps=GLOBAL_RPS,
                                       global_rps_fluctuation_percentage=GLOBAL_RPS_FLUCTUATION_PERCENTAGE,
                                       node_capacity_data=node_capacity_data, pg_conn=pg_conn,
                                       redis_conn=r, method=sim_mode,
                                       graph=graph, edges=edges,
                                       start_node=start_node,
                                       parameter=parameter, number_of_simulations=1000,
                                       increment_amount=increment_amount, markovian=False)

exit()

time.sleep(3)

FULL_PROBABILITY = True
REMOVE_STARTING_NODE = True

edge_probabilities = calculate_transition_probabilities(graph, start_node)
# full_probabilities = calculate_all_transition_probabilities(graph, start_node)


graph_cp = copy.deepcopy(graph)

# best_config, lowest_cv, best_adjusted_graph, best_new_edges = grid_search_replication(
#     graph_cp, start_node)
#
# print(best_config, lowest_cv)

# exit()

adjusted_graph, new_edges = replicate_nodes_in_graph_and_track_edges(graph_cp, start_node, replication_percentile=99,
                                                                     min_probability_threshold=0.1)
edges.extend(new_edges)  # Combine original edges with new edges

graph = adjusted_graph
# edges = new_edges

graph_analysis_1 = graph_analysis(graph=graph)
results_to_html_table(graph_analysis_1)

_good_green_hex = "#42f545"
updated_json_data = graph_to_json(graph)

# update_redis_with_graph(graph=adjusted_graph, redis_conn=r)

r.flushall()
json_to_redis(json_data=updated_json_data, redis_conn=r)

levels, full_probabilities = calculate_levels_and_probabilities(graph, start_node)

if FULL_PROBABILITY:
    transition_probabilities = full_probabilities
else:
    transition_probabilities = edge_probabilities

if REMOVE_STARTING_NODE:
    transition_probabilities.pop(start_node)

# pprint(transition_probabilities)

cv = calculate_evenness(transition_probabilities)
print(f"Coefficient of Variation: {cv:.2f}  (Closer to zero better)")

outlier_nodes = identify_outliers(transition_probabilities)  # TODO: Handle outlier nodes

# Find min and max probabilities
min_prob = min(transition_probabilities.values())
max_prob = max(transition_probabilities.values())

COLOR_IN_ADVANCE_EDGES = True  # The node ID you want to update


def color_the_pre_nodes():
    color_map = adjust_color_by_level(probabilities=transition_probabilities, node_levels=levels)
    # Display or use the colors
    for node, color in color_map.items():
        update_node_style_parameter_in_redis(redis_connection=r, node_id=node,
                                             parameter="fillColor", new_value=color)


if COLOR_IN_ADVANCE_EDGES:
    color_the_pre_nodes()

traverse_any_given_graph_and_visualize(global_rps=GLOBAL_RPS, redis_conn=r, method=sim_mode, graph=graph, edges=edges,
                                       start_node=start_node,
                                       parameter=parameter, number_of_simulations=number_of_simulations,
                                       increment_amount=increment_amount)

# new_value = "#ff0000"  # Specify the new color
# new_value = "#42f545"  # Specify the new color
#
# colors = ["#ff0000", "#42f545"]
#
# node_id = node_name_id[node_name]
# nodes_g = list(graph.nodes)
#
#
# for _ in range(1_000_000):
#     random_color = random.choice(colors)
#     random_graph_name = random.choice(nodes_g)
#     random_node_id = node_name_id[random_graph_name]
#
# update_node_style_parameter_in_redis(r, random_node_id, parameter, random_color)
