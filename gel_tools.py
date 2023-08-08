import numpy as np
import networkx as nx
# read patch_network file with multiple timestamps


def read_bonds(filen):
    first_line_pair = [0, 0, 0, 0]
    cut = False
    with open(filen, 'r') as f:
        network_list = []
        for line in f:
            if "#" in line:
                network_list.append([])
                first_line_pair = [0, 0, 0, 0]
                cut = False

            else:
                line_counter = len(network_list[-1])
                pairs = list(map(int, line.split(" ")))
                if pairs == first_line_pair or cut == True:
                    cut = True
                else:
                    network_list[-1].append(np.array(pairs))

                if line_counter == 0:
                    first_line_pair = pairs
    network_list = [np.array(item) for item in network_list]

    return network_list


def get_domain_lengths(G):
    domains = list(nx.connected_components(G))
    domain_lengths = np.array([len(domain) for domain in domains])
    #d_id = np.argmax(domain_length)
    #particles_max_domain = np.array(list(domains[d_id]))
    return domain_lengths


def make_cycles(G, size):
    DG = nx.DiGraph(G)
    loops = list(nx.simple_cycles(DG))

    len_loops = [len(loop) for loop in loops]
    cluster = np.array([loop for loop in loops if len(loop) == size]).flatten()

    return cluster


def find_cliques(G):
    cliques = list(nx.find_cliques(G))
    length_N3_loops = len([item for item in cliques if len(item) == 3])
    return length_N3_loops


def get_particles_in_largest_cluster(G):
    domains = list(nx.connected_components(G))
    domain_lengths = np.array([len(domain) for domain in domains])
    d_id = np.argmax(domain_lengths)
    particles_max_domain = np.array(list(domains[d_id]))
    return particles_max_domain
