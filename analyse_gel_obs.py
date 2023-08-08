from collections import Counter
import numpy as np
import glob
import pandas as pd
import argparse
from spanning_func import get_spanning
import gel_tools as gt
import re
from scipy.optimize import curve_fit
import pandas as pd
import configparser
from os.path import exists
import networkx as nx
from collections import defaultdict
import multiprocessing


def generator_from_fsys(fsys_iterator):
    for dir_i in fsys_iterator:
        print(dir_i)
        run_id = dir_i.split("_")[-1].split("/")[0]
        config = configparser.ConfigParser()

        try:
            config.read("{}para.ini".format(dir_i))

            N = int(config["System"]["Number_of_Particles"])
            phi = float(config["System"]["Packing_Fraction"])
            temperature = float(config["System"]["Temperature"])
            ptype = config["Rhombus"]["rhombus_type"]
            delta = config["Rhombus"]["patch_delta"]
            patch_size = config["Rhombus"]["patch_size"]

            pos_files = glob.glob("{}positions_*.bin".format(dir_i))

            # get the last value from the string
            def g(x):
                return int(re.findall(r"\d+", x)[-1])

            mc_times = list(map(g, pos_files))
            last_time = np.max(mc_times)

            pos_file = "{}positions_{}.bin".format(dir_i, last_time)
            pos = np.fromfile(pos_file)
            pos = np.reshape(pos, (-1, 3))
            pos = pos[:, :2]

            box_file = "{}Box_{}.bin".format(dir_i, last_time)
            box = np.fromfile(box_file)

        except:
            print("error reading config")
            N = np.nan
            phi = np.nan
            temperature = np.nan
            delta = np.nan
            last_time = np.nan
            pos = None
            box = None
            ptype = ""

        yield (dir_i, ptype, N, phi, temperature, delta, run_id, last_time, pos, box)


def calculate_np(connections_patch, orient_dict):
    orientd = [orient_dict[(i, j)] for i, j in connections_patch]
    frac_np = np.sum(orientd) / len(orientd)
    return frac_np


def get_loop_stats(G, N_particles, loop_type):
    if loop_type == "non_parallel":
        # make subgraph of np particles
        subgraph = nx.Graph(
            (
                (source, target, attr)
                for source, target, attr in G.edges(data=True)
                if attr["bond_type"] == 1
            )
        )

    if loop_type == "parallel":
        # make subgraph of np particles
        subgraph = nx.Graph(
            (
                (source, target, attr)
                for source, target, attr in G.edges(data=True)
                if attr["bond_type"] == 0
            )
        )

    if loop_type == "mixed":
        subgraph = nx.Graph(
            ((source, target, attr) for source, target, attr in G.edges(data=True))
        )

    # get loops via all simple paths
    loop_set_list = []
    for node in subgraph.nodes:
        neigh = list(subgraph.neighbors(node))
        for nni in neigh:
            subgraph.remove_edge(node, nni)
            all_simple_paths = list(nx.all_simple_paths(subgraph, nni, node, cutoff=6))
            if all_simple_paths:
                all_simple_paths_set = [set(s) for s in all_simple_paths if s]
                loop_set_list.extend(all_simple_paths)
            subgraph.add_edge(node, nni)

    # get rid of duplicates
    freqs = Counter(frozenset(sub) for sub in loop_set_list)
    res = [key for key, val in freqs.items() if val > 1]

    # calculate stats for different loops sizes
    def get_percent_loop(res, size, N_particles):
        unique_list = [list(s) for s in res if len(list(s)) == size]
        nodes = np.unique([s for sublist in unique_list for s in sublist]).tolist()
        percent_nodes_in_loop = len(nodes) / N_particles
        return percent_nodes_in_loop

    p3_loops = get_percent_loop(res, 3, N_particles)
    p4_loops = get_percent_loop(res, 4, N_particles)
    p5_loops = get_percent_loop(res, 5, N_particles)
    p6_loops = get_percent_loop(res, 6, N_particles)

    return p3_loops, p4_loops, p5_loops, p6_loops


def calculate(vals):
    dir_name, ptype, N, phi, temperature, delta, run_id, last_time, pos, box = vals
    new_results = {}
    file_name = "{}/patch_network.dat".format(dir_name)

    # 0: parallel
    # 1: non-parallel
    orient_dict = {
        (0, 0): 0,
        (0, 1): 1,
        (0, 2): 1,
        (0, 3): 0,
        (1, 0): 1,
        (1, 1): 0,
        (1, 2): 0,
        (1, 3): 1,
        (2, 0): 1,
        (2, 1): 0,
        (2, 2): 0,
        (2, 3): 1,
        (3, 0): 0,
        (3, 1): 1,
        (3, 2): 1,
        (3, 3): 0,
    }

    if exists(file_name) and pos is not None and len(pos) != 0 and len(box) != 0:
        subdir_name = "{}_phi_{}_delta_{}_temp_{}_run_{}".format(
            ptype, phi, delta, temperature, run_id
        )

        new_results["id"] = subdir_name
        new_results["ptype"] = ptype
        new_results["delta"] = delta
        new_results["phi"] = phi
        new_results["run_id"] = run_id
        new_results["temperature"] = temperature
        new_results["current_time"] = last_time
        new_results["N"] = N

        connections = gt.read_bonds(file_name)[-1]
        frac_np = calculate_np(connections[:, 2:], orient_dict)
        new_results["frac_np"] = frac_np
        connections_pid = connections[:, :2]

        G = nx.Graph()
        G.add_edges_from(connections_pid)

        attrs = {}
        for entry in connections:
            attrs[(entry[0], entry[1])] = orient_dict[(entry[2], entry[3])]
        nx.set_edge_attributes(G, attrs, name="bond_type")

        frac_largest, virtual_frac_largest = get_spanning(
            pos,
            box,
            connections_pid,
            G,
        )
        new_results["frac_largest"] = frac_largest
        new_results["frac_largest_virtual"] = virtual_frac_largest

        degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
        dmax = max(degree_sequence)

        degrees, degrees_N = np.unique(degree_sequence, return_counts=True)

        N_particles = len(pos)
        degrees_f = defaultdict(lambda: 0)
        for di, dn in zip(degrees, degrees_N):
            degrees_f[di] = dn / N_particles

        degree_0_percent = 1 - (len(np.unique(connections_pid.flatten())) / N_particles)

        new_results["frac_degree_0"] = degree_0_percent
        new_results["frac_degree_1"] = degrees_f[1]
        new_results["frac_degree_2"] = degrees_f[2]
        new_results["frac_degree_3"] = degrees_f[3]
        new_results["frac_degree_4"] = degrees_f[4]

        (
            new_results["p3_loops"],
            new_results["p4_loops"],
            new_results["p5_loops"],
            new_results["p6_loops"],
        ) = get_loop_stats(G, N_particles, "non_parallel")

        _, _, _, new_results["parallel_p6_loops"] = get_loop_stats(
            G, N_particles, "parallel"
        )

        _, _, _, new_results["mixed_p6_loops"] = get_loop_stats(G, N_particles, "mixed")

    else:
        print("Warning: Folder {} has issues. Results not evaluated".format(dir_name))

        new_results["id"] = np.nan
        new_results["ptype"] = ""
        new_results["delta"] = np.nan
        new_results["phi"] = np.nan
        new_results["run_id"] = np.nan
        new_results["temperature"] = np.nan
        new_results["current_time"] = np.nan
        new_results["N"] = np.nan

        new_results["frac_largest"] = np.nan
        new_results["frac_largest_virtual"] = np.nan

        new_results["frac_degree_0"] = np.nan
        new_results["frac_degree_1"] = np.nan
        new_results["frac_degree_2"] = np.nan
        new_results["frac_degree_3"] = np.nan
        new_results["frac_degree_4"] = np.nan

        new_results["frac_np"] = np.nan

        new_results["p3_loops"] = np.nan
        new_results["p4_loops"] = np.nan
        new_results["p5_loops"] = np.nan
        new_results["p6_loops"] = np.nan
        new_results["parallel_p6_loops"] = np.nan
        new_results["mixed_p6_loops"] = np.nan

    new_results = pd.DataFrame.from_dict(new_results, orient="index").T
    return new_results


if __name__ == "__main__":
    # read data either through files system via glob or via db
    parser = argparse.ArgumentParser()
    parser.add_argument("-ana_id", type=str)
    parser.add_argument("-ncores", type=int)

    args = parser.parse_args()
    gen_fsys = generator_from_fsys(glob.glob("batch*double*/double*/"))
    gen_dict = {"fsys": gen_fsys}
    columns = ["id", "ptype", "delta", "phi", "temperature", "current_time", "run_id"]

    columns.append("frac_largest")
    columns.append("frac_largest_virtual")

    columns.append("frac_degree_0")
    columns.append("frac_degree_1")
    columns.append("frac_degree_2")
    columns.append("frac_degree_3")
    columns.append("frac_degree_4")

    columns.append("p3_loops")
    columns.append("p4_loops")
    columns.append("p5_loops")
    columns.append("p6_loops")
    columns.append("parallel_p6_loops")
    columns.append("mixed_p6_loops")

    df = pd.DataFrame(columns=columns)
    gen = gen_dict["fsys"]

    N_CORES = int(args.ncores)
    N_CORES_MAX = 8

    if N_CORES > 1 and N_CORES <= N_CORES_MAX:
        print("Multiprocessing with {} cores".format(N_CORES))
        pool = multiprocessing.Pool(N_CORES)
        new_results = pool.map(calculate, gen)
        pool.close()
        pool.join()
        df = pd.concat(new_results)

    if N_CORES == 1:
        print("single core job")
        for vals in gen:
            new_results = calculate(vals)
            df = df.append(new_results, ignore_index=True)

    if N_CORES > N_CORES_MAX:
        print(
            "Too many cores allocated, please do not use more than {} cores".format(
                N_CORES_MAX
            )
        )

    df.to_pickle("results_gel_{}.pickle".format(args.ana_id))
