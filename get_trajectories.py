from collections import Counter
import numpy as np
import glob
import pandas as pd
import argparse
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

        except:
            print("error reading config")
            N = np.nan
            phi = np.nan
            temperature = np.nan
            delta = np.nan
            ptype = ""

        yield (dir_i, ptype, N, phi, temperature, delta, run_id)


def calculate(vals):
    dir_name, ptype, N, phi, temperature, delta, run_id = vals
    new_results = {}
    file_name = "{}patch_network.dat".format(dir_name)
    if exists(file_name):
        subdir_name = "{}_phi_{}_delta_{}_temp_{}_run_{}".format(
            ptype, phi, delta, temperature, run_id
        )
        connections = gt.read_bonds(file_name)
        frac_largest = []
        n_t = len(connections)
        n_times = 0
        if n_t > 1:
            for ci in range(n_t):
                if connections[ci].size > 0:
                    connections_pid = connections[ci][:, :2]
                    G = nx.Graph()
                    G.add_edges_from(connections_pid)
                    size_largest = len(gt.get_particles_in_largest_cluster(G))
                    frac_largest.append(size_largest / N)
                    n_times = n_times + 1 
        if n_t == 1:
            connections_pid = connections[:, :2]
            G = nx.Graph()
            G.add_edges_from(connections_pid)
            size_largest = len(gt.get_particles_in_largest_cluster(G))
            frac_largest.append(size_largest / N)
            n_times = 1 

        if n_t == 0:
            frac_largest.append(np.nan)
            n_times = 1 

        new_results["frac_largest"] = frac_largest
        new_results["itime"] = list(range(n_times))
        new_results["id"] = [subdir_name] * n_times
        new_results["ptype"] = [ptype] * n_times
        new_results["delta"] = [delta] * n_times
        new_results["phi"] = [phi] * n_times
        new_results["run_id"] = [run_id] * n_times
        new_results["temperature"] = [temperature] * n_times
        new_results["N"] = [N] * n_times

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
        new_results["itime"] = np.nan

        new_results["frac_largest"] = np.nan

    new_results = pd.DataFrame.from_dict(new_results, orient="index").T
    return new_results


if __name__ == "__main__":
    # read data either through files system via glob or via db
    parser = argparse.ArgumentParser()
    parser.add_argument("-ana_id", type=str)
    parser.add_argument("-ncores", type=int)

    args = parser.parse_args()
    gen_fsys = generator_from_fsys(glob.glob("batch*double*/double*/"))
    #gen_fsys = generator_from_fsys(glob.glob("batch*double*/double_manta_asymm_1_phi_0.03_delta_0.2_temp_0.14_run_1*/"))

    
    gen_dict = {"fsys": gen_fsys}
    columns = [
        "id",
        "ptype",
        "delta",
        "phi",
        "temperature",
        "current_time",
        "run_id",
        "frac_largest",
        "itime",
    ]

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

    df.to_pickle("traj_{}.pickle".format(args.ana_id))
