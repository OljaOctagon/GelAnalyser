import numpy as np
import gel_tools as gt
import networkx as nx
from itertools import islice
from itertools import cycle


def push(i, j, rot, cell_lists, N_particles):
    network_list = []
    for klist in cell_lists:

        nklist = klist
        if rot == -1:
            nklist = klist[::-1]

        for ki, k in enumerate(nklist):
            im_j = j + k*N_particles

            starts_at_k = islice(cycle(nklist), ki+1, None)
            l = next(starts_at_k)

            im_i = i + l*N_particles
            network_list.append([im_i, im_j])

    return network_list


def get_spanning(pos, box, connections, G):

    frac_largest = []
    virtual_frac_largest = []

    cell_dict = {'top-bottom': [[2, 1, 8], [3, 0, 7], [4, 5, 6]],
                 'left-right': [[2, 3, 4], [1, 0, 5], [8, 7, 6]],
                 'diag-top-bottom': [[2, 0, 6], [3, 5, 8], [4, 1, 7]],
                 'diag-bottom-top': [[8, 0, 4], [1, 3, 6], [2, 7, 5]]}

    box_l = box[3:5]
    particles_max_domain = gt.get_particles_in_largest_cluster(G)
    N_largest = len(particles_max_domain)

    N_particles = len(pos)
    frac_largest = N_largest/N_particles

    virtual_frac_largest = 0
    virtual_patch_network = []
    N_images = 9

    for conn_j in connections:
        i, j = conn_j

        # Calculate distances between next neighbours
        # (bonded particles according to patch_network)
        pdist = pos[j]-pos[i]

        x = pdist[0]
        y = pdist[1]

        # define some shorthands
        x_abs = np.fabs(x)
        x_sign = np.sign(x)
        y_abs = np.fabs(y)
        y_sign = np.sign(y)

        box_x = box_l[0]/2
        box_y = box_l[1]/2

        # if pbc not hit:
        # particles remain neighbours and are filled into new network edge array
        if (x_abs < box_x) and (y_abs < box_y):
            for k in range(N_images):
                virtual_patch_network.append(
                    [i+k*N_particles, j+k*N_particles])

        # pbc are hit and particles are neighbours with image according to cell lists

        else:
            # top-bottom
            if (x_abs < box_x) and (y_abs > box_y):

                if y_sign > 0:
                    rot = 1

                if y_sign < 0:

                    rot = -1

                network_list_images = push(
                    i, j, rot, cell_dict['top-bottom'], N_particles)
                virtual_patch_network.extend(network_list_images)

            # left-right
            if (x_abs > box_x) and (y_abs < box_y):

                # to the left
                if x_sign > 0:
                    rot = -1
                # to the right
                if x_sign < 0:
                    rot = 1

                network_list_images = push(
                    i, j, rot, cell_dict['left-right'], N_particles)
                virtual_patch_network.extend(network_list_images)

            # diagognal
            if (x_abs > box_x) and (y_abs > box_y):
                # left top to right bottom
                if x_sign != y_sign:

                    if x_sign > 0 and y_sign < 0:
                        rot = -1
                    if x_sign < 0 and y_sign > 0:
                        rot = 1
                    network_list_images = push(
                        i, j, rot, cell_dict['diag-top-bottom'], N_particles)
                    virtual_patch_network.extend(network_list_images)

                # left bottom to right top
                if x_sign == y_sign:

                    if x_sign > 0:
                        rot = 1
                    if x_sign < 0:
                        rot = -1
                    network_list_images = push(
                        i, j, rot, cell_dict['diag-bottom-top'], N_particles)
                    virtual_patch_network.extend(network_list_images)

    # get virtual Graph
    G_virtual = nx.Graph()
    G_virtual.add_edges_from(virtual_patch_network)

    virtual_max_domain = gt.get_particles_in_largest_cluster(G_virtual)
    virtual_frac_largest = len(virtual_max_domain)/(N_particles*N_images)

    return frac_largest, virtual_frac_largest
