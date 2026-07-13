import numba as nb
import numpy as np
import matplotlib.pyplot as plt
import math

@nb.jit(nopython=True)
def get_relative_distance_angle(target_x, target_y, focal_agent_x, focal_agent_y):
    r_x = target_x - focal_agent_x
    r_y = target_y - focal_agent_y
    r_length = np.sqrt(r_x ** 2 + r_y ** 2)
    r_angle = np.arctan2(r_y, r_x)  # [-pi, pi]
    return r_length, r_angle


@nb.jit(nopython=True)
def create_distance_matrix(agents):
    num_agents = agents.shape[0]
    # create a metric with N * N dimension
    distance_matrix = np.zeros(shape=(num_agents, num_agents))
    # loop agents
    for agent_index in range(num_agents):
        for neighbor_index in range(num_agents):
            if agent_index != neighbor_index:
                r_ij, theta_ij = get_relative_distance_angle(agents[agent_index][0], agents[agent_index][1],
                                                             agents[neighbor_index][0], agents[neighbor_index][1])
                distance_matrix[agent_index][neighbor_index] = r_ij
            else:
                distance_matrix[agent_index][agent_index] = np.nan
    # np.fill_diagonal(distance_matrix, float("NaN"))
    return distance_matrix


# @nb.jit(nopython=True)
def transform_angle(theta):
    '''Casts any (radian) angle to the
                    equivalent in the interval (-pi, pi)'''
    # while theta >= np.pi:
    #     theta = theta - 2 * np.pi
    # while theta <= -np.pi:
    #     theta = theta + 2 * np.pi

    new_theta = (theta + np.pi) % (2. * np.pi)
    new_theta -= np.pi
    return theta


# @nb.jit(nopython=True)
def create_metric_network(agents, R, Fov):
    # R: length of attraction field: maximum interaction range for agent
    # FOV: field of view [0, pi]---- agents[agent_index][19] = np.pi/2
    # get number of agents
    # num_agents = agents.shape[0]
    # calculate distance matrix
    distance_matrix = create_distance_matrix(agents)
    # metric_network = np.zeros(shape=(num_agents, num_agents))
    metric_network = np.array(np.nan_to_num(distance_matrix, nan=np.inf) < R, dtype=int)
    # loop agents
    # for agent_index in range(num_agents):
    #     theta_i = agents[agent_index, 2]
    #     for neighbor_index in range(num_agents):
    #         if agent_index != neighbor_index:
    #             r_ij, theta_ij = get_relative_distance_angle(agents[agent_index][0], agents[agent_index][1],
    #                                                          agents[neighbor_index][0], agents[neighbor_index][1])
    #             turning_angle = transform_angle(theta_i - theta_ij)  # theta_i, theta_ij<-- [-pi, pi];
    #             if r_ij <= R and (abs(turning_angle) <= Fov):
    #                 metric_network[agent_index, neighbor_index] = 1
    #                 metric_network[neighbor_index, agent_index] = 1
    return metric_network


def create_topological_network(agents, Num_nearst_neighbor):
    # create matrix with distance
    distance_matrix = create_distance_matrix(agents)
    topological_network = np.array(np.argsort(np.argsort(distance_matrix, axis=1)) < Num_nearst_neighbor, dtype=int)
    return topological_network
