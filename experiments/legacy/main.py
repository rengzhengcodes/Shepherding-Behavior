# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os, sys
import numba as nb
import numpy as np
import matplotlib.pyplot as plt
from basic.initiation import initiate, initiate_shepherds
from basic.interaction import evolve, make_preodic_boundary
from basic.save_data import save_data
from basic.draw_states import draw_state, draw_state_single
from basic.draw import draw_single, draw_dynamic, plot_snapshot
from basic.create_network import create_metric_network

# N_sheep = 100
# N_shepherd = 1
Space_x = 150
Space_y = 150

Target_place_x = 400
Target_place_y = 400
Target_size = 100  # radius

Boundary_x = Target_place_x + Target_size
Boundary_y = Target_place_y + Target_size

# Iterations = 100
# TICK = 10000
# Repetition = 5     #1
# f = open("result.txt", "x")
if __name__ == '__main__':
    # get parameters from .sh script
    parameter = {"N_sheep": int(sys.argv[1]),
                 "N_shepherd": int(sys.argv[2]),
                 "Repetition": int(sys.argv[3]),
                 "Iterations": int(sys.argv[4]),
                 "TICK": int(sys.argv[5])}

    N_sheep = parameter["N_sheep"]
    N_shepherd = parameter["N_shepherd"]
    Repetition = parameter["Repetition"]
    Iterations = parameter["Iterations"]
    TICK = parameter["TICK"]
    # Initiation
    agents = initiate(N_sheep, Space_x, Space_y, Target_size)
    shepherd = initiate_shepherds(0, N_sheep)
    Data_agents = np.zeros((agents.shape[0], agents.shape[1], Iterations), float)
    Data_shepherds = np.zeros((shepherd.shape[0], shepherd.shape[1], Iterations), float)
    # Map_agents = np.zeros((N_sheep, N_sheep, Iterations), float)
    Max_agents_indexes = np.zeros((shepherd.shape[0], Iterations), int)
    Final_iterations = Iterations
    # Loop the function of evolve
    for tick in range(Iterations):
        if tick == TICK:
            # Insert shepherd
            shepherd = initiate_shepherds(N_shepherd, N_sheep)  # add shepherd at certain tick
            Data_shepherds = np.zeros((shepherd.shape[0], shepherd.shape[1], Iterations), float)
            Max_agents_indexes = np.zeros((N_shepherd, Iterations), int)

        agents_update, shepherd_update, max_agents_indexes = evolve(agents, shepherd, Target_place_x, Target_place_y,
                                                                    Target_size)
        # Update agent information
        agents = agents_update
        shepherd = shepherd_update
        Data_agents[:, :, tick] = agents
        Data_shepherds[:, :, tick] = shepherd
        # Map_agents[:, :, tick] = map
        Max_agents_indexes[:, tick] = max_agents_indexes  # only two dimension

        if sum(agents[:, 21]) == N_sheep:  # finish
            Final_iterations = tick
            break

    # draw_dynamic(Final_iterations, Data_agents, Data_shepherds, Map_agents, Boundary_x, Boundary_y, Target_place_x, Target_place_y, Target_size)
    # print("Repetition=", Repetition, "N_Shepherd=", N_shepherd, "N_sheep=", N_sheep, "Final_tick=", Final_iterations)
    # draw_state(Final_iterations, Data_shepherds)

    print("Repetition=", Repetition, "N_Shepherd=", N_shepherd, "N_sheep=", N_sheep, "Final_tick=", Final_iterations)
    print("L0=", shepherd[0][3], "L1=", shepherd[0][5], "L2=", shepherd[0][12], "L3=", shepherd[0][19])

    # draw_state_single(Final_iterations, Data_shepherds)
    # plot_snapshot(agents, shepherd, Repetition, Boundary_x, Boundary_y, Target_place_x, Target_place_y, Target_size)
    save_data(N_sheep, N_shepherd, Repetition, Final_iterations, Data_agents, Data_shepherds)
