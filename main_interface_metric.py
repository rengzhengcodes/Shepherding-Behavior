import os, sys
from timeit import default_timer as timer
from datetime import timedelta
import numba as nb
import numpy as np
import matplotlib.pyplot as plt
from basic.initiation import initiate, initiate_shepherds
from basic.interaction import evolve
from basic.save_data import save_data, save_data_L3
from basic.draw import draw_single, draw_dynamic, plot_snapshot
from basic.vision_functions import drive_the_herd_using_vision, plot_snapshot_of_vision_field_dynamic


# N_sheep = 200
# N_shepherd = 2
Space_x = 150
Space_y = 150

Target_place_x = 450  # x:400
Target_place_y = 450  # y: 400
Target_size = 120  # radius 100

Boundary_x = Target_place_x + Target_size
Boundary_y = Target_place_y + Target_size

# L3 = 0

# Repetition = 0

# Num_nearst_neighbor = 5

VISION_HERD = False  #True

TICK = 10000

# Iterations = 200000



# start = timer()
if __name__ == '__main__':
    parameter = {"N_sheep": int(sys.argv[1]),
                 "N_shepherd": int(sys.argv[2]),
                 "Iterations": int(sys.argv[3]),
                 "L3": int(sys.argv[4]),
                 "Repetitions": int(sys.argv[5])}

    N_sheep = parameter["N_sheep"]
    N_shepherd = parameter["N_shepherd"]
    Iterations = parameter["Iterations"]
    L3 = parameter["L3"]
    runs = parameter["Repetitions"]
    # for N_sheep in range(100, 500, 100):
    #     for N_shepherd in range(4, 6, 1):
    for Repetition in range(0, runs, 1):
        agents = initiate(N_sheep, Space_x, Space_y, Target_size)
        shepherd = initiate_shepherds(0, N_sheep, L3)
        # self-organized flocking
        for tick in range(TICK):
            agents_update, shepherd_update, max_agents_indexes = evolve(agents, shepherd, Target_place_x,
                                                                        Target_place_y, Target_size, VISION_HERD)
            agents = agents_update
            shepherd = shepherd_update
        # prepare the shepherd and record data
        shepherd = initiate_shepherds(N_shepherd, N_sheep, L3)
        Data_agents = np.zeros((agents.shape[0], agents.shape[1], Iterations), float)
        Data_shepherds = np.zeros((shepherd.shape[0], shepherd.shape[1], Iterations), float)
        Max_agents_indexes = np.zeros((N_shepherd, Iterations), int)
        Final_tick = Iterations
        # continue the sheep data with shepherd
        for tick in range(Iterations):
            # start evolve function
            agents_update, shepherd_update, max_agents_indexes = evolve(agents, shepherd, Target_place_x, Target_place_y,
                                                                        Target_size, VISION_HERD)
            # update data
            agents = agents_update
            shepherd = shepherd_update
            # save data
            Data_agents[:, :, tick] = agents
            Data_shepherds[:, :, tick] = shepherd
            Max_agents_indexes[:, tick] = max_agents_indexes  # only two dimension
            # stop program if all the sheep are in the "staying" mode;
            if sum(agents[:, 21]) == N_sheep:   # finish
                Final_tick = tick
                break

        # Interval = 100
        # plot_snapshot_of_vision_field_dynamic(Final_tick, Data_agents, Data_shepherds, Target_place_x, Target_place_y,
        #                                       Target_size, Interval)
        # draw_dynamic(Final_tick, Data_agents, Data_shepherds, Boundary_x, Boundary_y, Target_place_x, Target_place_y, Target_size, L3)

        save_data_L3(N_sheep, N_shepherd, Repetition, Final_tick, Data_agents, Data_shepherds, L3)
        # print("N_Shepherd=", N_shepherd, "N_sheep=", N_sheep, "L3=", L3, "Repetition_", Repetition, "Final_tick=", Final_tick)
