import os, sys, random
from timeit import default_timer as timer
from datetime import timedelta
import numba as nb
import numpy as np
import matplotlib.pyplot as plt
from basic.initiation import initiate, initiate_shepherd
from basic.interaction import evolve, make_periodic_boundary
from basic.save_data import save_data, save_data_L3, save_all
from basic.draw import draw_single, draw_dynamic, plot_snapshot
from basic.create_network import create_metric_network, create_topological_network


N_sheep = 300
N_shepherd = 3
Space_x = 150
Space_y = 150

Target_place_x = 400
Target_place_y = 400
Target_size = 125  # radius

Boundary_x = Target_place_x + Target_size + 300
Boundary_y = Target_place_y + Target_size + 300


TICK = 5000
Iterations = 200000

L3 = np.sqrt(N_sheep / N_shepherd) * 5 # average flock radius per shepherd
MODE = 4

Repetition = 0
reps = 10

Num_nearst_neighbor = 5

# Function seeds numpy rng in numba code.
@nb.jit(nopython=True)
def seed_run(seed):
    random.seed(seed)
    np.random.seed(seed)


start = timer()
if __name__ == '__main__':
    successes = 0
    for rep in range(reps):
        print("Repetition:", rep)
        seed_run(rep)
        agents = initiate(N_sheep, N_shepherd, Space_x, Space_y, Target_size)
        shepherd = initiate_shepherd(0, N_shepherd, L3)
        # self-organized flocking
        for tick in range(TICK):
            agents_update, shepherd_update, max_agents_indexes = evolve(agents, shepherd, Target_place_x,
                                                                        Target_place_y, Target_size, MODE=MODE)
            agents = agents_update
            shepherd = shepherd_update
        # prepare the shepherd and record data
        shepherd = initiate_shepherd(N_shepherd, N_sheep, L3)
        Data_agents = np.zeros((agents.shape[0], agents.shape[1], Iterations), float)
        Data_shepherds = np.zeros((shepherd.shape[0], shepherd.shape[1], Iterations), float)
        Max_agents_indexes = np.zeros((N_shepherd, Iterations), int)
        Final_tick = Iterations
        # continue the sheep data with shepherd
        for tick in range(Iterations):
            # start evolve function
            agents_update, shepherd_update, max_agents_indexes = evolve(agents, shepherd, Target_place_x, 
                                                                        Target_place_y, Target_size, MODE=MODE)
            # update data
            agents = agents_update
            shepherd = shepherd_update
            # save data
            Data_agents[:, :, tick] = agents
            Data_shepherds[:, :, tick] = shepherd
            Max_agents_indexes[:, tick] = max_agents_indexes  # only two dimension
            # print(tick)
            # stop program if all the sheep are in the "staying" mode;
            if sum(agents[:, 21]) == N_sheep:   # finish
                Final_tick = tick
                break
            # try to create network
            # topological_network = create_topological_network(agents, Num_nearst_neighbor)
            # metric_network = create_metric_network(agents, agents[0][5], np.pi)   # Fov not used
            # print("topological_network:", topological_network)
            # print("metric_network:", metric_network)

        # draw_dynamic(Final_tick, Data_agents, Data_shepherds, Boundary_x, Boundary_y, Target_place_x, Target_place_y, Target_size, L3, MODE=MODE)
        # save_data(N_sheep, N_shepherd, Repetition, Final_tick, Data_agents, Data_shepherds)
        # save_data_L3(N_sheep, N_shepherd, rep, Final_tick, Data_agents, Data_shepherds, L3)
        # save_all(N_sheep, N_shepherd, rep, Final_tick, Data_agents, Data_shepherds, L3, MODE)
        print("N_Shepherd=",N_shepherd,"N_sheep=",N_sheep,"L3=",L3,"Repetition_",rep,"Final_tick=",Final_tick)
        end = timer()
        print("program takes:", timedelta(seconds=end-start), "seconds")
        if np.all(agents[:, 21] == 1):
            successes += 1
            print(f"Successes: {successes}/{reps}")
    
    print(f"Success rate: {successes/reps}")
            

    #However, depending on the specific formulation of the shepherding task and model parameters,
    # we also observed scenarios with an optimal number of shepherds where the guiding time becomes minimal. This appears to be related to possible obstruction of the shepherds by themselves.
