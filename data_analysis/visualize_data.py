import os, sys
from timeit import default_timer as timer
from datetime import timedelta
import numba as nb
import numpy as np
import matplotlib.pyplot as plt
import h5py
from basic.draw import draw_single, draw_dynamic, plot_snapshot, calculate_mass_center


@nb.jit(nopython=True)
def calculate_mass_center(agents, agents_state):
    sum_x = 0
    sum_y = 0
    n = 0
    for index in range(agents.shape[0]):
        # agent state: staying -> 1; moving -> 0;
        if agents_state[index] == 0:
            n = n + 1
            sum_x = sum_x + agents[index][0]
            sum_y = sum_y + agents[index][1]
    if n != 0:
        sum_x = sum_x / n
        sum_y = sum_y / n
    return sum_x, sum_y


def draw_network(swarm):
    # draw_network
    N = swarm.shape[0]
    for i in range(N):
        for j in range(N):
            if map[i, j] == 1:
                plt.plot([swarm[i][0], swarm[j][0]], [swarm[i][1], swarm[j][1]], linewidth=1, color='g',
                         alpha=0.4)  # '#e6e6fa'

    return


# @nb.jit(nopython=True)
def draw_single(swarm, agents_state, shepherd, shepherd_state,
                    Space_x, Space_y, Target_place_x, Target_place_y, Target_size):
    # draw sheep
    N = swarm.shape[0]
    Agent_size = 5
    for index in range(N):
        if agents_state[index] == 1:  # staying state radius=2.5
            circles = plt.Circle((swarm[index, 0], swarm[index, 1]), radius=Agent_size, facecolor='none', edgecolor='b',
                                 alpha=0.8)
        else:  # moving state radius=2.5
            circles = plt.Circle((swarm[index, 0], swarm[index, 1]), radius=Agent_size, facecolor='none', edgecolor='g',
                                 alpha=0.8)
        plt.gca().add_patch(circles)
    plt.quiver(swarm[:, 0], swarm[:, 1], np.cos(swarm[:, 2]), np.sin(swarm[:, 2]), headwidth=3, headlength=4,
               headaxislength=3.5, minshaft=4, minlength=1, color='g', scale_units='inches', scale=10)
    # draw shepherd and its collect point
    N_shepherd = shepherd.shape[0]
    for i in range(N_shepherd):
        if shepherd_state[i] == 1:  # drive mode
            plt.plot(shepherd[i, 0], shepherd[i, 1], marker='o', color='r', markersize=Agent_size * 2, alpha=0.2)
            plt.quiver(shepherd[i, 0], shepherd[i, 1], np.cos(shepherd[i, 2]), np.sin(shepherd[i, 2]), headwidth=3,
                       headlength=3, headaxislength=3.5, minshaft=4, minlength=1, color='r', scale_units='inches', scale=10)
        else:
            plt.plot(shepherd[i, 0], shepherd[i, 1], marker='o', color='b', markersize=Agent_size * 2, alpha=0.2)
            plt.quiver(shepherd[i, 0], shepherd[i, 1], np.cos(shepherd[i, 2]), np.sin(shepherd[i, 2]), headwidth=3,
                       headlength=3, headaxislength=3.5, minshaft=4, minlength=1, color='r', scale_units='inches', scale=10)
    # draw center of mass
    center_of_mass_x, center_of_mass_y = calculate_mass_center(swarm, agents_state)
    plt.plot(center_of_mass_x, center_of_mass_y, "r*", markersize=5)
    # draw target center
    plt.plot(Target_place_x, Target_place_y, "b*")
    target_circle = plt.Circle((Target_place_x, Target_place_y), radius=Target_size, facecolor='none', edgecolor='b',
                               alpha=0.5)
    plt.gca().add_patch(target_circle)
    plt.xlim(xmin=0, xmax=Boundary_x)
    plt.ylim(ymin=0, ymax=Boundary_y)
    plt.xticks([])
    plt.yticks([])
    plt.axis("off")
    # plt.axis('equal')
    plt.axis('square')


def plot_trajectory(shepherd, shepherd_states, memory_volume, current_tick, Agent_size):
    N_shepherd = shepherd.shape[0]
    for mem_tick in range(memory_volume):
        absolute_tick = current_tick - mem_tick
        if absolute_tick >= 0:
            for index in range(N_shepherd):
                if shepherd_states[index, absolute_tick] == 1:  # drive mode
                    # if index == 0:
                    #     print(shepherd[index, 0, absolute_tick], shepherd[index, 1, absolute_tick])
                    plt.plot(shepherd[index, 0, absolute_tick], shepherd[index, 1, absolute_tick], marker='o', color='r', markersize=Agent_size * 2, alpha=0.01)
                else:
                    plt.plot(shepherd[index, 0, absolute_tick], shepherd[index, 1, absolute_tick], marker='o', color='b', markersize=Agent_size * 2, alpha=0.01)
    return


def Visualize_dynamic(Iterations, agents_pos, agents_state, shepherd_pos, shepherd_state, repetition, Space_x, Space_y, Target_place_x, Target_place_y, Target_size, Interval, Mem_Volume, Echo_size):
    N_sheep = agents_pos.shape[0]
    plt.figure(figsize=(8, 6), dpi=300)
    plt.ion()
    folder_path = os.getcwd() + "/images/"
    # print(folder_path)
    file_list = os.listdir(folder_path)
    for file in file_list:
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            file_ext = os.path.splitext(file_path)[1]
            if file_ext.lower() in ['.png', '.mp4']:
                os.remove(file_path)

    for tick in range(0, Iterations, Interval):
        plt.cla()
        draw_single(agents_pos[:, :, tick], agents_state[:, tick], shepherd_pos[:, :, tick], shepherd_state[:, tick],
                    Space_x, Space_y, Target_place_x, Target_place_y, Target_size)
        # print("shepherd_pos_0:", shepherd_pos[0, :, tick])

        plot_trajectory(shepherd_pos, shepherd_state, Mem_Volume, tick, Echo_size)

        # plt.title("N_sheep = " + str(N_sheep) + "L3 = 20" + "_tick = " + str(tick))
        plt.savefig(folder_path + str(int(tick / Interval)) + ".png")
    plt.ioff()
    return


def read_hdf5_data(file_path):
    with h5py.File(file_path, "r") as f:
        # print(f.keys())
        agents = f.get("agent_data")[:]
        shepherd = f.get("shepherd_data")[:]
        agents_pos = agents[:, 0:3, :]
        # agents_angle = agents[:, 2, :]
        agents_state = agents[:, 3, :]
        shepherd_pos = shepherd[:, 0:3, :]
        shepherd_state = shepherd[:, 3, :]
        # print(agents_pos, agents_pos.shape)
        # print(agents_pos[0,0,0])
        # print(shepherd_pos, shepherd_pos.shape)
        # print(shepherd_state, shepherd_state.shape)
        f.close()
    return agents_pos, agents_state, shepherd_pos, shepherd_state


def get_final_tick(file_name):
    file_name_string = os.path.splitext(file_name)[0].split("_")
    for item in file_name_string:
        if "tick" in item:
            tick = int(item.split("=")[1])
    return tick


def get_mode(file_name):
    file_name_string = os.path.splitext(file_name)[0].split("_")
    for item in file_name_string:
        if "MODE" in item:
            mode = int(item.split("=")[1])
    return mode
    

def visualize_from_all(file_path):
    with h5py.File(file_path, "r") as f:
        # print(f.keys())
        agents = f.get("agent_data")[:]
        shepherd = f.get("shepherd_data")[:]


###############show shepherd states in one single_data########
Space_x = 150
Space_y = 150

Target_place_x = 450  # x:400
Target_place_y = 450  # y: 400
Target_size = 120  # radius 100

Boundary_x = Target_place_x + Target_size
Boundary_y = Target_place_y + Target_size

### get directory for the path
directory = "/home/yateng/Workspace/Shepherd_Behavior/Data_Metric_Model/L3=0/"
file_name = "N_sheep=200_N_shepherd=2_L3=0_Final_tick=105073_Repetition=22" + ".hdf5"
n_sheep = 200
# directory = "/media/yateng/Extreme SSD/L3=20/"
# file_name = "N_sheep=300_N_shepherd=3_L3=20_Final_tick=105701_Repetition=18" + ".hdf5"
# n_sheep = 300

path = directory + "N_sheep=" + str(n_sheep) + "/" + file_name
agents_pos, agents_state, shepherd_pos, shepherd_state = read_hdf5_data(path)
Final_tick = get_final_tick(file_name)
n_sheep = 200
repetition = 22
repetition = 3
Interval = 200
Mem_Volume = 500
Agent_size = 1.5
Visualize_dynamic(Final_tick, agents_pos, agents_state, shepherd_pos, shepherd_state, repetition, Boundary_x, Boundary_y, Target_place_x, Target_place_y, Target_size, Interval, Mem_Volume, Agent_size)
print("Draw Fertig!")

#### test data in folder data ##########
# directory = "/home/yateng/Workspace/Shepherd_Behavior/data/"
# file_name = "N_sheep=10_N_shepherd=2_L3=0_Final_tick=1000_Repetition=3" + ".hdf5"
# path = directory + file_name
# agents_pos, agents_state, shepherd_pos, shepherd_state = read_hdf5_data(path)
# Final_tick = get_final_tick(file_name)
# print(Final_tick)
# Interval = 100
# repetition = 3
# Mem_Volume = 100
# Agent_size = 2.5
# Visualize_dynamic(Final_tick, agents_pos, agents_state, shepherd_pos, shepherd_state, repetition, Boundary_x, Boundary_y, Target_place_x, Target_place_y, Target_size, Interval, Mem_Volume, Agent_size)

# print(shepherd_pos[0, :, 1])