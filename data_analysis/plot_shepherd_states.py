import csv
import matplotlib.pyplot as plt
import os, sys
import numpy as np
import h5py
import seaborn as sns
from timeit import default_timer as timer
from datetime import timedelta
from collections import defaultdict


def plot_multi_states(shepherd_state, N_sheep, Iterations):
    plt.figure(figsize=(8, 6), dpi=300)
    N_shepherd = shepherd_state.shape[0]
    x = [item * 0.01 for item in range(Iterations)]
    # subplot shepherd state
    for shepherd_index in range(N_shepherd):
        # print(shepherd_index)
        plt.subplot(N_shepherd + 1, 1, shepherd_index + 1)
        y = shepherd_state[shepherd_index, 0:Iterations]  # 1.0  drive_mode_true
        plt.plot(x, y, 'b-')
        plt.ylabel("Shepherd" + str(shepherd_index))
        if shepherd_index == 0:
            plt.title("N_shepherd=" + str(N_shepherd) + "_N_sheep=" + str(N_sheep) + "_Final_tick=" + str(Iterations))
    # subplot difference state
    difference = np.zeros(Iterations)
    for index in range(Iterations):
        if sum(shepherd_state[:, index]) != 0 and sum(shepherd_state[:, index]) != N_shepherd:
            difference[index] = 1
    # print("difference:", sum(difference))
    plt.subplot(N_shepherd + 1, 1, N_shepherd + 1)
    plt.plot(x, difference, 'r-')
    plt.ylabel("difference")
    plt.xlabel("Time (s)")
    plt.savefig("./figures/state_of_shepherd=" + str(N_shepherd)+"_N_sheep="+str(N_sheep) + ".png")
    plt.show()
    return


def calculate_pairwise_differ_states(shepherd_states_data, Iterations):
    N_shepherd = shepherd_states_data.shape[0]
    iterations = Iterations
    difference = np.zeros(iterations)
    for tick in range(iterations):
        count = 0
        for start_index in range(N_shepherd - 1):
            for end_index in range(start_index + 1, N_shepherd):
                count = count + 1
                if shepherd_states_data[start_index, tick] != shepherd_states_data[end_index, tick]:
                    difference[tick] = difference[tick] + 1
        difference[tick] = difference[tick] / count  # N_shepherd
    value = sum(difference)
    print("N_shepherd:", N_shepherd, "count:", count, "difference:", value, "iterations", iterations)
    return value


# calculate coordination time when shepherds are in different state;
def calculate_differ_states(shepherd_states_data, Iterations):
    N_shepherd = shepherd_states_data.shape[0]
    iterations = Iterations
    difference = np.zeros(iterations)
    for index in range(iterations):
        if sum(shepherd_states_data[:, index]) != 0 and sum(shepherd_states_data[:, index]) != N_shepherd:
            difference[index] = 1
    value = sum(difference)
    print("N_shepherd:", N_shepherd, "difference:", value, "iterations", iterations)
    return value


def read_hdf5_data(path):
    with h5py.File(path, "r") as f:
        # print(f.keys())
        agents = f.get("agent_data")[:]
        shepherd = f.get("shepherd_data")[:]
        # agents_num = agents.shape[0]
        # shepherd_num = shepherd.shape[0]
        agents_pos = agents[:, 0:2, :]
        agents_state = agents[:, 3, :]
        shepherd_pos = shepherd[:, 0:2, :]
        shepherd_state = shepherd[:, 3, :]
        # print("shepherd_state", shepherd_state.shape)
        # print("agents_state", agents_state.shape)
        f.close()
    return agents_pos, agents_state, shepherd_pos, shepherd_state


def get_final_tick(file_name):
    file_name_string = os.path.splitext(file_name)[0].split("_")
    for item in file_name_string:
        if "tick" in item:
            tick = int(item.split("=")[1])
            # print("tick:", tick)
    return tick


################show shepherd states in one single_data########
### get directory for the path
# directory = "/home/yateng/Workspace/Shepherd_Behavior/Data_Metric_Model/L3=0/N_sheep=100/" #os.getcwd() + "/../data/"
# file_name = "N_sheep=100_N_shepherd=2_L3=0_Final_tick=59666_Repetition=6.hdf5"
###########################################################################
directory = "/media/yateng/Extreme SSD/L3=20/N_sheep=300/"
file_name = "N_sheep=300_N_shepherd=3_L3=20_Final_tick=105701_Repetition=18" + ".hdf5"
###########################################################################
path = directory + file_name
agents_pos, agents_state, shepherd_pos, shepherd_state = read_hdf5_data(path)
N_sheep = agents_pos.shape[0]
Iterations = get_final_tick(file_name)
plot_multi_states(shepherd_state, N_sheep, Iterations)


################# calculate time spent in different state #######
# # get directory for the path
def calculate_time_in_different_modes(N_shepherd):
    directory = os.getcwd() + "/../Data/"
    files = os.listdir(directory)
    count = 0
    drive_mode_list = []
    collect_mode_list = []
    drive_mode = dict([])
    collect_mode = dict([])
    L3 = 0
    for sheep_index in range(100, 500, 100):
        keyword = "N_sheep=" + str(sheep_index) + "_N_shepherd=" + str(N_shepherd) + "_L3=" + str(L3)
        # print(keyword)
        drive_ratio = []
        collect_ratio = []
        for file in files:
            if not os.path.isdir(file):
                if keyword in file:
                    print(file)
                    path = directory + file
                    agents_pos, agents_state, shepherd_pos, shepherd_state = read_hdf5_data(path)
                    Iterations = get_final_tick(file)
                    # plot_multi_states(shepherd_state, sheep_index, Iterations)
                    # print("Iterations:", Iterations)
                    drive_ratio.append(np.sum(shepherd_state) / Iterations)
                    collect_ratio.append(1 - np.sum(shepherd_state) / Iterations)
                    # print(len(drive_ratio))
                    # print(shepherd_state)
        count = count + 1
        drive_mode_list.append(drive_ratio)
        collect_mode_list.append(collect_ratio)
        drive_mode["Ns=" + str(sheep_index)] = drive_ratio
        collect_mode["Ns=" + str(sheep_index)] = collect_ratio

    drive_mode_array = np.array(drive_mode_list)
    collect_mode_array = np.array(collect_mode_list)
    return drive_mode_array, collect_mode_array


###########################generate mediate result for shepherd in different mode###
# print("keys:", list(drive_mode.values()))
def save_file(file, data):
    with open(file, "wb") as f:
        np.save(f, data)
    f.close()
    return


# N_shepherd = 1
# drive_mode_arr, collect_mode_arr = calculate_time_in_different_modes(N_shepherd)
# file1 = os.getcwd() + "/mediate_result/drive_mode.npy"
# file2 = os.getcwd() + "/mediate_result/collect_mode.npy"
# save_file(file1, drive_mode_arr)
# save_file(file2, collect_mode_arr)


#####used in box_plot_new file############
# def read_mode_data(file):
#     with open(file, "rb") as f:
#         data = np.load(f)
#         f.close()
#         # print("data:", data)
#     return data
# drive_data = read_mode_data(file1)
# collect_data = read_mode_data(file2)


##########calculate difference value for multiple data#######
def calculate_multi_difference(N_sheep, L3):
    directory = os.getcwd() + "/../Data/"
    files = os.listdir(directory)
    count = 0
    matrix = np.zeros(4, dtype=float)
    for n_shepherd in range(2, 6):  # 6
        keyword = "N_sheep=" + str(N_sheep) + "_N_shepherd=" + str(n_shepherd) + "_L3=" + str(L3)
        print(keyword)
        states = []
        for file in files:
            if not os.path.isdir(file):
                if keyword in file:
                    # print(file)
                    path = directory + file
                    agents_pos, agents_state, shepherd_pos, shepherd_state = read_hdf5_data(path)
                    Iterations = get_final_tick(file)
                    # print("Iterations:", Iterations)
                    # value = calculate_differ_states(shepherd_state, Iterations) / Iterations    # ratio over running time
                    value = calculate_pairwise_differ_states(shepherd_state, Iterations) / Iterations
                    states.append(value)
                    # print("value:", value)
            else:
                print("Please enter a file path!")
        mean_value = np.mean(np.array(states))  # ratio over repetitions
        # print(keyword, "repetitions=", np.array(states).shape[0], "mean_different_state_value",
        #       np.mean(np.array(states)))
        matrix[count] = mean_value
        count = count + 1
    return matrix


def plot_coordination_ratio(coordinate_rate_1, coordinate_rate_2, coordinate_rate_3, coordinate_rate_4, L3):
    plt.figure(figsize=(8, 6), dpi=300)
    x = np.arange(1, 5)
    plt.plot(x, coordinate_rate_1, '^-', color='orange', label="N_sheep = 100")
    plt.plot(x, coordinate_rate_2, '^-', color='cyan', label="N_sheep = 200")
    plt.plot(x, coordinate_rate_3, '^-', color='green', label="N_sheep = 300")
    plt.plot(x, coordinate_rate_4, '^-', color='blue', label="N_sheep = 400")

    plt.xticks(x, ["N = 2", "N = 3", "N = 4", "N = 5"])
    plt.title("L3 = " + str(L3))
    plt.xlabel("Shepherd number")
    plt.ylabel("Coordination Ratio")

    plt.legend(loc="best")
    plt.savefig("./figures/Coordination_Ratio_L3=" + str(L3) + ".png")
    plt.show()


# l3 = 20
# n_sheep = 300
# Matrix = calculate_multi_difference(n_sheep, l3)
# print(Matrix)

##################################################
# l3 = 0
# list_of_L3 = [5, 10] #[i for i in range(0, 25, 5)]
# list_of_sheep = [i for i in range(100, 500, 100)]
# for l3 in list_of_L3:
#     for n_sheep in list_of_sheep:
#         Matrix = calculate_multi_difference(n_sheep, l3)
#         print(Matrix)

#################################################
# # coordination ratio plot#
# L3 = 0 #
# Coordinate_rate_1 = [0.11446127, 0.24226493, 0.28600385, 0.38381578]  ###N_shepherd = 2,3,4,5 N_sheep = 100_L3=0
# Coordinate_rate_2 = [0.02219488, 0.14015552, 0.29542325, 0.30980389]  ###N_shepherd = 2,3,4,5 N_sheep = 200_L3=0
# Coordinate_rate_3 = [0.05145715, 0.16570792, 0.18534092, 0.2756973]   ###N_shepherd = 2,3,4,5 N_sheep = 300_L3=0
# Coordinate_rate_4 = [0.01473957, 0.03488040, 0.12400992, 0.16809813]  ###N_shepherd = 2,3,4,5 N_sheep = 400_L3=0
#
# L3 = 5 #
# Coordinate_rate_1 = [0.4488077,  0.16863035, 0.26614935, 0.34868498] ###N_shepherd = 2,3,4,5 N_sheep = 100_L3=5
# Coordinate_rate_2 = [0.08492622, 0.11463437, 0.13364688, 0.2439555 ] ###N_shepherd = 2,3,4,5 N_sheep = 200_L3=5
# Coordinate_rate_3 = [0.06963093, 0.15978579, 0.21912787, 0.2202473 ] ###N_shepherd = 2,3,4,5 N_sheep = 300_L3=5
# Coordinate_rate_4 = [0.01073654, 0.08148325, 0.07265147, 0.23638927] ###N_shepherd = 2,3,4,5 N_sheep = 400_L3=5
#
# L3 = 10 #
# Coordinate_rate_1 = [0.35838987, 0.29392029, 0.36125527, 0.39790295] ###N_shepherd = 2,3,4,5 N_sheep = 100_L3=10
# Coordinate_rate_2 = [0.23044335, 0.35019742, 0.27576224, 0.49213392] ###N_shepherd = 2,3,4,5 N_sheep = 200_L3=10
# Coordinate_rate_3 = [0.05542843, 0.26633775, 0.23940488, 0.23745046] ###N_shepherd = 2,3,4,5 N_sheep = 300_L3=10
# Coordinate_rate_4 = [0.04579048, 0.003995,   0.02836194, 0.11593542] ###N_shepherd = 2,3,4,5 N_sheep = 400_L3=10
# L3 = 15 #
# Coordinate_rate_1 = [0.2497901,  0.51541905, 0.33528089, 0.45353701] ###N_shepherd = 2,3,4,5 N_sheep = 100_L3=15
# Coordinate_rate_2 = [0.19527684, 0.40387749, 0.49802415, 0.39411142] ###N_shepherd = 2,3,4,5 N_sheep = 200_L3=15
# Coordinate_rate_3 = [0.27665068, 0.13830231, 0.25089963, 0.30122779] ###N_shepherd = 2,3,4,5 N_sheep = 300_L3=15
# Coordinate_rate_4 = [0.004141,   0.10507848, 0.12908428, 0.26038203] ###N_shepherd = 2,3,4,5 N_sheep = 400_L3=15

# L3 = 20 #
# Coordinate_rate_1 = [0.12928679, 0.37901797, 0.54283535, 0.50823672] ###N_shepherd = 2,3,4,5 N_sheep = 100_L3=20
# Coordinate_rate_2 = [0.1625286,  0.15084054, 0.54076582, 0.43316742] ###N_shepherd = 2,3,4,5 N_sheep = 200_L3=20
# Coordinate_rate_3 = [0.10899191, 0.07678632, 0.34647055, 0.3899159 ] ###N_shepherd = 2,3,4,5 N_sheep = 300_L3=20
# Coordinate_rate_4 = [0.014689,   0.01714197, 0.04700806, 0.37046774]  ###N_shepherd = 2,3,4,5 N_sheep = 400_L3=20
###############################################################################
##############################New difference method: calculate_pairwise_differ_states = Cij/N #################################################
# l3 = 0  #
# Coordinate_rate_1 = [0.11446127, 0.16150995, 0.15245258, 0.18980776]  ###N_shepherd = 2,3,4,5 N_sheep = 100_L3=0
# Coordinate_rate_2 = [0.02219488, 0.09343701, 0.16257676, 0.14920585]  ###N_shepherd = 2,3,4,5 N_sheep = 200_L3=0
# Coordinate_rate_3 = [0.05145715, 0.11047195, 0.10683229, 0.13695883]  ###N_shepherd = 2,3,4,5 N_sheep = 300_L3=0
# Coordinate_rate_4 = [0.01473957, 0.02325361, 0.06825137, 0.07935163]  ###N_shepherd = 2,3,4,5 N_sheep = 400_L3=0
#
# l3 = 5 #
# Coordinate_rate_1 =  [0.4488077,  0.11242023, 0.14513916, 0.19033357] ###N_shepherd = 2,3,4,5 N_sheep = 100_L3=5
# Coordinate_rate_2 =  [0.08492622, 0.07642292, 0.07232158, 0.12994346] ###N_shepherd = 2,3,4,5 N_sheep = 200_L3=5
# Coordinate_rate_3 =  [0.06963093, 0.10652386, 0.11478791, 0.11081657] ###N_shepherd = 2,3,4,5 N_sheep = 300_L3=5
# Coordinate_rate_4 =  [0.01073654, 0.05432217, 0.0376154, 0.12386883] ###N_shepherd = 2,3,4,5 N_sheep = 400_L3=5

# l3 = 10 #
# Coordinate_rate_1 =  [0.35838987, 0.19594686, 0.20509263, 0.18446148] ###N_shepherd = 2,3,4,5 N_sheep = 100_L3=10
# Coordinate_rate_2 =  [0.23044335, 0.23346495, 0.14255964, 0.25671146] ###N_shepherd = 2,3,4,5 N_sheep = 200_L3=10
# Coordinate_rate_3 =  [0.05542843, 0.1775585, 0.13453253, 0.12049661] ###N_shepherd = 2,3,4,5 N_sheep = 300_L3=10
# Coordinate_rate_4 =  [0.04579048, 0.00266333, 0.0153928,  0.0605764 ]###N_shepherd = 2,3,4,5 N_sheep = 400_L3=10

# l3 = 15 #
# Coordinate_rate_1 =  [0.2497901,  0.3436127,  0.18447026, 0.20830424] ###N_shepherd = 2,3,4,5 N_sheep = 100_L3=15
# Coordinate_rate_2 =  [0.19527684, 0.26925166, 0.27917186, 0.19795973] ###N_shepherd = 2,3,4,5 N_sheep = 200_L3=15
# Coordinate_rate_3 =  [0.27665068, 0.09220154, 0.13620125, 0.15892195] ###N_shepherd = 2,3,4,5 N_sheep = 300_L3=15
# Coordinate_rate_4 =  [0.004141,   0.07005232, 0.06996356, 0.12717596] ###N_shepherd = 2,3,4,5 N_sheep = 400_L3=15

# l3 = 20 #
# Coordinate_rate_1 = [0.12928679, 0.25267865, 0.30998367, 0.27725155]    ###N_shepherd = 2,3,4,5 N_sheep = 100_L3=20
# Coordinate_rate_2 = [0.1625286,  0.10056036, 0.31199312, 0.21253594]    ###N_shepherd = 2,3,4,5 N_sheep = 200_L3=20
# Coordinate_rate_3 = [0.10899191, 0.05119088, 0.19994449, 0.18298071]    ###N_shepherd = 2,3,4,5 N_sheep = 300_L3=20
# Coordinate_rate_4 = [0.014689,   0.01142798, 0.02625245, 0.18189089]    ###N_shepherd = 2,3,4,5 N_sheep = 400_L3=20

###############################################################################
# plot_coordination_ratio(Coordinate_rate_1, Coordinate_rate_2, Coordinate_rate_3, Coordinate_rate_4, l3)
