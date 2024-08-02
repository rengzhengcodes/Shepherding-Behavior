import csv
import matplotlib.pyplot as plt
import os, sys
import numpy as np
import h5py
import seaborn as sns
from timeit import default_timer as timer
from datetime import timedelta
from collections import defaultdict


def read_hdf5_data(file_path):
    with h5py.File(file_path, "r") as f:
        # print(f.keys())
        agents = f.get("agent_data")[:]
        shepherd = f.get("shepherd_data")[:]
        agents_pos = agents[:, 0:2, :]
        agents_state = agents[:, 3, :]
        shepherd_pos = shepherd[:, 0:2, :]
        shepherd_state = shepherd[:, 3, :]
        f.close()
    return agents_pos, agents_state, shepherd_pos, shepherd_state


def get_final_tick(file_name):
    file_name_string = os.path.splitext(file_name)[0].split("_")
    for item in file_name_string:
        if "tick" in item:
            tick = int(item.split("=")[1])
    return tick


def calculate_differ_states(shepherd_states_data, N_shepherd, Iterations):
    # N_shepherd = shepherd_states_data.shape[0]
    iterations = Iterations
    difference = np.zeros(iterations)
    for index in range(iterations):
        if sum(shepherd_states_data[:, index]) != 0 and sum(shepherd_states_data[:, index]) != N_shepherd:
            difference[index] = 1
    value = sum(difference) / (Iterations + 1)
    # print("N_shepherd:", N_shepherd, "difference:", value, "iterations", iterations)
    # drive_ratio.append(np.sum(shepherd_state) / Iterations)
    return value


def get_drive_mode_ratio(shepherd_states_data, Iterations):
    difference = np.zeros(Iterations)
    for index in range(Iterations):
        if sum(shepherd_states_data[:, index]) != 0:  # == 1.0:  #????
            difference[index] = 1
    value = sum(difference) / Iterations
    return value


def calculate_drive_matrix(List_of_L3, N_sheep, List_of_N_shepherd, Directory):
    mean_matrix = np.zeros(len(List_of_N_shepherd), dtype=float)
    std_matrix = np.zeros(len(List_of_N_shepherd), dtype=float)
    for L3 in List_of_L3:
        # for N_sheep in List_of_N_sheep:
        data_folder = Directory + "/L3=" + str(L3) + "/N_sheep=" + str(N_sheep) + "/"
        files = os.listdir(data_folder)
        count = 0
        for N_shepherd in List_of_N_shepherd:
            keyword = "N_sheep=" + str(N_sheep) + "_N_shepherd=" + str(N_shepherd) + "_L3=" + str(L3)
            # print(keyword)
            states = []
            for file in files:
                if not os.path.isdir(file):
                    if keyword in file:  # and "Repetition=0" in file:
                        # print(file)
                        file_path = data_folder + file
                        agents_pos, agents_state, shepherd_pos, shepherd_state = read_hdf5_data(file_path)
                        Iterations = get_final_tick(file)
                        # print("Iterations:", Iterations)
                        if Iterations != 200000:
                            value = get_drive_mode_ratio(shepherd_state, Iterations)
                            states.append(value)
                            # print("value:", value)
                else:
                    print("Please enter a file path!")
            mean_value = np.mean(np.array(states))  # ratio over repetitions
            std_value = np.std(np.array(states))
            print(keyword, "repetitions=", np.array(states).shape[0], "mean drive ratio:",
                  mean_value, "std drive ratio:", std_value)
            mean_matrix[count] = mean_value
            std_matrix[count] = std_value
            count = count + 1
    return mean_matrix, std_matrix


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
        # print("count:", count)
        difference[tick] = difference[tick] / count
    value = sum(difference) / Iterations
    # print("N_shepherd:", N_shepherd, "count:", count, "difference:", value, "iterations", iterations)
    return value


##########calculate difference value for multiple data#######
def calculate_multi_difference(List_of_L3, N_sheep, List_of_N_shepherd, Directory):
    mean_matrix = np.zeros(len(List_of_N_shepherd), dtype=float)
    std_matrix = np.zeros(len(List_of_N_shepherd), dtype=float)
    for L3 in List_of_L3:
        # for N_sheep in List_of_N_sheep:
        data_folder = Directory + "/L3=" + str(L3) + "/N_sheep=" + str(N_sheep) + "/"
        files = os.listdir(data_folder)
        count = 0
        for N_shepherd in List_of_N_shepherd:
            keyword = "N_sheep=" + str(N_sheep) + "_N_shepherd=" + str(N_shepherd) + "_L3=" + str(L3)
            # print(keyword)
            states = []
            for file in files:
                if not os.path.isdir(file):
                    if keyword in file:  # and "Repetition=17" in file:
                        # print(file)
                        file_path = data_folder + file
                        agents_pos, agents_state, shepherd_pos, shepherd_state = read_hdf5_data(file_path)
                        Iterations = get_final_tick(file)
                        # print("Iterations:", Iterations)
                        if Iterations != 200000:
                            # value1 = calculate_pairwise_differ_states(shepherd_state, Iterations)
                            value = calculate_differ_states(shepherd_state, N_shepherd, Iterations)
                            states.append(value)
                            # print("value:", value)
                else:
                    print("Please enter a file path!")
            mean_value = np.mean(np.array(states))  # ratio over repetitions
            std_value = np.std(np.array(states))
            print(keyword, "repetitions=", np.array(states).shape[0], "mean_different_state_value", mean_value,
                  "std difference ratio:", std_value)
            mean_matrix[count] = mean_value
            std_matrix[count] = std_value
            count = count + 1
    return mean_matrix, std_matrix


def plot_drive_ratio(list_of_L3, list_of_N_sheep, list_of_N_shepherd, path):
    for L3 in list_of_L3:
        plt.figure(figsize=(8, 6), dpi=300)
        x = list_of_N_shepherd
        labels = ["N = " + str(n_shepherd) for n_shepherd in list_of_N_shepherd]
        for N_sheep in list_of_N_sheep:
            mean_matrix, std_matrix = calculate_drive_matrix(list_of_L3, N_sheep, list_of_N_shepherd, path)
            # plt.plot(x, coordinate_ratio, 's-', label="N_sheep = " + str(N_sheep))
            plt.errorbar(x, mean_matrix, yerr=std_matrix, fmt="-o", label="N_sheep = " + str(N_sheep))
        plt.xticks(x, labels)
        plt.title("L3 = " + str(L3))
        plt.ylim(0, 1)
        plt.xlabel("Shepherd number")
        plt.ylabel("Drive Ratio")
        plt.legend(loc="best")
        plt.savefig("./figures/Drive_Ratio_L3=" + str(L3) + ".png")
        # plt.savefig("./figures/Drive_Ratio_Vision_Model" + ".png")
        plt.show()
        plt.clf()


def plot_coordination_ratio(list_of_L3, list_of_N_sheep, list_of_N_shepherd, path):
    for L3 in list_of_L3:
        plt.figure(figsize=(8, 6), dpi=300)
        x = list_of_N_shepherd
        labels = ["N = " + str(n_shepherd) for n_shepherd in list_of_N_shepherd]
        for N_sheep in list_of_N_sheep:
            mean_matrix, std_matrix = calculate_multi_difference(list_of_L3, N_sheep, list_of_N_shepherd, path)
            # plt.plot(x, mean_matrix, '^-', label="N_sheep = " + str(N_sheep))
            plt.errorbar(x, mean_matrix, yerr=std_matrix, fmt="-o", label="N_sheep = " + str(N_sheep))
        plt.xticks(x, labels)
        plt.ylim(0, 0.7)
        plt.title("L3 = " + str(L3))
        # plt.title("Vision Model")
        plt.xlabel("Shepherd number")
        plt.ylabel("Coordination Ratio")
        plt.legend(loc="best")
        plt.savefig("./figures/Coordination_Ratio_L3=" + str(L3) + ".png")
        # plt.savefig("./figures/Coordination_Ratio_Vision_Model" + ".png")
        plt.show()
        plt.clf()


def calculate_time_in_different_modes(list_of_L3, list_of_N_sheep, list_of_N_shepherd, path):
    for L3 in list_of_L3:
        for N_shepherd in list_of_N_shepherd:
            drive_mode_list = []
            collect_mode_list = []
            drive_mode = dict([])
            collect_mode = dict([])
            for N_sheep in list_of_N_sheep:
                data_folder = path + "/L3=" + str(L3) + "/N_sheep=" + str(N_sheep) + "/"
                files = os.listdir(data_folder)
                keyword = "N_sheep=" + str(N_sheep) + "_N_shepherd=" + str(N_shepherd) + "_L3=" + str(L3)
                # print(keyword)
                drive_ratio = []
                collect_ratio = []
                count = 0
                for file in files:
                    if not os.path.isdir(file):
                        if keyword in file:
                            # print(file)
                            file_path = data_folder + file
                            agents_pos, agents_state, shepherd_pos, shepherd_state = read_hdf5_data(file_path)
                            Iterations = get_final_tick(file)
                            # print("Iterations:", Iterations)
                            drive_ratio.append(np.sum(shepherd_state) / Iterations)
                            collect_ratio.append(1 - np.sum(shepherd_state) / Iterations)
                            # print(len(drive_ratio))
                            # print(shepherd_state)
                            count = count + 1
                # print("drive_ratio:", drive_ratio)
                # print("collect_ratio", collect_ratio)
                drive_mode_list.append(drive_ratio)
                collect_mode_list.append(collect_ratio)
                drive_mode["Ns=" + str(N_sheep)] = drive_ratio
                collect_mode["Ns=" + str(N_sheep)] = collect_ratio
                print(keyword, "Repetitions:", count)

    drive_mode_array = np.array(drive_mode_list)
    collect_mode_array = np.array(collect_mode_list)
    return drive_mode_array, collect_mode_array


def plot_multi_states(shepherd_state, N_sheep, N_shepherd, Iterations):
    plt.figure(figsize=(8, 6), dpi=300)
    # N_shepherd = shepherd_state.shape[0]
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
    plt.savefig("./figures/state_of_shepherd=" + str(N_shepherd) + "_N_sheep=" + str(N_sheep) + ".png")
    plt.show()
    return


def save_file(file, data):
    with open(file, "wb") as f:
        np.save(f, data)
    f.close()
    return


# (i) L3 = 0 N_shepherd = 1 N_sheep = 100, 150, 200, 250, 300,
# box_plot_new
# path = os.getcwd() + "/../Data_Metric_Model"
# list_of_L3 = [0]
# list_of_N_sheep = [i for i in range(100, 350, 50)]
# list_of_N_shepherd = [1]
#
# drive_mode_arr, collect_mode_arr = calculate_time_in_different_modes(list_of_L3, list_of_N_sheep, list_of_N_shepherd, path)
# file1 = os.getcwd() + "/mediate_result/drive_mode.npy"
# file2 = os.getcwd() + "/mediate_result/collect_mode.npy"
# save_file(file1, drive_mode_arr)
# save_file(file2, collect_mode_arr)

###(ii) single file test
# path = directory = os.getcwd() + "/../Data_Metric_Model"
# file_1 = path + "/L3=0/N_sheep=150/" + "N_sheep=150_N_shepherd=4_L3=0_Final_tick=79160_Repetition=19" + ".hdf5"
# file_2 = "../data/N_sheep=150_N_shepherd=4_L3=0_Final_tick=79160_Repetition=19.hdf5"
# agents_pos, agents_state, shepherd_pos, shepherd_state = read_hdf5_data(file_2)
# # print(shepherd_state)
# N_sheep = 150
# N_shepherd = 4
# Iterations = 79160
# # plot_multi_states(shepherd_state, N_sheep, N_shepherd, Iterations)
# value1 = calculate_differ_states(shepherd_state, N_shepherd, Iterations)
# value2 = calculate_pairwise_differ_states(shepherd_state,  Iterations)
# value3 = get_drive_mode_ratio(shepherd_state, Iterations)
# print(value1*Iterations, value2*Iterations, value3*Iterations)

#######(iii) L3 = 0
# path = os.getcwd() + "/../Data_Metric_Model"
# # path = "/media/yateng/Extreme SSD/Vision_Model"
# list_of_L3 = [0]
# list_of_N_sheep = [i for i in range(100, 350, 50)]  # 100, 150, 200, 250, 300,
# list_of_N_shepherd = [i for i in range(2, 6)]
#
# # plot_drive_ratio(list_of_L3, list_of_N_sheep, list_of_N_shepherd, path)
# plot_coordination_ratio(list_of_L3, list_of_N_sheep, list_of_N_shepherd, path)

###############plot time as a function of the number of shepherds
# Matrix = calculate_multi_difference(list_of_L3, list_of_N_sheep, list_of_N_shepherd, path)
# print(Matrix)


####(iiii) L3 = 20
path = "/media/yateng/Extreme SSD/"
list_of_L3 = [20]
list_of_N_sheep = [i for i in range(100, 350, 50)]  # 100, 150, 200, 250, 300,
list_of_N_shepherd = [i for i in range(2, 6)]

# plot_drive_ratio(list_of_L3, list_of_N_sheep, list_of_N_shepherd, path)
plot_coordination_ratio(list_of_L3, list_of_N_sheep, list_of_N_shepherd, path)
#

# result_vision model
