import matplotlib.pyplot as plt
import os, sys
import numpy as np
import h5py
from collections import defaultdict
from timeit import default_timer as timer
from datetime import timedelta
import glob


def plot_multi_states(shepherd_state):
    plt.figure(figsize=(8, 6), dpi=300)
    N_shepherd = shepherd_state.shape[0]
    # print("N_shepherd", N_shepherd)
    Iterations = shepherd_state.shape[1]
    x = [item * 0.01 for item in range(Iterations)]
    # subplot shepherd state
    for shepherd_index in range(N_shepherd):
        # print(shepherd_index)
        plt.subplot(N_shepherd + 1, 1, shepherd_index + 1)
        y = shepherd_state[shepherd_index, :]  # 1.0  drive_mode_true
        plt.plot(x, y, 'b-')
        plt.ylabel("Shepherd" + str(shepherd_index))
        if shepherd_index == 0:
            plt.title("Shepherd states")
    # subplot difference state
    difference = np.zeros(Iterations)
    for index in range(Iterations):
        if sum(shepherd_state[:, index]) != 0 and sum(shepherd_state[:, index]) != N_shepherd:
            difference[index] = 1
    print("difference:", sum(difference))
    plt.subplot(N_shepherd + 1, 1, N_shepherd + 1)
    plt.plot(x, difference, 'r-')
    plt.ylabel("difference")
    # plt.savefig("./figures/state_of_shepherd.png")
    plt.xlabel("Time (s)")
    plt.show()
    return


def read_hdf5_data(path):
    # get directory for the path
    # directory = os.getcwd() + "/../data/"
    # file_name = "N_sheep=400_N_shepherd=3_Final_tick=200000_Repetition=0.hdf5"
    # path = directory + file_name
    with h5py.File(path, "r") as f:
        # print(f.keys())
        agents = f.get("agent_data")[:]
        shepherd = f.get("shepherd_data")[:]
        agents_num = agents.shape[0]
        shepherd_num = shepherd.shape[0]
        agents_pos = agents[:, 0:2, :]
        agents_state = agents[:, 3, :]
        shepherd_pos = shepherd[:, 0:2, :]
        shepherd_state = shepherd[:, 3, :]
        # print("shepherd_state", shepherd_state.shape)
        # print("agents_state", agents_state.shape)
    return agents_pos, agents_state, shepherd_pos, shepherd_state


def data_plot(data, N_sheep):
    N_shepherd = len(data)
    # repetition = len(data[0])
    X = np.arange(N_shepherd) + 1
    Time = [np.mean(data[item]) * 0.01 for item in range(N_shepherd)]
    plt.plot(X, Time, 's-', label="N_sheep = " + str(N_sheep))
    plt.xticks(X, ["N = 1", "N = 2", "N = 3", "N = 4", "N = 5"])
    return

def success_rate_plot(data, N_sheep, n_shepherd):
    # N_shepherd = len(data)
    N_shepherd = n_shepherd
    # repetition = len(data[0])
    X = np.arange(N_shepherd) + 1
    Time = [1/(np.mean(data[item]) * 0.01) for item in range(N_shepherd)]
    plt.plot(X, Time, 's-', label="N_sheep = " + str(N_sheep))
    plt.xticks(X, ["N = 1", "N = 2", "N = 3", "N = 4", "N = 5"])
    # plt.axis("square")
    return

def get_dict_of_data(N_sheep):
    directory = os.getcwd() + "/../Data"
    files = os.listdir(directory)
    dic = defaultdict(list)
    # matrix = np.zeros(len(data + 1))
    for N_shepherd in range(1, 6):
        count = 0
        for file in files:
            if not os.path.isdir(file):
                keyword = "N_sheep=" + str(N_sheep) + "_N_shepherd=" + str(N_shepherd)
                if keyword in file:
                    count = count + 1
                    file_name_string = os.path.splitext(file)[0].split("_")
                    for item in file_name_string:
                        if "tick" in item:  # get final ticks from the file name "tick = ..."
                            dic[keyword].append(int(item.split("=")[1]))
        # print(keyword)
        # print(count)
    return dic


# plot time as a function of the number of shepherds
def plot_time_shepherd(list_of_sheep_number):
    plt.figure(figsize=(8, 6), dpi=300)
    for N_sheep in list_of_sheep_number:
        dic = get_dict_of_data(N_sheep)
        data = list(dic.values())
        repetition = len(data[0])
        print("N_sheep=:", N_sheep, "repetition=:", repetition)
        data_plot(data, N_sheep)
    plt.xlabel("Shepherd number")
    plt.ylabel("Time (s)")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13),
               ncol=3, fancybox=True, shadow=True)
    plt.savefig("./figures/Time_and_N_shepherd.png")
    plt.show()
    return


###############################################################################################
def get_dict_of_data_with_L3(N_sheep, l3):
    directory = os.getcwd() + "/../Data_using_vision"  # +"/../Data"
    files = os.listdir(directory)
    dic = defaultdict(list)
    # matrix = np.zeros(len(data + 1))
    for N_shepherd in range(1, 6):
        count = 0
        for file in files:
            if not os.path.isdir(file):
                keyword = "N_sheep=" + str(N_sheep) + "_N_shepherd=" + str(N_shepherd) + "_L3=" + str(l3)
                if keyword in file:
                    count = count + 1
                    file_name_string = os.path.splitext(file)[0].split("_")
                    for item in file_name_string:
                        if "tick" in item:  # get final ticks from the file name "tick = ..."
                            dic[keyword].append(int(item.split("=")[1]))
        # print(keyword)
        # print(count)
    return dic


def plot_time_shepherd_force(list_of_sheep_number, list_of_force):
    for l3 in list_of_force:
        # plt.figure(figsize=(8, 6), dpi=300)
        plt.figure()
        for N_sheep in list_of_sheep_number:
            dic = get_dict_of_data_with_L3(N_sheep, l3)
            data = list(dic.values())
            repetition = len(data[0])
            print("L3:", l3, "N_sheep:", N_sheep, repetition)
            data_plot(data, N_sheep)
        # plt.title("L3 = 0")
        plt.xlabel("Shepherd number (L3 = " + str(l3) + ")")
        plt.ylabel("Time (s)")
        plt.legend()
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13),
        #            ncol=3, fancybox=True, shadow=True)
        plt.savefig("./figures/Time_and_N_shepherd_L3=" + str(l3) + ".png")
        plt.show()
    return

#################################################################################

def plot_sr_shepherd_force(list_of_sheep_number, list_of_force, n_shepherd):
    for l3 in list_of_force:
        # plt.figure(figsize=(4, 4), dpi=300)
        plt.figure()
        for N_sheep in list_of_sheep_number:
            dic = get_dict_of_data_with_L3(N_sheep, l3)
            data = list(dic.values())
            repetition = len(data[0])
            # print("L3:", l3, "N_sheep:", N_sheep, repetition)
            # print("data:", data)
            success_rate_plot(data, N_sheep, n_shepherd)
        plt.title("L3 = 0")
        plt.xlabel("Shepherd number (L3 = " + str(l3) + ")")
        plt.ylabel("Success Rate")
        plt.legend()
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=3, fancybox=True, shadow=True)
        # plt.axis("square")
        plt.savefig("./figures/SR_and_N_shepherd_L3=" + str(l3) + ".png")
        plt.show()
    return

#########################################################################################
def get_dict_of_data_force(N_sheep, N_shepherd, list_of_force):
    directory = os.getcwd() + "/../Data"
    files = os.listdir(directory)
    dic = defaultdict(list)
    L3 = list_of_force
    # matrix = np.zeros(len(data + 1))
    for l3 in L3:
        count = 0
        for file in files:
            if not os.path.isdir(file):
                keyword = "N_sheep=" + str(N_sheep) + "_N_shepherd=" + str(N_shepherd) + "_L3=" + str(l3)
                if keyword in file:
                    count = count + 1
                    file_name_string = os.path.splitext(file)[0].split("_")
                    for item in file_name_string:
                        if "tick" in item:  # get final ticks from the file name "tick = ..."
                            dic[keyword].append(int(item.split("=")[1]))
        print(keyword)
        print(count)
    return dic


######################
# plot SUCCESS RATE as a function of L3, under fixed shepherd number
def plot_success_rate_with_force(N_sheep, List_of_N_shepherd, List_of_force):
    plt.figure(figsize=(8, 6), dpi=300)
    n = len(List_of_force)
    for N_shepherd in List_of_N_shepherd:
        dic = get_dict_of_data_force(N_sheep, N_shepherd, List_of_force)
        data = list(dic.values())
        repetition = len(data[0])
        print("N_sheep=", N_sheep, "N_shepherd=", N_shepherd, "repetition=", repetition)
        # data_plot(data, N_sheep)
        X = List_of_force
        # Time = [np.mean(data[item]) * 0.01 for item in range(n)]
        success_rate = [1 / (np.mean(data[item]) * 0.01) for item in range(n)]
        plt.plot(X, success_rate, 's-', label="N_shepherd = " + str(N_shepherd))
        plt.xticks(X, ["L3 = 0", "L3 = 5", "L3 = 10", "L3 = 15", "L3 = 20"])

    plt.title("N_sheep = " + str(N_sheep))
    plt.xlabel("Equilibrium distance between shepherd")
    plt.ylabel("Success Rate")
    plt.legend()
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13),ncol=3, fancybox=True, shadow=True)
    plt.savefig("./figures/SR_and_L3_Ns=" + str(N_sheep) + "N_shepherd=" + str(N_shepherd) + ".png")
    plt.show()
    return

#######################################################################################################
######################
# plot SUCCESS RATE as a function of L3, under fixed shepherd number
def plot_success_rate_with_force_fix_shepherd(List_of_N_sheep, N_shepherd, List_of_force):
    plt.figure(figsize=(8, 6), dpi=300)
    n = len(List_of_force)
    for N_sheep in list_of_sheep_number:
        dic = get_dict_of_data_force(N_sheep, N_shepherd, List_of_force)
        data = list(dic.values())
        repetition = len(data[0])
        print("N_sheep=", N_sheep, "N_shepherd=", N_shepherd, "repetition=", repetition)
        # data_plot(data, N_sheep)
        X = List_of_force
        # Time = [np.mean(data[item]) * 0.01 for item in range(n)]
        success_rate = [1 / (np.mean(data[item]) * 0.01) for item in range(n)]
        plt.plot(X, success_rate, 's-', label="N_sheep = " + str(N_sheep))
        plt.xticks(X, ["L3 = 0", "L3 = 5", "L3 = 10", "L3 = 15", "L3 = 20"])

    plt.title("N_shepherd = " + str(N_shepherd))
    plt.xlabel("Equilibrium distance between shepherd")
    plt.ylabel("Success Rate")
    plt.legend()
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13),ncol=3, fancybox=True, shadow=True)
    plt.savefig("./figures/SR_and_L3_" + "N_shepherd=" + str(N_shepherd) + ".png")
    # plt.show()
    plt.close()
    return


########################################################################################################

# plot finish time as a function of L3, under fixed shepherd number
def plot_time_force(N_sheep, List_of_N_shepherd, List_of_force):
    plt.figure(figsize=(8, 6), dpi=300)
    n = len(List_of_force)
    for N_shepherd in List_of_N_shepherd:
        dic = get_dict_of_data_force(N_sheep, N_shepherd, List_of_force)
        data = list(dic.values())
        repetition = len(data[0])
        print("N_sheep=", N_sheep, "N_shepherd=", N_shepherd, "repetition=", repetition)
        # data_plot(data, N_sheep)
        X = List_of_force
        Time = [np.mean(data[item]) * 0.01 for item in range(n)]
        plt.plot(X, Time, 's-', label="N_shepherd = " + str(N_shepherd))
        plt.xticks(X, ["L3 = 0", "L3 = 5", "L3 = 10", "L3 = 15", "L3 = 20"])

    plt.title("N_sheep = " + str(N_sheep))
    plt.xlabel("Equilibrium distance between shepherd")
    plt.ylabel("Time (s)")
    plt.legend()
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13),ncol=3, fancybox=True, shadow=True)
    plt.savefig("./figures/Time_and_L3_Ns=" + str(N_sheep) + ".png")
    plt.show()
    return


######################################################################################################

def get_dict_of_data_force_2(N_sheep, N_shepherd, list_of_force):
    if N_sheep == 400:
        directory = os.getcwd() + "/../Data_N_sheep=400/N_sheep=400_N_shepherd=5"
    else:
        directory = os.getcwd() + "/../Data"
    files = os.listdir(directory)
    dic = defaultdict(list)
    L3 = list_of_force
    # matrix = np.zeros(len(data + 1))
    for l3 in L3:
        count = 0
        for file in files:
            if not os.path.isdir(file):
                keyword = "N_sheep=" + str(N_sheep) + "_N_shepherd=" + str(N_shepherd) + "_L3=" + str(l3)
                if keyword in file:
                    count = count + 1
                    file_name_string = os.path.splitext(file)[0].split("_")
                    for item in file_name_string:
                        if "tick" in item:  # get final ticks from the file name "tick = ..."
                            dic[keyword].append(int(item.split("=")[1]))
        print(keyword)
        print(count)
    return dic


def plot_time_force_2(N_sheep, List_of_N_shepherd, List_of_force):
    plt.figure(figsize=(8, 6), dpi=300)
    n = len(List_of_force)
    for N_shepherd in List_of_N_shepherd:
        dic = get_dict_of_data_force_2(N_sheep, N_shepherd, List_of_force)
        data = list(dic.values())
        repetition = len(data[0])
        print("N_sheep=", N_sheep, "N_shepherd=", N_shepherd, "repetition=", repetition)
        # data_plot(data, N_sheep)
        X = List_of_force
        Time = [np.mean(data[item]) * 0.01 for item in range(n)]
        plt.plot(X, Time, 's-', label="N_shepherd = " + str(N_shepherd))
        plt.xticks(X, ["L3 = 0", "L3 = 5", "L3 = 10", "L3 = 15", "L3 = 20", "L3 = 25"])

    plt.title("N_sheep = " + str(N_sheep))
    plt.xlabel("Equilibrium distance between shepherd")
    plt.ylabel("Time (s)")
    plt.legend()
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13),ncol=3, fancybox=True, shadow=True)
    plt.savefig("./figures/Time_and_L3_Ns=" + str(N_sheep) + "N_shepherd=" + str(N_shepherd) + ".png")
    plt.show()
    return


#########
def plot_success_rate_force_2(N_sheep, List_of_N_shepherd, List_of_force):
    plt.figure(figsize=(8, 6), dpi=300)
    n = len(List_of_force)
    for N_shepherd in List_of_N_shepherd:
        dic = get_dict_of_data_force_2(N_sheep, N_shepherd, List_of_force)
        data = list(dic.values())
        repetition = len(data[0])
        print("N_sheep=", N_sheep, "N_shepherd=", N_shepherd, "repetition=", repetition)
        # data_plot(data, N_sheep)
        X = List_of_force
        Time = [1 / (np.mean(data[item]) * 0.01) for item in range(n)]
        plt.plot(X, Time, 's-', label="N_shepherd = " + str(N_shepherd))
        plt.xticks(X, ["L3 = 0", "L3 = 5", "L3 = 10", "L3 = 15", "L3 = 20", "L3 = 25"])

    plt.title("N_sheep = " + str(N_sheep))
    plt.xlabel("Equilibrium distance between shepherd")
    plt.ylabel("Success Rate")
    plt.legend()
    # plt.axis('equal')
    # plt.axis('square')
    plt.savefig("./figures/SR_and_L3_Ns=" + str(N_sheep) + "N_shepherd=" + str(N_shepherd) + ".png")
    plt.show()
    return


##########################################################################################
def sucess_rate_plot(mean_success_rate, N_sheep):
    N_shepherd = len(mean_success_rate)
    print("N_shepherd:", N_shepherd)
    # repetition = len(data[0])
    X = np.arange(N_shepherd) + 1
    plt.plot(X, mean_success_rate, 's-', label="N_sheep = " + str(N_sheep))
    plt.xticks(X, ["N = 1", "N = 2", "N = 3", "N = 4", "N = 5"])
    return


def get_success_rate(N_sheep):
    dic = get_dict_of_data(N_sheep)
    data = list(dic.values())
    Data = np.array(data, dtype=float)
    Data = 1 / (Data * 0.01)
    mean_success_rate = np.mean(Data, axis=1)  # calculate the mean value according to lines;
    print("N_sheep", N_sheep, "mean_success_rate", mean_success_rate)
    return mean_success_rate


def draw_success_rate(list_of_sheep_number):
    plt.figure(figsize=(8, 6), dpi=300)
    for N_sheep in list_of_sheep_number:
        mean_success_rate = get_success_rate(N_sheep)
        sucess_rate_plot(mean_success_rate, N_sheep)
    plt.xlabel("Shepherd number")
    plt.ylabel("Success rate")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13),
               ncol=3, fancybox=True, shadow=True)
    plt.savefig("./figures/Mean_success_rate.png")
    plt.show()

    return


#################################################################################################

######### plot time as a function of the number of shepherds
# list_of_force = [i for i in range(0, 25, 5)] #[0]
# list_of_sheep_number = [i for i in range(100, 500, 100)]
# # plot_time_shepherd(list_of_sheep_number)
# # draw_success_rate(list_of_sheep_number)
# # plot_time_shepherd_force(list_of_sheep_number, list_of_force)
# plot_sr_shepherd_force(list_of_sheep_number, list_of_force)

###########################################################
###### plot time as function of differnt L3 value under the same number of shepherd ##########
# list_of_sheep_number = [i for i in range(100, 500, 100)]
# list_of_force = [i for i in range(0, 25, 5)]
# list_of_N_shepherd = [i for i in range(1, 6, 1)]
# # plot_time_force(N_sheep, list_of_N_shepherd, list_of_force)
# for n_sheep in list_of_sheep_number:
#     # plot_time_force(n_sheep, list_of_N_shepherd, list_of_force)
#     plot_success_rate_with_force(n_sheep, list_of_N_shepherd, list_of_force)

##########################Fix N_shepherd##################
# list_of_sheep_number = [i for i in range(100, 500, 100)]
# list_of_force = [i for i in range(0, 25, 5)]
# list_of_shepherd = [i for i in range(1, 2, 1)]
# for n_shepherd in list_of_shepherd:
#     plot_success_rate_with_force_fix_shepherd(list_of_sheep_number, n_shepherd, list_of_force)

####################### Ns = 400 ##############
# n_sheep = 300
# list_of_force = [i for i in range(0, 25, 5)]
# list_of_N_shepherd = [i for i in range(1, 6, 1)]
# plot_time_force(n_sheep, list_of_N_shepherd, list_of_force)

#################### Ns = 400, N_shepherd = 5 ################
# n_sheep = 400
# list_of_N_shepherd = [5]
# list_of_force = [i for i in range(0, 30, 5)]
# # plot_time_force_2(n_sheep, list_of_N_shepherd, list_of_force)
# plot_success_rate_force_2(n_sheep, list_of_N_shepherd, list_of_force)

##########################vision data analysis ####################
######### plot time as a function of the number of shepherds
list_of_force = [0]
list_of_sheep_number = [i for i in range(100, 500, 100)]
n_shepherd = 5
# list_of_shepherd_number = [i for i in range(1, 5, 1)]
# plot_time_shepherd(list_of_sheep_number)
# draw_success_rate(list_of_sheep_number)
plot_time_shepherd_force(list_of_sheep_number, list_of_force)
# plot_sr_shepherd_force(list_of_sheep_number, list_of_force, n_shepherd)
