import os, sys
import numpy as np
import pandas as pd
import h5py


def save_data_L3(N_sheep, N_shepherd, Repetition, Final_tick, Data_agents, Data_shepherds, L3):
    # get directory for the path
    directory = os.getcwd() + "/data"

    if not os.path.exists(directory):
        os.mkdir(directory)
    file_name = ("N_sheep=" + str(N_sheep) + "_N_shepherd=" + str(N_shepherd) + "_L3=" + str(L3)
                 + "_Final_tick=" + str(Final_tick) + "_Repetition=" + str(Repetition))

    # grab meaningful data
    agent_pos = Data_agents[:, 0:3, :]
    agent_state = Data_agents[:, 21:22, :]  # 1: staying mode
    # print("agent_state:", agent_state.shape)
    agent_data = np.concatenate((agent_pos, agent_state), axis=1)

    shepherd_pos = Data_shepherds[:, 0:3, :]
    shepherd_state = Data_shepherds[:, 13:14, :]  # 1: drive mode
    shepherd_data = np.concatenate((shepherd_pos, shepherd_state), axis=1)
    # save_data
    with h5py.File(directory + "/" + file_name + ".hdf5", "w") as f:
        f.create_dataset("agent_data", data=agent_data, compression="gzip", compression_opts=9)
        f.create_dataset("shepherd_data", data=shepherd_data, compression="gzip", compression_opts=9)

    # # save agents to .npy file
    # Data = np.vstack((Data_agents, Data_shepherds))
    # np.save(file=directory + "/" + file_name + ".npy", arr=Data)
    return


def save_data(N_sheep, N_shepherd, Repetition, Final_tick, Data_agents, Data_shepherds):
    # get directory for the path
    directory = os.getcwd() + "/data"

    if not os.path.exists(directory):
        os.mkdir(directory)
    file_name = ("N_sheep=" + str(N_sheep) + "_N_shepherd=" + str(N_shepherd)
                 + "_Final_tick=" + str(Final_tick) + "_Repetition=" + str(Repetition))

    # grab meaningful data
    agent_pos = Data_agents[:, 0:3, :]
    agent_state = Data_agents[:, 21:22, :]  # 1: staying mode
    # print("agent_state:", agent_state.shape)
    agent_data = np.concatenate((agent_pos, agent_state), axis=1)

    shepherd_pos = Data_shepherds[:, 0:3, :]
    shepherd_state = Data_shepherds[:, 13:14, :]  # 1: drive mode
    shepherd_data = np.concatenate((shepherd_pos, shepherd_state), axis=1)
    # save_data
    with h5py.File(directory + "/" + file_name + ".hdf5", "w") as f:
        f.create_dataset("agent_data", data=agent_data, compression="gzip", compression_opts=9)
        f.create_dataset("shepherd_data", data=shepherd_data, compression="gzip", compression_opts=9)

    # # save agents to .npy file
    # Data = np.vstack((Data_agents, Data_shepherds))
    # np.save(file=directory + "/" + file_name + ".npy", arr=Data)
    return


def read_hdf5_data():
    # get directory for the path
    directory = os.getcwd() + "/data/"
    file_name = 'N_sheep=200_N_shepherd=1_Final_tick=13_Repetition=0.hdf5'
    with h5py.File(directory + file_name, "r") as f:
        print(f.keys())
        agents = f.get("agent_data")[:]
        shepherd = f.get("shepherd_data")[:]
    return


def save_file(N_sheep, N_shepherd, Repetition, Final_tick):
    # write simple data to txt file
    directory = os.getcwd() + "/data/"
    file_name = ("N_sheep=" + str(N_sheep) + "_N_shepherd=" + str(N_shepherd)
                 + "_Final_tick=" + str(Final_tick))  # + "_Repetition=" + str(Repetition)
    file_txt = directory + file_name + ".txt"
    # print(file_txt)
    if not os.path.exists(file_txt):
        f = open(file_txt, "w")
    else:
        f = open(file_txt, "a")
    # f.write("Repetition=%d_Final=tick=%d\n" % int(Repetition) % int(Final_tick))
    f.write("Repetition=%d\n" % int(Repetition))
    f.write("Final=tick=%d\n" % int(Final_tick))
    f.close()
    return


def save_all(n_sheep, n_shepherd, rep, final_tick, agents, shepherds, L3, MODE):
    # get directory for the path
    directory = f"{os.getcwd()}/data"

    # Makes directory if it does not exist.
    if not os.path.exists(directory):
        os.mkdir(directory)
    # Defines the file name.
    file_name = f"N_sheep={n_sheep}_N_shepherd={n_shepherd}_Final_tick={final_tick}_Repetition={rep}_L3={L3}_MODE={MODE}"

    # save_data
    with h5py.File(directory + "/" + file_name + ".hdf5", "w") as f:
        f.create_dataset("agents", data=agents, compression="gzip", compression_opts=9)
        f.create_dataset("shepherds", data=shepherds, compression="gzip", compression_opts=9)