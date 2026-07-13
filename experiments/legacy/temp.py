import numpy as np
from basic.initiation import initiate, initiate_shepherds
import os, sys
import h5py
from timeit import default_timer as timer
from datetime import timedelta
from basic.create_network import create_metric_network, create_topological_network

# dis = np.array([[1,28,14,25,9,3],[1,28,14,np.nan,9,3],[1,28,14,np.inf,9,3],[1,2,14,25,9,3]])
# print(dis)
#
# # temp1 = np.array(np.nan_to_num(dis, nan=np.inf) < 5, dtype=int)
#
# temp2 = np.argsort(dis, axis=1)  # rank line by line
# print("temp2=:", temp2)
#
# temp3 = np.array(np.argsort(temp2) < 3, dtype = int)
# print("temp3=:", temp3)


# def calculate_mass_center(agents):
#     position_matrix = np.array([agents[:, 0], agents[:, 1]]).T  #list
#     agents[2][21] = 1
#     for index in range(agents.shape[0]):
#         # agent state: staying -> 1; moving -> 0;
#         if agents[index][21] == 1:
#             position_matrix[index, :] = 0.0
#             print(position_matrix[index, :])
#     center_x = np.mean(position_matrix[:, 0])
#     center_y = np.mean(position_matrix[:, 1])
#     return center_x, center_y
#
# agents = initiate(10, 100, 100, 50)
#
# center_x, center_y = calculate_mass_center(agents)
# print(np.mean(agents[:,0]),np.mean(agents[:,1]))
# print("center of mass:", center_x, center_y)

# a = np.load('/home/yateng/Workspace/Shepherd_Behavior/data/.npy')
# print(a[:, :, 12000])

# agents = initiate(10, 100, 100, 50)
# shepherds = initiate_shepherds(2, 10)
# try to create network
# topological_network = create_topological_network(agents, 5)
# metric_network = create_metric_network(agents, agents[0][5], np.pi)  # Fov not used
# print("topological_network:", topological_network)
# print("metric_network:", metric_network)

# a = np.array([[1, 2, 3, 1], [4, 5, 6, 1]])
# b = np.array([[7, 8, 9, 1], [10, 11, 12, 1]])
# c = np.array([[13, 14, 15, 1], [16, 17, 18, 1]])
#
# d = np.hstack((a, b, c))
# print(d)

def connect_data(Data_agents, Data_shepherds):

    # grab meaningful data
    agent_pos = Data_agents[:, 0:3, :]
    agent_state = Data_agents[:, 20:21, :]         #1: staying mode

    shepherd_pos = Data_shepherds[:, 0:3, :]
    shepherd_state = Data_shepherds[:, 12:13, :]   #1: drive mode

    Data = np.hstack((agent_pos, agent_state, shepherd_pos, shepherd_state))
    print(Data.shape)
    return Data

def read_hdf5_data():
    # get directory for the path
    directory = os.getcwd() + "/data/"
    file_name = 'N_sheep=400_N_shepherd=2_Final_tick=200000_Repetition=0.hdf5'
    with h5py.File(directory + file_name, "r") as f:
        print(f.keys())
        agents = f.get("agent_data")[:]
        shepherd = f.get("shepherd_data")[:]
        agents_pos = agents[:, 0:2, :]
        agents_state = agents[:, 3, :]
        shepherd_pos = agents[:, 0:2, :]
        shepherd_state = agents[:, 3, :]
        print(agents.shape)
        print(shepherd.shape)
        print(agents_pos.shape)
        print(agents_state.shape)
    return

start = timer()
read_hdf5_data()

end = timer()
print(timedelta(seconds=end-start))
# print(timer())