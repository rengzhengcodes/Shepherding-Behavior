import os, json, random, shutil, subprocess, sys
from joblib import Parallel, delayed
from timeit import default_timer as timer
import datetime
from datetime import timedelta
import numba as nb
import numpy as np
import matplotlib.pyplot as plt
from basic.initiation import initiate, initiate_shepherd
from basic.interaction import evolve, make_periodic_boundary
from basic.save_data import save_data, save_data_L3, save_all
from basic.draw import draw_single, draw_dynamic, plot_snapshot
from basic.create_network import create_metric_network, create_topological_network

from basic import MODE, MORPHOLOGY

THREADS = -1
DRAW = False

N_sheep = 300
N_shepherd = 6
Space_x = 150
Space_y = 150

Target_place_x = 400
Target_place_y = 400
Target_size = 125  # radius

Boundary_x = Target_place_x + Target_size + 300
Boundary_y = Target_place_y + Target_size + 300


TICK = 1000
Iterations = 200000

L3 = np.sqrt(N_sheep / N_shepherd) * 5 # average flock radius per shepherd
reps = 1000
seeds = range(reps)

Num_nearst_neighbor = 5

# Function seeds numpy rng in numba code.
@nb.jit(nopython=True)
def seed_run(seed):
    random.seed(seed)
    np.random.seed(seed)


def run_target(rep):
    print("Starting repetition", rep)
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

    # Output logging, print the final tick.
    results = {
        # Static parameters, for reference.
        "Space_x": Space_x,
        "Space_y": Space_y,
        "Target_place_x": Target_place_x,
        "Target_place_y": Target_place_y,
        "Target_size": Target_size,

        # Viewing parameters.
        "Boundary_x": Boundary_x,
        "Boundary_y": Boundary_y,
        "TICK": TICK,
        "Iterations": Iterations,

        # Model parameters
        "N_shepherd": N_shepherd,
        "N_sheep": N_sheep,
        "L3": L3,
        "Repetition": rep,
        "MODE": MODE,

        # Results
        "Final_tick": Final_tick,
        "Success": bool(np.all(agents[:, 21] == 1)),
        "Experiment_type": "Target"
    }

    return results


def run_morph(rep):
    print("Starting repetition", rep)
    seed_run(rep)
    agents = initiate(N_sheep, N_shepherd, Space_x, Space_y, Target_size)
    shepherd = initiate_shepherd(0, N_shepherd, L3)
    # self-organized flocking
    for tick in range(TICK):
        # Defines the target as the global center of mass.
        Target_place_x = np.mean(agents[:, 0])
        Target_place_y = np.mean(agents[:, 1])
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
        # Defines the target as the global center of mass.
        Target_place_x = np.mean(agents[:, 0])
        Target_place_y = np.mean(agents[:, 1])
        # start evolve function
        #! @note L2 is defined in initiate_agent and copied here for brevity.
        agents_update, shepherd_update, max_agents_indexes = evolve(agents, shepherd, Target_place_x, 
                                                                    Target_place_y, L2 := 10*(np.sqrt(N_sheep))*2/3, MODE=MODE)
        # update data
        agents = agents_update
        shepherd = shepherd_update
        # save data
        Data_agents[:, :, tick] = agents
        Data_shepherds[:, :, tick] = shepherd
        Max_agents_indexes[:, tick] = max_agents_indexes  # only two dimension
        # print(tick)
        # stop program if all the sheep are within L2 of the center of mass.
        if np.all(np.sqrt((agents[:, 0] - Target_place_x) ** 2 + (agents[:, 1] - Target_place_y) ** 2) < L2):   # finish
            Final_tick = tick
            success = True
            break
        
    # Draws the results.
    if DRAW:
        folder_path = f"results/morphology_attraction_naïve/{MODE}"
        draw_dynamic(Final_tick, Data_agents, Data_shepherds, 
                    Boundary_x, Boundary_y, 
                    Target_place_x, Target_place_y, Target_size, 
                    L3, MODE=MODE, folder_path=f"{folder_path}/repetition_{rep}")
        # Runs the ffmpeg command to create a video.
        # ffmpeg -framerate 10 -start_number 0 -i %d.png -c:v libx264 -r 30 -pix_fmt yuv420p output.mp4
        subprocess.run(["ffmpeg", "-framerate", "10", 
                        "-start_number", "0", "-i", 
                        f"{folder_path}/repetition_{rep}/%d.png",
                        "-c:v", "libx264", "-r", "30", "-pix_fmt", "yuv420p", 
                        f"{folder_path}/MODE_{MODE}|Rep_{rep}|Final_{Final_tick}.mp4"])
        # Deletes all the images.
        shutil.rmtree(f"{folder_path}/repetition_{rep}")

    # Output logging, print the final tick.
    results = {
        # Static parameters, for reference.
        "Space_x": Space_x,
        "Space_y": Space_y,
        "Target_place_x": Target_place_x,
        "Target_place_y": Target_place_y,
        "Target_size": Target_size,

        # Viewing parameters.
        "Boundary_x": Boundary_x,
        "Boundary_y": Boundary_y,
        "TICK": TICK,
        "Iterations": Iterations,

        # Model parameters
        "N_shepherd": N_shepherd,
        "N_sheep": N_sheep,
        "L3": L3,
        "Repetition": rep,
        "MODE": MODE,

        # Results
        "Final_tick": Final_tick,
        "Success": success,
        "Experiment_type": "Morphology"
    }

    return results


start = timer()
if __name__ == '__main__':
    successes = 0
    
    start = timer()
    results = Parallel(n_jobs=THREADS)(delayed(run_morph if MORPHOLOGY else run_target)(seed) for seed in seeds)
    end = timer()
    print(f"Elapsed time: ", timedelta(seconds=end-start))
    
    # Saves stats if we're not making figures.
    if not DRAW:
        # Creates results folder if it does not exist
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        res_dir = f"{cur_dir}/results/morphology_attraction_naïve/{MODE}"
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

        # Create a file with a text list of results
        with open(f"{res_dir}/{(cur_time := datetime.datetime.now())}|{N_sheep}_sheep|{N_shepherd}_shepherds.txt", "w") as f:
            json.dump(results, f)

        # Prints out result summary.
        successes = sum([result["Success"] for result in results])
        print(f"Success rate: {successes/reps}")
        
        # Retrieves all final ticks.
        final_ticks = [result["Final_tick"] for result in results]
        print(f"Average final tick: {np.mean(final_ticks)}")
        print(f"Standard deviation: {np.std(final_ticks)}")
        print(f"Minimum final tick: {np.min(final_ticks)}")
        print(f"Maximum final tick: {np.max(final_ticks)}")

        # Creates a histogram of final ticks.
        plt.figure()
        plt.title(f"Final Tick Distribution: Mode {MODE}, {N_sheep} Sheep, {N_shepherd} Shepherds")
        plt.xlabel("Final tick")
        plt.ylabel("Number of samples")
        plt.hist(final_ticks, bins=20)
        plt.savefig(f"{res_dir}/{cur_time}|{N_sheep}_sheep|{N_shepherd}_shepherds_hist.png")

    #However, depending on the specific formulation of the shepherding task and model parameters,
    # we also observed scenarios with an optimal number of shepherds where the guiding time becomes minimal. This appears to be related to possible obstruction of the shepherds by themselves.
