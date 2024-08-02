"""
Runs simulation parameters for the shepherding model being tested.
"""

import json
import os
import random
import shutil
import subprocess
import datetime
from datetime import timedelta
from timeit import default_timer as timer

from joblib import Parallel, delayed
import numba as nb
import numpy as np
import matplotlib.pyplot as plt
from basic.initiation import initiate, initiate_shepherds
from basic.interaction import evolve
from basic.draw import draw_dynamic

from basic import MODE, MORPHOLOGY

THREADS = -1
DRAW = False

N_SHEEP = 300
N_SHEPHERD = 6
SPACE_X = 150
SPACE_Y = 150

TARGET_X = 400
TARGET_Y = 400
TARGET_SIZE = 125  # radius

BOUNDARY_X = TARGET_X + TARGET_SIZE + 300
BOUNDARY_Y = TARGET_Y + TARGET_SIZE + 300


TICK = 1000
ITERATIONS = 200000

L3 = np.sqrt(N_SHEEP / N_SHEPHERD) * 5  # average flock radius per shepherd
REPS = 1000
seeds = range(REPS)

NUM_NEAREST_NEIGHBOR = 5


# Function seeds numpy rng in numba code.
@nb.jit(nopython=True)
def seed_run(seed: int):
    """
    Seeds the random number generators.
    Args:
        @param seed: The seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)


def run_target(rep):
    """
    Runs the point-to-point herding simulation.
    Args:
        @param rep: The repetition number, used as a seed.
    """
    print("Starting repetition", rep)
    seed_run(rep)
    agents = initiate(N_SHEEP, N_SHEPHERD, SPACE_X, SPACE_Y, TARGET_SIZE)
    shepherd = initiate_shepherds(0, N_SHEPHERD, L3)
    # self-organized flocking
    for tick in range(TICK):
        agents_update, shepherd_update, max_agents_indexes = evolve(
            agents, shepherd, TARGET_X, TARGET_Y, TARGET_SIZE, MODE=MODE
        )
        agents = agents_update
        shepherd = shepherd_update
    # prepare the shepherd and record data
    shepherd = initiate_shepherds(N_SHEPHERD, N_SHEEP, L3)
    data_agents = np.zeros((agents.shape[0], agents.shape[1], ITERATIONS), float)
    data_shepherds = np.zeros((shepherd.shape[0], shepherd.shape[1], ITERATIONS), float)
    max_agents_indexes = np.zeros((N_SHEPHERD, ITERATIONS), int)
    final_tick = ITERATIONS
    # continue the sheep data with shepherd
    for tick in range(ITERATIONS):
        # start evolve function
        agents_update, shepherd_update, max_agents_indexes = evolve(
            agents, shepherd, TARGET_X, TARGET_Y, TARGET_SIZE, MODE=MODE
        )
        # update data
        agents = agents_update
        shepherd = shepherd_update
        # save data
        data_agents[:, :, tick] = agents
        data_shepherds[:, :, tick] = shepherd
        max_agents_indexes[:, tick] = max_agents_indexes  # only two dimension
        # print(tick)
        # stop program if all the sheep are in the "staying" mode;
        if sum(agents[:, 21]) == N_SHEEP:  # finish
            final_tick = tick
            break

    # Output logging, print the final tick.
    results = {
        # Static parameters, for reference.
        "SPACE_X": SPACE_X,
        "SPACE_Y": SPACE_Y,
        "TARGET_X": TARGET_X,
        "TARGET_Y": TARGET_Y,
        "TARGET_SIZE": TARGET_SIZE,
        # Viewing parameters.
        "BOUNDARY_X": BOUNDARY_X,
        "BOUNDARY_Y": BOUNDARY_Y,
        "TICK": TICK,
        "ITERATIONS": ITERATIONS,
        # Model parameters
        "N_SHEPHERD": N_SHEPHERD,
        "N_SHEEP": N_SHEEP,
        "L3": L3,
        "Repetition": rep,
        "MODE": MODE,
        # Results
        "final_tick": final_tick,
        "Success": bool(np.all(agents[:, 21] == 1)),
        "Experiment_type": "target",
    }

    return results


def run_morph(rep):
    """
    Runs the morphology herding simulation.
    Args:
        @param rep: The repetition number, used as a seed.
    """
    print("Starting repetition", rep)
    seed_run(rep)
    agents = initiate(N_SHEEP, N_SHEPHERD, SPACE_X, SPACE_Y, TARGET_SIZE)
    shepherds = initiate_shepherds(0, N_SHEPHERD, L3)
    # self-organized flocking
    for tick in range(TICK):
        # Defines the target as the global center of mass.
        center_x: float = np.mean(agents[:, 0])
        center_y: float = np.mean(agents[:, 1])
        agents, shepherds, max_agents_indexes = evolve(
            agents, shepherds, center_x, center_y, TARGET_SIZE, MODE=MODE
        )
    # prepare the shepherds and record data
    shepherds = initiate_shepherds(N_SHEPHERD, N_SHEEP, L3)
    data_agents = np.zeros((agents.shape[0], agents.shape[1], ITERATIONS), float)
    data_shepherds = np.zeros(
        (shepherds.shape[0], shepherds.shape[1], ITERATIONS), float
    )
    max_agents_indexes = np.zeros((N_SHEPHERD, ITERATIONS), int)
    final_tick = ITERATIONS
    # continue the sheep data with shepherds
    for tick in range(ITERATIONS):
        # Defines the target as the global center of mass.
        center_x = np.mean(agents[:, 0])
        center_y = np.mean(agents[:, 1])
        # start evolve function
        #! @note L2 is defined in initiate_agent and copied here for brevity.
        agents, shepherds, max_agents_indexes = evolve(
            agents,
            shepherds,
            center_x,
            center_y,
            l2 := 10 * (np.sqrt(N_SHEEP)) * 2 / 3,
            MODE=MODE,
        )
        # save data
        data_agents[:, :, tick] = agents
        data_shepherds[:, :, tick] = shepherds
        max_agents_indexes[:, tick] = max_agents_indexes  # only two dimension
        # print(tick)
        # stop program if all the sheep are within L2 of the center of mass.
        if np.all(
            np.sqrt((agents[:, 0] - center_x) ** 2 + (agents[:, 1] - center_y) ** 2)
            < l2
        ):  # finish
            final_tick = tick
            success = True
            break

    # Draws the results.
    if DRAW:
        folder_path = f"results/morphology_attraction_naïve/{MODE}"
        draw_dynamic(
            final_tick,
            data_agents,
            data_shepherds,
            BOUNDARY_X,
            BOUNDARY_Y,
            center_x,
            center_y,
            TARGET_SIZE,
            L3,
            MODE=MODE,
            folder_path=f"{folder_path}/repetition_{rep}",
        )
        # Runs the ffmpeg command to create a video.
        # ffmpeg -framerate 10 -start_number 0 -i %d.png -c:v libx264 \
        #        -r 30 -pix_fmt yuv420p output.mp4
        subprocess.run(
            [
                "ffmpeg",
                "-framerate",
                "10",
                "-start_number",
                "0",
                "-i",
                f"{folder_path}/repetition_{rep}/%d.png",
                "-c:v",
                "libx264",
                "-r",
                "30",
                "-pix_fmt",
                "yuv420p",
                f"{folder_path}/MODE_{MODE}|Rep_{rep}|final_{final_tick}.mp4",
            ],
            check=True,
        )
        # Deletes all the images.
        shutil.rmtree(f"{folder_path}/repetition_{rep}")

    # Output logging, print the final tick.
    results = {
        # Static parameters, for reference.
        "SPACE_X": SPACE_X,
        "SPACE_Y": SPACE_Y,
        "center_x": center_x,
        "center_y": center_y,
        "TARGET_SIZE": TARGET_SIZE,
        # Viewing parameters.
        "BOUNDARY_X": BOUNDARY_X,
        "BOUNDARY_Y": BOUNDARY_Y,
        "TICK": TICK,
        "ITERATIONS": ITERATIONS,
        # Model parameters
        "N_SHEPHERD": N_SHEPHERD,
        "N_SHEEP": N_SHEEP,
        "L3": L3,
        "Repetition": rep,
        "MODE": MODE,
        # Results
        "final_tick": final_tick,
        "Success": success,
        "Experiment_type": "Morphology",
    }

    return results


if __name__ == "__main__":
    start = timer()
    sims: dict = Parallel(n_jobs=THREADS)(
        delayed(run_morph if MORPHOLOGY else run_target)(seed) for seed in seeds
    )
    end = timer()
    print(f"Elapsed time: {timedelta(seconds=end - start)}")

    # Saves stats if we're not making figures.
    if not DRAW:
        # Creates results folder if it does not exist
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        res_dir = f"{cur_dir}/results/morphology_attraction_naïve/{MODE}"
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

        # Create a file with a text list of results
        with open(
            f"{res_dir}/{(cur_time := datetime.datetime.now())}"
            + f"|{N_SHEEP}_sheep|{N_SHEPHERD}_shepherds.txt",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(sims, f)

        # Prints out result summary.
        successes = sum(result["Success"] for result in sims)
        print(f"Success rate: {successes/REPS}")

        # Retrieves all final ticks.
        final_ticks = [result["final_tick"] for result in sims]
        print(f"Average final tick: {np.mean(final_ticks)}")
        print(f"Standard deviation: {np.std(final_ticks)}")
        print(f"Minimum final tick: {np.min(final_ticks)}")
        print(f"maximum final tick: {np.max(final_ticks)}")

        # Creates a histogram of final ticks.
        plt.figure()
        plt.title(
            f"final Tick Distribution: Mode {MODE}, {N_SHEEP} Sheep, {N_SHEPHERD} Shepherds"
        )
        plt.xlabel("final tick")
        plt.ylabel("Number of samples")
        plt.hist(final_ticks, bins=20)
        plt.savefig(
            f"{res_dir}/{cur_time}|{N_SHEEP}_sheep|{N_SHEPHERD}_shepherds_hist.png"
        )

    # However, depending on the specific formulation of the shepherding task and
    # model parameters, we also observed scenarios with an optimal number of
    # shepherds where the guiding time becomes minimal. This appears to be related
    # to possible obstruction of the shepherds by themselves.
