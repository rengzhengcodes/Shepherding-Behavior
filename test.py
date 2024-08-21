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

from basic import MODE, MORPHOLOGY, TARGET_X, TARGET_Y, TARGET_SIZE

THREADS = -1
DRAW = True

N_SHEEP = 300
N_SHEPHERD = 6
SPACE_X = 150
SPACE_Y = 150

BOUNDARY_X = TARGET_X + TARGET_SIZE + 300
BOUNDARY_Y = TARGET_Y + TARGET_SIZE + 300


TICK = 1000
ITERATIONS = 200000

L3 = np.sqrt(N_SHEEP / N_SHEPHERD) * 5  # average flock radius per shepherd
REPS = 1000
seeds = range(REPS)

NUM_NEAREST_NEIGHBOR = 5

# Defines where to save the results.
cur_dir = os.path.dirname(os.path.realpath(__file__))
res_dir = f"{cur_dir}/results/morphology_attraction_naïve/{MODE}"


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


def run_mode(rep: int, evolver: callable, terminator: callable, summarizer: callable):
    """
    Generic function that can run some herding model given an evolver, terminator
    and summarizer function.
    Args:
        @param rep: The repetition number, used as a seed.
        @param evolver: The function that evolves the agents and shepherds.
        @param terminator: The function that determines if the simulation is successful.
        @param summarizer: The function that summarizes the results.
    Returns:
        A dictionary with the results of the simulation, produced by the summarizer.
    """
    print("Starting repetition", rep)
    seed_run(rep)
    agents = initiate(N_SHEEP, N_SHEPHERD, SPACE_X, SPACE_Y, TARGET_SIZE)
    shepherds = initiate_shepherds(0, N_SHEPHERD, L3)
    # self-organized flocking
    for tick in range(TICK):
        # Defines the target as the global center of mass.
        agents, shepherds, max_agents_indices = evolver(agents, shepherds)

    # prepare the shepherds and record data
    shepherds = initiate_shepherds(N_SHEPHERD, N_SHEEP, L3)
    # Only record data per tick if we're drawing.
    if DRAW:
        data_agents: np.ndarray = np.zeros(
            (agents.shape[0], agents.shape[1], ITERATIONS), float
        )
        data_shepherds: np.ndarray = np.zeros(
            (shepherds.shape[0], shepherds.shape[1], ITERATIONS), float
        )
        data_max_agents_indices: np.ndarray = np.zeros((N_SHEPHERD, ITERATIONS), int)
    final_tick: int = ITERATIONS

    # continue the sheep data with shepherds
    success: bool = False  # whether the simulation was successful
    for tick in range(ITERATIONS):
        agents, shepherds, max_agents_indices = evolver(agents, shepherds)
        # save data
        if DRAW:
            data_agents[:, :, tick] = agents
            data_shepherds[:, :, tick] = shepherds
            data_max_agents_indices[:, tick] = max_agents_indices  # only two dimension
        # stop program if all the sheep are within L2 of the center of mass.
        if terminator(agents, shepherds):  # finish
            final_tick = tick
            success = True
            break

    # Summarizes the results.
    results = summarizer(agents, shepherds, final_tick, success)

    # Draws the results.
    if DRAW:
        folder_path = cur_dir
        draw_dynamic(
            final_tick,
            data_agents,
            data_shepherds,
            results["BOUNDARY_X"],
            results["BOUNDARY_Y"],
            results["TARGET_X"],
            results["TARGET_Y"],
            results["TARGET_SIZE"],
            results["L3"],
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

    return results


def run_target(rep: int):
    """
    Runs the point-to-point herding simulation.
    Args:
        @param rep: The repetition number, used as a seed.
    """

    def evolver(agents, shepherds, *args, **kwargs):
        del args, kwargs
        return evolve(agents, shepherds, TARGET_X, TARGET_Y, TARGET_SIZE)

    def terminator(agents, shepherds, *args, **kwargs):
        del shepherds, args, kwargs
        return np.all(agents[:, 21])

    def summarizer(agents, shepherds, final_tick, success, *args, **kwargs):
        del agents, shepherds, args, kwargs
        return {
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
            "Success": success,
            "Experiment_type": "target",
        }

    return run_mode(rep, evolver, terminator, summarizer)


def run_morph(rep):
    """
    Runs the morphology herding simulation.
    Args:
        @param rep: The repetition number, used as a seed.
    """
    # L2 is the distance from the center of mass that all sheep must be within.
    # Used as a termination condition for morphology herding as it is the point
    # where it is deemed okay to stop herding.
    L2: float = 10 * (np.sqrt(N_SHEEP)) * 2 / 3  # pylint: disable=invalid-name

    def evolver(agents, shepherds, *args, **kwargs):
        del args, kwargs
        center: tuple[float, float] = (np.mean(agents[:, 0]), np.mean(agents[:, 1]))
        return evolve(agents, shepherds, *center, L2)

    def successor(agents, shepherds, *args, **kwargs):
        del shepherds, args, kwargs
        center: tuple[float, float] = (np.mean(agents[:, 0]), np.mean(agents[:, 1]))
        return np.all(
            np.sqrt((agents[:, 0] - center[0]) ** 2 + (agents[:, 1] - center[1]) ** 2)
            < L2
        )

    def summarizer(agents, shepherds, final_tick, success, *args, **kwargs):
        del agents, shepherds, args, kwargs
        center: tuple[float, float] = (np.mean(agents[:, 0]), np.mean(agents[:, 1]))
        return {
            # Static parameters, for reference.
            "SPACE_X": SPACE_X,
            "SPACE_Y": SPACE_Y,
            "TARGET_X": center[0],
            "TARGET_Y": center[1],
            "TARGET_SIZE": L2,
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
            "Experiment_type": "morphology",
        }

    return run_mode(rep, evolver, successor, summarizer)


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
