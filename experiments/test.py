"""
Runs simulation parameters for the shepherding model being tested.
"""

import json
import os
import random
import shutil
import datetime
from datetime import timedelta
from timeit import default_timer as timer

from joblib import Parallel, delayed
import numba as nb
import numpy as np
import matplotlib.pyplot as plt
from basic.herding.initiation import initiate, initiate_shepherds
from basic.herding.interaction import evolve
from basic.drawing.draw import draw_dynamic

from basic import (
    DRAW_INTERVAL,
    MODE,
    MORPHOLOGY,
    FENCE,
    TARGET_X,
    TARGET_Y,
    TARGET_SIZE,
    TARGET,
)

THREADS = -1
DRAW = False

N_SHEEP = 300
N_SHEPHERD = 3
SPACE_X = 150
SPACE_Y = 150

BOUNDARY_X = TARGET_X + TARGET_SIZE + 300
BOUNDARY_Y = TARGET_Y + TARGET_SIZE + 300


TICK = 1000
ITERATIONS = 200000

L3 = np.sqrt(N_SHEEP / N_SHEPHERD) * 5  # average flock radius per shepherd
REPS = 64
seeds = range(REPS)

NUM_NEAREST_NEIGHBOR = 5

# Defines where to save the results, anchored to the repo root (one level up
# from experiments/) so results/ stays where parse.py expects it.
cur_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
res_dir = f"{cur_dir}/results/fence/{MODE}"


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
    # Only record data if we're drawing, and only every DRAW_INTERVAL ticks
    # (plus the run's true final tick) rather than every tick.
    # Design: recording every tick at N_SHEEP=300 allocated a ~12 GB
    # (300, 25, 200000) float64 buffer per rep -- an OOM risk once joblib
    # fans multiple DRAW reps out across cores -- when draw_dynamic only
    # ever consumed every DRAW_INTERVAL-th tick anyway. Sampling at record
    # time instead of over-allocating and sub-sampling later shrinks this to
    # a few hundred frames (well under 200 MB; see n_slots below).
    if DRAW:
        n_slots = ITERATIONS // DRAW_INTERVAL + 2
        data_agents: np.ndarray = np.zeros(
            (agents.shape[0], agents.shape[1], n_slots), float
        )
        data_shepherds: np.ndarray = np.zeros(
            (shepherds.shape[0], shepherds.shape[1], n_slots), float
        )
        # Real simulation tick recorded into each filled frame slot. Frames
        # are not evenly spaced in tick-space when the final-state frame is
        # appended out of cadence (see the loop below), so draw_dynamic
        # needs the real tick per frame, not just a frame count.
        frame_ticks: np.ndarray = np.zeros(n_slots, int)
        frame: int = 0  # monotonic slot cursor, independent of `tick`
    final_tick: int = ITERATIONS

    # continue the sheep data with shepherds
    success: bool = False  # whether the simulation was successful
    for tick in range(ITERATIONS):
        agents, shepherds, max_agents_indices = evolver(agents, shepherds)
        # Evaluate termination once per tick; both the record-check and the
        # break below need the result, and terminator() is not free.
        done = terminator(agents, shepherds)
        # save data: every DRAW_INTERVAL-th tick, plus (per user decision)
        # the exact termination tick and the last tick of the run, so a
        # rendered video always ends on the true final state instead of
        # potentially stopping up to DRAW_INTERVAL - 1 ticks early.
        if DRAW and (tick % DRAW_INTERVAL == 0 or done or tick == ITERATIONS - 1):
            data_agents[:, :, frame] = agents
            data_shepherds[:, :, frame] = shepherds
            frame_ticks[frame] = tick
            frame += 1
        # stop program if all the sheep are within L2 of the center of mass.
        if done:  # finish
            final_tick = tick
            success = True
            break

    # Summarizes the results.
    results = summarizer(agents, shepherds, final_tick, success)

    # Draws the results.
    if DRAW:
        print(f"Drawing repetition {rep}")
        # res_dir previously existed in DRAW mode only as a side effect of
        # the (now-removed) per-rep PNG folder's makedirs; draw_dynamic no
        # longer creates any folder, so it must be created explicitly here
        # or ffmpeg's output-file open inside draw_dynamic would fail.
        os.makedirs(res_dir, exist_ok=True)
        # Naming is byte-identical to the previous implementation's contract.
        video_path = f"{res_dir}/MODE_{MODE}|Rep_{rep}|final_{final_tick}.mp4"
        draw_dynamic(
            (frame_ticks[:frame], results["L3"], MODE),
            (data_agents[:, :, :frame], data_shepherds[:, :, :frame]),
            (results["BOUNDARY_X"], results["BOUNDARY_Y"]),
            (results["TARGET_X"], results["TARGET_Y"], results["TARGET_SIZE"]),
            video_path,
        )

    return results


def run_target(rep: int):
    """
    Runs the point-to-point herding simulation.
    Args:
        @param rep: The repetition number, used as a seed.
    """

    def evolver(agents, shepherds, *args, **kwargs):
        del args, kwargs
        return evolve(agents, shepherds, *TARGET)

    def terminator(agents, shepherds, *args, **kwargs):
        del shepherds, args, kwargs
        return np.all(agents[:, 21] == 1)

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
            "FENCE": FENCE,
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
    # Fail fast: a missing ffmpeg binary would otherwise only surface once
    # draw_dynamic tries to Popen it, potentially after hours of simulation
    # work across every rep has already completed.
    if DRAW and shutil.which("ffmpeg") is None:
        raise SystemExit(
            "DRAW=True but ffmpeg was not found on PATH; aborting before "
            "running the simulation."
        )

    start = timer()
    sims: tuple[dict] = Parallel(n_jobs=THREADS)(
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
