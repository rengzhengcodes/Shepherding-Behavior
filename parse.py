# Creates results folder if it does not exist
import json
import os
import numpy as np

reps = 1000
with open(f'{os.getcwd()}/results/morphology_attraction_naïve/2/2024-07-27 00:27:36.948050|300_sheep|3_shepherds.txt', "r") as f:
    results = json.load(f)

# Prints out result summary.
successes = sum([result["Success"] for result in results])
print(f"Success rate: {successes}/{reps}")

# Retrieves all final ticks.
final_ticks = [result["Final_tick"] for result in results]
print(f"Average final tick: {np.mean(final_ticks)}")
print(f'Median final tick: {np.median(final_ticks)}')
print(f"Standard deviation: {np.std(final_ticks)}")
print(f"Minimum final tick: {np.min(final_ticks)}")
print(f"Maximum final tick: {np.max(final_ticks)}")

# Gets the rep number of the 0th, 25th, 50th, 75th, and 100th percentile of final ticks.
percentiles = [25, 50, 75]
reps_sorted = np.argsort(final_ticks)
isolated_reps = [reps_sorted[int(len(reps_sorted) * percentile / 100)] for percentile in percentiles]
isolated_reps += [reps_sorted[0], reps_sorted[-1]]
print(isolated_reps)
print([final_ticks[rep] for rep in isolated_reps])
