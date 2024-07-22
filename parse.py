# Creates results folder if it does not exist
import json
import numpy as np

reps = 1000
with open('/Users/rengz/Documents/UHumboldt/Shepherding-Behavior/results/4/2024-07-19 17:14:14.040857|300_sheep|3_shepherds.txt', "r") as f:
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