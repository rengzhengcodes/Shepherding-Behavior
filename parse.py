# Creates results folder if it does not exist
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from basic import MODE

# Mode aliases.
ALIAS = {
    0: "Center of Mass",
    2: "Convex Hull",
    3: "Visible Convex Hull",
    4: "Local Visible Convex Hull"
}
# Goes through all modes.
mode_runs = {}
for MODE in [0, 2, 3, 4]:
    # Gets the folder of the latest results.
    folder = f"{os.getcwd()}/results/p2p/{MODE}"
    # Gets latest txt file in folder.
    filename = [file for file in os.listdir(folder) if file.endswith('.txt')][-1]
    with open(os.path.join(folder, filename), "r") as f:
        results = json.load(f)

    # Prints out result summary.
    successes = sum([result["Success"] for result in results])
    print(f"Success rate: {successes}/{len(results)}")
    assert len(results) == 1000, "Not enough results."

    # Retrieves all final ticks.
    final_ticks = [result["final_tick"] for result in results]
    print(f"Average final tick: {np.mean(final_ticks)}")
    print(f'Median final tick: {np.median(final_ticks)}')
    print(f"Standard deviation: {np.std(final_ticks)}")
    print(f"Minimum final tick: {np.min(final_ticks)}")
    print(f"Maximum final tick: {np.max(final_ticks)}")
    print(f"Variance: {np.var(final_ticks)}")

    # Gets the rep number of the 0th, 25th, 50th, 75th, and 100th percentile of final ticks.
    percentiles = [25, 50, 75]
    reps_sorted = np.argsort(final_ticks)
    isolated_reps = [reps_sorted[int(len(reps_sorted) * percentile / 100)] for percentile in percentiles]
    isolated_reps += [reps_sorted[0], reps_sorted[-1]]
    # Converts all reps to integers.
    isolated_reps = [int(rep) for rep in isolated_reps]
    print(isolated_reps)
    print([final_ticks[rep] for rep in isolated_reps])
    mode_runs[MODE] = np.array(final_ticks)

# Plots them all in a histogram.
bins = np.linspace(np.array(list(mode_runs.values())).min(), results[0]["ITERATIONS"], 100)
for i, (mode, final_ticks) in enumerate(mode_runs.items()):
    # Plots the mean and median.
    c = plt.cm.tab10(i)
    plt.hist(final_ticks, bins, alpha=0.5, label=f"{ALIAS[mode]}", color=c)
    plt.axvline(final_ticks.mean(), color='k', linestyle='dashed', linewidth=1, label=f"{ALIAS[mode]} Mean", c=c)
    plt.axvline(np.median(final_ticks), color='r', linestyle='dotted', linewidth=1, label=f"{ALIAS[mode]} Median", c=c)



plt.legend(loc='upper right')
plt.xlabel("Final Tick")
plt.ylabel("Frequency")
plt.title("Final Tick Distribution for all Modes")
plt.show()

# plots a boxplot
for i, (mode, final_ticks) in enumerate(mode_runs.items()):
    c = plt.cm.tab10(i)
    plt.boxplot(final_ticks, positions=[i], patch_artist=True, boxprops=dict(facecolor=c))
    plt.scatter([i] * len(final_ticks), final_ticks, alpha=0.5, c=c)
plt.xticks(list(mode_runs.keys()), [ALIAS[mode] for mode in mode_runs.keys()])
plt.xlabel("Mode")
plt.ylabel("Final Tick")
plt.title("Final Tick Distribution for all Modes")
plt.show()