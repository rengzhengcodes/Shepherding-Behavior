"""
The visualization functions for the simulation.
"""

import os
import numba as nb
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from matplotlib import patches
from joblib import Parallel, delayed

from basic import DRAW_THREADS, FENCE

if FENCE:
    from basic import FENCE_MIDDLE_ANGLE, GATE_ANGULAR_WIDTH


@nb.jit(nopython=True)
def calculate_mass_center(agents: np.ndarray):
    """
    Calculates the center of mass of a group of agents.

    Args:
        @param agents: The agents to calculate the CoM of.
    """
    sum_x = 0
    sum_y = 0
    n = 0
    for index in range(agents.shape[0]):
        # agent state: staying -> 1; moving -> 0;
        if agents[index][21] == 0:
            n = n + 1
            sum_x = sum_x + agents[index][0]
            sum_y = sum_y + agents[index][1]
    if n != 0:
        sum_x = sum_x / n
        sum_y = sum_y / n
    return sum_x, sum_y


def draw_network(swarm):
    # draw_network
    N = swarm.shape[0]
    for i in range(N):
        for j in range(N):
            if map[i, j] == 1:
                plt.plot(
                    [swarm[i][0], swarm[j][0]],
                    [swarm[i][1], swarm[j][1]],
                    linewidth=1,
                    color="g",
                    alpha=0.4,
                )  # '#e6e6fa'


# @nb.jit(nopython=True)
def draw_single(swarm, shepherds, boundary: tuple[int], target: tuple[int], mode: int):
    """
    Draws one frame of the simulation.
    Args:
        @param swarm: The flock agents at this point in time.
        @param shepherds: The shepherds
    """
    # draw sheep
    for agent in swarm:
        if agent[21] == 1:  # staying state radius=2.5
            circles = plt.Circle(
                (agent[0], agent[1]),
                radius=agent[7],
                facecolor="none",
                edgecolor="b",
                alpha=0.8,
                lw=0.5,
            )
        else:  # moving state radius=2.5
            # Determines edge color based on if the sheep is in the hull or not.
            facecolor = "g" if agent[22] != 0 else "none"
            circles = plt.Circle(
                (agent[0], agent[1]),
                radius=agent[7],
                facecolor=facecolor,
                edgecolor="g",
                alpha=0.8,
                lw=0.5,
            )
            # if index == 0:
            #     plt.text(swarm[index, 0] * 1.05, swarm[index, 1] * 1.05, "agent_0", fontsize = 10)
        plt.gca().add_patch(circles)
    plt.quiver(
        *swarm[:, :2],
        np.cos(swarm[:, 2]),
        np.sin(swarm[:, 2]),
        headwidth=3,
        headlength=4,
        headaxislength=3.5,
        minshaft=4,
        minlength=1,
        color="g",
        scale_units="inches",
        scale=10,
    )

    # draw shepherds
    # plt.plot(shepherds[:, 0], shepherds[:, 1],
    #          marker='o', color='r', markersize=Agent_size * 2, alpha=0.2)
    # plt.quiver(shepherds[:, 0], shepherds[:, 1],
    #            np.cos(shepherds[:, 2]), np.sin(shepherds[:, 2]), headwidth=3,
    #            headlength=3, headaxislength=3.5, minshaft=4, minlength=1,
    #            color='r', scale_units='inches', scale=10)
    # draw shepherds and its collect point
    for shepherd in shepherds:
        # shepherds to collect_x/drive_x collect_y/drive_y
        plt.plot(
            [shepherd[14], shepherd[0]],
            [shepherd[15], shepherd[1]],
            color="cyan",
        )
        shepherd_state = shepherd[13]
        if shepherd_state == 1:  # drive mode
            plt.plot(
                shepherd[0],
                shepherd[1],
                marker="o",
                color="r",
                markersize=shepherd[7],
                alpha=0.2,
            )
            plt.quiver(
                shepherd[0],
                shepherd[1],
                np.cos(shepherd[2]),
                np.sin(shepherd[2]),
                headwidth=3,
                headlength=3,
                headaxislength=3.5,
                minshaft=4,
                minlength=1,
                color="r",
                scale_units="inches",
                scale=10,
            )
        else:
            plt.plot(
                shepherd[0],
                shepherd[1],
                marker="o",
                color="b",
                markersize=agent[7],
                alpha=0.2,
            )
            plt.quiver(
                shepherd[0],
                shepherd[1],
                np.cos(shepherd[2]),
                np.sin(shepherd[2]),
                headwidth=3,
                headlength=3,
                headaxislength=3.5,
                minshaft=4,
                minlength=1,
                color="r",
                scale_units="inches",
                scale=10,
            )

    # draw center of mass
    center_of_mass_x, center_of_mass_y = calculate_mass_center(swarm)
    plt.plot(center_of_mass_x, center_of_mass_y, "r*", markersize=5)

    # Plots the agent being collected.
    collecting_shepherdss = shepherds[shepherds[:, 13] == 0]
    for agent in collecting_shepherdss:
        collecting_agent = swarm[int(agent[16])]
        plt.plot(
            (agent[0], collecting_agent[0]),
            (agent[1], collecting_agent[1]),
            color="y",
            linestyle=":",
            lw=2,
        )

    match mode:
        # draw center of convex hull.
        case 2:
            hull = swarm[swarm[:, 22] != 0]
            if np.any(hull):
                # Sorts hull by CCW order for plotting.
                hull = hull[np.argsort(hull[:, 22])]

                # Calculates and plots the center of the hull.
                center_of_hull_x, center_of_hull_y = np.mean(hull[:, 0]), np.mean(
                    hull[:, 1]
                )
                plt.plot(center_of_hull_x, center_of_hull_y, "k*", markersize=5)

                # Draws the convex hull.
                plt.fill(
                    hull[:, 0], hull[:, 1], color="g", linestyle=":", lw=2, fill=False
                )
        # Draw the direct line between shepherds and agent it can see.
        case 3:
            # Manually calculates entire hull.
            moving_swarm = swarm[swarm[:, 21] == 0]
            if moving_swarm.shape[0] > 2:
                hull = ConvexHull(moving_swarm[:, :2]).vertices
            else:
                hull = np.arange(moving_swarm.shape[0])
            plt.fill(
                moving_swarm[hull, 0],
                moving_swarm[hull, 1],
                color="g",
                linestyle=":",
                lw=2,
                fill=False,
            )

            for i, shepherd in enumerate(shepherds):
                relevant_swarm = swarm[
                    (swarm[:, 23].view("uint64") & (0b01 << i)) != 0b0
                ]
                plt.plot(
                    [
                        np.repeat(shepherd[0], relevant_swarm.shape[0]),
                        relevant_swarm[:, 0],
                    ],
                    [
                        np.repeat(shepherd[1], relevant_swarm.shape[0]),
                        relevant_swarm[:, 1],
                    ],
                    color="m",
                    lw=1,
                    alpha=0.25,
                )
                # draw center of visible sheep. If no visible sheep it assumes self as CoM.
                center_of_visible_sheep_x = np.mean(relevant_swarm[:, 0])
                center_of_visible_sheep_y = np.mean(relevant_swarm[:, 1])
                plt.plot(
                    center_of_visible_sheep_x,
                    center_of_visible_sheep_y,
                    "m*",
                    markersize=5,
                )
        case 4:
            # Goes through each flock and plots the hull.
            for i in range(1, int(np.max(swarm[:, 24])) + 1):
                flock = swarm[swarm[:, 24] == i]
                if flock.shape[0] > 2:
                    hull = ConvexHull(flock[:, :2]).vertices
                else:
                    hull = np.arange(flock.shape[0])
                plt.fill(
                    flock[hull, 0],
                    flock[hull, 1],
                    color="g",
                    linestyle=":",
                    lw=2,
                    fill=False,
                )

            for shepherd in shepherds:
                relevant_swarm = swarm[
                    (swarm[:, 23].view("uint64") & (0b01 << i)) != 0b0
                ]
                plt.plot(
                    [
                        np.repeat(shepherd[0], relevant_swarm.shape[0]),
                        relevant_swarm[:, 0],
                    ],
                    [
                        np.repeat(shepherd[1], relevant_swarm.shape[0]),
                        relevant_swarm[:, 1],
                    ],
                    color="m",
                    lw=1,
                    alpha=0.25,
                )
                # draw center of visible sheep. If no visible sheep it assumes self as CoM.
                center_of_visible_sheep_x = np.mean(relevant_swarm[:, 0])
                center_of_visible_sheep_y = np.mean(relevant_swarm[:, 1])
                plt.plot(
                    center_of_visible_sheep_x,
                    center_of_visible_sheep_y,
                    "m*",
                    markersize=5,
                )

    # draw target center
    plt.plot(*target[:2], "b*")
    target_circle = plt.Circle(
        target[:2], radius=target[-1], facecolor="none", edgecolor="b", alpha=0.5
    )
    plt.gca().add_patch(target_circle)
    plt.xlim(xmin=-boundary[0] // 4, xmax=boundary[0])
    plt.ylim(ymin=-boundary[1] // 4, ymax=boundary[1])
    # draw gate to the fence.
    if FENCE:
        # Calculate the angles of the fence.
        theta_1: float = FENCE_MIDDLE_ANGLE - GATE_ANGULAR_WIDTH / 2
        theta_2: float = FENCE_MIDDLE_ANGLE + GATE_ANGULAR_WIDTH / 2
        # Converts to degrees. Rotates becaue theta = 0 is down, instead of right.
        theta_1 = np.degrees(theta_1) - 90
        theta_2 = np.degrees(theta_2) - 90
        fence = patches.Arc(
            target[:2],
            2 * target[-1],
            2 * target[-1],
            theta1=theta_1,
            theta2=theta_2,
            color="r",
            lw=2,
        )
        plt.gca().add_patch(fence)
    # plt.axis('equal')
    # plt.axis('square')


def draw_dynamic(
    setup: tuple[int, int, int],
    data: tuple[np.ndarray, np.ndarray],
    space: tuple[int],
    target: tuple[int],
    folder_path: str = None,
):
    data_agents: np.ndarray
    data_shepherds: np.ndarray
    data_agents, data_shepherds = data
    iterations: int
    l3: int
    mode: int
    iterations, l3, mode = setup
    plt.figure(figsize=(8, 6), dpi=300)
    plt.ion()

    if folder_path is None:
        folder_path: str = f"{os.getcwd()}/images"

    # If the folder does not exist, create it.
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_list = os.listdir(folder_path)
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            file_ext = os.path.splitext(file_path)[1]
            if file_ext.lower() in [".png", ".mp4"]:
                os.remove(file_path)

    def savefig(index: int):
        plt.cla()
        draw_single(
            data_agents[:, :, index], data_shepherds[:, :, index], space, target, mode
        )

        plt.title(
            f"N_sheep = {data_agents[:, :, 0].shape[0]} | L3 = {l3} | Tick = {index}"
        )
        plt.savefig(f"{folder_path}/{int(index / 100)}.png")

    Parallel(n_jobs=DRAW_THREADS)(
        delayed(savefig)(index) for index in range(0, iterations, 100)
    )

    plt.ioff()


def plot_snapshot(
    experiment: tuple,
    boundary: tuple[int],
    target: tuple[int],
    filepath: str = None,
    title: str = None,
):
    final_tick, swarm, shepherds, repetition = experiment
    # create folder
    folder_path = os.getcwd() + "/snapshot"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # create figure
    plt.figure(figsize=(8, 6), dpi=300)
    # plot sheep
    for index in range(swarm.shape[0]):
        if swarm[index, 21] == 1:
            # staying state
            circles = plt.Circle(
                (swarm[index, 0], swarm[index, 1]),
                radius=2.5,
                facecolor="none",
                edgecolor="b",
                alpha=0.8,
            )
        else:
            # moving state
            circles = plt.Circle(
                (swarm[index, 0], swarm[index, 1]),
                radius=2.5,
                facecolor="none",
                edgecolor="g",
                alpha=0.8,
            )
        plt.gca().add_patch(circles)
    # add arrow
    plt.quiver(
        swarm[:, 0],
        swarm[:, 1],
        np.cos(swarm[:, 2]),
        np.sin(swarm[:, 2]),
        headwidth=3,
        headlength=4,
        headaxislength=3.5,
        minshaft=4,
        minlength=1,
        color="g",
        scale_units="inches",
        scale=10,
    )
    # plot shepherds
    plt.plot(
        shepherds[:, 0], shepherds[:, 1], marker="o", color="r", markersize=5, alpha=0.2
    )
    plt.quiver(
        shepherds[:, 0],
        shepherds[:, 1],
        np.cos(shepherds[:, 2]),
        np.sin(shepherds[:, 2]),
        headwidth=3,
        headlength=3,
        headaxislength=3.5,
        minshaft=4,
        minlength=1,
        color="r",
        scale_units="inches",
        scale=10,
    )
    # draw direct line between shepherds
    for shepherd in shepherds:
        plt.plot(
            [shepherd[14], shepherd[0]],
            [shepherd[15], shepherd[1]],
            color="cyan",
        )

    center_of_mass_x, center_of_mass_y = calculate_mass_center(swarm)

    plt.plot(center_of_mass_x, center_of_mass_y, "r*", markersize=5)
    plt.plot(*target[:2], "b*")
    target_circle = plt.Circle(
        target[:2],
        radius=target[-1],
        facecolor="none",
        edgecolor="b",
        alpha=0.5,
    )
    plt.gca().add_patch(target_circle)

    plt.xlim(xmin=-100, xmax=boundary[0])
    plt.ylim(ymin=-100, ymax=boundary[1])

    if title is None:
        title = (
            f"Ns = {swarm.shape[0]} | N = {shepherds.shape[0]} | "
            + f"tick = {final_tick} | R = {repetition}"
        )
    plt.title(title)

    if filepath is None:
        filepath = f"{folder_path}/N_sheep={swarm.shape[0]}_N_shepherds={shepherds.shape[0]}_repetition={repetition}.png"
    plt.savefig(filepath)


# ffmpeg -framerate 10 -start_number 0 -i %d.png -c:v libx264 -r 30 -pix_fmt yuv420p output.mp4
## ffplay output.mp4
