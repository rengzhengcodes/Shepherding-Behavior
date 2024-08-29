"""
Contains all the driving algorithms for the shepherds in the simulation, along
with their helper functions.
"""

import numba as nb
import numpy as np
from scipy.spatial import ConvexHull


@nb.jit(nopython=True)
def get_relative_distance_angle(vector_head, vector_end):
    """Gets the relative distance and angle, with 0 degrees being i_hat."""
    r = vector_head - vector_end
    r_length = np.linalg.norm(r)
    r_angle = np.arctan2(r[1], r[0])  # range[-pi, pi]
    return r_length, r_angle


@nb.jit(nopython=True)
def calculate_mass_center(agents):
    """Calculates the mass center of the moving agents."""
    sum_x = 0
    sum_y = 0
    n = 0  # number of agents in moving state;
    for index in range(agents.shape[0]):
        # agent state: staying -> 1; moving -> 0;
        if agents[index][21] == 0.0:
            n = n + 1
            sum_x = sum_x + agents[index][0]
            sum_y = sum_y + agents[index][1]
    if n != 0.0:
        sum_x = sum_x / n
        sum_y = sum_y / n
    return n, np.array([sum_x, sum_y])


@nb.jit(nopython=True)
def drive_the_herd(agents, shepherd_pos, target_pos):
    """
    Drives the herd towards the target using the center of mass model described
    in Yating's paper.
    """
    # get the center of only moving mass, not concluding the staying mass;
    num_agents_moving, center_of_mass = calculate_mass_center(agents)
    # calculate the distance, angle between the center of the mass and the
    # shepherd;
    _, angle_mass_target = get_relative_distance_angle(
        center_of_mass[0], center_of_mass[1], *target_pos
    )
    # update the safe drive distance to the center according to the CURRENT num of moving agents,
    # initial parameter of shepherd swarm[:,5];
    if num_agents_moving >= 10:
        l1_new = (2 / 3) * np.sqrt(num_agents_moving) * 10  # 7.5
    else:
        l1_new = 15
    # L1: drive point: from shepherd to mass center
    # angle_mass_target: from the target place to the mass
    drive_point = center_of_mass + np.array(
        [l1_new * np.cos(angle_mass_target), l1_new * np.sin(angle_mass_target)]
    )
    # the shepherd should be attracted by the drive point
    distance_drive_herd, angle_drive_herd = get_relative_distance_angle(
        drive_point, shepherd_pos
    )
    # print("distance_drive_herd", distance_drive_herd)
    # the drive force is linear to the distance between the shepherd and the
    # drive point;
    force = np.array(
        [
            distance_drive_herd
            * np.cos(
                angle_drive_herd
            ),  # angle_drive_herd: from shepherd to drive point;
            distance_drive_herd * np.sin(angle_drive_herd),
        ]  # !!! Attention: the vector (force_x, force_y) is not unit;
    )

    return drive_point, force


@nb.jit(nopython=True)
def drive_the_herd_using_convex_hull(agents, shepherd_pos, target_pos):
    """
    Drives the herd ina  method similar to Yating's paper, but using the center
    of the convex hull of the flock (estimated through the average of the vertices
    of the convex hull) instead of the center of mass.
    """
    # Gets the precalculated convex hull of the flock.
    hull = np.where(agents[:, 22] != 0)[0]

    # Gets center of mass estimate as average of the convex hull vertices.
    center_of_hull: np.ndarray = (np.mean(agents[hull, 0]), np.mean(agents[hull, 1]))

    # calculate the distance, angle between the center of the mass and the
    # shepherd;
    _, angle_mass_target = get_relative_distance_angle(*center_of_hull, *target_pos)

    # update the safe drive distance to the center according to the CURRENT num of moving agents,
    # initial parameter of shepherd swarm[:,5];
    num_agents_moving = agents[agents[:, 21] == 0].shape[0]
    if num_agents_moving >= 10:
        l1_new = (2 / 3) * np.sqrt(num_agents_moving) * 10  # 7.5
    else:
        l1_new = 15

    # L1: drive point: from shepherd to mass center
    # angle_mass_target: from the target place to the mass
    drive_point: np.ndarray = (
        center_of_hull[0] + l1_new * np.cos(angle_mass_target),
        center_of_hull[1] + l1_new * np.sin(angle_mass_target),
    )

    # the shepherd should be attracted by the drive point
    distance_drive_herd, angle_drive_herd = get_relative_distance_angle(
        drive_point, shepherd_pos
    )

    # the drive force is linear to the distance between the shepherd and the
    # drive point;
    force = np.array(
        [
            distance_drive_herd
            * np.cos(
                angle_drive_herd
            ),  # angle_drive_herd: from shepherd to drive point;
            distance_drive_herd * np.sin(angle_drive_herd),
        ]  # !!! Attention: the vector (force_x, force_y) is not unit;
    )

    return drive_point, force


@nb.jit(nopython=True)
def drive_the_herd_using_visible_convex_hull(
    agents, shepherd_pos: np.ndarray, shepherd_index, target_pos
):
    """
    Drives the herd using the average of the positions of the visible convex hull
    vertices of the flock to the shepherd. l1_new is decreased as the estimated
    center of mass is closer to the edge of the flock.
    """
    # Calculate the convex hull of the flock not staying.
    roaming_agents = agents[agents[:, 21] == 0]
    sheperd_and_sheep_coordinates = np.concatenate(
        (np.expand_dims(shepherd_pos, axis=1), roaming_agents[:, :2])
    )

    with nb.objmode(visible_hull="int64[:]"):
        if roaming_agents.shape[0] <= 2:
            visible_hull = np.where(agents[:, 21] == 0)[0]
        else:
            visible_hull = ConvexHull(
                sheperd_and_sheep_coordinates, qhull_options="QG0"
            )
            # Takes the visible simplices (edges)/
            visible_hull = visible_hull.simplices[visible_hull.good]
            # Accounts for the shepherd inserted into the sheep swarm skewing
            # indices.
            visible_hull -= 1
            visible_hull = np.unique(visible_hull).flatten()
            # If there is no visible hull, e.g., if the shepherd is inside, the shepherd
            # assumes the nearest sheep as the center of mass.
            if visible_hull.shape[0] == 0:
                visible_hull = np.array(
                    [
                        np.argmin(
                            np.sqrt(
                                (roaming_agents[:, 0] - shepherd_pos[0]) ** 2
                                + (roaming_agents[:, 1] - shepherd_pos[1]) ** 2
                            )
                        )
                    ]
                )

            # Converts visible_hull back to the original indices.
            visible_hull = np.where(agents[:, 21] == 0)[0][visible_hull]

    # Set hull and visibility status.
    agents[visible_hull, 22] = np.arange(1, visible_hull.shape[0] + 1)
    # Very suspicious little endian coding that should be rewritten using the following numpy trick:
    # https://stackoverflow.com/a/40249859
    # unstable if float dtype ever changes in the array.
    with nb.objmode():
        agents[visible_hull, 23] = (
            agents[visible_hull, 23].view("uint64") | (0b01 << shepherd_index)
        ).view("float64")

    # Gets center of mass estimate as average of the visible convex hull
    # vertices.
    center_of_hull: np.ndarray = np.array(
        [np.mean(agents[visible_hull, 0]), np.mean(agents[visible_hull, 1])]
    )

    # calculate the distance, angle between the center of the mass and the
    # shepherd;
    _, angle_mass_target = get_relative_distance_angle(center_of_hull, target_pos)

    # update the safe drive distance to the center according to the CURRENT num of moving agents,
    # initial parameter of shepherd swarm[:,5];
    num_agents_moving = agents[agents[:, 21] == 0].shape[0]
    if num_agents_moving >= 10:
        l1_new = (2 / 3) * np.sqrt(num_agents_moving) * 2  # 7.5
    else:
        l1_new = 15

    # L1: drive point: from shepherd to mass center
    # angle_mass_target: from the target place to the mass
    drive_point: np.ndarray = center_of_hull + np.array(
        [l1_new * np.cos(angle_mass_target), l1_new * np.sin(angle_mass_target)]
    )

    # the shepherd should be attracted by the drive point
    distance_drive_herd, angle_drive_herd = get_relative_distance_angle(
        drive_point, shepherd_pos
    )

    # the drive force is linear to the distance between the shepherd and the
    # drive point;
    force = np.array(
        [
            distance_drive_herd
            * np.cos(
                angle_drive_herd
            ),  # angle_drive_herd: from shepherd to drive point;
            distance_drive_herd * np.sin(angle_drive_herd),
        ]  # !!! Attention: the vector (force_x, force_y) is not unit;
    )

    return (
        drive_point,
        force,
        center_of_hull,
        visible_hull,
    )


@nb.jit(nopython=True)
def drive_the_herd_using_subflock_convex_hulls(
    agents, shepherd_pos: np.ndarray, shepherd_index, target_pos
):
    """
    Drives the herd using the visible convex hull method on the nearest subflock.
    """
    # Goes through every flock and calculates the visible hull agents.
    visible_hulls_section = np.zeros(0, dtype="int64")
    # Calculates the number of flocks.
    num_flocks = np.max(agents[:, 24])
    # Does compute of visible hulls for each flock and chooses the one to
    # attend to.
    closest_distance = np.inf
    for flock_index in range(1, num_flocks + 1):
        # Gets the flock.
        flock = agents[agents[:, 24] == flock_index]
        # Gets the visible convex hull.
        _, _, _, visible_flock_hull = drive_the_herd_using_visible_convex_hull(
            flock,
            shepherd_pos,
            shepherd_index,
            target_pos,
        )
        # Corrects the indices.
        visible_flock_hull = np.where(agents[:, 24] == flock_index)[0][
            visible_flock_hull
        ]
        # Calculates the minimal distance to a point in the hull.
        min_distance = np.min(
            np.sqrt(
                (agents[visible_flock_hull, 0] - shepherd_pos[0]) ** 2
                + (agents[visible_flock_hull, 1] - shepherd_pos[1]) ** 2
            )
        )
        # If it trumps the previous minimal distance, updates the visible hull.
        if min_distance < closest_distance:
            closest_distance = min_distance
            visible_hulls_section = visible_flock_hull

    # Set hull and visibility status.
    agents[visible_hulls_section, 22] = np.arange(1, visible_hulls_section.shape[0] + 1)

    # Very suspicious little endian coding that should be rewritten using the following numpy trick:
    # https://stackoverflow.com/a/40249859
    # unstable if float dtype ever changes in the array.
    with nb.objmode():
        agents[visible_hulls_section, 23] = (
            agents[visible_hulls_section, 23].view("uint64") | (0b01 << shepherd_index)
        ).view("float64")

    # Calculates the center of mass of the shepherd flock.
    center_of_hull: np.ndarray = np.array(
        np.mean(agents[visible_hulls_section, 0]),
        np.mean(agents[visible_hulls_section, 1]),
    )

    # calculate the distance, angle between the center of the mass and the
    # shepherd;
    _, angle_mass_target = get_relative_distance_angle(*center_of_hull, *target_pos)

    # update the safe drive distance to the center according to the CURRENT num of moving agents,
    # initial parameter of shepherd swarm[:,5];
    num_agents_moving = visible_hulls_section.shape[0]
    if num_agents_moving >= 10:
        l1_new = (2 / 3) * np.sqrt(num_agents_moving) * 2  # 7.5
    else:
        l1_new = 15

    # L1: drive point: from shepherd to mass center
    # angle_mass_target: from the target place to the mass
    drive_point = center_of_hull + np.array(
        [l1_new * np.cos(angle_mass_target), l1_new * np.sin(angle_mass_target)]
    )

    # the shepherd should be attracted by the drive point
    distance_drive_herd, angle_drive_herd = get_relative_distance_angle(
        drive_point, shepherd_pos
    )

    # the drive force is linear to the distance between the shepherd and the
    # drive point;
    force: np.ndarray = np.array(
        [
            distance_drive_herd
            * np.cos(
                angle_drive_herd
            ),  # angle_drive_herd: from shepherd to drive point;
            distance_drive_herd * np.sin(angle_drive_herd),
        ]  # !!! Attention: the vector (force_x, force_y) is not unit;
    )

    return (
        drive_point,
        force,
        center_of_hull,
        visible_hulls_section,
    )
