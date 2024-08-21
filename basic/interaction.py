"""
All collective shepherding herding interactions are defined here.
"""

import math

import numba as nb
import numpy as np
from scipy.spatial import ConvexHull, distance
from basic.vision_functions import (
    drive_the_herd_using_vision,
    collect_the_herd_using_vision,
)
from . import MODE, MORPHOLOGY, TARGET, FENCE
if FENCE:
    from . import FENCE_MIDDLE_ANGLE, GATE_ANGULAR_WIDTH


@nb.jit(nopython=True)
def transform_angle(theta):  # [-pi, pi]
    """
    Limits the angle to the range [-pi, pi].
    Args:
        @param theta: The angle to be transformed.
    Returns:
        The transformed angle.
    """
    # new_theta = (theta + np.pi) % (2. * np.pi)
    # new_theta -= np.pi
    while theta >= np.pi:
        theta = theta - 2 * np.pi
    while theta <= -np.pi:
        theta = theta + 2 * np.pi
    return theta


@nb.jit(nopython=True)
def reflect_angle(angle):  # [-2pi, 2pi]
    """
    Reflects the angle.
    Args:
        @param angle: The angle to be reflected.
    Returns:
        The reflected angle.
    """
    while angle >= 2 * np.pi:
        angle = angle - 2 * np.pi
    while angle <= 0:
        angle = angle + 2 * np.pi
    return angle


@nb.jit(nopython=True)
def get_attraction_force(
    agents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the attraction force between agents.
    Args:
        @param agents: The agents to calculate the attraction forces between.
    Returns:
        num_att: The number of agents attracted to the agent.
        f_attraction_x: The x-component of the attraction force.
        f_attraction_y: The y-component of the attraction force.
    """
    num_att = np.zeros(agents.shape[0])
    f_attraction_x = np.zeros(agents.shape[0])
    f_attraction_y = np.zeros(agents.shape[0])
    for agent_index in range(agents.shape[0]):
        neighbor_num = 0
        r_x = 0
        r_y = 0
        x_i = agents[agent_index][0]
        y_i = agents[agent_index][1]
        for neighbor_index in range(agents.shape[0]):
            if (
                agent_index != neighbor_index
            ):  # and (map_att[agent_index, neighbor_index] == 1)
                x_j = agents[neighbor_index][0]
                y_j = agents[neighbor_index][1]
                distance = np.sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2)
                if (distance >= agents[0][3]) and (distance <= agents[0][5]):
                    neighbor_num = neighbor_num + 1
                    r_x = r_x + (x_j - x_i) / distance  # unit vector
                    r_y = r_y + (y_j - y_i) / distance  # unit vector
        num_att[agent_index] = neighbor_num
        f_attraction_x[agent_index] = r_x
        f_attraction_y[agent_index] = r_y

    return num_att, f_attraction_x, f_attraction_y


@nb.jit(nopython=True)
def get_repulsion_force(
    agents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the repulsion force between agents.
    Args:
        @param agents: The agents to calculate the repulsion forces between.
    Returns:
        num_avoid: The number of agents repelled by the agent.
        f_avoid_x: The x-component of the repulsion force.
        f_avoid_y: The y-component of the repulsion force
    """
    num_avoid = np.zeros(agents.shape[0])
    f_avoid_x = np.zeros(agents.shape[0])
    f_avoid_y = np.zeros(agents.shape[0])
    for agent_index in range(agents.shape[0]):
        neighbor_num = 0
        r_x = 0
        r_y = 0
        x_i = agents[agent_index][0]
        y_i = agents[agent_index][1]
        for neighbor_index in range(agents.shape[0]):
            if agent_index != neighbor_index:
                x_j = agents[neighbor_index][0]
                y_j = agents[neighbor_index][1]
                distance = np.sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2)
                if distance <= agents[0][3]:  # R_repulsion
                    neighbor_num = neighbor_num + 1
                    r_x = r_x + (x_i - x_j) / distance  # unit vector
                    r_y = r_y + (y_i - y_j) / distance  # unit vector
        num_avoid[agent_index] = neighbor_num
        f_avoid_x[agent_index] = r_x
        f_avoid_y[agent_index] = r_y

    return num_avoid, f_avoid_x, f_avoid_y


@nb.jit(nopython=True)
def get_shepherd_force(agents, shepherd):
    """
    Calculates the repulsion force between agents and shepherds.
    Args:
        @param agents: The agents to calculate the repulsion force for.
        @param shepherd: The shepherds repulsing.
    Returns:
        num_shepherd_avoid: The number of shepherds repelling the agent.
        f_shepherd_force_x: The x-component of the repulsion force.
        f_shepherd_force_y: The y-component of the repulsion force.
    """
    num_shepherd_avoid = np.zeros(agents.shape[0])
    f_shepherd_force_x = np.zeros(agents.shape[0])
    f_shepherd_force_y = np.zeros(agents.shape[0])

    safe_distance = agents[0][17]  # safe_distance
    for agent_index in range(agents.shape[0]):
        agent_x = agents[agent_index][0]
        agent_y = agents[agent_index][1]
        r_x = 0
        r_y = 0
        num_shepherd = 0
        for shepherd_index in range(shepherd.shape[0]):
            shepherd_x = shepherd[shepherd_index][0]
            shepherd_y = shepherd[shepherd_index][1]
            distance = np.sqrt(
                (agent_x - shepherd_x) ** 2 + (agent_y - shepherd_y) ** 2
            )
            if distance <= safe_distance and distance != 0.0:
                num_shepherd = num_shepherd + 1
                r_x = (
                    r_x + (agent_x - shepherd_x) / distance
                )  # unit vector  ?? check distance == 0 ?
                r_y = r_y + (agent_y - shepherd_y) / distance  # unit vector
        num_shepherd_avoid[agent_index] = num_shepherd
        f_shepherd_force_x[agent_index] = r_x
        f_shepherd_force_y[agent_index] = r_y
    return num_shepherd_avoid, f_shepherd_force_x, f_shepherd_force_y


@nb.jit(nopython=True)
def get_fence_force(agents: np.ndarray, shepherds: np.ndarray, 
                    target: tuple[float, float, float], fence: np.ndarray[float, float]
                    ) -> tuple[np.ndarray[float, float], np.ndarray[float, float]]:
    """
    We model a fence as an impassible barrier around the pen that the agents and
    shepherds cannot pass. We calculate the repulsion force between agents and
    the fence, necessary for the agents to not pass through the fence.
    Args:
        @param agents: The agents to calculate the repulsion force for.
        @param shepherd: The shepherds repulsing.
        @param target: The target to calculate the repulsion force for.
        @param fence: The fence to calculate the repulsion force for.

    Returns:
        f_fence_force_x: The x-component of the repulsion force.
        f_fence_force_y: The y-component of the repulsion force.
    """
    t_x, t_y, t_r = target
    # Calculates the distance between the agents and the fence.
    agent_dist: np.ndarray = np.array([
        np.linalg.norm(agents[i, :2] - fence) for i in range(agents.shape[0])
    ])
    # Calculates the angle the agent is approaching the target, from the target's perspective.
    agent_angle: np.ndarray = np.arctan2(t_x - agents[:, 1], t_y - agents[:, 0])
    # Calculates the distance between the shepherds and the fence.
    shepherd_dist: np.ndarray = np.array([
        np.linalg.norm(shepherds[i, :2] - fence) for i in range(shepherds.shape[0])
    ])
    # Calculates the angle the shepherd is approaching the target, from the target's perspective.
    shepherd_angle: np.ndarray = np.arctan2(t_x - shepherds[:, 1], t_y - shepherds[:, 0])

    # Calculates the repulsion force between the agents and the fence.
    unaffected_agents: np.ndarray = agents[:, 21] == 1 or (
        FENCE_MIDDLE_ANGLE - GATE_ANGULAR_WIDTH / 2 <= agent_angle <= FENCE_MIDDLE_ANGLE + GATE_ANGULAR_WIDTH / 2
    ) # agents in the target or in the gate
    fence_range_agents: np.ndarray = agent_dist <= t_r + agents[:, 7]
    affected_agents: np.ndarray = np.logical_not(unaffected_agents) and fence_range_agents
    # Casts the affected agents to a 2D array.
    affected_agents: np.ndarray = np.expand_dims(affected_agents, axis=1)
    f_fence_on_sheep: np.ndarray = np.where(
        affected_agents, fence - agents[:, :2], np.zeros((agents.shape[0], 2))
    )

    # Calculates the repulsion force between the shepherds and the fence.
    unaffected_shepherds: np.ndarray = np.logical_not(
        FENCE_MIDDLE_ANGLE - GATE_ANGULAR_WIDTH / 2 <= shepherd_angle <= FENCE_MIDDLE_ANGLE + GATE_ANGULAR_WIDTH / 2
    )
    fence_range_shepherds: np.ndarray = shepherd_dist <= t_r + shepherds[:, 7]
    affected_shepherds: np.ndarray = np.logical_not(unaffected_shepherds) and fence_range_shepherds
    # Casts the affected shepherds to a 2D array.
    affected_shepherds: np.ndarray = np.expand_dims(affected_shepherds, axis=1)
    f_fence_on_shepherds = np.where(
        affected_shepherds, 
        fence - shepherds[:, :2], 
        np.zeros((shepherds.shape[0], 2))
    )

    return f_fence_on_sheep, f_fence_on_shepherds


@nb.jit(nopython=True)
def update_agents_state(
    agents: np.ndarray, target_x: float, target_y: float, target_size: float
) -> np.ndarray:
    """
    Updates the state of the agents if they are within the target.
    Args:
        @param agents: The agents to update the state of.
        @param target_x: The x-coordinate of the target.
        @param target_y: The y-coordinate of the target.
        @param target_size: The size of the target.
    Returns:
        agents: The agents with updated states.
    """
    for agent_index in range(agents.shape[0]):
        agent_x = agents[agent_index][0]
        agent_y = agents[agent_index][1]
        distance, _ = get_relative_distance_angle(target_x, target_y, agent_x, agent_y)
        if distance < target_size and not MORPHOLOGY:
            # agent state: 0 -> moving; 1 -> staying;
            agents[agent_index][21] = 1.0
        else:
            agents[agent_index][21] = 0.0
    return agents


# @nb.jit(nopython=True)
def update(agents, shepherd, target_x, target_y):
    # get variables
    v0 = agents[0][6]
    k_repulsion_agent = agents[0][10]  # k_repulsion_agent
    k_attraction_agent = agents[0][11]  # k_attraction_agent
    k_repulsion_shepherd = agents[0][12]  # k_repulsion_shepherd
    k_dr = agents[0][13]  # noise_strength
    tick_time = agents[0][14]  # tick_time
    max_turning_angle = agents[0][18]  # np.pi*2/3

    # calculate agent-agent repulsion force
    num_avoid, f_avoid_x, f_avoid_y = get_repulsion_force(agents)
    # calculate agent-agent attraction force
    _, f_attraction_x, f_attraction_y = get_attraction_force(agents)
    # calculate agent-shepherd repulsion force
    _, f_shepherd_force_x, f_shepherd_force_y = get_shepherd_force(agents, shepherd)

    for agent_index in range(agents.shape[0]):
        if num_avoid[agent_index] != 0:  # first priority!!!
            f_x = f_avoid_x[agent_index] * k_repulsion_agent
            f_y = f_avoid_y[agent_index] * k_repulsion_agent
        else:
            f_x = (
                f_attraction_x[agent_index] * k_attraction_agent
                + f_shepherd_force_x[agent_index] * k_repulsion_shepherd
            )
            f_y = (
                f_attraction_y[agent_index] * k_attraction_agent
                + f_shepherd_force_y[agent_index] * k_repulsion_shepherd
            )

        if (
            agents[agent_index][21] == 1 and num_avoid[agent_index] == 0
        ):  # staying state and no repulsion
            v0 = 0.5
            distance_agent_target, angle_agent_target = get_relative_distance_angle(
                target_x, target_y, agents[agent_index][0], agents[agent_index][1]
            )
            # if abs(target_size - distance_agent_target) < 20:  # near the
            # wall
            f_x = np.cos(angle_agent_target) * distance_agent_target * 0.1
            f_y = np.sin(angle_agent_target) * distance_agent_target * 0.1

        if FENCE:
            f_fence, _ = get_fence_force(agents, shepherd, TARGET, np.array(TARGET[:2]))
            f_x += f_fence[agent_index][0]
            f_y += f_fence[agent_index][1]

        v_dot = f_x * np.cos(agents[agent_index][2]) + f_y * np.sin(
            agents[agent_index][2]
        )
        w_dot = (
            -f_x * np.sin(agents[agent_index][2]) + f_y * np.cos(agents[agent_index][2])
        ) * (
            1 / v0
        )  # inertia

        w_dot = min(w_dot, max_turning_angle)
        w_dot = max(w_dot, -max_turning_angle)

        dr = np.random.normal(0, 1) * np.sqrt(2 * k_dr) / (tick_time**0.5)

        agents[agent_index][0] = (
            agents[agent_index][0]
            + (v0 + v_dot) * np.cos(agents[agent_index][2]) * tick_time
        )
        agents[agent_index][1] = (
            agents[agent_index][1]
            + (v0 + v_dot) * np.sin(agents[agent_index][2]) * tick_time
        )
        agents[agent_index][2] = transform_angle(
            agents[agent_index][2] + (w_dot + dr) * tick_time
        )
    return agents


@nb.jit(nopython=True)
def get_relative_distance_angle(
    vector_head_x, vector_head_y, vector_end_x, vector_end_y
):
    r_x = vector_head_x - vector_end_x
    r_y = vector_head_y - vector_end_y
    r_length = np.sqrt(r_x**2 + r_y**2)
    r_angle = np.arctan2(r_y, r_x)  # range[-pi, pi]
    return r_length, r_angle


@nb.jit(nopython=True)
def get_furthest_agent(agents, shepherd_x, shepherd_y, target_x, target_y):
    num_agents = agents.shape[0]
    angle_herd_agents = np.zeros(agents.shape[0])
    distance_herd_agents = np.zeros(agents.shape[0])
    dirt_angles_of_target_to_agent = np.zeros(agents.shape[0])

    _, angle_target_herd = get_relative_distance_angle(
        target_x, target_y, shepherd_x, shepherd_y
    )
    for agent_index in range(num_agents):
        # the furthest agent should only in the moving state;
        if agents[agent_index][21] == 0:
            agent_x = agents[agent_index][0]
            agent_y = agents[agent_index][1]
            r_agent_herd, angle_agent_herd = get_relative_distance_angle(
                agent_x, agent_y, shepherd_x, shepherd_y
            )
            angle_herd_agents[agent_index] = angle_agent_herd  # [-pi, pi]
            dirt_angle_from_target_to_herd = transform_angle(
                angle_target_herd - angle_agent_herd
            )  # [-pi, pi]
            # angle between two vector: [-np.pi, np.pi] negative: the agent its
            # on the left side of the target
            dirt_angles_of_target_to_agent[agent_index] = dirt_angle_from_target_to_herd
            distance_herd_agents[agent_index] = r_agent_herd

    # max_agent_index = int(np.argmax(np.absolute(angle_herd_agents)))  # +:
    # clockwise, -: anti-clockwise
    max_agent_index = int(
        np.argmax(np.absolute(dirt_angles_of_target_to_agent))
    )  # +: clockwise, -: anti-clockwise
    max_angle_target_to_agent = dirt_angles_of_target_to_agent[max_agent_index]
    return (
        max_agent_index,
        distance_herd_agents[max_agent_index],
        max_angle_target_to_agent,
    )


@nb.jit(nopython=True)
def collect_furthest_agent(
    agent_x, agent_y, shepherd_x, shepherd_y, target_x, target_y, l0
):
    # get the angle from agent to target first;
    _, angle_agent_target = get_relative_distance_angle(
        agent_x, agent_y, target_x, target_y
    )
    # keep l0 distance from the collect agent;
    collect_point_x = agent_x + l0 * np.cos(angle_agent_target)
    collect_point_y = agent_y + l0 * np.sin(angle_agent_target)
    # attracted by the collect point;
    distance_cp_herd, angle_cp_herd = get_relative_distance_angle(
        collect_point_x, collect_point_y, shepherd_x, shepherd_y
    )
    # print("distance_cp_herd:", distance_cp_herd)
    # attraction force is linear with the distance between the herd and the
    # collect point;
    force_x = distance_cp_herd * np.cos(angle_cp_herd)  #
    force_y = distance_cp_herd * np.sin(angle_cp_herd)  #
    return collect_point_x, collect_point_y, force_x, force_y


@nb.jit(nopython=True)
def calculate_mass_center(agents):
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
    return n, sum_x, sum_y


@nb.jit(nopython=True)
def drive_the_herd_using_convex_hull(
    agents, shepherd_x, shepherd_y, target_x, target_y
):
    # Gets the precalculated convex hull of the flock.
    hull = np.where(agents[:, 22] != 0)[0]

    # Gets center of mass estimate as average of the convex hull vertices.
    center_of_hull_x = np.mean(agents[hull, 0])
    center_of_hull_y = np.mean(agents[hull, 1])

    # calculate the distance, angle between the center of the mass and the
    # shepherd;
    _, angle_mass_target = get_relative_distance_angle(
        center_of_hull_x, center_of_hull_y, target_x, target_y
    )

    # update the safe drive distance to the center according to the CURRENT num of moving agents,
    # initial parameter of shepherd swarm[:,5];
    num_agents_moving = agents[agents[:, 21] == 0].shape[0]
    if num_agents_moving >= 10:
        l1_new = (2 / 3) * np.sqrt(num_agents_moving) * 10  # 7.5
    else:
        l1_new = 15

    # L1: drive point: from shepherd to mass center
    # angle_mass_target: from the target place to the mass
    drive_point_x = center_of_hull_x + l1_new * np.cos(angle_mass_target)
    drive_point_y = center_of_hull_y + l1_new * np.sin(angle_mass_target)

    # the shepherd should be attracted by the drive point
    distance_drive_herd, angle_drive_herd = get_relative_distance_angle(
        drive_point_x, drive_point_y, shepherd_x, shepherd_y
    )

    # the drive force is linear to the distance between the shepherd and the
    # drive point;
    force_x = distance_drive_herd * np.cos(
        angle_drive_herd
    )  # angle_drive_herd: from shepherd to drive point;
    force_y = distance_drive_herd * np.sin(angle_drive_herd)  #
    # !!! Attention: the vector (force_x, force_y) is not unit;

    return drive_point_x, drive_point_y, force_x, force_y


@nb.jit(nopython=True)
def drive_the_herd_using_visible_convex_hull(
    agents, shepherd_x, shepherd_y, shepherd_index, target_x, target_y
):
    # Calculate the convex hull of the flock not staying.
    roaming_agents = agents[agents[:, 21] == 0]
    sheperd_and_sheep_coordinates = np.concatenate(
        (np.array([[shepherd_x, shepherd_y]]), roaming_agents[:, :2])
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
                                (roaming_agents[:, 0] - shepherd_x) ** 2
                                + (roaming_agents[:, 1] - shepherd_y) ** 2
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
    center_of_hull_x = np.mean(agents[visible_hull, 0])
    center_of_hull_y = np.mean(agents[visible_hull, 1])

    # calculate the distance, angle between the center of the mass and the
    # shepherd;
    _, angle_mass_target = get_relative_distance_angle(
        center_of_hull_x, center_of_hull_y, target_x, target_y
    )

    # update the safe drive distance to the center according to the CURRENT num of moving agents,
    # initial parameter of shepherd swarm[:,5];
    num_agents_moving = agents[agents[:, 21] == 0].shape[0]
    if num_agents_moving >= 10:
        l1_new = (2 / 3) * np.sqrt(num_agents_moving) * 2  # 7.5
    else:
        l1_new = 15

    # L1: drive point: from shepherd to mass center
    # angle_mass_target: from the target place to the mass
    drive_point_x = center_of_hull_x + l1_new * np.cos(angle_mass_target)
    drive_point_y = center_of_hull_y + l1_new * np.sin(angle_mass_target)

    # the shepherd should be attracted by the drive point
    distance_drive_herd, angle_drive_herd = get_relative_distance_angle(
        drive_point_x, drive_point_y, shepherd_x, shepherd_y
    )

    # the drive force is linear to the distance between the shepherd and the
    # drive point;
    force_x = distance_drive_herd * np.cos(
        angle_drive_herd
    )  # angle_drive_herd: from shepherd to drive point;
    force_y = distance_drive_herd * np.sin(angle_drive_herd)  #
    # !!! Attention: the vector (force_x, force_y) is not unit;

    return (
        drive_point_x,
        drive_point_y,
        force_x,
        force_y,
        center_of_hull_x,
        center_of_hull_y,
        visible_hull,
    )


@nb.jit(nopython=True)
def identify_flocks(agents, flock_distance):
    # Calculates the flocks using full DFS.
    i = 0  # Flock number
    agents[agents[:, 21] == 1, 24] = -1.0  # Clears flock membership.
    agents[agents[:, 21] == 0, 24] = 0.0  # Clears flock membership.
    # Tracks agents remaining.
    remaining_agents = np.where(agents[:, 21] == 0)[0]

    while remaining_agents.shape[0] > 0:
        # Increments counter.
        i += 1
        # Gets the first unvisited agent.
        seed = remaining_agents[0]
        # Marks the seed as visited.
        agents[seed, 24] = i
        # Initializes the stack.
        stack = [seed]
        # DFS.
        while stack:
            current = stack.pop()
            for agent in remaining_agents:
                if agents[agent, 24] == 0 and (
                    np.sqrt(
                        (agents[current, 0] - agents[agent, 0]) ** 2
                        + (agents[current, 1] - agents[agent, 1]) ** 2
                    )
                    <= flock_distance
                ):
                    agents[agent, 24] = i
                    stack.append(agent)

        # Updates the remaining agents.
        remaining_agents = np.where(agents[:, 24] == 0)[0]


@nb.jit(nopython=True)
def drive_the_herd_using_subflock_convex_hulls(
    agents, shepherd_x, shepherd_y, shepherd_index, target_x, target_y
):
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
        _, _, _, _, _, _, visible_flock_hull = drive_the_herd_using_visible_convex_hull(
            flock,
            shepherd_x,
            shepherd_y,
            shepherd_index,
            target_x,
            target_y,
        )
        # Corrects the indices.
        visible_flock_hull = np.where(agents[:, 24] == flock_index)[0][
            visible_flock_hull
        ]
        # Calculates the minimal distance to a point in the hull.
        min_distance = np.min(
            np.sqrt(
                (agents[visible_flock_hull, 0] - shepherd_x) ** 2
                + (agents[visible_flock_hull, 1] - shepherd_y) ** 2
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
    center_of_hull_x = np.mean(agents[visible_hulls_section, 0])
    center_of_hull_y = np.mean(agents[visible_hulls_section, 1])

    # calculate the distance, angle between the center of the mass and the
    # shepherd;
    _, angle_mass_target = get_relative_distance_angle(
        center_of_hull_x, center_of_hull_y, target_x, target_y
    )

    # update the safe drive distance to the center according to the CURRENT num of moving agents,
    # initial parameter of shepherd swarm[:,5];
    num_agents_moving = visible_hulls_section.shape[0]
    if num_agents_moving >= 10:
        l1_new = (2 / 3) * np.sqrt(num_agents_moving) * 2  # 7.5
    else:
        l1_new = 15

    # L1: drive point: from shepherd to mass center
    # angle_mass_target: from the target place to the mass
    drive_point_x = center_of_hull_x + l1_new * np.cos(angle_mass_target)
    drive_point_y = center_of_hull_y + l1_new * np.sin(angle_mass_target)

    # the shepherd should be attracted by the drive point
    distance_drive_herd, angle_drive_herd = get_relative_distance_angle(
        drive_point_x, drive_point_y, shepherd_x, shepherd_y
    )

    # the drive force is linear to the distance between the shepherd and the
    # drive point;
    force_x = distance_drive_herd * np.cos(
        angle_drive_herd
    )  # angle_drive_herd: from shepherd to drive point;
    force_y = distance_drive_herd * np.sin(angle_drive_herd)  #
    # !!! Attention: the vector (force_x, force_y) is not unit;

    return (
        drive_point_x,
        drive_point_y,
        force_x,
        force_y,
        center_of_hull_x,
        center_of_hull_y,
        visible_hulls_section,
    )


@nb.jit(nopython=True)
def drive_the_herd(agents, shepherd_x, shepherd_y, target_x, target_y):
    # get the center of only moving mass, not concluding the staying mass;
    num_agents_moving, center_of_mass_x, center_of_mass_y = calculate_mass_center(
        agents
    )
    # calculate the distance, angle between the center of the mass and the
    # shepherd;
    _, angle_mass_target = get_relative_distance_angle(
        center_of_mass_x, center_of_mass_y, target_x, target_y
    )
    # update the safe drive distance to the center according to the CURRENT num of moving agents,
    # initial parameter of shepherd swarm[:,5];
    if num_agents_moving >= 10:
        l1_new = (2 / 3) * np.sqrt(num_agents_moving) * 10  # 7.5
    else:
        l1_new = 15
    # L1: drive point: from shepherd to mass center
    # angle_mass_target: from the target place to the mass
    drive_point_x = center_of_mass_x + l1_new * np.cos(angle_mass_target)
    drive_point_y = center_of_mass_y + l1_new * np.sin(angle_mass_target)
    # the shepherd should be attracted by the drive point
    distance_drive_herd, angle_drive_herd = get_relative_distance_angle(
        drive_point_x, drive_point_y, shepherd_x, shepherd_y
    )
    # print("distance_drive_herd", distance_drive_herd)
    # the drive force is linear to the distance between the shepherd and the
    # drive point;
    force_x = distance_drive_herd * np.cos(
        angle_drive_herd
    )  # angle_drive_herd: from shepherd to drive point;
    force_y = distance_drive_herd * np.sin(angle_drive_herd)  #
    # !!! Attention: the vector (force_x, force_y) is not unit;

    return drive_point_x, drive_point_y, force_x, force_y


@nb.jit(nopython=True)
def keep_distance_from_other_shepherd(shepherd):
    angle_other_shepherd = np.zeros(shepherd.shape[0])
    distance_other_shepherd = np.zeros(shepherd.shape[0])
    l3 = shepherd[0][19]  # L3 Equilibrium distance from other shepherd
    for shepherd_index in range(shepherd.shape[0]):
        x_i = shepherd[shepherd_index][0]
        y_i = shepherd[shepherd_index][1]
        neighbor_num = 0
        r_x = 0
        r_y = 0
        for neighbor_index in range(shepherd.shape[0]):
            if shepherd_index != neighbor_index:
                x_j = shepherd[neighbor_index][0]
                y_j = shepherd[neighbor_index][1]
                distance = np.sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2)
                if distance <= l3:  # Distance_from_other_shepherd
                    neighbor_num = neighbor_num + 1
                    r_x = r_x + (x_i - x_j)
                    r_y = r_y + (y_i - y_j)
        if neighbor_num != 0:
            r_x = r_x / neighbor_num
            r_y = r_y / neighbor_num
            angle = math.atan2(r_y, r_x)
            angle_other_shepherd[shepherd_index] = reflect_angle(
                angle
            )  # Angle of the repulsion vector
            distance_other_shepherd[shepherd_index] = np.sqrt(
                r_x**2 + r_y**2
            )  # Distance of the repulsion vector
    return distance_other_shepherd, angle_other_shepherd


@nb.jit(nopython=True)
def herd(
    agents, shepherd, target: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Herds the agents using the shepherds by some specified mode.
    Args:
        @param agents: The agents to be herded.
        @param shepherd: The shepherds herding the agents.
        @param target: The target location.
    Returns:
        shepherd: The shepherds after herding.
        max_indexes: The agents being collected.
    """
    # record the furthest agent index
    max_agents_indexes = np.zeros(shepherd.shape[0])
    l0 = shepherd[0][3]
    # k = shepherd[0][4]
    # l1 = shepherd[0][5]  # distance from the center of mass to the drive
    # point ###related to N
    v0 = shepherd[0][6]  # 4
    alpha = shepherd[0][7]  # acceleration rate
    beta = shepherd[0][8]  # turning rate
    dr = shepherd[0][9]
    tick_time = shepherd[0][10]
    # max_turning_rate = shepherd[0][11]
    # HALF FOV threshold for collect mode;
    Angle_Threshold_Collection = shepherd[0][17]
    # k_attraction_target = 0.01  # shepherd[0][18]  # k_attraction_target
    # 0.01

    match MODE:
        case 0 | 1:
            # first get the position of the center of the mass
            num_agents_moving, center_of_mass_x, center_of_mass_y = (
                calculate_mass_center(agents)
            )
        case 2:
            # Reset hull status.
            agents[:, 22] = 0
            # Finds the convex hull of the flock.
            with nb.objmode(hull="int64[:]"):
                if agents[agents[:, 21] == 0].shape[0] <= 2:
                    hull = np.where(agents[:, 21] == 0)[0]
                else:
                    hull = ConvexHull(agents[agents[:, 21] == 0, :2]).vertices
                    # Returns it back to the original indices.
                    hull = np.where(agents[:, 21] == 0)[0][hull]

            # Sets the hull items in their CCW order.
            agents[hull, 22] = np.arange(1, hull.shape[0] + 1)

            # Finds the center of the hull.
            num_agents_moving, center_of_hull_x, center_of_hull_y = (
                np.count_nonzero(agents[:, 21] == 0),
                np.mean(agents[hull, 0]),
                np.mean(agents[hull, 1]),
            )
            # Proxy for code concision later.
            center_of_mass_x, center_of_mass_y = center_of_hull_x, center_of_hull_y
        case 3:
            # Reset hull status.
            agents[:, 22] = 0
            # Resets who is visible to the shepherd, must be done outside of loop
            # or else each shepherd erases information for all other shepherds in this
            # call of herd.
            agents[:, 23] = 0.0
            # Sets center of mass values for error handling.
            center_of_mass_x, center_of_mass_y = None, None
            # Sets the number of moving agents.
            num_agents_moving = np.count_nonzero(agents[:, 21] == 0)
        case 4:
            # Reset hull status.
            agents[:, 22] = 0
            # Resets who is visible to the shepherd, must be done outside of loop
            # or else each shepherd erases information for all other shepherds in this
            # call of herd.
            agents[:, 23] = 0.0
            # Sets center of mass values for error handling.
            center_of_mass_x, center_of_mass_y = None, None
            # Sets the number of moving agents.
            num_agents_moving = np.count_nonzero(agents[:, 21] == 0)
            # Resets the flock membership.
            identify_flocks(agents, max(agents[0][3], agents[0][5]))
        case _:
            raise NotImplementedError(
                "Mode {MODE} does not have pre-processing implemented."
            )

    # d_furthest = shepherd[0][12]    # L2
    if num_agents_moving >= 50:
        d_furthest = 10 * (np.sqrt(num_agents_moving)) * 2 / 3  # 7.5 *
        # Although it could be an issue when the agent number = 1, d_furthest =
        # 5
    else:
        d_furthest = (
            50  # 35 ## related to l1, and was also used in drive the herd function
        )

    # avoid the other shepherd first!
    distance_other_shepherd, angle_other_shepherd = keep_distance_from_other_shepherd(
        shepherd
    )
    # Gets force from the fence.
    if FENCE:
        _, f_fence_shepherd = get_fence_force(agents, shepherd, TARGET, np.array(TARGET[:2]))

    for shepherd_index in range(shepherd.shape[0]):
        shepherd_pos = (shepherd[shepherd_index][0], shepherd[shepherd_index][1])
        shepherd_angle = shepherd[shepherd_index][2]

        # repulsion force from other shepherd
        f_x_other_shepherd = distance_other_shepherd[shepherd_index] * np.cos(
            angle_other_shepherd[shepherd_index]
        )
        f_y_other_shepherd = distance_other_shepherd[shepherd_index] * np.sin(
            angle_other_shepherd[shepherd_index]
        )

        # drive_mode: attract by the mass center and the target, repulsion from
        # other shepherd;
        if shepherd[shepherd_index][13] == 1.0:
            current_drive_agent_id = int(shepherd[shepherd_index][20])
            match MODE:
                case 0:
                    # find the drive point and calculate the force attraction
                    # from the drive point; drive_point_x,
                    drive_point_x, drive_point_y, drive_force_x, drive_force_y = (
                        drive_the_herd(agents, *shepherd_pos, *target)
                    )
                case 1:
                    # using vision
                    (
                        drive_point_x,
                        drive_point_y,
                        drive_force_x,
                        drive_force_y,
                        drive_agent_id,
                    ) = drive_the_herd_using_vision(agents, *shepherd_pos, *target)
                    shepherd[shepherd_index][20] = drive_agent_id
                case 2:
                    # using convex hull
                    (drive_point_x, drive_point_y, drive_force_x, drive_force_y) = (
                        drive_the_herd_using_convex_hull(agents, *shepherd_pos, *target)
                    )
                case 3:
                    # using visible convex hull
                    (
                        drive_point_x,
                        drive_point_y,
                        drive_force_x,
                        drive_force_y,
                        center_of_hull_x,
                        center_of_hull_y,
                        visible_hull,
                    ) = drive_the_herd_using_visible_convex_hull(
                        agents, *shepherd_pos, shepherd_index, *target
                    )
                    center_of_mass_x = center_of_hull_x
                    center_of_mass_y = center_of_hull_y
                case 4:
                    # using subflock convex hulls
                    (
                        drive_point_x,
                        drive_point_y,
                        drive_force_x,
                        drive_force_y,
                        center_of_hull_x,
                        center_of_hull_y,
                        visible_hulls_section,
                    ) = drive_the_herd_using_subflock_convex_hulls(
                        agents, *shepherd_pos, shepherd_index, *target
                    )
                    center_of_mass_x = center_of_hull_x
                    center_of_mass_y = center_of_hull_y
                case _:
                    raise NotImplementedError(
                        "Mode {MODE} does not have drive mode implemented."
                    )

            f_x = drive_force_x + f_x_other_shepherd
            f_y = drive_force_y + f_y_other_shepherd
            if FENCE:
                f_x += f_fence_shepherd[shepherd_index][0]
                f_y += f_fence_shepherd[shepherd_index][1]

            shepherd[shepherd_index][14] = drive_point_x
            shepherd[shepherd_index][15] = drive_point_y

            # check the current furthest agent which triggers the switch of collect mode;
            # get the info of the furthest agent;
            match MODE:
                case 0 | 1 | 2:
                    # Case 2 degenerates to this due to furthest agents needing
                    # to be an extreme point.
                    max_agent_index, _, max_angle_target_to_agent = get_furthest_agent(
                        agents, *shepherd_pos, *target
                    )

                case 3:
                    max_agent_index, _, max_angle_target_to_agent = get_furthest_agent(
                        agents[visible_hull], *shepherd_pos, *target
                    )
                    # Converts max agent index in visible hull to the original
                    # index.
                    max_agent_index = visible_hull[max_agent_index]
                case 4:
                    max_agent_index, _, max_angle_target_to_agent = get_furthest_agent(
                        agents[visible_hulls_section], *shepherd_pos, *target
                    )
                    # Converts max agent index in visible hull to the original
                    # index.
                    max_agent_index = visible_hulls_section[max_agent_index]
                case _:
                    raise NotImplementedError(
                        "Mode {MODE} does not have furthest agent identification implemented."
                    )

            match MODE:
                case 1:
                    # max_angle_target_to_agent +: clockwise, -:
                    # anti-clockwise; threshold = np.pi/3
                    if (
                        np.absolute(max_angle_target_to_agent)
                        > Angle_Threshold_Collection
                    ) and (agents[max_agent_index][21] == 0.0):
                        # collect_mode = true
                        shepherd[shepherd_index][13] = 0.0
                        # lock the ID of the furthest agent for the collect
                        # mode;
                        shepherd[shepherd_index][16] = int(max_agent_index)
                case 0 | 2 | 3 | 4:
                    agent_x = agents[int(max_agent_index)][0]
                    agent_y = agents[int(max_agent_index)][1]
                    max_agents_indexes[shepherd_index] = int(max_agent_index)
                    distance_agent_mass, _ = get_relative_distance_angle(
                        agent_x, agent_y, center_of_mass_x, center_of_mass_y
                    )
                    # switch to the collect mode if the furthest agent are far
                    # enough from the center, and moving outside the target
                    # circle/
                    if (distance_agent_mass > d_furthest) and (
                        agents[max_agent_index][21] == 0.0
                    ):
                        # collect_mode = true
                        shepherd[shepherd_index][13] = 0.0
                        # lock the ID of the furthest agent for the collect
                        # mode;
                        shepherd[shepherd_index][16] = int(max_agent_index)
                case _:
                    raise NotImplementedError(
                        "Mode {MODE} does not have collect agent identification implemented."
                    )

            # if the drive agent is staying, then switch to collect mode:  ???
            # to be checked;
            if agents[current_drive_agent_id][21]:
                # collect_mode = true
                shepherd[shepherd_index][13] = 0.0
                # lock the ID of the furthest agent for the collect mode;
                shepherd[shepherd_index][16] = int(max_agent_index)
        else:
            # collect mode: attract by the furthest agent and repulsion from other shepherd;
            # get the info of the furthest agent;
            collect_agent_id = shepherd[shepherd_index][16]
            agent_pos = (agents[int(collect_agent_id)][0], agents[int(collect_agent_id)][1])

            match MODE:
                case 1:
                    # attract by the furthest agent out of FOV;
                    # using target place: x/y;
                    collect_point_x, collect_point_y, force_x, force_y = (
                        collect_furthest_agent(*agent_pos, *shepherd_pos, *target, l0)
                    )
                case 3:
                    # using visible convex hull
                    _, _, _, _, center_of_hull_x, center_of_hull_y, visible_hull = (
                        drive_the_herd_using_visible_convex_hull(
                            agents, *shepherd_pos, shepherd_index, *target
                        )
                    )
                    collect_point_x, collect_point_y, force_x, force_y = (
                        collect_furthest_agent(*agent_pos, *shepherd_pos, *target, l0)
                    )
                    # Aliased for code concision.
                    center_of_mass_x, center_of_mass_y = (
                        center_of_hull_x,
                        center_of_hull_y,
                    )
                case 4:
                    # using subflock convex hulls
                    (
                        _,
                        _,
                        _,
                        _,
                        center_of_hull_x,
                        center_of_hull_y,
                        visible_hulls_section,
                    ) = drive_the_herd_using_subflock_convex_hulls(
                        agents, *shepherd_pos, shepherd_index, *target
                    )
                    collect_point_x, collect_point_y, force_x, force_y = (
                        collect_furthest_agent(*agent_pos, *shepherd_pos, *target, l0)
                    )
                    # Aliased for code concision.
                    center_of_mass_x, center_of_mass_y = (
                        center_of_hull_x,
                        center_of_hull_y,
                    )
                case 0 | 2:
                    # attract by the furthest agent;
                    # using center of mas: x/y;
                    collect_point_x, collect_point_y, force_x, force_y = (
                        collect_furthest_agent(
                            *agent_pos,
                            *shepherd_pos,
                            center_of_mass_x,
                            center_of_mass_y,
                            l0,
                        )
                    )
                case _:
                    raise NotImplementedError(
                        "Mode {MODE} does not have collect mode implemented."
                    )

            # repulsion from other shepherd and attraction from the furthest
            # agent;
            f_x = force_x + f_x_other_shepherd
            f_y = force_y + f_y_other_shepherd
            if FENCE:
                f_x += f_fence_shepherd[shepherd_index][0]
                f_y += f_fence_shepherd[shepherd_index][1]

            shepherd[shepherd_index][14] = collect_point_x  # collect_x
            shepherd[shepherd_index][15] = collect_point_y  # collect_y

            distance_agent_mass, _ = get_relative_distance_angle(
                collect_point_x, collect_point_y, center_of_mass_x, center_of_mass_y
            )
            # !!! switch to the drive mode:
            match MODE:
                case 1:
                    # if the agent is closer enough to ANY AGENT in the GROUP or the agents are staying inside the circe;
                    # get the center of projection of the GROUP
                    angle_difference_agent_mass = collect_the_herd_using_vision(
                        collect_agent_id, agents, *shepherd_pos
                    )
                    if (angle_difference_agent_mass <= np.pi / 3) or (
                        agents[int(shepherd[shepherd_index][16])][21] == 1.0
                    ):
                        shepherd[shepherd_index][13] = 1.0  # drive_mode_true
                case 0 | 2 | 3 | 4:
                    # if the agent is closer enough to the center or the agents
                    # are staying inside the circle;
                    if (
                        distance_agent_mass <= d_furthest
                        or agents[int(shepherd[shepherd_index][16])][21] == 1.0
                    ):
                        shepherd[shepherd_index][13] = 1.0  # drive_mode_true
                case _:
                    raise NotImplementedError(
                        "Mode {MODE} does not have a way to exit collect mode."
                    )

        # calculate the linear speed and angular speed;
        v_dot = f_x * np.cos(shepherd_angle) + f_y * np.sin(
            shepherd_angle
        )  # heading_direction_acceleration
        w_dot = -f_x * np.sin(shepherd_angle) + f_y * np.cos(
            shepherd_angle
        )  # angular_acceleration
        # alpha: acceleration rate; beta: turning rate;
        noise = (
            np.sqrt(2 * dr) / (tick_time**0.5) * np.random.normal(0, 1)
        )  # (mean, std_deviation) dr = 0.1
        shepherd[shepherd_index][0] = (
            shepherd_pos[0]
            + ((v0 + v_dot * alpha) * np.cos(shepherd_angle)) * tick_time
        )
        shepherd[shepherd_index][1] = (
            shepherd_pos[1]
            + ((v0 + v_dot * alpha) * np.sin(shepherd_angle)) * tick_time
        )
        shepherd[shepherd_index][2] = reflect_angle(
            shepherd_angle + (w_dot / v0 * beta + noise) * tick_time
        )  # [-2pi, 2pi]

    return shepherd, max_agents_indexes


def make_periodic_boundary(agents, space_x, space_y):
    # how to calculate periodic distance?
    num_agents = agents.shape[0]
    for agent_index in range(num_agents):
        agent_x = agents[agent_index][0]
        agent_y = agents[agent_index][1]
        if agent_x < -space_x / 2:
            agent_x += space_x
        elif agent_x >= space_x / 2:
            agent_x -= space_x
        if agent_y < -space_y / 2:
            agent_y += space_y
        elif agent_y >= space_y / 2:
            agent_y -= space_y
        agents[agent_index][0] = agent_x
        agents[agent_index][1] = agent_y
    return agents


# @nb.jit(nopython=True)
def evolve(agents, shepherd, target_x, target_y, target_size):
    target = (target_x, target_y)
    # network_matrix = create_metric_network((agents, R, Fov))
    # agent-agent, agent-shepherd interaction;
    agents_update = update(agents, shepherd, *target)
    # shepherd switch between collect and drive mode;
    shepherd_update, max_agents_indexes = herd(agents, shepherd, target)
    # update agents state
    agents_update = update_agents_state(agents_update, *target, target_size)

    return agents_update, shepherd_update, max_agents_indexes
