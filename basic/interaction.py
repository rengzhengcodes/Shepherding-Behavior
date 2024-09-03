"""
All collective shepherding herding interactions are defined here.
"""

import numba as nb
import numpy as np
from scipy.spatial import ConvexHull
from basic.vision_functions import (
    drive_the_herd_using_vision,
    collect_the_herd_using_vision,
)
from . import MODE, MORPHOLOGY, TARGET, FENCE
from .herd.forces import (
    get_attraction_force,
    get_repulsion_force,
    get_shepherd_force,
    get_fence_force,
)
from .herd.driver import (
    get_relative_distance_angle,
    get_relative_distance_angle_vectorized,
    calculate_mass_center,
    drive_the_herd,
    drive_the_herd_using_convex_hull,
    drive_the_herd_using_visible_convex_hull,
    drive_the_herd_using_subflock_convex_hulls,
)

from . import FENCE_MIDDLE_ANGLE, GATE_ANGULAR_WIDTH

if not FENCE:
    FENCE_MIDDLE_ANGLE = 0
    GATE_ANGULAR_WIDTH = 2 * np.pi


@nb.jit(nopython=True)
def transform_angle(theta):  # [-pi, pi]
    """
    Limits the angle to the range [-pi, pi].

    @param theta: The angles to be transformed.

    @returns The transformed angles.
    """
    return np.atan2(np.sin(theta), np.cos(theta))


@nb.jit(nopython=True)
def reflect_angle(angle):  # [0, 2pi)
    """
    Reflects the angle.

    @param angle: The angle to be reflected.

    @returns The reflected angle.
    """
    return angle % (2 * np.pi)


@nb.jit(nopython=True)
def update_agents_state(
    agents: np.ndarray, target_pos: np.ndarray, target_size: float
) -> np.ndarray:
    """
    Updates the state of the agents if they are within the target.

    @param agents: The agents to update the state of.
    @param target_pos: The target coordinates.
    @param target_size: The size of the target.

    @return: The agents with updated states.
    """
    for agent_index in range(agents.shape[0]):
        agent_pos: np.ndarray = agents[agent_index][:2]
        distance, _ = get_relative_distance_angle(target_pos, agent_pos)
        if not MORPHOLOGY and (
            (distance < target_size)
            or (
                FENCE
                and (
                    FENCE_MIDDLE_ANGLE - GATE_ANGULAR_WIDTH / 2
                    <= np.arctan2(
                        agent_pos[0] - target_pos[0], agent_pos[1] - target_pos[1]
                    )
                    <= FENCE_MIDDLE_ANGLE + GATE_ANGULAR_WIDTH / 2
                )
                or agents[agent_index][21] == 1
            )
        ):
            # agent state: 0 -> moving; 1 -> staying;
            agents[agent_index][21] = 1.0
        else:
            agents[agent_index][21] = 0.0
    return agents


@nb.jit(nopython=True)
def update(agents, shepherd, target_x, target_y):
    """
    Updates the force, angular velocity, and velocity of the agents.

    @param agents: The agents to update.
    @param shepherd: The shepherds herding the agents.
    @param target_x: The x-coordinate of the target.
    @param target_y: The y-coordinate of the target.

    @return: The updated agents.
    """

    target: np.ndarray = np.array((target_x, target_y))

    # calculate agent-agent repulsion force
    num_avoid, f_avoid = get_repulsion_force(agents)
    # calculate agent-agent attraction force
    _, f_attraction = get_attraction_force(agents)
    # calculate agent-shepherd repulsion force
    _, f_shepherd_force = get_shepherd_force(agents, shepherd)
    # Determine the velocity and angular velocity of the agents.
    v0: np.ndarray = np.where(
        (agents[:, 21] == 1) & (num_avoid == 0), 0.5, agents[:, 6]
    )

    # Calculates the force, whether they are explicitly avoiding other shepherds
    # versus flocking behavior.
    force: np.ndarray = np.where(
        np.expand_dims(num_avoid != 0, axis=1),
        f_avoid * np.expand_dims(agents[:, 10], axis=1),
        f_attraction * np.expand_dims(agents[:, 11], axis=1)
        + f_shepherd_force * np.expand_dims(agents[:, 12], axis=1),
    )
    # Attraction to the target.
    force: np.ndarray = np.where(
        np.expand_dims((agents[:, 21] == 1) & (num_avoid == 0), axis=1),
        0.1 * (agents[:, :2] - target),
        force,
    )
    # Gets the force from the fences.
    if FENCE:
        f_fence, _ = get_fence_force(agents, shepherd, TARGET, np.array(TARGET[:2]))
        force += f_fence

    # Calculates v_dot and w_dot for each agent.
    v_dot: np.ndarray = (
        force * np.stack((np.cos(agents[:, 2]), np.sin(agents[:, 2])), axis=1)
    ).sum(axis=1)
    w_dot: np.ndarray = (
        force * np.stack((-np.sin(agents[:, 2]), np.cos(agents[:, 2])), axis=1)
    ).sum(axis=1)
    w_dot *= (1 / v0)  # inertia
    w_dot = np.clip(w_dot, -agents[:, 18], agents[:, 18])

    # Calculates the random noise.
    dr: np.ndarray = (
        np.random.normal(0, 1) * np.sqrt(2 * agents[:, 13]) / np.sqrt(agents[:, 14])
    )

    # Updates the agents.
    agents[:, 0] += (v0 + v_dot) * np.cos(agents[:, 2]) * agents[:, 14]
    agents[:, 1] += (v0 + v_dot) * np.sin(agents[:, 2]) * agents[:, 14]
    agents[:, 2] = transform_angle(agents[:, 2] + (w_dot + dr) * agents[:, 14])

    return agents


@nb.jit(nopython=True)
def get_furthest_agent(agents, shepherd_pos: np.ndarray, target_pos: np.ndarray):
    """
    Sets which agent is the furthest from the shepherd and collects it.
    """
    num_agents = agents.shape[0]
    angle_herd_agents = np.zeros(agents.shape[0])
    distance_herd_agents = np.zeros(agents.shape[0])
    dirt_angles_of_target_to_agent = np.zeros(agents.shape[0])

    _, angle_target_herd = get_relative_distance_angle(target_pos, shepherd_pos)
    for agent_index in range(num_agents):
        # the furthest agent should only in the moving state;
        if agents[agent_index][21] == 0:
            agent_pos: np.ndarray = agents[agent_index][:2]
            r_agent_herd, angle_agent_herd = get_relative_distance_angle(
                agent_pos, shepherd_pos
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
    agent_pos: np.ndarray, shepherd_pos: np.ndarray, target_pos: np.ndarray, l0: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Identifies the collect point and attraction force of a shepherd  to the collect
    point given the agent to be collected's position and shepherd's position.

    @param agent_pos: The position of the agent to be collected.
    @param shepherd_pos: The shepherd doing the collecting.
    @param target_pos: The target position.
    @param l0:

    @returns collect_point: The point at which to collec the agent.
    @returns force: The shepherd's attraction force towards the collect point.
    """
    # get the angle from agent to target first;
    _, angle_agent_target = get_relative_distance_angle(agent_pos, target_pos)
    # keep l0 distance from the collect agent;
    collect_point: np.ndarray = agent_pos + l0 * np.array(
        [np.cos(angle_agent_target), np.sin(angle_agent_target)]
    )
    # attracted by the collect point;
    distance_cp_herd, angle_cp_herd = get_relative_distance_angle(
        collect_point, shepherd_pos
    )
    # print("distance_cp_herd:", distance_cp_herd)
    # attraction force is linear with the distance between the herd and the
    # collect point;
    force = distance_cp_herd * np.array([np.cos(angle_cp_herd), np.sin(angle_cp_herd)])
    return collect_point, force


@nb.jit(nopython=True)
def identify_flocks(agents, flock_distance):
    """
    Identifies the individual agents into distinct flocks of all agents.

    @param agents: The agents to be sorted into flocks.
    @param flock_distance: The distance an agent has to be within to a member of
    a flock to also be considered part of that flock.

    @postcondition: The agents array contains information about which agent is
    in which flock.
    """
    # Calculates the flocks using full DFS.
    i = 1  # Flock number
    agents[agents[:, 21] == 1, 24] = -1.0  # Clears flock membership.
    agents[agents[:, 21] == 0, 24] = 0.0  # Clears flock membership.
    # Tracks agents remaining.
    remaining_agents = np.where(agents[:, 21] == 0)[0]

    while remaining_agents.shape[0] > 0:
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
        # Increments counter.
        i += 1


@nb.jit(nopython=True)
def keep_distance_from_other_shepherd(shepherd):
    angle_other_shepherd = np.zeros(shepherd.shape[0])
    distance_other_shepherd = np.zeros(shepherd.shape[0])
    if shepherd.shape[0] > 0:
        l3 = shepherd[0][19]  # L3 Equilibrium distance from other shepherd
    for shepherd_index in range(shepherd.shape[0]):
        us = shepherd[shepherd_index][:2]
        neighbor_num = 0
        r = np.zeros(2)
        for neighbor_index in range(shepherd.shape[0]):
            if shepherd_index != neighbor_index:
                them = shepherd[neighbor_index][:2]
                distance = np.linalg.norm(us - them)
                if distance <= l3:  # Distance_from_other_shepherd
                    neighbor_num += 1
                    r += (us - them)
        if neighbor_num != 0:
            r /= neighbor_num
            angle = np.arctan2(r[1], r[0])
            angle_other_shepherd[shepherd_index] = reflect_angle(
                angle
            )  # Angle of the repulsion vector
            distance_other_shepherd[shepherd_index] = np.linalg.norm(
                r
            )  # Distance of the repulsion vector
    return distance_other_shepherd, angle_other_shepherd


@nb.jit(nopython=True)
def herd_preprocessor(agents: np.ndarray) -> tuple[int, tuple[float, float]]:
    """
    Calculates the number of agents still moving and the default [estimated]
    center of mass for the flock.

    @param agents: The agents to be herded.

    @return: The number of agents still moving and the default [estimated] center of mass.
    """
    match MODE:
        case 0 | 1:
            # first get the position of the center of the mass
            num_agents_moving, center_of_mass = calculate_mass_center(agents)
            center_of_mass_x, center_of_mass_y = center_of_mass
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
            center_of_mass_x, center_of_mass_y = np.nan, np.nan
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
            center_of_mass_x, center_of_mass_y = np.nan, np.nan
            # Sets the number of moving agents.
            num_agents_moving = np.count_nonzero(agents[:, 21] == 0)
            # Resets the flock membership.
            identify_flocks(agents, max(agents[0][3], agents[0][5]))
        case _:
            raise NotImplementedError(
                "Mode {MODE} does not have pre-processing implemented."
            )

    return num_agents_moving, np.array([center_of_mass_x, center_of_mass_y])


@nb.jit(nopython=True)
def herd_trigger_collect(
    agents: np.ndarray,
    shepherd_pos: np.ndarray,
    target: np.ndarray,
    subset: np.ndarray = None,
):
    """
    Checks the furthest agent and triggers the collect mode if necessary.

    @param agents: The agents to be herded.
    @param shepherd_pos: The position of the shepherd.
    @param target: The target location.
    @param subset: The subset of agents to consider.

    @return: The furthest agent index, the distance to the furthest agent, and
    the angle from the target to the furthest agent.
    """
    if subset is None:
        max_agent_index, _, max_angle_target_to_agent = get_furthest_agent(
            agents, shepherd_pos, target
        )
    else:
        max_agent_index, _, max_angle_target_to_agent = get_furthest_agent(
            agents[subset], shepherd_pos, target
        )
        # Converts max agent index in visible hull to the original index.
        max_agent_index = subset[max_agent_index]

    return max_agent_index, max_angle_target_to_agent


@nb.jit(nopython=True)
def herd(
    agents, shepherd, target: np.ndarray[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Herds the agents using the shepherds by some specified mode.

    @param agents: The agents to be herded.
    @param shepherd: The shepherds herding the agents.
    @param target: The target location.

    @returns shepherd: The shepherds after herding.
    @returns max_indexes: The agents being collected.
    """
    # record the furthest agent index
    max_agents_indexes = np.zeros(shepherd.shape[0])
    if shepherd.shape[0] > 0:
        l0 = shepherd[0][3]
        v0 = shepherd[0][6]  # 4
        alpha = shepherd[0][7]  # acceleration rate
        beta = shepherd[0][8]  # turning rate
        dr = shepherd[0][9]
        tick_time = shepherd[0][10]
        # HALF FOV threshold for collect mode;
        angle_threshold_collection = shepherd[0][17]
    else:
        l0 = 15
        v0 = 1
        alpha = 1
        beta = 0.1
        dr = 0.1
        tick_time = 0.01
        angle_threshold_collection = np.pi / 2

    # Determines the number of agents still moving and the [estimated] CoM.
    num_agents_moving, center_of_mass = herd_preprocessor(agents)
    assert center_of_mass is not None, "Center of mass is None."

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
        _, f_fence_shepherd = get_fence_force(
            agents, shepherd, TARGET, np.array(TARGET[:2])
        )

    # Calculates the repulsion force for each shepherd on each other.
    f_other_shepherd = np.expand_dims(distance_other_shepherd, axis=1) * np.stack(
        (np.cos(angle_other_shepherd), np.sin(angle_other_shepherd)), axis=1
    )

    for shepherd_index in range(shepherd.shape[0]):
        shepherd_pos = shepherd[shepherd_index][:2]
        shepherd_angle = shepherd[shepherd_index][2]

        # drive_mode: attract by the mass center and the target, repulsion from
        # other shepherd;
        if shepherd[shepherd_index][13] == 1.0:
            current_drive_agent_id = int(shepherd[shepherd_index][20])
            subset: np.ndarray = (
                None  # Subset of agents to consider for CoM estimation.
            )
            match MODE:
                case 0:
                    # find the drive point and calculate the force attraction
                    # from the drive point; drive_point_x,
                    drive_point, drive_force = drive_the_herd(
                        agents, shepherd_pos, target
                    )
                case 1:
                    # using vision
                    (
                        drive_point,
                        drive_force,
                        drive_agent_id,
                    ) = drive_the_herd_using_vision(
                        agents, shepherd_pos[0], shepherd_pos[1], *target
                    )
                    shepherd[shepherd_index][20] = drive_agent_id
                case 2:
                    # using convex hull
                    (drive_point, drive_force) = drive_the_herd_using_convex_hull(
                        agents, shepherd_pos, target
                    )
                case 3:
                    # using visible convex hull
                    (
                        drive_point,
                        drive_force,
                        center_of_mass,
                        subset,
                    ) = drive_the_herd_using_visible_convex_hull(
                        agents, shepherd_pos, shepherd_index, target
                    )
                case 4:
                    # using subflock convex hulls
                    (
                        drive_point,
                        drive_force,
                        center_of_mass,
                        subset,
                    ) = drive_the_herd_using_subflock_convex_hulls(
                        agents, shepherd_pos, shepherd_index, target
                    )
                case _:
                    raise NotImplementedError(
                        "Mode {MODE} does not have drive mode implemented."
                    )

            f_net = drive_force + f_other_shepherd[shepherd_index]
            if FENCE:
                f_net += f_fence_shepherd[shepherd_index]

            shepherd[shepherd_index][14:16] = drive_point

            # check the current furthest agent which triggers the switch of collect mode;
            # get the info of the furthest agent;
            max_agent_index, max_angle_target_to_agent = herd_trigger_collect(
                agents, shepherd_pos, target, subset
            )

            match MODE:
                case 1:
                    # max_angle_target_to_agent +: clockwise, -:
                    # anti-clockwise; threshold = np.pi/3
                    if (
                        np.absolute(max_angle_target_to_agent)
                        > angle_threshold_collection
                    ) and (agents[max_agent_index][21] == 0.0):
                        # collect_mode = true
                        shepherd[shepherd_index][13] = 0.0
                        # lock the ID of the furthest agent for the collect
                        # mode;
                        shepherd[shepherd_index][16] = int(max_agent_index)
                case 0 | 2 | 3 | 4:
                    agent_pos = agents[int(max_agent_index)][:2]
                    max_agents_indexes[shepherd_index] = int(max_agent_index)
                    distance_agent_mass, _ = get_relative_distance_angle(
                        agent_pos, center_of_mass
                    )
                    # switch to the collect mode if the furthest agent are far
                    # enough from the center, and moving outside the target
                    # circle.
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
                        f"Mode {MODE} does not have collect agent identification implemented."
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
            collect_agent_id: float = shepherd[shepherd_index][16]
            agent_pos: np.ndarray = agents[int(collect_agent_id)][:2]

            match MODE:
                case 1:
                    # attract by the furthest agent out of FOV;
                    # using target place: x/y;
                    collect_point, force = collect_furthest_agent(
                        agent_pos,
                        shepherd_pos,
                        target,
                        l0,
                    )
                case 3:
                    # using visible convex hull
                    _, _, center_of_mass, subset = (
                        drive_the_herd_using_visible_convex_hull(
                            agents, shepherd_pos, shepherd_index, target
                        )
                    )
                    collect_point, force = collect_furthest_agent(
                        agent_pos,
                        shepherd_pos,
                        target,
                        l0,
                    )
                case 4:
                    # using subflock convex hulls
                    (
                        _,
                        _,
                        center_of_mass,
                        subset,
                    ) = drive_the_herd_using_subflock_convex_hulls(
                        agents, shepherd_pos, shepherd_index, target
                    )
                    collect_point, force = collect_furthest_agent(
                        agent_pos,
                        shepherd_pos,
                        target,
                        l0,
                    )
                case 0 | 2:
                    # attract by the furthest agent;
                    # using center of mas: x/y;
                    collect_point, force = collect_furthest_agent(
                        agent_pos,
                        shepherd_pos,
                        center_of_mass,
                        l0,
                    )
                case _:
                    raise NotImplementedError(
                        "Mode {MODE} does not have collect mode implemented."
                    )

            # repulsion from other shepherd and attraction from the furthest
            # agent;
            f_net = force + f_other_shepherd[shepherd_index]

            if FENCE:
                f_net += f_fence_shepherd[shepherd_index]

            shepherd[shepherd_index][14:16] = collect_point

            distance_agent_mass, _ = get_relative_distance_angle(
                collect_point, center_of_mass
            )
            # !!! switch to the drive mode:
            match MODE:
                case 1:
                    # if the agent is closer enough to ANY AGENT in the GROUP or
                    # the agents are staying inside the circle; get the center of
                    # projection of the GROUP
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
                        f"Mode {MODE} does not have a way to exit collect mode."
                    )

        # calculate the linear speed and angular speed;
        v_dot = f_net[0] * np.cos(shepherd_angle) + f_net[1] * np.sin(
            shepherd_angle
        )  # heading_direction_acceleration
        w_dot = -f_net[0] * np.sin(shepherd_angle) + f_net[1] * np.cos(
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


@nb.jit(nopython=True)
def evolve(agents, shepherd, target_x, target_y, target_size):
    """
    Evolves the agents and shepherds for one time step.
    """
    target_pos = np.array([target_x, target_y], dtype=np.float64)
    # network_matrix = create_metric_network((agents, R, Fov))
    # agent-agent, agent-shepherd interaction;
    agents_update = update(agents, shepherd, target_pos[0], target_pos[1])
    ## shepherd switch between collect and drive mode;
    # Changes target to center of group if FENCE.
    if FENCE:
        target_pos += target_size * np.array(
            [
                np.cos(FENCE_MIDDLE_ANGLE),
                np.sin(FENCE_MIDDLE_ANGLE),
            ]
        )
    shepherd_update, max_agents_indexes = herd(agents, shepherd, target_pos)
    # update agents state
    agents_update = update_agents_state(agents_update, target_pos, target_size)

    return agents_update, shepherd_update, max_agents_indexes
