"""
Calculates the social forces for the herding problem.
"""

import numpy as np
import numba as nb
from .. import FENCE, FENCE_MIDDLE_ANGLE, GATE_ANGULAR_WIDTH, K_FENCE

if not FENCE:
    K_FENCE = 0
    FENCE_MIDDLE_ANGLE = 0
    GATE_ANGULAR_WIDTH = 2 * np.pi


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
        f_attraction: The attraction force.
    """
    num_att: np.ndarray = np.zeros(agents.shape[0])
    f_attraction: np.ndarray = np.zeros((agents.shape[0], 2))
    for agent_index in range(agents.shape[0]):
        neighbor_num = 0
        r: np.ndarray = np.zeros(2)
        agent_pos: np.ndarray = agents[agent_index][:2]
        for neighbor_index in range(agents.shape[0]):
            if (
                agent_index != neighbor_index
            ):  # and (map_att[agent_index, neighbor_index] == 1)
                neighbor_pos = agents[neighbor_index][:2]
                distance = np.linalg.norm(agent_pos - neighbor_pos)
                if agents[0][3] <= distance <= agents[0][5]:
                    neighbor_num = neighbor_num + 1
                    r += (neighbor_pos - agent_pos) / distance
        num_att[agent_index] = neighbor_num
        f_attraction[agent_index, :2] = r

    return num_att, f_attraction


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
        f_avoid: The repulsion force.
    """
    num_avoid = np.zeros(agents.shape[0])
    f_avoid = np.zeros((agents.shape[0], 2))
    for agent_index in range(agents.shape[0]):
        neighbor_num = 0
        r: np.ndarray = np.zeros(2)
        agent_pos: np.ndarray = agents[agent_index][:2]
        for neighbor_index in range(agents.shape[0]):
            if agent_index != neighbor_index:
                neighbor_pos: np.ndarray = agents[neighbor_index][:2]
                distance = np.linalg.norm(agent_pos - neighbor_pos)
                if distance <= agents[0][3]:  # R_repulsion
                    neighbor_num = neighbor_num + 1
                    r += (agent_pos - neighbor_pos) / distance  # unit vector
        num_avoid[agent_index] = neighbor_num
        f_avoid[agent_index] = r

    return num_avoid, f_avoid


@nb.jit(nopython=True)
def get_sheep_forces(
    agents: np.ndarray, shepherd: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the agent-agent repulsion, agent-agent attraction, and
    agent-shepherd repulsion forces in a single fused pass. Replaces the
    get_repulsion_force / get_attraction_force / get_shepherd_force trio on the
    hot path: those walked the O(N^2) agent pairs once per force and computed
    each distance with np.linalg.norm, which numba lowers to a BLAS dnrm2 call
    plus a heap-allocated temporary (~80x the cost of a scalar
    sqrt(dx*dx + dy*dy) per pair, ~97% of the tick at N=300).

    Numerical contract relative to the replaced functions:
      * Pair visit order (ascending neighbor index, self skipped) and therefore
        every accumulation order is identical.
      * Distances are sqrt(dx*dx + dy*dy) instead of dnrm2; the two round
        differently by 1 ULP on a sizable fraction of pairs, so trajectories
        are NOT bit-identical. Steps-to-completion distributions were
        revalidated per mode instead (KS/MWU over 100 seeds).
      * Pairs with d2 > attraction_distance^2 skip the sqrt entirely. With
        attraction_distance = 25, both 25 and 625 are exactly representable
        and sqrt is correctly rounded, so d2 > 625 implies sqrt(d2) > 25
        exactly: the early-out can never drop a pair the full comparison would
        have kept. Surviving pairs use the original comparisons on the
        distance itself (this exactness needs the square of the attraction
        radius to be representable; it holds for any integer radius < 2^26).
      * An agent exactly at the repulsion radius contributes to BOTH forces,
        and coincident agents produce the same 0/0 = nan repulsion, as before.

    Args:
        @param agents: The agents to calculate the pairwise forces between.
        @param shepherd: The shepherds repulsing the agents.
    Returns:
        num_avoid: The number of agents repelled by the agent.
        f_avoid: The repulsion force.
        f_attraction: The attraction force.
        f_shepherd_force: The agent-shepherd repulsion force.
    """
    num_agents = agents.shape[0]
    num_shepherds = shepherd.shape[0]
    num_avoid: np.ndarray = np.zeros(num_agents)
    f_avoid: np.ndarray = np.zeros((num_agents, 2))
    f_attraction: np.ndarray = np.zeros((num_agents, 2))
    f_shepherd_force: np.ndarray = np.zeros((num_agents, 2))

    r_repulsion: float = agents[0, 3]
    r_attraction: float = agents[0, 5]
    safe_distance: float = agents[0, 17]
    r_attraction_sq: float = r_attraction * r_attraction

    for agent_index in range(num_agents):
        agent_x: float = agents[agent_index, 0]
        agent_y: float = agents[agent_index, 1]

        avoid_num: int = 0
        avoid_x: float = 0.0
        avoid_y: float = 0.0
        att_x: float = 0.0
        att_y: float = 0.0
        for neighbor_index in range(num_agents):
            if agent_index == neighbor_index:
                continue
            dx: float = agents[neighbor_index, 0] - agent_x
            dy: float = agents[neighbor_index, 1] - agent_y
            d2: float = dx * dx + dy * dy
            if d2 > r_attraction_sq:  # exact: can never skip a d <= 25 pair
                continue
            distance: float = np.sqrt(d2)
            # np.divide, not the / operator: numba gives scalar / Python
            # semantics (raises on 0.0), while the replaced array division
            # follows IEEE and yields nan/inf for coincident agents.
            if distance <= r_repulsion:  # R_repulsion
                avoid_num = avoid_num + 1
                avoid_x += np.divide(-dx, distance)  # unit vector, agent - neighbor
                avoid_y += np.divide(-dy, distance)
            if r_repulsion <= distance <= r_attraction:
                att_x += np.divide(dx, distance)  # unit vector, neighbor - agent
                att_y += np.divide(dy, distance)
        num_avoid[agent_index] = avoid_num
        f_avoid[agent_index, 0] = avoid_x
        f_avoid[agent_index, 1] = avoid_y
        f_attraction[agent_index, 0] = att_x
        f_attraction[agent_index, 1] = att_y

        shepherd_x: float = 0.0
        shepherd_y: float = 0.0
        for shepherd_index in range(num_shepherds):
            sdx: float = agent_x - shepherd[shepherd_index, 0]
            sdy: float = agent_y - shepherd[shepherd_index, 1]
            s_distance: float = np.sqrt(sdx * sdx + sdy * sdy)
            if s_distance <= safe_distance and s_distance != 0.0:
                shepherd_x += np.divide(sdx, s_distance)
                shepherd_y += np.divide(sdy, s_distance)
        f_shepherd_force[agent_index, 0] = shepherd_x
        f_shepherd_force[agent_index, 1] = shepherd_y

    return num_avoid, f_avoid, f_attraction, f_shepherd_force


@nb.jit(nopython=True)
def get_shepherd_force(agents, shepherd):
    """
    Calculates the repulsion force between agents and shepherds.
    Args:
        @param agents: The agents to calculate the repulsion force for.
        @param shepherd: The shepherds repulsing.
    Returns:
        num_shepherd_avoid: The number of shepherds repelling the agent.
        f_shepherd_force: The repulsion force.
    """
    num_shepherd_avoid = np.zeros(agents.shape[0])
    f_shepherd_force = np.zeros((agents.shape[0], 2))

    safe_distance = agents[0][17]  # safe_distance
    for agent_index in range(agents.shape[0]):
        agent_pos: np.ndarray = agents[agent_index][:2]
        r_pos: np.ndarray = np.zeros(2)
        num_shepherd: int = 0
        for shepherd_index in range(shepherd.shape[0]):
            shepherd_pos: np.ndarray = shepherd[shepherd_index][:2]
            distance = np.linalg.norm(agent_pos - shepherd_pos)
            if distance <= safe_distance and distance != 0.0:
                num_shepherd = num_shepherd + 1
                r_pos = r_pos + (agent_pos - shepherd_pos) / distance
        num_shepherd_avoid[agent_index] = num_shepherd
        f_shepherd_force[agent_index] = r_pos
    return num_shepherd_avoid, f_shepherd_force


@nb.jit(nopython=True)
def get_fence_force(
    agents: np.ndarray,
    shepherds: np.ndarray,
    target: tuple[float, float, float],
    fence: np.ndarray[float, float],
) -> tuple[np.ndarray[float, float], np.ndarray[float, float]]:
    """
    We model a fence as an impassible barrier around the pen that the agents and
    shepherds cannot pass. We calculate the repulsion force between agents and
    the fence, necessary for the agents to not pass through the fence.

    @param agents: The agents to calculate the repulsion force for.
    @param shepherd: The shepherds repulsing.
    @param target: The target to calculate the repulsion force for.
    @param fence: The fence to calculate the repulsion force for.

    @return f_fence_force: The repulsion force of the fence.
    """
    # Cutoff force
    f_max: float = 5e2
    # Calculates the distance between the agents and the fence.
    agent_dist: np.ndarray = np.array(
        [np.linalg.norm(agents[i, :2] - fence) for i in range(agents.shape[0])]
    )
    # Calculates the angle the agent is approaching the target, from the target's perspective.
    agent_angle: np.ndarray = np.arctan2(
        target[0] - agents[:, 0], target[1] - agents[:, 1]
    )
    # Calculates the distance between the shepherds and the fence.
    shepherd_dist: np.ndarray = np.array(
        [np.linalg.norm(shepherds[i, :2] - fence) for i in range(shepherds.shape[0])]
    )
    # Calculates the angle the shepherd is approaching the target, from the target's perspective.
    shepherd_angle: np.ndarray = np.arctan2(
        target[0] - shepherds[:, 0], target[1] - shepherds[:, 1]
    )

    # Calculates the repulsion force between the agents and the fence.
    affected_agents = np.logical_not(
        np.logical_or(  # agents not in the target or in the gate
            agents[:, 21] == 1,
            np.logical_and(
                FENCE_MIDDLE_ANGLE - GATE_ANGULAR_WIDTH / 2 <= agent_angle,
                agent_angle <= FENCE_MIDDLE_ANGLE + GATE_ANGULAR_WIDTH / 2,
            ),
        )
    ) & (  # agents within the fence's range.
        agent_dist <= target[-1] + 3 * agents[:, 7]
    )
    f_fence_on_sheep: np.ndarray = np.zeros((agents.shape[0], 2), dtype=float)
    for i in range(agents.shape[0]):
        if affected_agents[i]:
            # Direction of repulsion (normal to fence).
            vec: np.ndarray = agents[i, :2] - fence
            # Unit vector in the direction of the fence.
            vec = vec / np.linalg.norm(vec)
            # Calculate the repulsion force.
            f_fence_on_sheep[i] = (K_FENCE / (agent_dist[i] - target[-1])) * vec
            # Force should not be infinite, cap it.
            if np.linalg.norm(f_fence_on_sheep[i]) > f_max:
                f_fence_on_sheep[i] = f_max * vec
            # Assert the force is not nan or inf.
            # assert (np.all(np.isfinite(f_fence_on_sheep[i])),
            # f"Force: {f_fence_on_sheep[i]}")

    # Calculates the repulsion force between the shepherds and the fence.
    affected_shepherds: np.ndarray = np.logical_not(
        np.logical_and(  # Shepherds not entering via the gate.
            FENCE_MIDDLE_ANGLE - GATE_ANGULAR_WIDTH / 2 <= shepherd_angle,
            shepherd_angle <= FENCE_MIDDLE_ANGLE + GATE_ANGULAR_WIDTH / 2,
        )
    ) & (  # Shepherds within the fence's range.
        shepherd_dist <= target[-1] + 3 * shepherds[:, 7]
    )
    f_fence_on_shepherds: np.ndarray = np.zeros((shepherds.shape[0], 2), dtype=float)
    for i in range(shepherds.shape[0]):
        if affected_shepherds[i]:
            # Vector parallel to the repulsion force (normal from fence).
            vec: np.ndarray = shepherds[i, :2] - fence
            # Unit vector in the direction of the fence.
            vec = vec / np.linalg.norm(vec)
            # Calculate the repulsion force.
            f_fence_on_shepherds[i] = (K_FENCE / (shepherd_dist[i] - target[-1])) * vec
            # Force should not be infinite, cap it.
            if np.linalg.norm(f_fence_on_shepherds[i]) > f_max:
                f_fence_on_shepherds[i] = f_max * vec
            # Assert the force is not nan or inf.
            # assert (np.all(np.isfinite(f_fence_on_shepherds[i])),
            # f"Force: {f_fence_on_shepherds[i]}")

    return f_fence_on_sheep, f_fence_on_shepherds
