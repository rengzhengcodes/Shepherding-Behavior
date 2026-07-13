"""
The visualization functions for the simulation.
"""

import os
import subprocess
import numba as nb
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import patches
from matplotlib.collections import EllipseCollection
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from basic import DRAW_DPI, DRAW_FIGSIZE, DRAW_FPS, FENCE

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


# @nb.jit(nopython=True)
def draw_single(
    swarm,
    shepherds,
    boundary: tuple[int, int],
    target: tuple[int, int],
    mode: int,
    ax=None,
):
    """
    Draws one frame of the simulation.
    Args:
        @param swarm: The flock agents at this point in time.
        @param shepherds: The shepherds at this point in time.
        @param boundary: Max x and y to draw.
        @param target: Target location.
        @param mode: What mode the sim is from.
        @param ax: The matplotlib Axes to draw into. Defaults to the current
        pyplot Axes (`plt.gca()`) when omitted, which preserves the historical
        pyplot-global calling convention used by `plot_snapshot` and the smoke
        driver (`.claude/skills/run-shepherding-behavior/driver.py`). Callers
        that own an explicit Figure/Axes (e.g. `draw_dynamic`) should pass it
        explicitly to avoid touching pyplot's global figure/axes stack.
    """
    # Design: default to plt.gca() rather than requiring ax so every existing
    # positional caller (plot_snapshot below, driver.py's snapshot()) keeps
    # working unchanged, while draw_dynamic can pass its own owned Axes to
    # avoid registering a figure with pyplot (which would leak across
    # loky-reused worker processes under joblib parallelism).
    if ax is None:
        ax = plt.gca()

    # Draw sheep as one EllipseCollection instead of one Circle patch per
    # sheep. Design: a per-sheep `ax.add_patch(plt.Circle(...))` loop measured
    # ~98 ms/frame at N=300 in profiling; a single EllipseCollection with
    # vectorized offsets/colors measures ~5 ms (20x) for the identical visual
    # result. `radius` on the old Circle is a half-width, so the collection's
    # widths/heights (full diameter) must be doubled to match.
    diameter = 2 * swarm[0, 7]
    # Colors are precomputed RGBA rows with the 0.8 alpha baked in, and no
    # collection-level `alpha=` is set: a scalar collection alpha is forced
    # onto every color by mcolors.to_rgba_array(colors, alpha), which turns
    # "none" (0,0,0,0) into 80%-opaque BLACK -- unlike the old per-patch
    # Circle path, where a "none" facecolor stayed transparent under alpha.
    in_hull = (swarm[:, 22] != 0) & (swarm[:, 21] != 1)  # Color face if in hull
    staying = swarm[:, 21] == 1  # Blue if staying
    face_rgba = np.where(
        in_hull[:, None], mcolors.to_rgba("g", 0.8), (0.0, 0.0, 0.0, 0.0)
    )
    edge_rgba = np.where(
        staying[:, None], mcolors.to_rgba("b", 0.8), mcolors.to_rgba("g", 0.8)
    )
    sheep_collection = EllipseCollection(
        widths=diameter,
        heights=diameter,
        angles=0,
        units="xy",  # sizes in data units, matching the old Circle(radius=...) sizing
        offsets=swarm[:, :2],
        # matplotlib 3.11 renamed/removed the old `transOffset` kwarg in favor
        # of `offset_transform`; ax.transData maps offsets from data space.
        offset_transform=ax.transData,
        facecolors=face_rgba,
        edgecolors=edge_rgba,
        linewidths=0.5,
    )
    ax.add_collection(sheep_collection)
    ax.quiver(
        *(swarm[:, :2].T),
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

    # draw shepherds and its collect point
    # shepherds to collect_x/drive_x collect_y/drive_y
    ax.plot(shepherds[:, (14, 0)].T, shepherds[:, (15, 1)].T, color="cyan")
    # Plots collect vs drive mode.
    ax.set_prop_cycle(plt.cycler(color=np.where(shepherds[:, 13] == 1, "r", "b")))
    ax.plot(
        *(shepherds[:, :2].T),
        marker="o",
        markersize=swarm[0, 7],
        alpha=0.2,
    )
    ax.set_prop_cycle(None)
    # Plots orientation of movement.
    ax.quiver(
        *(shepherds[:, :2].T),
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

    # Draw center of mass
    ax.plot(*calculate_mass_center(swarm), "r*", markersize=5)

    # Plots the agent being collected.
    for agent in shepherds[shepherds[:, 13] == 0]:
        collecting_agent = swarm[int(agent[16])]
        ax.plot(
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
                ax.plot(np.mean(hull[:, 0]), np.mean(hull[:, 1]), "k*", markersize=5)

                # Draws the convex hull.
                ax.fill(
                    hull[:, 0], hull[:, 1], color="g", linestyle=":", lw=2, fill=False
                )
        # Draw the direct line between shepherds and agent it can see.
        case 3:
            # Manually calculates entire hull.
            moving_swarm = swarm[swarm[:, 21] == 0]
            hull = (
                ConvexHull(moving_swarm[:, :2]).vertices
                if moving_swarm.shape[0] > 2
                else np.arange(moving_swarm.shape[0])
            )
            # Plots the convex hull.
            ax.fill(
                moving_swarm[hull, 0],
                moving_swarm[hull, 1],
                color="g",
                linestyle=":",
                lw=2,
                fill=False,
            )
            # Plots the hull agents visible to each shepherd.
            for i, shepherd in enumerate(shepherds):
                relevant_swarm = swarm[
                    (swarm[:, 23].view("uint64") & (0b01 << i)) != 0b0
                ]
                ax.plot(
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
                # NOTE: np.mean of an empty relevant_swarm slice is a deliberate
                # no-op RuntimeWarning->nan (matches pre-existing behavior; not
                # "fixed" here per spec).
                ax.plot(
                    np.mean(relevant_swarm[:, 0]),
                    np.mean(relevant_swarm[:, 1]),
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
                ax.fill(
                    flock[hull, 0],
                    flock[hull, 1],
                    color="g",
                    linestyle=":",
                    lw=2,
                    fill=False,
                )

            # Bug fix: this loop previously read `for shepherd in shepherds:`
            # while indexing the visibility bitmask with `i`, the stale loop
            # variable left over from the flock-hull loop above (always
            # `int(np.max(swarm[:, 24]))`, not the shepherd's own index).
            # Ported the mode-3 pattern (`enumerate`) so each shepherd's
            # bit-index matches its own visibility bit, as originally intended.
            for i, shepherd in enumerate(shepherds):
                relevant_swarm = swarm[
                    (swarm[:, 23].view("uint64") & (0b01 << i)) != 0b0
                ]
                ax.plot(
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
                ax.plot(
                    np.mean(relevant_swarm[:, 0]),
                    np.mean(relevant_swarm[:, 1]),
                    "m*",
                    markersize=5,
                )

    # draw target center
    ax.plot(*target[:2], "b*")
    ax.add_patch(
        plt.Circle(
            target[:2], radius=target[-1], facecolor="none", edgecolor="b", alpha=0.5
        )
    )
    ax.set_xlim(xmin=-boundary[0] // 4, xmax=boundary[0])
    ax.set_ylim(ymin=-boundary[1] // 4, ymax=boundary[1])
    # draw gate to the fence.
    if FENCE:
        # Calculate the angles of the fence.
        theta_1: float = FENCE_MIDDLE_ANGLE - GATE_ANGULAR_WIDTH / 2
        theta_2: float = FENCE_MIDDLE_ANGLE + GATE_ANGULAR_WIDTH / 2
        # Converts to degrees. Rotates becaue theta = 0 is down, instead of right.
        theta_1 = np.degrees(theta_1) - 90
        theta_2 = np.degrees(theta_2) - 90
        # Draws arc that represents the gate.
        ax.add_patch(
            patches.Arc(
                target[:2],
                2 * target[-1],
                2 * target[-1],
                theta1=theta_1,
                theta2=theta_2,
                color="r",
                lw=2,
            )
        )
    # plt.axis('equal')
    # plt.axis('square')


def _ffmpeg_pipe_cmd(w: int, h: int, out_path: str) -> list[str]:
    """
    Build the argv for an ffmpeg process that consumes raw RGBA frames on
    stdin and encodes them to an H.264/yuv420p mp4.

    Parameters
    ----------
    w : int
        Frame width in pixels, as actually rendered (see `draw_dynamic`,
        which measures this from the first rendered frame rather than
        recomputing it from `DRAW_FIGSIZE * DRAW_DPI`).
    h : int
        Frame height in pixels, as actually rendered.
    out_path : str
        Destination path for the encoded mp4.

    Returns
    -------
    list of str
        The ffmpeg command in list (argv) form, suitable for
        `subprocess.Popen(..., shell=False)`.

    Notes
    -----
    This is returned as a list, never a shell string, and callers must never
    pass it through `shell=True`: result mp4 paths built by this project
    embed a literal `|` character (see `test.py`'s
    `MODE_{MODE}|Rep_{rep}|final_{final_tick}.mp4` naming), which a shell
    would interpret as a pipe operator.

    No `-r 30` output frame-rate override is used (unlike the previous
    PNG-based pipeline) -- with one input frame per `DRAW_INTERVAL` ticks,
    tripling to 30 fps only re-encoded every frame 3x for no visual gain.
    `-framerate DRAW_FPS` on the *input* is sufficient to set playback speed.

    `-threads 1` caps each ffmpeg process to one encoder thread. Reps are
    parallelized at the joblib level (potentially one ffmpeg process per
    CPU core running concurrently); an unbounded per-process thread pool
    would oversubscribe the machine.
    """
    return [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgba",
        "-video_size",
        f"{w}x{h}",
        "-framerate",
        str(DRAW_FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-threads",
        "1",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-movflags",
        "+faststart",
        out_path,
    ]


def _tail_log(log_path: str, n_lines: int = 30) -> str:
    """
    Reads the last few lines of a log file, for embedding in error messages.
    Args:
        @param log_path: Path to the log file.
        @param n_lines: Number of trailing lines to return.
    """
    with open(log_path, "rb") as log_file:
        lines = log_file.read().splitlines()[-n_lines:]
    return b"\n".join(lines).decode("utf-8", errors="replace")


def draw_dynamic(
    setup: tuple[np.ndarray, float, int],
    data: tuple[np.ndarray, np.ndarray],
    space: tuple[int, int],
    target: tuple[int, int, int],
    video_path: str,
):
    """
    Stream-renders an experiment run directly to an mp4 via a piped ffmpeg process.

    Renders one frame per column of `data`'s trailing (frame) axis into a
    single, reused Matplotlib Axes that is never registered with pyplot, and
    writes each frame's raw RGBA pixel buffer straight to an ffmpeg
    subprocess's stdin, which encodes it as H.264 video as frames arrive.
    This replaces a previous implementation that wrote one PNG per frame to
    disk (via `plt.savefig`), had ffmpeg re-decode every PNG, and then
    deleted them -- a pure temp-file round trip that also could not run
    safely under joblib parallelism (allocating one full-resolution PNG
    write per rep, per frame, concurrently).

    Parameters
    ----------
    setup : tuple of (numpy.ndarray, float, int)
        `(frame_ticks, l3, mode)`. `frame_ticks` is a 1-D int array giving
        the real simulation tick each frame corresponds to; frames need not
        be evenly spaced in tick-space (callers may append an out-of-cadence
        final frame showing the exact termination state). `l3` and `mode`
        are simulation parameters shown in each frame's title and forwarded
        to `draw_single`, respectively.
    data : tuple of (numpy.ndarray, numpy.ndarray)
        `(data_agents, data_shepherds)`, each shaped `(n_agents, n_fields,
        n_frames)` with `n_frames == len(frame_ticks)`. Must already be
        trimmed by the caller to exactly the frames to render -- this
        function does no further slicing along the frame axis.
    space : tuple of int
        `(boundary_x, boundary_y)` forwarded to `draw_single` as the axis
        boundary.
    target : tuple of int
        `(target_x, target_y, target_size)` forwarded to `draw_single`.
    video_path : str
        Destination `.mp4` path. May contain characters that are unsafe to
        pass through a shell (this project's naming convention embeds `|`);
        ffmpeg is always invoked via the list form of `subprocess.Popen`
        (never `shell=True`), so this is safe. The ffmpeg stderr log is
        written alongside it at `f"{video_path}.log"`.

    Returns
    -------
    None
        The video is written to `video_path` as a side effect.

    Raises
    ------
    ValueError
        If `frame_ticks` (and therefore `data`) contains zero frames --
        there is nothing to render and no frame to measure pixel dimensions
        from.
    RuntimeError
        If the ffmpeg subprocess exits with a nonzero return code, including
        the case where it exits mid-stream and writing to its stdin raises
        `BrokenPipeError`. The message includes the last ~30 lines of
        ffmpeg's stderr log to aid diagnosis.

    Notes
    -----
    Owns an explicit `Figure`/`FigureCanvasAgg`/`Axes` rather than using
    `plt.figure()` / `plt.gca()`: this function runs inside joblib/loky
    worker processes that are reused across repetitions, and a
    pyplot-registered figure would leak for the worker's remaining lifetime.
    There is no `plt.ion()`/`plt.ioff()` for the same reason those calls
    were dead in the original implementation -- Agg is a non-interactive,
    headless backend.

    `canvas.buffer_rgba()` returns a top-down, contiguous RGBA8888 buffer,
    which is exactly the `-f rawvideo -pix_fmt rgba` layout declared to
    ffmpeg by `_ffmpeg_pipe_cmd` -- no flip or channel reorder is needed.

    Rendering and encoding overlap: ffmpeg consumes and encodes frames from
    its stdin pipe while this process is still rendering and writing later
    frames, so wall-clock cost is close to `max(render_time, encode_time)`
    rather than their sum. Encoding runs single-threaded (`-threads 1`)
    because many reps' ffmpeg processes may run concurrently under joblib.

    On success, the ffmpeg stderr log file is deleted. On failure it is left
    on disk for inspection (in addition to being embedded in the raised
    exception).
    """
    # Unpacks setup variables for easier use.
    frame_ticks: np.ndarray
    l3: float
    mode: int
    frame_ticks, l3, mode = setup
    # Unpacks data for easier use.
    data_agents: np.ndarray
    data_shepherds: np.ndarray
    data_agents, data_shepherds = data

    n_frames = frame_ticks.shape[0]
    if n_frames == 0:
        raise ValueError("draw_dynamic received zero frames to render")
    n_sheep = data_agents.shape[0]

    # Design: an explicit Figure + FigureCanvasAgg + one Axes, created
    # without ever touching pyplot's global figure stack (no plt.figure(),
    # no plt.gca()). See the "Notes" section of this function's docstring
    # for why (loky worker figure leaks).
    fig = Figure(figsize=DRAW_FIGSIZE, dpi=DRAW_DPI)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot()

    def render(frame_index: int):
        """Renders one frame into the shared Axes; returns its Agg buffer."""
        ax.cla()
        draw_single(
            data_agents[:, :, frame_index],
            data_shepherds[:, :, frame_index],
            space,
            target,
            mode,
            ax=ax,
        )
        ax.set_title(
            f"N_sheep = {n_sheep} | L3 = {l3} | Tick = {int(frame_ticks[frame_index])}"
        )
        canvas.draw()
        return canvas.buffer_rgba()

    # Render frame 0 before spawning ffmpeg: the exact pixel dimensions of
    # an Agg-rendered figure can differ by rounding from a naive
    # figsize * dpi computation, and ffmpeg needs the true -video_size up
    # front (rawvideo has no header to carry it).
    first_buffer = render(0)
    height, width, _ = np.asarray(first_buffer).shape
    first_frame_bytes = bytes(first_buffer)

    log_path = f"{video_path}.log"
    log_file = open(log_path, "wb")
    try:
        proc = subprocess.Popen(
            _ffmpeg_pipe_cmd(width, height, video_path),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            # Design: stderr goes to a file, not PIPE and not DEVNULL. A
            # PIPE's OS buffer is only tens of KB; nothing here reads it
            # concurrently (this process is busy writing stdin), so ffmpeg's
            # routine progress/diagnostic chatter on stderr would eventually
            # fill the buffer and deadlock ffmpeg against us. DEVNULL would
            # avoid the deadlock but discard diagnostics needed to debug a
            # failed render. A file write has no such bound and survives the
            # process for inspection.
            stderr=log_file,
        )
    except BaseException:
        # Popen itself failed (e.g. ffmpeg vanished from PATH after the
        # caller's fail-fast check): there is no ffmpeg stderr to keep, so
        # close and remove the just-created empty log rather than orphaning
        # a zero-byte file next to a video that was never started.
        log_file.close()
        os.remove(log_path)
        raise

    try:
        try:
            proc.stdin.write(first_frame_bytes)
            for frame_index in range(1, n_frames):
                proc.stdin.write(bytes(render(frame_index)))
        except BrokenPipeError:
            # ffmpeg exited early (bad args, codec error, disk full, ...).
            # Nothing to do here but stop writing; the shared
            # close/wait/returncode check below raises with the log tail
            # regardless of whether we got here via BrokenPipeError or a
            # clean write loop that ffmpeg still failed after.
            pass
        finally:
            # Closing stdin signals EOF to ffmpeg so it can finish encoding
            # and exit; must happen even on exception so wait() cannot hang.
            try:
                proc.stdin.close()
            except BrokenPipeError:
                pass
            proc.wait()
    finally:
        log_file.close()

    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg exited with code {proc.returncode} while writing "
            f"{video_path!r}. Last log lines:\n{_tail_log(log_path)}"
        )

    # Only reached on success; a failed run's log stays on disk for
    # inspection (and is already embedded in the exception above).
    os.remove(log_path)


def plot_snapshot(
    experiment: tuple,
    boundary: tuple[int],
    target: tuple[int],
    filepath: str = None,
    title: str = None,
) -> None:
    """
    Plots ending state of the experiment
    Args:
        @param experiment: Experiment setup. Gives final_tick, the swarm state,
        the shepherd state, and the seed (repetition) in that order.
        @param boundary: x and y boundary to plot.
        @param target: x and y of the target area.
        @param filepath: Where to store the figure.
        @param title: What to title the figure.
    """
    final_tick: int
    swarm: np.ndarray
    shepherds: np.ndarray
    repetition: int
    final_tick, swarm, shepherds, repetition = experiment
    # create folder
    folder_path = os.getcwd() + "/snapshot"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # create figure
    plt.figure(figsize=(8, 6), dpi=300)
    # plot sheep
    for agent in swarm[:, :2] + swarm[:, 21]:
        plt.gca().add_patch(
            plt.Circle(
                agent,
                radius=2.5,
                facecolor="none",
                edgecolor="b" if agent[3] == 1 else "g",
                alpha=0.8,
            )
        )
    # Add arrow for orientation.
    plt.quiver(
        *(swarm[:, :2].T),
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
    plt.plot(*(shepherds[:, :2].T), marker="o", color="r", markersize=5, alpha=0.2)
    plt.quiver(
        *(shepherds[:, :2].T),
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

    center_of_mass: tuple[int, int] = calculate_mass_center(swarm)

    plt.plot(*center_of_mass, "r*", markersize=5)
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
        filepath = (
            f"{folder_path}/"
            + f"N_sheep={swarm.shape[0]}_N_shepherds={shepherds.shape[0]}_"
            + f"repetition={repetition}.png"
        )
    plt.savefig(filepath)
