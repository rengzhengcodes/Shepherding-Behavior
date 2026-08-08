"""
mp4 export of PyGame-rendered simulation frames via a piped ffmpeg subprocess.

This module is the pygame-backed counterpart to `draw_dynamic`, formerly in
`basic/drawing/draw.py`, which rendered frames with Matplotlib.
`basic/drawing/draw.py` was deleted when this pygame port landed; it
survives in git history for anyone who needs to consult it directly.
`write_video` below is a drop-in replacement with the same signature and
semantics as the old `draw_dynamic`, but sources each frame's pixels from a
`pygame.Surface` rendered by `basic.drawing.pygame_draw.render_frame`
instead of a Matplotlib Figure/Axes, and streams the raw RGBA bytes
straight to ffmpeg's stdin -- never touching disk with an intermediate
frame file.

Notes
-----
This module must never call `pygame.display.*` or `pygame.init()`. Every
surface it creates or touches is a plain, off-screen `pygame.Surface`,
which -- unlike the display module -- requires no windowing/video backend
and performs no I/O with an X server, framebuffer, or SDL video subsystem.
That makes it safe to import and use inside headless containers (this
environment has no display) and inside `joblib`/`loky` worker processes,
where initializing a display subsystem per worker would be both wrong
(there is nothing to display to) and a likely source of crashes or hangs
under process-pool reuse.
"""

# Design: PYGAME_HIDE_SUPPORT_PROMPT must be set before `import pygame` --
# pygame prints its "Hello from the pygame community" banner as an import
# side effect the first time the `pygame` package is imported in a
# process, so the env var has to land before that import, not after.
import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import shutil
import subprocess

import numpy as np
import pygame

from basic import DRAW_FPS
from basic.drawing.pygame_draw import DEFAULT_CONFIG, render_frame


def ffmpeg_exe() -> str | None:
    """
    Locate an invocable ffmpeg executable.

    First checks `PATH` via `shutil.which`, then falls back to the
    `imageio_ffmpeg` package's bundled static binary, which this project's
    virtual environment provides even when no system-wide ffmpeg install
    exists.

    Returns
    -------
    str or None
        An absolute (or otherwise invocable) path to an ffmpeg executable,
        or `None` if neither `PATH` nor `imageio_ffmpeg` yields one.

    Notes
    -----
    The `imageio_ffmpeg` fallback is wrapped in a bare `except Exception`
    because the package may be absent (`ImportError`) or, in principle,
    fail in other ways while resolving/extracting its bundled binary; any
    such failure should degrade to "no ffmpeg found" rather than propagate,
    since this function's whole purpose is a best-effort lookup that
    callers (`FfmpegWriter.__enter__`) turn into an actionable error.
    """
    exe = shutil.which("ffmpeg")
    if exe is not None:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _ffmpeg_pipe_cmd(
    w: int, h: int, out_path: str, ffmpeg_path: str, fps: int = DRAW_FPS
) -> list[str]:
    """
    Build the argv for an ffmpeg process that consumes raw RGBA frames on
    stdin and encodes them to an H.264/yuv420p mp4.

    Parameters
    ----------
    w : int
        Frame width in pixels. For this module, always
        `DEFAULT_CONFIG.window_size[0]` -- pygame surfaces have an exact,
        caller-chosen pixel size (unlike a DPI-rendered Matplotlib canvas),
        so there is no rounding uncertainty to resolve by measuring a
        rendered frame first.
    h : int
        Frame height in pixels; see `w`.
    out_path : str
        Destination path for the encoded mp4.
    ffmpeg_path : str
        Path to the ffmpeg executable to invoke, as resolved by
        `ffmpeg_exe()`. Used as `argv[0]` instead of the bare string
        `"ffmpeg"` so this works even when ffmpeg is not on `PATH` (e.g.
        the `imageio_ffmpeg`-bundled binary).
    fps : int, optional
        Playback frame rate declared to ffmpeg via `-framerate` on the
        *input* (not `-r` on the output -- see Notes). Defaults to
        `DRAW_FPS` (`basic.DRAW_FPS`).

    Returns
    -------
    list of str
        The ffmpeg command in list (argv) form, suitable for
        `subprocess.Popen(..., shell=False)`.

    Notes
    -----
    This is returned as a list, never a shell string, and callers must never
    pass it through `shell=True`: result mp4 paths built by this project
    embed a literal `|` character (see `experiments/test.py`'s
    `MODE_{MODE}|Rep_{rep}|final_{final_tick}.mp4` naming), which a shell
    would interpret as a pipe operator.

    No `-r 30` output frame-rate override is used (unlike an early
    PNG-based pipeline this project no longer uses) -- with one input frame
    per `DRAW_INTERVAL` ticks, tripling to 30 fps would only re-encode every
    frame 3x for no visual gain. `-framerate fps` on the *input* is
    sufficient to set playback speed.

    `-threads 1` caps each ffmpeg process to one encoder thread. Reps are
    parallelized at the joblib level (potentially one ffmpeg process per
    CPU core running concurrently); an unbounded per-process thread pool
    would oversubscribe the machine.

    # NOTE: draw.py's `_ffmpeg_pipe_cmd` (which this is ported from) hard-coded
    # `str(DRAW_FPS)` inline rather than taking an `fps` parameter, since it
    # had no caller-configurable frame rate. `FfmpegWriter.__init__` below
    # *does* expose a public `fps` parameter (defaulting to `DRAW_FPS`, so
    # behavior is unchanged unless a caller opts in), and threading it
    # through here is the only way to honor a non-default value instead of
    # silently ignoring it. This is the one argv-construction difference
    # from draw.py's version beyond argv[0]; when `fps == DRAW_FPS` (the
    # default) the produced argv is identical.
    """
    return [
        ffmpeg_path,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgba",
        "-video_size",
        f"{w}x{h}",
        "-framerate",
        str(fps),
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


class FfmpegWriter:
    """
    Context manager wrapping a piped ffmpeg process that encodes raw RGBA
    frames written to its stdin into an H.264 mp4.

    Frames are supplied one at a time via `write(surface)`, where `surface`
    is a `pygame.Surface`; ffmpeg consumes and encodes them from its stdin
    pipe concurrently with rendering, so no frame or intermediate video
    segment is ever written to disk by this class.

    Parameters
    ----------
    size : tuple of int
        `(width, height)` in pixels of every frame that will be written.
        Must match the actual pixel dimensions of every `pygame.Surface`
        passed to `write` -- ffmpeg is told this size up front via
        `-video_size` (raw video has no per-frame header to carry it) and
        will misinterpret or reject a stream of differently sized frames.
    video_path : str
        Destination `.mp4` path. May contain characters that are unsafe to
        pass through a shell (this project's naming convention embeds `|`);
        ffmpeg is always invoked via the list form of `subprocess.Popen`
        (never `shell=True`), so this is safe. The ffmpeg stderr log is
        written alongside it at `f"{video_path}.log"`.
    fps : int, optional
        Playback frame rate forwarded to `_ffmpeg_pipe_cmd`. Defaults to
        `DRAW_FPS` (`basic.DRAW_FPS`).

    Raises
    ------
    RuntimeError
        From `__enter__`, if no ffmpeg executable can be resolved, or if
        the underlying ffmpeg process exits with a nonzero return code
        (raised from `__exit__`, unless another exception is already
        propagating out of the `with` block -- see `__exit__`).

    Notes
    -----
    Not safe to reuse across multiple videos or to enter more than once:
    construct a fresh `FfmpegWriter` per output file.
    """

    def __init__(self, size: tuple[int, int], video_path: str, fps: int = DRAW_FPS):
        """
        Store the writer's configuration; does not touch the filesystem or
        spawn any process (see `__enter__` for that).

        Parameters
        ----------
        size : tuple of int
            `(width, height)` in pixels of every frame to be written.
        video_path : str
            Destination `.mp4` path.
        fps : int, optional
            Playback frame rate. Defaults to `DRAW_FPS`.
        """
        self._size = size
        self._video_path = video_path
        self._fps = fps
        self._log_path = f"{video_path}.log"
        self._log_file = None
        self._proc: subprocess.Popen | None = None

    def __enter__(self) -> "FfmpegWriter":
        """
        Resolve an ffmpeg executable, open the stderr log file, and spawn
        the ffmpeg subprocess with its stdin piped for frame writes.

        Returns
        -------
        FfmpegWriter
            `self`, per the context manager protocol.

        Raises
        ------
        RuntimeError
            If `ffmpeg_exe()` cannot locate an ffmpeg executable.
        OSError
            If the log file at `f"{video_path}.log"` cannot be opened (e.g.
            its parent directory does not exist).

        Notes
        -----
        stderr is redirected to the log FILE, not `PIPE` and not
        `DEVNULL`. A `PIPE`'s OS buffer is only tens of KB; nothing here
        reads it concurrently (this process is busy writing stdin via
        `write()`), so ffmpeg's routine progress/diagnostic chatter on
        stderr would eventually fill the buffer and deadlock ffmpeg against
        us. `DEVNULL` would avoid the deadlock but discard diagnostics
        needed to debug a failed render. A file write has no such bound and
        survives the process for inspection.

        If `Popen` itself raises (e.g. ffmpeg vanished from disk between
        `ffmpeg_exe()`'s check and now), there is no ffmpeg stderr to keep,
        so the just-created empty log is closed and removed rather than
        orphaning a zero-byte file next to a video that was never started.
        """
        ffmpeg_path = ffmpeg_exe()
        if ffmpeg_path is None:
            raise RuntimeError(
                "No ffmpeg executable found. Install ffmpeg and ensure it is "
                "on PATH, or run `uv pip install imageio-ffmpeg` to obtain a "
                "bundled binary usable without a system-wide install."
            )

        width, height = self._size
        self._log_file = open(self._log_path, "wb")
        try:
            cmd = _ffmpeg_pipe_cmd(
                width, height, self._video_path, ffmpeg_path, self._fps
            )
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self._log_file,
            )
        except BaseException:
            self._log_file.close()
            os.remove(self._log_path)
            raise
        return self

    def write(self, surface: "pygame.Surface") -> None:
        """
        Encode one frame from a pygame Surface.

        Parameters
        ----------
        surface : pygame.Surface
            The frame to write. Must have the same `(width, height)` as
            `size` passed to `__init__` (see that parameter's docstring for
            why a mismatch is a problem for ffmpeg's `-video_size`).

        Returns
        -------
        None

        Notes
        -----
        No intermediate files anywhere: `pygame.image.tobytes(surface,
        "RGBA")` returns a tightly packed, top-down buffer in R, G, B, A
        byte order -- exactly the `-f rawvideo -pix_fmt rgba` layout
        declared to ffmpeg by `_ffmpeg_pipe_cmd`. No flip, channel reorder,
        or on-disk staging of any kind is needed; the bytes go straight
        from the Surface into ffmpeg's stdin pipe. (On the alpha-less
        surfaces this module renders into, the A byte's *value* is
        arbitrary -- pygame fabricates it from the surface's pixel word --
        but that is harmless: swscale discards alpha in the rgba->yuv420p
        conversion, so only the R/G/B bytes ever reach the encoder.)

        `BrokenPipeError` is swallowed here: it means ffmpeg has already
        exited (bad args, codec error, disk full, ...). There is nothing
        productive to do at the point of a single failed frame write --
        `__exit__`'s returncode check raises with full diagnostics
        regardless of whether we got here via a `BrokenPipeError` or ffmpeg
        instead fails only after all frames were accepted.
        """
        try:
            self._proc.stdin.write(pygame.image.tobytes(surface, "RGBA"))
        except BrokenPipeError:
            pass

    def __exit__(self, exc_type, exc, tb) -> bool | None:
        """
        Close ffmpeg's stdin (signaling EOF so it can finish encoding and
        exit), wait for it, close the log file, and surface any encoding
        failure.

        Parameters
        ----------
        exc_type : type or None
            Exception type propagating out of the `with` block, if any.
        exc : BaseException or None
            The exception instance, if any.
        tb : types.TracebackType or None
            The exception traceback, if any.

        Returns
        -------
        bool or None
            Always `None`/falsy: this method never suppresses an exception
            that was already propagating out of the `with` block.

        Raises
        ------
        RuntimeError
            If ffmpeg exited with a nonzero return code and no other
            exception is already propagating. The message embeds the last
            ~30 lines of ffmpeg's stderr log (`_tail_log`).

        Notes
        -----
        stdin-close/wait/log-close all run unconditionally, on every exit
        path (normal or exceptional), mirroring the guarantees of
        `draw.py`'s `draw_dynamic`, which nested these in `try`/`finally`
        blocks so the subprocess can never be left running past this
        function's return and the log fd can never leak.

        The ffmpeg-failure `RuntimeError` is raised only when `exc_type is
        None`, i.e. only on a normal (non-exceptional) exit from the `with`
        block. This reproduces `draw_dynamic`'s control flow, where the
        equivalent returncode check sits textually *after* -- not inside --
        the frame-write loop's `try`/`finally`, so it is only ever reached
        when that loop completed without raising. If the caller's own code
        (e.g. `render_frame`) raises something other than a swallowed
        `BrokenPipeError`, that original exception must surface as itself
        rather than being replaced by a secondary ffmpeg diagnostic --
        cleanup still happens, but the ffmpeg-specific error and log
        deletion are skipped so the log remains available for inspection
        alongside the real failure.
        """
        try:
            try:
                self._proc.stdin.close()
            except BrokenPipeError:
                pass
            self._proc.wait()
        finally:
            self._log_file.close()

        if exc_type is not None:
            # Another exception is already unwinding the `with` block;
            # don't mask it with an ffmpeg diagnostic, and leave the log on
            # disk in case it's relevant to that failure too.
            return None

        if self._proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg exited with code {self._proc.returncode} while "
                f"writing {self._video_path!r}. Last log lines:\n"
                f"{_tail_log(self._log_path)}"
            )

        # Only reached on success; a failed run's log stays on disk for
        # inspection (and is already embedded in the exception above).
        os.remove(self._log_path)
        return None


def _fence_params() -> tuple[float, float] | None:
    """
    Read the current fence-gate parameters from `basic`, if fencing is
    enabled.

    Returns
    -------
    tuple of (float, float) or None
        `(basic.FENCE_MIDDLE_ANGLE, basic.GATE_ANGULAR_WIDTH)` if
        `basic.FENCE` is truthy, else `None`.

    Notes
    -----
    Design: this does `import basic` and reads `basic.FENCE` /
    `basic.FENCE_MIDDLE_ANGLE` / `basic.GATE_ANGULAR_WIDTH` as module
    attribute accesses at call time, rather than `from basic import FENCE`
    at module import time (which is what the now-removed `draw.py` did,
    lines 17-20). A `from ... import` binds the value once, at import
    time, into this module's namespace; a later `basic.FENCE = True`
    (e.g. a test monkeypatching the flag) would not be seen. Reading
    `basic.FENCE` fresh on every call keeps the flag monkeypatchable for
    callers/tests that flip it after this module has already been
    imported.
    """
    import basic

    if basic.FENCE:
        return (basic.FENCE_MIDDLE_ANGLE, basic.GATE_ANGULAR_WIDTH)
    return None


def write_video(
    setup: tuple[np.ndarray, float, int],
    data: tuple[np.ndarray, np.ndarray],
    space: tuple[int, int],
    target: tuple[int, int, int],
    video_path: str,
) -> None:
    """
    Stream-renders an experiment run directly to an mp4 via a piped ffmpeg
    process, using pygame as the frame source.

    Renders one frame per column of `data`'s trailing (frame) axis into a
    single, reused, off-screen `pygame.Surface`, and writes each frame's
    raw RGBA pixel buffer straight to an ffmpeg subprocess's stdin, which
    encodes it as H.264 video as frames arrive. This is a drop-in
    replacement for `draw.py`'s `draw_dynamic` -- identical signature and
    semantics -- but renders with `basic.drawing.pygame_draw.render_frame`
    instead of Matplotlib, and never writes a PNG or any other intermediate
    frame file to disk.

    Parameters
    ----------
    setup : tuple of (numpy.ndarray, float, int)
        `(frame_ticks, l3, mode)`. `frame_ticks` is a 1-D int array giving
        the real simulation tick each frame corresponds to; frames need not
        be evenly spaced in tick-space (callers may append an out-of-cadence
        final frame showing the exact termination state). `l3` and `mode`
        are simulation parameters shown in each frame's HUD title and
        forwarded to `render_frame`, respectively.
    data : tuple of (numpy.ndarray, numpy.ndarray)
        `(data_agents, data_shepherds)`, each shaped `(n_agents, n_fields,
        n_frames)` with `n_frames == len(frame_ticks)`. Must already be
        trimmed by the caller to exactly the frames to render -- this
        function does no further slicing along the frame axis.
    space : tuple of int
        `(boundary_x, boundary_y)` forwarded to `render_frame` as the
        `boundary` argument.
    target : tuple of int
        `(target_x, target_y, target_size)` forwarded to `render_frame`.
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
        there is nothing to render.
    RuntimeError
        If no ffmpeg executable can be found, or if the ffmpeg subprocess
        exits with a nonzero return code. The message includes the last
        ~30 lines of ffmpeg's stderr log to aid diagnosis (see
        `FfmpegWriter`).

    Notes
    -----
    Unlike `draw_dynamic`, this function does not need to render frame 0
    before spawning ffmpeg to measure pixel dimensions: a `pygame.Surface`
    has an exact, caller-chosen size (`DEFAULT_CONFIG.window_size`), with
    no DPI-driven rounding the way a Matplotlib Agg canvas has. The same
    surface object is reused across all frames and overwritten in place by
    `render_frame` each iteration -- `render_frame` is responsible for
    clearing/repainting it fully each call.

    Never touches `pygame.display` or `pygame.init()` -- see this module's
    docstring. That makes this function safe to run inside joblib/loky
    worker processes, which may run many repetitions' renders concurrently
    across a process pool.

    Rendering and encoding overlap: ffmpeg consumes and encodes frames from
    its stdin pipe while this process is still rendering and writing later
    frames, so wall-clock cost is close to `max(render_time, encode_time)`
    rather than their sum. Encoding runs single-threaded (`-threads 1`)
    because many reps' ffmpeg processes may run concurrently under joblib.

    On success, the ffmpeg stderr log file is deleted. On failure it is
    left on disk for inspection (in addition to being embedded in the
    raised exception). This function does not mutate `data_agents` or
    `data_shepherds`: it only reads per-frame slices from them.
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
        raise ValueError("write_video received zero frames to render")
    n_sheep = data_agents.shape[0]

    # Design: one off-screen Surface, reused for every frame and repainted
    # in place by render_frame -- avoids a per-frame allocation, matching
    # draw_dynamic's reuse of a single Figure/Axes across all frames.
    surface = pygame.Surface(DEFAULT_CONFIG.window_size)

    with FfmpegWriter(DEFAULT_CONFIG.window_size, video_path) as writer:
        for frame_index in range(n_frames):
            render_frame(
                surface,
                data_agents[:, :, frame_index],
                data_shepherds[:, :, frame_index],
                space,
                target,
                mode,
                hud=(
                    f"N_sheep = {n_sheep} | L3 = {l3} | "
                    f"Tick = {int(frame_ticks[frame_index])}"
                ),
                fence=_fence_params(),
            )
            writer.write(surface)
