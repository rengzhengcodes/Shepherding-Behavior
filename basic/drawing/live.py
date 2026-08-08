"""
Live PyGame viewer for the shepherding simulation.

This is the ONLY module in `basic.drawing` allowed to touch `pygame.display`
(or any other pygame subsystem tied to owning a window/surface). Every other
module in this package renders into a caller-owned Surface
(`basic.drawing.pygame_draw.render_frame`) and never opens a display of its
own; centralizing display ownership here keeps "does this process have a
window" a single, auditable question instead of one every drawing helper
would otherwise have to answer for itself. (`basic/drawing/draw.py`, the
matplotlib-based renderer this package's pygame port replaced, owned a
Matplotlib Figure rather than a pygame display; it was deleted when this
port landed and survives in git history for anyone who needs to consult it
directly.)

Pacing design
--------------
The simulation itself (`basic.herding.interaction.evolve`) always runs
uncapped -- this module never throttles the caller's tick rate directly.
Only every `render_every`-th tick is drawn, and only *drawn* frames are
subject to the `max_fps` clock cap (`pygame.time.Clock.tick`), so on-screen
rendering can never slow the simulation down by more than
`render_every * max_fps` ticks per second even in the worst case (every
single tick landing on the render cadence). Ticks that fall between render
points cost only an event-queue drain, which is cheap relative to a
JIT-compiled `evolve()` call.

Hard invariants
----------------
- Never mutates the `swarm`/`shepherds` arrays passed into `tick()` /
  `freeze()`. This viewer is a pure, read-only observer of simulation state.
- Never calls `random` / `np.random` (or anything that would consume from
  them). The simulation's determinism depends on a single, unperturbed
  global RNG stream shared between numpy and numba (see
  `basic/herding/initiation.py`'s `seed_run` and the driver scripts that
  call it); a viewer that quietly drew a random number would desync a run
  that attaches it from an otherwise-identical run that doesn't.
"""

import os

# Must be set before `import pygame`: pygame reads this env var exactly once,
# at import time, to decide whether to print its "Hello from the pygame
# community" support banner to stdout. Setting it here (rather than relying
# on the caller to have set it first) keeps the suppression in force
# regardless of what imports this module first.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from basic.drawing.pygame_draw import DEFAULT_CONFIG, render_frame
from basic.drawing.video import FfmpegWriter

import numpy as np


class LiveViewer:
    """
    Owns a pygame window (or an invisible dummy-driver surface) and drives
    it from a caller's simulation loop, one `tick()` call per simulation
    tick.

    Typical usage (mirrors `experiments/live.py`)::

        viewer = LiveViewer(BOUNDARY, TARGET, MODE, headless_ok=args.headless_ok)
        try:
            for tick in range(max_iterations):
                agents, shepherds, _ = evolve(agents, shepherds, *target)
                if not viewer.tick(agents, shepherds, tick):
                    break  # user quit
                if terminated(agents):
                    viewer.freeze(agents, shepherds, tick, f"SUCCESS at tick {tick}")
                    break
        finally:
            viewer.close()

    Notes:
        Not thread-safe and not meant to be: pygame's display/event APIs are
        only valid from the thread that initialized them, which is assumed
        to be the same thread driving the simulation loop.
    """

    # Keys that request an immediate quit, checked identically everywhere
    # events are handled (see `_process_events`).
    _QUIT_KEYS = (pygame.K_ESCAPE, pygame.K_q)

    def __init__(
        self,
        boundary: tuple[float, float],
        target: tuple[float, float, float],
        mode: int,
        *,
        config=DEFAULT_CONFIG,
        render_every: int = 50,
        max_fps: int = 60,
        record_path: str | None = None,
        screenshot_dir: str | None = None,
        headless_ok: bool = False,
        fence: tuple[float, float] | None = None,
    ) -> None:
        """
        Opens a pygame window (or, headless, an invisible dummy-driver
        surface) and prepares it to receive rendered simulation frames via
        `tick()`.

        Args:
            @param boundary: `(boundary_x, boundary_y)` forwarded to every
            `render_frame` call as the drawing area's extent -- the same
            role `boundary` played for `draw_single`/`draw_dynamic` in the
            since-removed `basic/drawing/draw.py` (see the module
            docstring above).
            @param target: `(target_x, target_y, target_size)` forwarded to
            every `render_frame` call. Fixed for the life of the viewer; a
            caller running morphology mode (where the target is the flock's
            own moving center of mass) is responsible for constructing a new
            `LiveViewer` -- or otherwise keeping its own `self.target` in
            sync -- rather than this class re-deriving it, mirroring how
            `evolve()` itself takes a fresh target argument every call
            instead of owning one.
            @param mode: Simulation MODE (see `basic/__init__.py`),
            forwarded to `render_frame` so mode-specific overlays (hulls,
            shepherd-visibility lines, ...) are drawn correctly.
            @param config: Rendering configuration (colors, window size,
            ...). Defaults to `pygame_draw.DEFAULT_CONFIG`.
            @param render_every: Draw a frame on every Nth call to `tick()`;
            every other call only drains the event queue. Must be >= 1.
            Adjustable at runtime via the `+`/`=` (halve) and `-` (double)
            keys -- see `_process_events`.
            @param max_fps: Cap, in frames per second, applied only to
            *drawn* frames via `pygame.time.Clock.tick`. Never throttles the
            simulation itself; see the module docstring's "Pacing design"
            section.
            @param record_path: If given, every drawn frame -- including the
            final `freeze()` frame -- is also appended to an mp4 at this
            path via `basic.drawing.video.FfmpegWriter`. `None` (default)
            disables recording.
            @param screenshot_dir: Directory `screenshot()` (and the `s`
            key) saves PNGs into. Created (with parents) if it doesn't
            already exist. `None` defaults to the current working directory
            at the time a screenshot is taken.
            @param headless_ok: If `True`, permits constructing the viewer
            even when no display is detected (see `Raises` below), rendering
            into an invisible SDL "dummy" surface -- useful for recording or
            screenshotting a run in a container or CI with no X/Wayland
            server. If `False` (the default), a headless environment raises
            instead of silently rendering into a window nobody can see.
            @param fence: `(FENCE_MIDDLE_ANGLE, GATE_ANGULAR_WIDTH)` when
            `basic.FENCE` is enabled, else `None`. Forwarded verbatim to
            `render_frame` for the fence-gate overlay.

        Raises:
            RuntimeError: If no display is detected (`DISPLAY` and
            `WAYLAND_DISPLAY` are both unset in the environment) and
            `headless_ok` is `False`.
            ValueError: If `render_every < 1`.

        Notes:
            Display detection is an env-var check, not a pygame probe:
            attempting `pygame.display.set_mode` under a real X11/Wayland
            driver with no server available blocks or raises with a far less
            actionable error than the message raised here, and a probe-then-
            reinitialize approach would initialize the video subsystem
            twice for no benefit.
        """
        # Design: headless is detected from the environment rather than by
        # attempting (and possibly failing) a real pygame.display.set_mode
        # call, so the failure mode is a clear, actionable RuntimeError
        # raised before any pygame state is touched, not an SDL error deep
        # inside set_mode. An explicit SDL_VIDEODRIVER=dummy counts as
        # headless even when DISPLAY is set -- the user has asked for an
        # invisible surface, and treating that as a windowed session would
        # send freeze() into a wait-for-window-close loop against a window
        # that can never receive a close event. Forcing the dummy driver is
        # itself explicit consent to invisible rendering, so it bypasses
        # the headless_ok gate (which exists to catch *accidentally*
        # display-less runs, not deliberate ones).
        forced_dummy = os.environ.get("SDL_VIDEODRIVER") == "dummy"
        no_display = not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )
        headless = forced_dummy or no_display
        if no_display and not forced_dummy and not headless_ok:
            raise RuntimeError(
                "LiveViewer: no display detected (DISPLAY and "
                "WAYLAND_DISPLAY are both unset). Pass headless_ok=True "
                "(or --headless-ok on the experiments/live.py CLI) to "
                "render into an invisible SDL 'dummy' surface instead -- "
                "you won't see a window, but --record and "
                "--screenshot-every still work. If you expected a real "
                "window, confirm this process actually has an X/Wayland "
                "display available."
            )
        self.headless = headless
        if headless:
            # setdefault, not a direct assignment: a caller who has already
            # forced a specific SDL_VIDEODRIVER (e.g. to work around a local
            # driver bug) must win over this default.
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        # Validate before touching any pygame subsystem: a constructor that
        # raises after pygame.display.init() would leak an initialized SDL
        # video subsystem with no LiveViewer object for the caller to
        # close().
        if render_every < 1:
            raise ValueError(f"render_every must be >= 1, got {render_every}")

        pygame.display.init()
        pygame.font.init()

        self.config = config
        self.boundary = boundary
        self.target = target
        self.mode = mode
        self.fence = fence
        self.render_every = render_every
        self.max_fps = max_fps

        # Everything past display.init() that can fail (set_mode, mkdir,
        # spawning ffmpeg for --record) is wrapped so a failed constructor
        # tears the display back down instead of leaking it -- the caller
        # never receives an object to call close() on.
        try:
            self.screen = pygame.display.set_mode(config.window_size)
            pygame.display.set_caption("Shepherding")
            self.clock = pygame.time.Clock()

            self.screenshot_dir = screenshot_dir
            if screenshot_dir:
                os.makedirs(screenshot_dir, exist_ok=True)

            self._writer: FfmpegWriter | None = None
            if record_path is not None:
                self._writer = FfmpegWriter(config.window_size, record_path)
                self._writer.__enter__()
        except BaseException:
            pygame.display.quit()
            raise

        # Playback-control state, mutated only by `_process_events` and read
        # only by `tick()`/`_draw()`.
        self._paused = False
        self._step_pending = False
        self._closed = False

    def tick(self, swarm: np.ndarray, shepherds: np.ndarray, tick_no: int) -> bool:
        """
        Advances the viewer by one simulation tick: services the window and,
        if this tick lands on the `render_every` cadence, draws a frame.

        Args:
            @param swarm: Current sheep/agent state array, as produced by
            `basic.herding.interaction.evolve`. Read-only: never mutated,
            and never mutated-via-a-copy, here.
            @param shepherds: Current shepherd state array; same read-only
            contract as `swarm`.
            @param tick_no: The caller's own tick counter, used for the HUD
            text, screenshot filenames, and the render_every cadence check.
            Need not start at 0, but should be monotonically non-decreasing
            for the HUD text and screenshot filenames to stay meaningful.

        Returns:
            `True` if the caller's simulation loop should continue; `False`
            if the user asked to quit (closed the window, or pressed Esc or
            `q`), in which case the caller should stop calling
            `evolve()`/`tick()` and proceed to `close()`.

        Notes:
            Event handling runs on every call, even non-render ticks, so the
            window stays responsive (and isn't flagged "Not Responding" by
            the OS) between renders; this is cheap relative to a JIT'd
            `evolve()` call. When paused, this method blocks the caller's
            loop entirely -- nothing else advances the simulation while
            `tick()` hasn't returned -- until the user unpauses, single-
            steps, or quits; see the module docstring's "Pacing design"
            section for why this is safe to do unconditionally.
        """
        if not self._process_events(tick_no):
            return False

        if tick_no % self.render_every == 0:
            self._draw(swarm, shepherds, tick_no)
            self.clock.tick(self.max_fps)

        # Pause loop: freezes the caller's simulation loop -- nothing calls
        # evolve() again until this returns -- while keeping the window
        # responsive at a modest, fixed 10fps. Deliberately independent of
        # max_fps, which paces *drawn simulation* frames, not this idle/
        # paused UI loop.
        while self._paused:
            self.clock.tick(10)
            if not self._process_events(tick_no):
                return False
            # Design: paused redraws are NOT written to the video recorder.
            # The recording should capture the run's actual progression, not
            # however many wall-clock seconds a human spent paused staring
            # at one frame; only the render_every-cadenced draw above feeds
            # the recorder.
            self._draw(swarm, shepherds, tick_no, write_to_recorder=False)
            if self._step_pending:
                self._step_pending = False
                # Let the caller advance exactly one more tick. self._paused
                # is deliberately left True, so the *next* tick() call
                # re-enters this same loop immediately (after its own
                # top-of-tick event/render housekeeping) -- i.e. "return
                # True once, re-pause next call".
                return True

        return True

    def screenshot(self, tick_no: int) -> str:
        """
        Saves the current on-screen frame to a PNG.

        Args:
            @param tick_no: Tick number folded into the filename
            (`tick_{tick_no:06d}.png`) so successive screenshots sort in
            simulation order and correlate with simulation progress.

        Returns:
            The path the PNG was written to.

        Notes:
            Saves `self.screen` exactly as it currently is -- i.e. whatever
            was last drawn to it, which may be stale if called between
            renders. Callers wanting a screenshot of a specific tick's state
            should prefer the `s` keypress (handled inside `tick()`'s event
            processing, so it always fires immediately after that tick's own
            draw) over calling this directly from outside a render.
        """
        directory = self.screenshot_dir or "."
        path = os.path.join(directory, f"tick_{tick_no:06d}.png")
        pygame.image.save(self.screen, path)
        return path

    def freeze(
        self, swarm: np.ndarray, shepherds: np.ndarray, tick_no: int, message: str
    ) -> None:
        """
        Renders one final, static frame and, outside a headless run, blocks
        until the user closes the window.

        Args:
            @param swarm: Final sheep/agent state to render.
            @param shepherds: Final shepherd state to render.
            @param tick_no: Final tick number. Used only to timestamp a
            screenshot if the user presses `s` while frozen; the HUD itself
            shows `message` verbatim rather than a tick-derived string,
            since the caller (e.g. `f"SUCCESS at tick {tick_no}"` or
            `"TIMEOUT"`) already encodes whatever tick context it wants.
            @param message: HUD text for the final frame.

        Notes:
            Under a headless (dummy-driver) run there is no window a human
            could ever close, so blocking here would hang the process
            forever with no way to signal completion; this method renders
            the frame (so a recording/screenshot still captures it) and
            returns immediately in that case instead. Under a real display,
            it idles at ~10fps servicing the event queue via
            `_process_events`, so Esc/`q`/closing the window all work
            exactly as they do mid-run, and `s` still screenshots the final
            frame if wanted.
        """
        self._draw(swarm, shepherds, tick_no, write_to_recorder=True, message=message)

        if self.headless:
            # Nothing external ever generates a QUIT/Esc/q event against an
            # invisible dummy-driver window, so waiting for one here would
            # hang forever. The frame above (and any recording/screenshot of
            # it) is already durable; there is nothing further to wait for.
            return

        while True:
            self.clock.tick(10)
            if not self._process_events(tick_no):
                return

    def close(self) -> None:
        """
        Idempotently releases display and recording resources.

        Safe to call multiple times (e.g. once from a `finally` block and
        again defensively elsewhere) and safe to call even if `tick()` was
        never invoked.

        Notes:
            Finalizes the recorder via `FfmpegWriter.__exit__(None, None,
            None)` before quitting the display. This ordering isn't
            load-bearing for correctness (the two resources are
            independent), but it finalizes the more failure-prone step --
            waiting on an external ffmpeg subprocess to flush and exit --
            while the display subsystem is still up, rather than after
            already tearing itself down.
        """
        if self._closed:
            return
        self._closed = True
        # The display must be released even when finalizing the recorder
        # raises (FfmpegWriter.__exit__ raises RuntimeError on a nonzero
        # ffmpeg exit) -- _closed is already latched, so without the
        # try/finally a failed encode would skip display.quit() forever.
        try:
            if self._writer is not None:
                self._writer.__exit__(None, None, None)
        finally:
            self._writer = None
            pygame.display.quit()

    def _draw(
        self,
        swarm: np.ndarray,
        shepherds: np.ndarray,
        tick_no: int,
        *,
        write_to_recorder: bool = True,
        message: str | None = None,
    ) -> None:
        """
        Renders one frame to `self.screen` and flips it to the display.

        Args:
            @param swarm: See `tick()`.
            @param shepherds: See `tick()`.
            @param tick_no: See `tick()`; used to build the default HUD text
            when `message` is not given.
            @param write_to_recorder: If `True` (default) and recording is
            enabled, the freshly drawn frame is also appended to the
            recording. `False` is used by `tick()`'s pause loop; see the
            design note there.
            @param message: Overrides the default HUD text; used by
            `freeze()` to show a fixed status string. `None` (default) shows
            `f"Tick = {tick_no} | render every {self.render_every}"`,
            annotated with `" | PAUSED"` while paused.
        """
        hud = message
        if hud is None:
            hud = f"Tick = {tick_no} | render every {self.render_every}"
            if self._paused:
                hud += " | PAUSED"
        render_frame(
            self.screen,
            swarm,
            shepherds,
            self.boundary,
            self.target,
            self.mode,
            hud=hud,
            fence=self.fence,
            config=self.config,
        )
        pygame.display.flip()
        if write_to_recorder and self._writer is not None:
            self._writer.write(self.screen)

    def _process_events(self, tick_no: int) -> bool:
        """
        Drains pygame's event queue and applies every keyboard/window
        command this viewer supports.

        This is the single helper `tick()` calls both on every ordinary call
        and from inside its pause loop (see `tick()`), so every key (quit,
        pause, step, speed, screenshot) behaves identically regardless of
        whether the simulation is currently paused. `freeze()` also reuses
        it while idling, for the same reason.

        Args:
            @param tick_no: Current tick, forwarded to `screenshot()` if the
            `s` key was pressed.

        Returns:
            `False` if the user asked to quit (closed the window, or
            pressed Esc or `q`); `True` otherwise.

        Notes:
            `pygame.event.get()` both pumps SDL's internal event queue
            (keeping the window responsive from the OS's perspective) and
            drains it for handling here, so no separate `pygame.event.pump()`
            call is needed in addition to this.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type != pygame.KEYDOWN:
                continue
            if event.key in self._QUIT_KEYS:
                return False
            if event.key == pygame.K_SPACE:
                self._paused = not self._paused
                # Unpausing cancels any not-yet-consumed single-step: a step
                # and an unpause drained from the same event batch would
                # otherwise leave the flag set with the pause loop never
                # running, and the stale request would fire a phantom step
                # the moment the user next pauses.
                if not self._paused:
                    self._step_pending = False
            elif event.key in (pygame.K_RIGHT, pygame.K_PERIOD):
                # Only meaningful while paused (see tick()'s pause loop).
                # Guarding here -- rather than unconditionally setting the
                # flag -- avoids a stale step request silently firing the
                # instant the user next pauses, long after this keypress.
                if self._paused:
                    self._step_pending = True
            elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                # Halving (floored at 1) approximates a perceived 2x
                # speed-up: half as many ticks are skipped between draws.
                self.render_every = max(1, self.render_every // 2)
            elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.render_every *= 2
            elif event.key == pygame.K_s:
                self.screenshot(tick_no)
        return True
