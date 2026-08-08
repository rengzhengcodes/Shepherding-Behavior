"""
Live-window driver for the shepherding simulation.

Runs the identical initiate -> flocking -> herding pipeline as
`.claude/skills/run-shepherding-behavior/driver.py` and `experiments/test.py`,
but renders every `--render-every`-th tick into a live (or headless,
recordable) PyGame window via `basic.drawing.live.LiveViewer` instead of
writing Matplotlib snapshots or an end-of-run mp4.

Usage (from the repo root, on a machine with a display):
    .venv/bin/python experiments/live.py
    .venv/bin/python experiments/live.py --sheep 30 --shepherds 2 --record out.mp4

Usage (headless container/CI -- no display, but recording/screenshots still
work against an invisible SDL "dummy" surface):
    SDL_VIDEODRIVER=dummy .venv/bin/python experiments/live.py --headless-ok \
        --record out.mp4 --screenshot-every 1000 --out images/live

While a window is open:
    space            pause / unpause
    right or .       single-step (while paused)
    + or =           halve render-every (draw more often)
    -                double render-every (draw less often)
    s                screenshot the current frame
    Esc, q, or close quit (exit code 0)

Honors MODE / MORPHOLOGY / FENCE from basic/__init__.py (edit that file to
switch modes; they are frozen into the numba-JIT'd code at import time).

Exit codes: 0 = herding succeeded, or the user quit early; 2 = max
iterations reached without success (a final frame is still shown/recorded);
nonzero otherwise = crash.
"""

import argparse
import os
import random
import sys
from timeit import default_timer as timer

# Repo root is one level up from this file (experiments/live.py).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numba as nb
import numpy as np

import basic
from basic import MODE, MORPHOLOGY, TARGET, TARGET_SIZE, TARGET_X, TARGET_Y
from basic.drawing.live import LiveViewer
from basic.herding.initiation import initiate, initiate_shepherds
from basic.herding.interaction import evolve

SPACE_X = 150
SPACE_Y = 150
BOUNDARY = (TARGET_X + TARGET_SIZE + 300, TARGET_Y + TARGET_SIZE + 300)


# Same as test.py: numba's RNG state is separate from the interpreter's, so
# seeding must happen inside JIT-compiled code to affect the jitted
# initiate/evolve. Seeds BOTH the `random`-module and `np.random` streams,
# byte-for-byte like test.py's seed_run -- seeding only one would let a
# live run diverge from a test.py rep with the same seed if any kernel
# draws from the other stream.
@nb.jit(nopython=True)
def seed_run(seed: int):
    """Seeds the random number generators, exactly as experiments/test.py does."""
    random.seed(seed)
    np.random.seed(seed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--sheep", type=int, default=30)
    parser.add_argument("--shepherds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--flock-ticks",
        type=int,
        default=1000,
        help="shepherd-free self-organization ticks before herding starts",
    )
    parser.add_argument("--max-iterations", type=int, default=100_000)
    parser.add_argument(
        "--render-every",
        type=int,
        default=50,
        help="draw (and, if recording, encode) one frame every N herding ticks",
    )
    parser.add_argument(
        "--max-fps",
        type=int,
        default=60,
        help="cap on drawn-frame rate; never throttles the simulation itself",
    )
    parser.add_argument(
        "--record",
        default=None,
        metavar="PATH",
        help="if given, path to write an mp4 recording of drawn frames to",
    )
    parser.add_argument(
        "--screenshot-every",
        type=int,
        default=0,
        help="also save a PNG every N herding ticks (0 = never)",
    )
    parser.add_argument(
        "--headless-ok",
        action="store_true",
        help="permit running with no display, rendering into an invisible "
        "SDL 'dummy' surface (--record/--screenshot-every still work)",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(REPO_ROOT, "images", "live"),
        help="directory for screenshot PNGs",
    )
    args = parser.parse_args()

    print(f"MODE={MODE} MORPHOLOGY={MORPHOLOGY} | {args.sheep} sheep, "
          f"{args.shepherds} shepherds, seed {args.seed}")

    seed_run(args.seed)
    l3 = np.sqrt(args.sheep / args.shepherds) * 5  # as in test.py
    # L2: morphology termination radius around the center of mass (test.py).
    l2 = 10 * np.sqrt(args.sheep) * 2 / 3

    print("initiating + JIT-compiling evolve (~20s on first call)...")
    t0 = timer()
    agents = initiate(args.sheep, args.shepherds, SPACE_X, SPACE_Y, TARGET_SIZE)
    shepherds = initiate_shepherds(0, args.sheep, l3)
    agents, shepherds, _ = evolver(agents, shepherds, l2)
    print(f"first evolve call took {timer() - t0:.1f}s")

    # Shepherd-free flocking phase.
    for _ in range(args.flock_ticks - 1):
        agents, shepherds, _ = evolver(agents, shepherds, l2)

    # Herding phase.
    shepherds = initiate_shepherds(args.shepherds, args.sheep, l3)

    fence = (
        (basic.FENCE_MIDDLE_ANGLE, basic.GATE_ANGULAR_WIDTH) if basic.FENCE else None
    )
    viewer = LiveViewer(
        BOUNDARY,
        current_target(agents, l2),
        MODE,
        render_every=args.render_every,
        max_fps=args.max_fps,
        record_path=args.record,
        screenshot_dir=args.out,
        headless_ok=args.headless_ok,
        fence=fence,
    )

    print("herding phase starting -- keys: space=pause, right/.=step, "
          "+/-=render-every, s=screenshot, esc/q=quit")

    # `outcome` distinguishes the three ways the herding loop below can end,
    # since each has a different exit-code / freeze() contract (see the
    # module docstring's "Exit codes" section):
    #   - "quit": the user closed the window / pressed Esc or q. No freeze()
    #     call: the window the user just asked to close is not the place to
    #     then block waiting for them to close it again.
    #   - "success": termination condition met. freeze() with a SUCCESS HUD.
    #   - "timeout": the loop ran out of iterations. freeze() with a TIMEOUT
    #     HUD. This is also the initial value, so a loop that somehow never
    #     executes (max_iterations == 0) still reports TIMEOUT rather than a
    #     stale "success".
    outcome = "timeout"
    final_tick = args.max_iterations
    t0 = timer()
    try:
        for tick in range(args.max_iterations):
            agents, shepherds, _ = evolver(agents, shepherds, l2)

            # NOTE: LiveViewer.tick()'s spec'd signature is
            # (swarm, shepherds, tick_no) -- it has no target parameter, and
            # LiveViewer.target is fixed at construction time. In MORPHOLOGY
            # mode the herding target is the flock's own moving center of
            # mass (see current_target() below), so it must be refreshed
            # every tick via the viewer's public `target` attribute rather
            # than through tick()'s argument list; this keeps the drawn
            # target circle from going stale as the flock moves, without
            # deviating from the given LiveViewer method signatures. In
            # non-MORPHOLOGY mode the target is the fixed TARGET constant
            # and this is a no-op.
            if MORPHOLOGY:
                viewer.target = current_target(agents, l2)

            if not viewer.tick(agents, shepherds, tick):
                print(f"user quit at tick {tick}")
                outcome, final_tick = "quit", tick
                break

            if args.screenshot_every and tick % args.screenshot_every == 0:
                viewer.screenshot(tick)

            if terminated(agents, l2):
                outcome, final_tick = "success", tick
                break

        if outcome != "quit":
            elapsed = timer() - t0
            rate = final_tick / elapsed if elapsed > 0 else float("inf")
            print(f"{'SUCCESS' if outcome == 'success' else 'TIMEOUT'}: "
                  f"final_tick={final_tick} ({rate:.0f} ticks/s)")
            viewer.freeze(
                agents,
                shepherds,
                final_tick,
                f"SUCCESS at tick {final_tick}"
                if outcome == "success"
                else "TIMEOUT",
            )
        return {"quit": 0, "success": 0, "timeout": 2}[outcome]
    finally:
        # Always torn down, including on a user quit, a timeout, or an
        # exception escaping the loop above -- finalizes the mp4 recording
        # (if any) and releases the display.
        viewer.close()


def current_target(agents, l2):
    """Target circle to herd toward / draw: fixed target, or the flock's
    center of mass in morphology mode (mirrors test.py's run_morph and
    the smoke driver's current_target)."""
    if MORPHOLOGY:
        return (np.mean(agents[:, 0]), np.mean(agents[:, 1]), l2)
    return TARGET


def evolver(agents, shepherds, l2):
    """One tick, aimed at the mode-appropriate target (as in test.py)."""
    return evolve(agents, shepherds, *current_target(agents, l2))


def terminated(agents, l2) -> bool:
    """Success condition, mirroring test.py's run_target / run_morph."""
    if MORPHOLOGY:
        center_x, center_y = np.mean(agents[:, 0]), np.mean(agents[:, 1])
        return np.all(
            np.sqrt((agents[:, 0] - center_x) ** 2 + (agents[:, 1] - center_y) ** 2)
            < l2
        )
    return np.all(agents[:, 21] == 1)


if __name__ == "__main__":
    sys.exit(main())
