"""
Smoke-run driver for the shepherding simulation.

Runs the same initiate -> flocking -> herding pipeline as test.py, but at a
small, bounded scale and with snapshot PNGs, so an agent (or human) can see
the simulation working end-to-end in about a minute instead of hours.

Usage (from the repo root):
    .venv/bin/python .claude/skills/run-shepherding-behavior/driver.py
    .venv/bin/python .claude/skills/run-shepherding-behavior/driver.py \
        --sheep 30 --shepherds 2 --max-iterations 60000 --out images/smoke

Honors MODE / MORPHOLOGY / FENCE from basic/__init__.py (edit that file to
switch modes; they are frozen into the numba-JIT'd code at import time).

Exit codes: 0 = herding succeeded, 2 = max iterations reached without
success (snapshot still written), nonzero otherwise = crash.
"""

import argparse
import os
import sys
from timeit import default_timer as timer

# Repo root is three levels up from this file (.claude/skills/run-*/driver.py).
REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, REPO_ROOT)

import matplotlib

matplotlib.use("Agg")  # headless container: never try to open a window
import matplotlib.pyplot as plt
import numba as nb
import numpy as np

from basic import MODE, MORPHOLOGY, TARGET, TARGET_SIZE, TARGET_X, TARGET_Y
from basic.draw import draw_single
from basic.initiation import initiate, initiate_shepherds
from basic.interaction import evolve

SPACE_X = 150
SPACE_Y = 150
BOUNDARY = (TARGET_X + TARGET_SIZE + 300, TARGET_Y + TARGET_SIZE + 300)


# Same as test.py: numba's RNG is separate from numpy's, so seeding must
# happen inside JIT-compiled code to affect the jitted initiate/evolve.
@nb.jit(nopython=True)
def seed_run(seed: int):
    """Seeds the random number generator numba actually uses."""
    np.random.seed(seed)


def snapshot(agents, shepherds, target, path: str, title: str):
    """Renders one frame of the simulation state to a PNG."""
    plt.figure(figsize=(8, 6), dpi=150)
    draw_single(agents, shepherds, BOUNDARY, target, MODE)
    plt.title(title)
    plt.savefig(path)
    plt.close()
    print(f"snapshot -> {path}")


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
        "--snapshot-every",
        type=int,
        default=0,
        help="also snapshot every N herding ticks (0 = start/end only)",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(REPO_ROOT, "images", "smoke"),
        help="directory for snapshot PNGs",
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
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
    snapshot(agents, shepherds, current_target(agents, l2),
             os.path.join(args.out, "start.png"), "herding start")

    t0 = timer()
    final_tick, success = args.max_iterations, False
    for tick in range(args.max_iterations):
        agents, shepherds, _ = evolver(agents, shepherds, l2)
        if args.snapshot_every and tick % args.snapshot_every == 0:
            snapshot(agents, shepherds, current_target(agents, l2),
                     os.path.join(args.out, f"tick_{tick:06d}.png"),
                     f"tick {tick}")
        if tick % 10_000 == 0:
            print(f"tick {tick}: {int(agents[:, 21].sum())}/{args.sheep} staying")
        if terminated(agents, l2):
            final_tick, success = tick, True
            break

    rate = final_tick / (timer() - t0)
    print(f"{'SUCCESS' if success else 'TIMEOUT'}: final_tick={final_tick} "
          f"({rate:.0f} ticks/s)")
    snapshot(agents, shepherds, current_target(agents, l2),
             os.path.join(args.out, "final.png"),
             f"{'success' if success else 'timeout'} at tick {final_tick}")
    return 0 if success else 2


def current_target(agents, l2):
    """Target circle to herd toward / draw: fixed target, or the flock's
    center of mass in morphology mode (mirrors test.py's run_morph)."""
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
