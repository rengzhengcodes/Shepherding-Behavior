# Shepherding-Behavior

Agent-based model of shepherding: a flock of self-propelled sheep governed by
social forces (short-range repulsion, mid-range attraction, noise, shepherd
avoidance) is driven by one or more shepherd agents. The core is numpy
accelerated with numba JIT.

Two experiment types are supported:

- **Target herding** — shepherds drive the flock into a fixed circular target
  region; a run succeeds when every sheep is in the "staying" state inside it.
  Optionally, a circular **fence** with an angular gate (`FENCE` in
  `basic/__init__.py`) constrains how the flock can enter.
- **Morphology** (`MORPHOLOGY = True`, developed on this branch) — there is no
  fixed target; shepherds compact the flock toward its own center of mass, and
  a run succeeds when all sheep are within radius L2 of it.

## Shepherd strategies (`MODE`)

Every strategy shares the same two-state controller per shepherd
(`herd()` in `basic/interaction.py`); what changes between modes is how the
shepherd *estimates the flock's center*, moving from global knowledge toward
information the shepherd could plausibly sense itself.

- **Drive mode** — the shepherd estimates a flock center, places a *drive
  point* a distance L1 behind that center (on the far side from the target,
  along the target→center ray), and steers toward it with a force linear in
  distance, plus repulsion from other shepherds and the fence force when
  `FENCE` is on. L1 grows with the number of moving sheep
  (`(2/3)·√n·10` in modes 0/2, `(2/3)·√n·2` in modes 3/4, floor of 15).
- **Collect mode** — each tick the shepherd finds the moving sheep whose
  bearing deviates most from its own line to the target
  (`get_furthest_agent`). If that straggler is more than L2 from the
  estimated center (mode 1: if its bearing deviation exceeds the half-FOV
  threshold π/2), the shepherd locks onto it and switches to collecting:
  it steers to a point L0 beyond the straggler — opposite the flock center
  (modes 0/2) or opposite the target (modes 1/3/4) — pushing it back toward
  the group. It returns to drive mode once the straggler is back within L2
  of the center (mode 1: within π/3 of the flock's mean bearing) or enters
  the staying state.

The center estimators (modes 0 and 2–4 in `basic/herd/driver.py`, mode 1 in
`basic/vision_functions.py`):

| MODE | Strategy | Center estimate |
|---|---|---|
| 0 | Center of mass | True mean position of all moving sheep (Yating's original model — full global knowledge). |
| 1 | Vision projection | No positions at all: each sheep is reduced to what it occupies on the shepherd's 1-D retina — a bearing (`arctan2` of relative position) and an angular width (`arctan(radius/distance)`, wider = closer). The shepherd drives behind the visually largest (nearest) sheep, and collect decisions compare bearings on the retina rather than distances. |
| 2 | Convex hull | Mean of the convex-hull vertices of the moving sheep (hull computed once per tick, stored in agent column 22) — only boundary sheep matter. |
| 3 | Visible convex hull | Per shepherd: the hull is computed with the shepherd included (qhull `QG0`), keeping only the facets *visible from its position*; the center is the mean of those visible vertices (biased toward the near edge, hence the smaller L1). Falls back to the nearest sheep when nothing is visible, e.g. with the shepherd inside the flock. Visibility is recorded per shepherd as a bitmask in column 23. |
| 4 | Local visible convex hull | Moving sheep are first clustered into subflocks by connected components with link distance equal to the attraction radius (`identify_flocks`, column 24); the shepherd computes each subflock's visible hull as in mode 3 and attends to the subflock with the closest visible vertex. |

Status: modes 0 and 3 are verified to herd to success end-to-end (see the
smoke driver below). Mode 1 is the least maintained — it appears to be broken
(`basic/vision_functions.py:239` builds the drive point as
`np.array(x, y)` instead of `np.array([x, y])`) and has no alias in
`parse.py`.

## Repository layout

- `basic/` — simulation core.
  - `__init__.py` — **experiment configuration**: `MODE`, `MORPHOLOGY`,
    `FENCE`, target position/size, `DEBUG` (disables numba JIT). These are
    frozen into JIT-compiled code at import time; edit the file and start a
    fresh process to change them.
  - `interaction.py` — `evolve()`, the per-tick update of all agents.
  - `initiation.py` — agent/shepherd array construction (column meanings are
    commented here).
  - `herd/` — shepherd driving strategies (`driver.py`) and social force
    calculations (`forces.py`).
  - `draw.py`, `save_data.py`, `vision_functions.py` — rendering, HDF5/JSON
    persistence, and an experimental vision-projection herding model.
- `test.py` — **the experiment runner** (despite the name): runs many seeded
  repetitions in parallel with joblib and writes per-run summaries to
  `results/fence/<MODE>/` as JSON plus a final-tick histogram.
- `parse.py` — cross-mode analysis of those results (histograms, boxplots,
  percentile runs).
- `data_analysis/` — plotting scripts for success rate, guiding time, and
  shepherd-state analyses.
- `.claude/skills/run-shepherding-behavior/` — smoke-run driver and verified
  run instructions (see below).

## Getting started

Requires Python 3.12 or newer. Every pinned dependency ships prebuilt wheels for
CPython 3.12–3.14, so no compiler is needed. Create a virtual environment and
install through *its* pip:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Install via `.venv/bin/pip` (or run `source .venv/bin/activate` first) rather
than a bare `pip install` — a bare `pip` runs against the system Python, which on
newer distros has no matching wheels and falls back to compiling the pins from
source (needs `python3-devel`, and fails without it). If your `python3` predates
3.12, create the venv with an explicit interpreter, e.g.
`python3.12 -m venv .venv`.

For a quick end-to-end run (~45 s: ~20 s of JIT compilation, then the
simulation) that herds 30 sheep with 2 shepherds and saves snapshot PNGs to
`images/smoke/`:

```bash
.venv/bin/python .claude/skills/run-shepherding-behavior/driver.py
```

See `.claude/skills/run-shepherding-behavior/SKILL.md` for options, timings,
and troubleshooting.

## Running full experiments

Edit the constants at the top of `test.py` (`N_SHEEP`, `N_SHEPHERD`, `REPS`,
`ITERATIONS`, `DRAW`, `THREADS`) and in `basic/__init__.py`, then:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python test.py
```

This is a batch job — the default 64 repetitions of 300 sheep for up to
200,000 ticks take hours across all cores. Results land in gitignored
`results/`. With `DRAW = True`, per-run frame videos are assembled with
ffmpeg instead of saving summaries.

## Legacy scripts

`main.py`, `main_interface.py`, `main_interface_metric.py`, `metric_test.py`,
`vision_test.py`, and `run_iterate_L3.sh` predate the current
`initiate`/`initiate_shepherds` signatures and no longer run unmodified; use
`test.py` or the smoke driver instead.

## License

MIT — see [LICENSE](LICENSE).
