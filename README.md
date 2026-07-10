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

Shepherds estimate where the flock is using one of several strategies,
selected by `MODE` in `basic/__init__.py` (implementations in
`basic/herd/driver.py`):

| MODE | Strategy |
|---|---|
| 0 | Center of mass (Yating's original model) |
| 1 | Vision projection (`basic/vision_functions.py`) |
| 2 | Convex hull center |
| 3 | Visible convex hull (only hull vertices the shepherd can see) |
| 4 | Local visible convex hull (nearest subflock's visible hull) |

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

Requires Python 3.12+. Set up a virtual environment:

```bash
python3 -m venv .venv
.venv/bin/pip install numba matplotlib joblib scipy
```

(`requirements.txt` carries stale 2024 pins; current numba/numpy work. The
`data_analysis/` scripts and `basic/save_data.py` additionally need
`pandas` and `h5py`.)

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
