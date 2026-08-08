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
(`herd()` in `basic/herding/interaction.py`); what changes between modes is how the
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

The center estimators (modes 0 and 2–4 in `basic/herding/driver.py`, mode 1 in
`basic/herding/vision_functions.py`):

| MODE | Strategy | Center estimate |
|---|---|---|
| 0 | Center of mass | True mean position of all moving sheep (Yating's original model — full global knowledge). |
| 1 | Vision projection | No positions at all: each sheep is reduced to what it occupies on the shepherd's 1-D retina — a bearing (`arctan2` of relative position) and an angular width (`arctan(radius/distance)`, wider = closer). The shepherd drives behind the visually largest (nearest) sheep, and collect decisions compare bearings on the retina rather than distances. |
| 2 | Convex hull | Mean of the convex-hull vertices of the moving sheep (hull computed once per tick, stored in agent column 22) — only boundary sheep matter. |
| 3 | Visible convex hull | Per shepherd: the hull is computed with the shepherd included (qhull `QG0`), keeping only the facets *visible from its position*; the center is the mean of those visible vertices (biased toward the near edge, hence the smaller L1). Falls back to the nearest sheep when nothing is visible, e.g. with the shepherd inside the flock. Visibility is recorded per shepherd as a bitmask in column 23. |
| 4 | Local visible convex hull | Moving sheep are first clustered into subflocks by connected components with link distance equal to the attraction radius (`identify_flocks`, column 24); the shepherd computes each subflock's visible hull as in mode 3 and attends to the subflock with the closest visible vertex. |

Status: modes 0 and 3 are verified to herd to success end-to-end (see the
smoke driver below). Mode 1 is the least maintained — it appears to be broken
(`basic/herding/vision_functions.py:239` builds the drive point as
`np.array(x, y)` instead of `np.array([x, y])`) and has no alias in
`data_analysis/parse.py`.

## Repository layout

- `basic/` — simulation core, split into sub-packages by concern.
  - `__init__.py` — **experiment configuration**: `MODE`, `MORPHOLOGY`,
    `FENCE`, target position/size, `DEBUG` (disables numba JIT). These are
    frozen into JIT-compiled code at import time; edit the file and start a
    fresh process to change them.
  - `herding/` — the herding mechanisms.
    - `interaction.py` — `evolve()`, the per-tick update of all agents.
    - `initiation.py` — agent/shepherd array construction (column meanings
      are commented here).
    - `driver.py` — shepherd drive strategies.
    - `forces.py` — social force calculations.
    - `hull.py` — convex-hull kernels backing modes 2–4.
    - `vision_functions.py` — the experimental mode-1 vision-projection
      herding model.
    - `create_network.py` — interaction network construction.
  - `drawing/` — the drawing mechanisms, PyGame-backed.
    - `pygame_draw.py` — offscreen frame renderer (`render_frame` draws one
      frame into a caller-owned `pygame.Surface`; `save_frame_png` wraps it
      for one-off snapshots).
    - `video.py` — streams pygame-rendered frames into a piped ffmpeg
      subprocess to assemble an mp4 (`write_video`), with no intermediate
      frame files ever written to disk; resolves an ffmpeg executable from
      `PATH` or, failing that, the bundled `imageio-ffmpeg` binary.
    - `live.py` — interactive window viewer (`LiveViewer`; see "Live
      viewer" below).

    This PyGame renderer replaced an earlier Matplotlib one (`draw.py`,
    removed; see git history); one deliberate visual change survives the
    port — the old Matplotlib figure never equalized its axis scales
    (~1.34:1 stretch), so bodies rendered as ellipses, while the PyGame
    renderer uses a single uniform scale and always draws true circles, so
    freshly rendered videos look slightly different from pre-port mp4s.
  - `analysis/` — the data mechanisms.
    - `save_data.py` — HDF5/JSON persistence.
- `experiments/` — **the experiment runners**.
  - `test.py` — the main experiment runner (despite the name): runs many
    seeded repetitions in parallel with joblib and writes per-run summaries
    to `results/fence/<MODE>/` (at the repo root) as JSON plus a final-tick
    histogram.
  - `legacy/` — broken legacy scripts predating the current package layout
    (see "Legacy scripts" below).
- `data_analysis/` — standalone analysis and plotting scripts, including
  `parse.py` (cross-mode analysis of `test.py` results: histograms,
  boxplots, percentile runs) and scripts for success rate, guiding time, and
  shepherd-state analyses.
- `notebooks/` — Jupyter walkthroughs. `basic_experiment.ipynb` runs one
  seeded repetition of the basic experiment end to end — flocking warm-up,
  herding, mp4 render via `save_frame_png`/`write_video` — and plays the
  resulting video inline (tooling: `uv pip install -e ".[notebook]"`).
- `.claude/skills/run-shepherding-behavior/` — smoke-run driver and verified
  run instructions (see below).

## Getting started

Requires Python 3.12 or newer. Every pinned dependency ships prebuilt wheels for
CPython 3.12–3.14, so no compiler is needed. This project's `.venv` is
`uv`-managed and has no `pip` binary inside it — create it, activate it, and
install with `uv pip`, not a bare `pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Alternatively, install the simulation core as an editable package (pulls in the
same dependencies — including `pygame` — and makes `import basic` work from any
directory, while edits to the constants in `basic/__init__.py` still take effect
immediately):

```bash
uv pip install -e .
```

If your `python3` predates 3.12, create the venv with an explicit interpreter,
e.g. `python3.12 -m venv .venv`. A bare `pip install` (or `python3 -m pip`)
against the system Python on newer distros has no matching wheels and falls
back to compiling the pins from source (needs `python3-devel`, and fails
without it); `uv pip install` against the activated `.venv` avoids both that
and the missing-`pip` problem above.

For a quick end-to-end run (~45 s: ~20 s of JIT compilation, then the
simulation) that herds 30 sheep with 2 shepherds and saves snapshot PNGs to
`images/smoke/`:

```bash
.venv/bin/python .claude/skills/run-shepherding-behavior/driver.py
```

See `.claude/skills/run-shepherding-behavior/SKILL.md` for options, timings,
and troubleshooting.

## Running full experiments

Edit the constants at the top of `experiments/test.py` (`N_SHEEP`,
`N_SHEPHERD`, `REPS`, `ITERATIONS`, `DRAW`, `THREADS`) and in
`basic/__init__.py`, then:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python experiments/test.py
```

This is a batch job — the default 64 repetitions of 300 sheep for up to
200,000 ticks take hours across all cores. `experiments/test.py`'s results
directory is re-anchored to the repo root, so results still land in
gitignored `results/fence/<MODE>/` at the repo root, same as before. With
`DRAW = True`, per-run frame videos are rendered by PyGame
(`basic/drawing/pygame_draw.py`) and streamed straight into a piped ffmpeg
subprocess (`basic/drawing/video.py`) instead of saving summaries. ffmpeg is
auto-resolved from `PATH` or, failing that, the bundled `imageio-ffmpeg`
binary, so no separate system-wide ffmpeg install is required.

## Live viewer

For an interactive, real-time view of a run instead of a batch mp4 render:

```bash
.venv/bin/python experiments/live.py
```

On a machine with a display, this opens a PyGame window and drives it live
from the simulation loop. Keys: `space` pause/resume, `right`/`.` single-step
while paused, `+`/`-` halve/double the render cadence, `s` screenshot, `esc`/`q`
quit. `--record out.mp4` additionally tees every drawn frame to an mp4 via the
same ffmpeg pipe `video.py` uses. `--headless-ok` permits running with no
display detected (renders into an invisible SDL "dummy" surface instead of
raising) — useful in headless containers/CI, where `--record` and
`--screenshot-every` still work normally. See `experiments/live.py --help` for
the full flag list (`--sheep`, `--shepherds`, `--seed`, `--flock-ticks`,
`--max-iterations`, `--render-every`, `--max-fps`, `--out`, ...).

## Legacy scripts

`main.py`, `main_interface.py`, `main_interface_metric.py`, `metric_test.py`,
`vision_test.py`, and `run_iterate_L3.sh` predate the current
`initiate`/`initiate_shepherds` signatures and no longer run unmodified; they
now live in `experiments/legacy/`. Use `experiments/test.py` or the smoke
driver instead.

## Related work

[`docs/collective-shepherding-literature-review.md`](docs/collective-shepherding-literature-review.md)
collects foundational and recent papers on collective shepherding — including
Zheng & Romanczuk's model that `MODE = 0` derives from. **It is an LLM-generated
literature review (Claude Code deep-research, 2026-07-12) and has not been
human-verified** — treat entries as leads and check primary sources before
citing.

[`docs/novelty-assessment.md`](docs/novelty-assessment.md) is a companion prior-art /
novelty analysis of this repo's convex-hull center estimators (`MODE` 2–4) — whether
they have been scooped and the nearest competing work. **Also LLM-generated and not
human-verified; not a novelty search of record.**

## License

MIT — see [LICENSE](LICENSE).
