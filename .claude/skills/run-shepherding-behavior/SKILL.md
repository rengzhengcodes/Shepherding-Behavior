---
name: run-shepherding-behavior
description: Run, smoke-test, and screenshot the Shepherding-Behavior simulation. Use when asked to run/start the simulation, verify a change to the herding model works, render/screenshot a simulation snapshot, or launch the full test.py experiment batch.
---

Headless agent-based shepherding simulation (numba-JIT'd numpy core, no GUI/server).
Drive it via `.claude/skills/run-shepherding-behavior/driver.py` — a bounded
smoke run of the real `initiate → evolve → terminate` pipeline that saves
snapshot PNGs. All paths below are relative to the repo root.

## Prerequisites

No apt packages needed (matplotlib renders headless via Agg; no xvfb, no GPU).
System Python is PEP 668 externally-managed, so use a venv:

```bash
python3 -m venv .venv
.venv/bin/pip install numba matplotlib joblib scipy
```

Ignore `requirements.txt` — its numba==0.60.0/numpy==2.0.0 pins are stale;
numba 0.66 + numpy 2.4.6 run this codebase fine (verified).

## Run (agent path)

```bash
.venv/bin/python .claude/skills/run-shepherding-behavior/driver.py
```

Takes ~45s: ~20s numba JIT compile (every process pays it — the `@nb.jit`
decorators have no `cache=True`), then ~2,600 ticks/s. Expected output ends
with `SUCCESS: final_tick=...` (≈48k ticks with the default seed) and writes
`images/smoke/start.png` and `images/smoke/final.png` — read the PNGs to see
the flock (green circles), shepherds (red), and target circle (blue). Sheep
turn blue when "staying" (inside the target). Exit 0 = herding succeeded,
2 = `--max-iterations` hit without success (snapshot still written).

| flag | default | meaning |
|---|---|---|
| `--sheep` / `--shepherds` | 30 / 2 | swarm sizes |
| `--seed` | 0 | RNG seed (seeded inside JIT code, as in test.py) |
| `--flock-ticks` | 1000 | shepherd-free self-organization phase |
| `--max-iterations` | 100000 | herding tick budget |
| `--snapshot-every` | 0 | also snapshot every N ticks (0 = start/end only) |
| `--out` | `images/smoke` | snapshot directory (`images/` is gitignored) |

Experiment config (MODE, MORPHOLOGY, FENCE, TARGET) lives in
`basic/__init__.py` as constants frozen into the JIT-compiled code at import —
there is no runtime switch. Edit the file, rerun the driver, revert. Verified:
MODE=0 (center of mass) and MODE=3 (visible convex hull) both herd to success.

## Direct invocation

For changes to one function in `basic/`, skip the full run:

```bash
.venv/bin/python -c "
from basic.initiation import initiate, initiate_shepherds
from basic.interaction import evolve
import numpy as np
agents = initiate(20, 2, 150, 150, 125)        # (N_sheep, N_shepherd, space_x, space_y, target_size)
shepherds = initiate_shepherds(2, 20, 10.0)    # (N_shepherd, agent_num, L3)
agents, shepherds, idx = evolve(agents, shepherds, 400, 400, 125)
print('OK', agents.shape, shepherds.shape)"
```

Agents are row-per-sheep float arrays; column meanings are commented in
`basic/initiation.py` (e.g. `[:, 21]` = staying flag, `[:, 22]` = hull membership).

## Run (full experiments)

`test.py` is the real experiment runner: 64 reps × 300 sheep × ≤200k
iterations, joblib-parallel across all cores — **hours** of runtime, each
worker paying its own ~20s JIT compile. It takes no CLI args; edit the
constants at the top of `test.py` (`N_SHEEP`, `N_SHEPHERD`, `REPS`,
`ITERATIONS`, `DRAW`, `THREADS`). Results land in gitignored
`results/fence/<MODE>/` as a JSON `.txt` + histogram PNG, written only after
all reps finish.

```bash
PYTHONUNBUFFERED=1 .venv/bin/python test.py   # prints "Starting repetition N" per rep
```

Killing it early leaves no partial results and no live orphan processes
(only reaped-on-exit zombies). `DRAW = True` in test.py additionally needs
ffmpeg (not installed here) to assemble frame videos.

## Test

No pytest suite exists — despite the name, `test.py` is the experiment
runner, and `metric_test.py` / `vision_test.py` / `temp.py` are stale scratch
scripts. The driver above is the smoke test.

## Gotchas

- **Legacy entry points are broken on this branch** — `main.py`,
  `main_interface.py`, `main_interface_metric.py`, `vision_test.py`,
  `metric_test.py` call `initiate()` with 4 args / `initiate_shepherds()`
  with 2; the current signatures (`basic/initiation.py`) take 5 and 3.
  Don't run or document them; use the driver or `test.py`.
- **`NUMBA_DISABLE_JIT=1` silently does nothing** — `basic/__init__.py` runs
  `config.DISABLE_JIT = DEBUG` at import, clobbering the env var. To run
  interpreted (for pdb/coverage), edit `DEBUG = True` in `basic/__init__.py`.
- **No test.py output when piped** — joblib's loky workers block-buffer
  stdout; run with `PYTHONUNBUFFERED=1` to see per-repetition progress.
- **Mode changes need a fresh process** — MODE/MORPHOLOGY/FENCE are baked
  into compiled code at import; re-importing in the same interpreter won't
  pick up edits.

## Troubleshooting

- **`error: externally-managed-environment` from pip**: PEP 668 system
  Python. Create the venv as in Prerequisites; don't use
  `--break-system-packages`.
- **First run appears to hang for ~20s after "initiating..."**: that's the
  numba JIT compile of `evolve`, not a hang. Post-compile it prints tick
  progress every 10k ticks.
