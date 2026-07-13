# Convex-Hull Shepherding — Prior Art & Novelty Positioning

> ⚠️ **LLM-GENERATED PRIOR-ART ASSESSMENT — NOT HUMAN-VERIFIED.**
> Produced on **2026-07-12** by an automated LLM deep-research process (Claude Code
> deep-research harness: fan-out web search → source fetch → multi-vote adversarial
> claim verification → synthesis; 25 claims verified, 24 unanimous / 1 split). **No
> human has checked it, and it is not a patent/novelty search of record.** It rests
> partly on argument-from-absence and is time-bounded to ~early 2025. **Do not rely
> on the novelty verdict for publication or IP decisions without an independent
> literature/patent search.** Verify every cited paper against its primary source.

## What is being assessed

This repo's **convex-hull-based drive-center estimators** (`basic/herd/driver.py`,
`basic/herd/hull.py`), which replace the omniscient global-center-of-mass (GCM)
drive-target of the Strömbom 2014 / Zheng & Romanczuk baseline with sensing-realistic
geometry:

- **Variant A — `MODE = 2` (convex-hull center):** flock center = **mean of the
  convex-hull vertices** of the moving flock (perimeter sheep only; interior ignored).
- **Variant B — `MODE = 3` (visible / occlusion-limited hull):** per-shepherd, keep only
  the **near-side hull arc visible from the shepherd** (Qhull `QG0`-equivalent
  visible-facet selection); center = mean of those visible vertices, biasing it toward
  the near edge (reduced L1); nearest-sheep fallback when nothing is visible.
- **Variant C — `MODE = 4` (nearest-subflock visible hull):** connected-component
  subflocking at the flocking-attraction radius, then attend to the subflock with the
  **nearest visible hull vertex**.

**Unifying claim under test:** driving-reference estimation from partial, sensing-realistic
geometry (perimeter → visible/occluded perimeter → nearest subflock's visible perimeter) as
a principled relaxation of the omniscient-GCM assumption.

## Verdict

**PARTIALLY ANTICIPATED in motivation — but the convex-hull MECHANISM appears NOVEL.**
No published shepherding/herding paper uses the convex hull (mean of hull vertices,
alpha-shape, or Qhull visible facets) to **define the shepherd's driving reference /
center estimate**. The *idea* of relaxing omniscient-GCM toward boundary/visible/subflock
perception has been reached by others, but always via a **different geometric primitive**
(bearings, extremal agents, graph/PSO), never the hull as center estimator.

- Confidence **high** on mechanism-level novelty (hull-as-center-estimator).
- Confidence **medium** on whether a very recent (mid-2025 → 2026) unindexed preprint has
  independently used a hull centroid / alpha-shape drive-center.

## The nearest neighbor — the near-scoop to cite and distinguish

**Li, Ogura & Wakamiya (2025), "Swarm shepherding using bearing-only measurements,"**
*Phil. Trans. R. Soc. A* 383(2289):20240145, DOI 10.1098/rsta.2024.0145 (open access,
PMC11779540). **Same Osaka lab** as the 2023 communication-free multi-shepherd paper. It
independently instantiates the spirit of all three variants — but via **angular / bearing**
geometry, with **no convex hull anywhere**:

| | This repo (hull) | Li et al. 2025 (bearing-only) |
|---|---|---|
| **A: center** | mean of **convex-hull vertices** | average of bearing vectors to **left + right extreme** sheep |
| **B: occlusion** | **visible hull arc** (Qhull `QG0` visible facets) | drop far/interior agents via nearest-first **angular threshold θ_occ** |
| **C: subflock** | connected components @ attraction radius; **nearest visible hull vertex** | angular partition θ_n; **largest angle to goal** |

**This is the reference most likely to be raised against a "first to relax omniscient-CoM"
framing.** The defensible distinction is purely mechanistic: two *extremal angular* agents
vs. the **mean of full hull vertices + visible-facet selection**, and nearest-visible-vertex
subflock choice vs. largest-angle-to-goal.

## Neighborhood map (all verified against sources; each differs)

- **Sebastián, Montijano & Sagüés (2022)**, *Multi-robot Implicit Control of Massive Herds*,
  arXiv:2209.09705 — **uses the convex hull, but only to SELECT which evaders to directly
  control** (convex-hull dynamic clustering + K-Means, furthest evader per cluster); the
  drive reference stays a plain global centroid `x_c = (1/m)Σx_j → x*`. This is the
  **hull-for-selection ≠ hull-for-center** distinction that separates it from Variant A.
- **Auletta, Fiore, Richardson & di Bernardo (2021)**, *Herding stochastic autonomous
  agents*, Auton. Robots, DOI 10.1007/s10514-021-10033-6 — convex hull appears **only as a
  spread metric** (mean-in-time of hull **area**); drive target is the farthest agent / CoM.
- **Hu, Turgut, Krajník, Lennox & Arvin (2020)**, *Occlusion-Based Coordination…*, IEEE
  TCDS, DOI 10.1109/TCDS.2020.3018549 — "occlusion" is a **motion-control positioning
  primitive** (Gauci/Groß paradigm), **not** a perception model estimating a center from a
  visible arc.
- **Mohamed, Elsayed, Hunjet & Abbass (2021)**, *Graph-based Shepherding with Limited
  Sensing Range*, IEEE CEC, DOI 10.1109/CEC45853.2021.9504706 — limited sensing is on the
  **sheep** (causing fragmentation); shepherd uses graph + PSO (DBSCAN), no hull.
- **Gade, Paranjape & Chung (2015)** n-wavefront; **Zhang et al. (2024)** *Distributed
  Outmost Push*, IEEE T-RO; **Tsunoda et al.** local-camera shepherd — all **select** a
  boundary/outermost agent to push (some FOV-limited), **not** center-from-visible-boundary.
- **Liu/Hussein, Elsayed, Abbass et al. (2023)**, *Planning-Assisted… Dispersed Swarms*,
  arXiv:2301.10363 — subflock clustering as **preprocessing** for an ACO/TSP herding
  sequence, not attention keyed on nearest visible boundary.
- **Survey — Long, Sammut, Sgarioto, Garratt & Abbass (2020)**, arXiv:1912.07796 — states
  "most shepherding approaches assume the shepherd(s) have **global knowledge**"; across ~81
  refs, **zero** mentions of convex hull / alpha-shape as a drive reference.

## How to position the contribution

- **Do** frame it as *the convex hull — and its occlusion-limited visible arc — as the
  shepherd's center estimator.* That specific mechanism is clear of the retrieved literature.
- **Don't** lead with *"first to relax omniscient-CoM via limited sensing"* — Li et al. 2025
  (and, weakly, Strömbom's own local-shepherd variant) will be held up against it.
- **Address head-on:** whether Variant B collapses, in the limit, to Li 2025's left/right-
  extreme + θ_occ bearing estimator (both bias toward the near boundary). A direct head-to-
  head showing the hull captures structure two extremal bearings cannot (near-arc curvature,
  multi-shepherd shared-hull geometry) is the strongest defense.

## Residual risks

- Argument-from-absence, time-bounded to ~early 2025; mid-2025 → 2026 preprints uncovered.
- Some IEEE sources (Hu 2020, Mohamed 2021, Zhang 2024) verified at **abstract level only**.
- The Gade n-wavefront detail comes via the 2020 survey, not the primary AIAA page.

## Provenance

Claude Code deep-research harness, run 2026-07-12: 6 search angles (orthodox hull-as-drive-
reference, occlusion/limited-FOV, subflock clustering, nearest-neighbor citation trail,
adjacent computational-geometry visibility, omniscience-assumption sentiment) → 22 sources
fetched → 90 claims extracted → 25 adversarially verified (25 confirmed, 0 refuted).
