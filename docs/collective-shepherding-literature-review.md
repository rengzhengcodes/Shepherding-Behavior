# Collective Shepherding — Literature Review

> ⚠️ **LLM-GENERATED LITERATURE REVIEW — NOT HUMAN-VERIFIED.**
> This document was produced on **2026-07-12** by an automated LLM deep-research
> process (Claude Code's deep-research harness: fan-out web search → source fetch →
> multi-vote adversarial claim verification → synthesis). **No human has checked it.**
> Author lists, venues, DOIs/arXiv ids, dates, and method characterizations may be
> wrong or hallucinated. **Verify every entry against the primary source before
> citing or relying on it.** Each entry is tagged:
> - **[VERIFIED]** — the specific factual claim survived automated 3-vote adversarial
>   cross-checking against fetched sources (still not a substitute for human review).
> - **[SURFACED]** — found in search and judged relevant, but *not* put through formal
>   verification. Treat as a lead to confirm.

## Scope

Recent + foundational work on collective shepherding, focused on models descending from
(a) the **three-zone** sheep/prey agent model (Couzin-style zones of repulsion /
orientation-alignment / attraction) and (b) **center-of-mass (CoM)** shepherding
heuristics (Strömbom et al. 2014 and successors), across four strands: bio-inspired
collective models, swarm robotics, control theory, and learning-based (RL).

## Relevance to this repository

This simulation implements exactly this lineage. `MODE = 0` (`basic/herd/driver.py`) is
described in the README as *"Yating's original model"* — i.e. the center-of-mass
collect/drive shepherding heuristic that Zheng & Romanczuk 2024 (below) formalize, run
over a three-zone-style social-force flock (`basic/interaction.py`,
`basic/herd/forces.py`). Modes 2–4 (convex-hull / visible-hull center estimators) are
this repo's own extensions to how the shepherd *estimates the flock center*.

For a focused prior-art / novelty analysis of those convex-hull extensions — whether they
have been scooped, and the nearest competing work (Li et al. 2025's bearing-only shepherd
and Sebastián et al. 2022's convex-hull clustering) — see
[`novelty-assessment.md`](novelty-assessment.md).

---

## Strongest starting points

1. **Zheng & Romanczuk (2024)** — *Bio-Inspired Agent-Based Model for Collective
   Shepherding* — the direct SCIoI reference and the model this repo is built on.
2. **Jadhav et al., incl. Zheng & Romanczuk (2024)** — real border-collie + sheep
   tracking reproduced by a CoM/barycenter shepherding model (*Communications Biology*).
3. **Poel, Couzin, Romanczuk et al. (2022)** — subcritical escape waves in fish
   (*Science Advances*): the collective-response physics three-zone prey models capture.
4. **Di Lorenzo, Maffettone & di Bernardo (2024)** — continuification/PDE control with
   global-convergence guarantees (strongest recent control-theory entry).
5. **Covone, …, di Bernardo (2025)** — hierarchical PPO for non-cohesive targets
   (strongest recent learning entry).

---

## Foundational lineage

- **[VERIFIED] Couzin, Krause, James, Ruxton & Franks (2002)** — *Collective Memory and
  Spatial Sorting in Animal Groups*, J. Theor. Biol. 218:1–11. Canonical **three-zone**
  model: local **repulsion**, **orientation/alignment**, **attraction**, repulsion
  highest-priority. PDF: https://jmvidal.cse.sc.edu/library/couzin02a.pdf
- **[VERIFIED] Vaughan, Sumpter, Henderson, Frost & Cameron (2000)** — *Experiments in
  automatic flock control* (Robot Sheepdog Project), Robotics and Autonomous Systems
  31:109–117, DOI 10.1016/S0921-8890(99)00084-6. Potential-field controller: robot orbits
  the **flock centre**; goal-repulsion tilts the field so the orbit minimum sits **behind
  the flock relative to the goal**. CoM-targeting precursor Strömbom cites; has drive
  positioning but no explicit collect/drive switch.
- **[VERIFIED] Strömbom, Mann, Nelson, Higgins, Sumpter & King (2014)** — *Solving the
  shepherding problem: heuristics for herding autonomous, interacting agents*, J. R. Soc.
  Interface 11(100):20140719, DOI 10.1098/rsif.2014.0719. **The heuristic this repo
  implements.** Reynolds boids / force-vector model: each sheep sums repulsion from the
  shepherd + attraction to the **local** CoM of its n nearest neighbours + inter-sheep
  repulsion + noise; the shepherd adaptively **switches between COLLECTING** outliers
  **and DRIVING** the aggregated herd from behind its **global** CoM.
- **[VERIFIED] Long, Sammut, Sgarioto, Garratt & Abbass (2020)** — *A Comprehensive Review
  of Shepherding as a Bio-inspired Swarm-Robotics Guidance Approach*, IEEE Access
  (arXiv:1912.07796). Best survey / map of the field.

## Strand 1 — Bio-inspired collective models (Romanczuk / Zheng / SCIoI)

- **[SURFACED — direct hit] Zheng & Romanczuk (2024)** — *Bio-Inspired Agent-Based Model
  for Collective Shepherding*, SAB 2024 / *From Animals to Animats 17*, Springer LNCS vol.
  14993, DOI 10.1007/978-3-031-71533-4_14. Agent-based **sheep-flock** + shepherd heuristic
  with **collecting** (keep flock together) and **driving** (move flock to target) modes —
  the Strömbom collect/drive dichotomy in an SCIoI collective model. Results: **multiple
  shepherds self-organize with no direct communication**; generalizes to **non-cohesive /
  passive agents without self-propulsion**. The paper by Yating Zheng + Pawel Romanczuk;
  the model `MODE = 0` in this repo derives from.
- **[VERIFIED] Jadhav et al., incl. Y. Zheng & P. Romanczuk (2024)** — *Collective
  responses of flocking sheep (Ovis aries) to a herding dog (border collie)*,
  *Communications Biology*, DOI 10.1038/s42003-024-07245-8 (bioRxiv 2024.05.24.595762;
  open data + code). 14 sheep + 1 collie tracked via UWB on a driving task, reproduced by
  an agent-based shepherding model (dog drives from behind; flock **barycenter** central).
  Finding: on short timescales directional info propagates from the **front** rearward,
  not from the rear-approaching dog forward.
- **[VERIFIED] Bartashevich, Herbert-Read, …, Krause & Romanczuk (2024)** — *Collective
  anti-predator escape manoeuvres through optimal attack and avoidance strategies*,
  *Communications Biology*, DOI 10.1038/s42003-024-07267-2 (bioRxiv 2024.03.26.586812).
  Prey use **repulsion / alignment / attraction** over first-Voronoi-shell neighbours;
  predator steers toward the prey group's **centroid (CoM)**. Caveat (only 2-1 vote): uses
  a single crossover distance, not two metric zones — "three-zone" is a loose paraphrase.
- **[VERIFIED] Poel, Winklmayr, …, Couzin & Romanczuk (2022)** — *Subcritical escape waves
  in schooling fish*, *Science Advances* 8:eabm6385, DOI 10.1126/sciadv.abm6385
  (arXiv:2108.05537). Schools are **subcritical** and **decrease distance to criticality
  as perceived risk rises**. Couzin + Romanczuk co-authors.
- **[SURFACED] Klamser & Romanczuk (2021)** — *Collective predator evasion: putting the
  criticality hypothesis to the test*, *PLOS Computational Biology*,
  DOI 10.1371/journal.pcbi.1008832 (arXiv:2009.02079). SCIoI three-zone prey + predator;
  predecessor to the Poel 2022 criticality line.
- **[SURFACED] Mezey, Bastien, Zheng, McKee, Stoll, Hamann & Romanczuk (2025)** — *Purely
  vision-based collective movement of robots*, *npj Robotics* (arXiv:2406.17106). Another
  Zheng-coauthored SCIoI paper; vision-only collective motion on real robots.

## Strand 2 — Swarm robotics (multi-robot / decentralized)

- **[VERIFIED] Sebastian, Montijano & Sagüés (2022)** — *Multi-robot Implicit Control of
  Massive Herds*, arXiv:2209.09705 (ROBOT 2022). **Convex-hull dynamic clustering**
  (Voronoi-inspired) picks *which* evaders to directly control; the rest follow via
  inter-agent repulsion. Caveat: the "decentralized at scale" framing was **refuted 0-3**
  (Implicit Control may be centralized) — treat as swarm-robotics/control hybrid.
- **[VERIFIED] Li, Ogura & Wakamiya (2023)** — *Communication-free shepherding navigation
  with multiple steering agents*, *Frontiers in Control Engineering* 4:989232,
  DOI 10.3389/fcteg.2023.989232 (arXiv:2205.08155). 1–10 shepherds, **no inter-agent
  communication**; each chases an **individual** target sheep rather than the global CoM.
- **[SURFACED] Reactive shepherding along a dynamic path (2024)** — *Scientific Reports*,
  DOI 10.1038/s41598-024-65894-5.
- **[SURFACED] Robotic Shepherding in Cluttered and Unknown Environments using Control
  Barrier Functions (2024)** — arXiv:2407.15701.

## Strand 3 — Control theory (formal guarantees)

- **[VERIFIED] Di Lorenzo, Maffettone & di Bernardo (2024)** — *A Continuification-Based
  Control Solution for Large-Scale Shepherding*, arXiv:2411.04791; *European Journal of
  Control* 86:101324 (2025), DOI 10.1016/j.ejcon.2025.101324. Microscopic dynamics →
  macroscopic **PDE**; leader herders confine follower targets to a goal region via
  indirect interaction. **Global (exponential) convergence guarantees**; mixed-reality
  validation.
- **[VERIFIED] Li, Li, Ji, Sun & Zhao (2025)** — *Multi-Robot Cooperative Herding through
  Backstepping Control Barrier Functions*, arXiv:2507.10249 (*Nonlinear Dynamics*). Two
  CBFs (goal-reaching + collision-avoidance) in one QP → provable **herding-completion +
  safety**. Evaders are **repulsion-only** (no orientation/attraction zone, no CoM switch).
- **[VERIFIED] Varava, Hang, Kragic & Pokorny (2017)** — *Herding by Caging: a Topological
  Approach…*, RSS 2017 (journal ext. *Autonomous Robots* 2021). Herders form a topological
  **"cage"** → **provably correct containment**. Pure potential-field; not Couzin/Strömbom.
- **[VERIFIED] Deng, Li, Ogura & Wakamiya (2023)** — *Collision-Free Shepherding Control of
  a Single Target within a Swarm*, arXiv:2306.12044 (IEEE). **Lyapunov-based** feedback with
  forward-invariance stability + provable collision-free guarantee.
- **[SURFACED] Shepherding control and herdability in complex multiagent systems (2024)** —
  *Phys. Rev. Research* 6, L032012 (arXiv:2307.16797). Defines **herdability**.
- **[SURFACED] Optimal Control Strategies for Multi-Agent Sheep Herding (2025)** —
  arXiv:2510.25115.
- **[SURFACED] Herding stochastic autonomous agents via local control rules and online
  target selection (2021)** — *Autonomous Robots*, DOI 10.1007/s10514-021-10033-6.

## Strand 4 — Learning-based (RL)

- **[VERIFIED] Covone, Napolitano, De Lellis & di Bernardo (2025)** — *Hierarchical
  Policy-Gradient RL for Multi-Agent Shepherding Control of Non-Cohesive Targets*,
  arXiv:2504.02479. Decentralized **PPO**; hierarchical target-**selection** +
  target-**driving**; targets **non-cohesive** (departs from the cohesive-herd CoM
  assumption).
- **[SURFACED] Napolitano, Lama, De Lellis & di Bernardo (2024)** — *Emergent Cooperative
  Strategies for Multi-Agent Shepherding via Reinforcement Learning*, arXiv:2411.05454.
- **[SURFACED] Hierarchical Learning-Based Control for Multi-Agent Shepherding of Stochastic
  Autonomous Agents (2025)** — arXiv:2508.02632.
- **[SURFACED] Decentralized Shepherding of Non-Cohesive Swarms Through Cluttered
  Environments via Deep RL (2025)** — arXiv:2511.21405.
- **[SURFACED] Zhi & Lien (2020)** — *Learning to Herd Agents Amongst Obstacles: Robust
  Shepherding via Deep RL*, arXiv:2005.09476. The **Lien** lineage in RL form.
- **[SURFACED] Nguyen, Nguyen, Barlow, Abbass et al. (2019)** — *A Deep Hierarchical RL
  Learner for Aerial Shepherding of Ground Swarms*, ICONIP 2019,
  DOI 10.1007/978-3-030-36708-4_54.

---

## Yating Zheng — specifically

SCIoI profile: https://www.scienceofintelligence.de/people/yating-zheng/. Three
co-authored papers surfaced:
1. **Zheng & Romanczuk (2024)** — *Bio-Inspired Agent-Based Model for Collective
   Shepherding* (SAB 2024) — the shepherding paper this repo's `MODE = 0` derives from.
2. **Jadhav et al. incl. Zheng & Romanczuk (2024)** — sheep + border-collie
   (*Communications Biology*).
3. **Mezey, …, Zheng, …, Romanczuk (2025)** — vision-based collective robot movement
   (*npj Robotics*).

## Caveats (beyond the top banner)

- Several 2024–2025 items straddle preprint/published versions — cite the journal version
  where noted.
- The two seminal papers (Strömbom 2014, Vaughan 2000) were verified partly via
  secondary/companion sources because journal pages were paywalled — faithful but not the
  exact journal PDFs.
- Many "relation to Strömbom/three-zone" contrasts are the synthesizer's positioning
  glosses, not the papers' own words.

## Provenance

Generated by the Claude Code deep-research harness (6 search angles → 25 sources fetched →
122 claims extracted → 25 adversarially verified: 24 confirmed, 1 refuted). Run date
2026-07-12. Prompted to prioritize the Romanczuk/Zheng SCIoI line and papers descending
from the three-zone and center-of-mass shepherding models.
