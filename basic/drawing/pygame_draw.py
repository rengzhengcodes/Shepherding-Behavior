"""
Headless PyGame frame renderer for the shepherding simulation.

This is a PyGame port of `draw_single`, formerly in `basic/drawing/draw.py`
(draw.py:47-306), and is now the sole frame renderer for the project.
`basic/drawing/draw.py` -- the matplotlib-based renderer this module
replaced -- was deleted when this pygame port landed; it survives in git
history for anyone who needs to consult it directly. Comments throughout
this module cite the specific draw.py line ranges it was ported from
wherever the translation is non-obvious.

Design goals
------------
- **Display-free by design.** This module never calls `pygame.init()` or
  any `pygame.display.*` function -- only `pygame.Surface`,
  `pygame.draw`/`pygame.gfxdraw`, and `pygame.font`, all of which work
  without an initialized video subsystem. That makes it safe to import and
  call from inside joblib/loky worker processes with no display, X server,
  or DRM device available. (draw.py's `draw_dynamic` docstring documented an
  analogous, if milder, concern for matplotlib/pyplot figures leaking
  across reused workers; an accidental `pygame.display.set_mode()` call in
  a headless container can outright fail or hang, so this module simply
  never makes one.)
- **Read-only on sim state, RNG-free (no-drift guarantee).** `render_frame`
  never writes to `swarm` or `shepherds`, including through views --
  boolean-mask/argsort/fancy-index selections (e.g. `swarm[swarm[:, 21] ==
  0]`) always return freshly allocated arrays in numpy and never alias the
  input. This module also never calls `random` or `np.random`. The
  simulation keeps a single global RNG in lockstep across every tick of
  every run; a stray draw from a renderer invoked mid-run (e.g. for a live
  preview) would desynchronize it from a resumed or parallel run using the
  same seed.

Deliberate visual changes vs. draw.py
--------------------------------------
1. **Uniform aspect ratio (true circles).** draw.py's matplotlib Axes never
   called `ax.set_aspect("equal")`, so its 8x6in @ 150dpi figure scaled
   world units by a different factor per axis (~1.34:1) -- sheep bodies and
   the target rendered as ellipses, not circles. `Camera.fit` here always
   picks a single scale shared by both axes (the smaller of the two
   per-axis fits), so this renderer always draws true circles.
2. **No bogus center-of-mass star at a degenerate mean.** draw.py's numba
   `calculate_mass_center` kernel (draw.py:24-43) returned `(0, 0)` when no
   sheep were moving (its accumulator's untouched initial value), which
   draw.py then plotted as a real red star at the world origin -- so, e.g.,
   the final frame of a run where every sheep has settled to `state ==
   1` (staying) drew a misleading star at (0, 0), which was never an
   actual computed center of mass. `flock_center` (below) returns `None`
   in that case, and `render_frame` skips the star entirely rather than
   drawing a wrong one.
3. **No polyline joining the shepherds, and per-shepherd body colors.**
   draw.py:131-136 drew all shepherd bodies with ONE `ax.plot((M,) xs,
   (M,) ys, marker="o", alpha=0.2)` call -- a single `Line2D`, which meant
   (a) a translucent solid line visibly connecting shepherd 0 -> 1 -> ...
   in array order, and (b) every marker painted with the FIRST shepherd's
   drive/collect color (the per-shepherd color cycle at draw.py:130 was
   consumed once by the single line). Both were artifacts of the one-call
   plotting style, not intent: this renderer draws no connecting line and
   colors each shepherd by its own drive/collect state (col 13), which is
   what the original color-cycle code was visibly trying to do.
4. **Z-order is paint order, not matplotlib's zorder sort.** matplotlib
   renders artists sorted by their `zorder` attribute (patches/collections
   at 1 below all `Line2D`s at 2), so draw.py's target circle, fence arc,
   and mode-hull outlines rendered UNDER every line/marker regardless of
   call order. This renderer paints strictly in call order (the element
   sequence in `render_frame`), so those outlines now sit above earlier
   elements. Purely cosmetic; kept because a faithful two-pass zorder
   emulation would complicate the renderer for no informational gain.
5. **Off-window and non-finite positions are skipped, not clipped.**
   matplotlib silently clipped far-outside points to the axes and dropped
   NaNs. `pygame.gfxdraw`'s C API takes signed 16-bit coordinates, so this
   renderer skips primitives whose pixel coordinates are non-finite or
   beyond +/-`_COORD_LIMIT_PX` instead -- one transiently exploded agent
   must not crash (and thereby abort) an hours-long video encode.

Data contract (read-only)
--------------------------
`swarm` : ndarray, shape (N, 25), float64
    Column 0/1 position (x, y); 2 heading in radians, WORLD convention
    (y-up); 7 body radius in WORLD units (homogeneous across all sheep --
    only row 0 is read); 21 agent state (1.0 staying, 0.0 moving); 22
    convex-hull membership (0.0 = not on hull, else a 1-based CCW rank
    along the hull); 23 a uint64 visibility bitmask bit-smuggled into a
    float64 lane (bit i set means shepherd i currently sees this sheep;
    recovered via `swarm[:, 23].view("uint64")`, which is valid even
    though that column is a non-contiguous strided view of `swarm`,
    because the view's itemsize (8 bytes) matches the source dtype's);
    24 subflock id (mode 4 only; -1 for staying sheep).
`shepherds` : ndarray, shape (M, 22), float64
    Column 0/1 position (x, y); 2 heading in radians; 13 state (1.0 drive,
    0.0 collect); 14/15 current drive-or-collect point (x, y); 16 index
    (float, cast to int) of the sheep currently being collected.

See draw.py's `draw_single` (removed; see the note above) for the reference
implementation this module was ported from.
"""

from __future__ import annotations

import math
import os

# Must be set before `import pygame`: pygame otherwise prints a "Hello from
# the pygame community" support-prompt banner to stdout on import, which
# would pollute the stdout of any calling process (e.g. a joblib worker's
# captured logs).
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame
import pygame.gfxdraw
from dataclasses import dataclass
from scipy.spatial import ConvexHull


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class RenderConfig:
    """
    Immutable bundle of window geometry, palette, and pixel-space sizing
    constants consumed by `render_frame`.

    Colors are matplotlib's named-color palette (as used throughout
    draw.py, e.g. `color="g"`) converted to 0-255 RGB triples; alpha is
    applied separately at each draw call via the `alpha_*` levels below,
    which are matplotlib alpha values (0-1) baked into 0-255 integers
    (`round(alpha * 255)`).

    Parameters
    ----------
    window_size : tuple of int
        `(width, height)` of the full render target in pixels, HUD band
        included. Both dimensions are even by default (960x1000) so the
        frame is directly yuv420p-safe if piped to an H.264 encoder, as
        draw.py's `_ffmpeg_pipe_cmd` does for its own frames.
    hud_height : int
        Height in pixels of the HUD text band reserved at the top of the
        window; the plot area occupies `window_size[1] - hud_height`.
    white, green, blue, red, cyan, yellow, magenta, black : tuple of int
        Opaque `(r, g, b)` triples, matplotlib-equivalent to `"w"`/`"g"`/
        `"b"`/`"r"`/`"cyan"`/`"y"`/`"m"`/`"k"`.
    alpha_80, alpha_20, alpha_25, alpha_50 : int
        0-255 alpha levels equivalent to matplotlib alphas 0.8, 0.2, 0.25,
        and 0.5 respectively (`round(alpha * 255)`).
    arrow_len_px : float
        Fixed pixel length of heading arrows (both sheep and shepherds).
        Approximates draw.py's `ax.quiver(..., scale_units="inches",
        scale=10)` heading indicator, which renders as a roughly
        fixed-size glyph at draw.py's fixed `DRAW_DPI` -- 14px chosen to
        match that glyph's approximate rendered length.
    arrow_head_px : float
        Length in pixels of each of the two arrowhead barbs.
    shepherd_radius_px : float
        FIXED pixel radius for shepherd body markers. NOT a world-space
        radius passed through `Camera.px` -- draw.py's original
        `markersize=swarm[0, 7]` is in matplotlib POINTS (a screen-space
        unit), not world/data units, so a fixed pixel size is the faithful
        port, not a `camera.px(...)` call.
    star_radius_px : float
        Outer-vertex pixel radius for the center-of-mass / hull-center /
        visibility-center star markers (draw.py's `markersize=5` "*"
        markers).
    target_star_radius_px : float
        Outer-vertex pixel radius for the target star marker (draw.py's
        `ax.plot(*target[:2], "b*")`, which uses matplotlib's default
        marker size of 6 rather than the explicit `markersize=5` used
        elsewhere).
    hud_font_size : int
        Point size forwarded to `pygame.font.Font(None, hud_font_size)`
        for HUD text.
    """

    # --- Window geometry -----------------------------------------------
    window_size: tuple[int, int] = (960, 1000)
    hud_height: int = 40

    # --- Palette (matplotlib named-color equivalents, 0-255 RGB) -------
    white: tuple[int, int, int] = (255, 255, 255)
    green: tuple[int, int, int] = (0, 128, 0)  # matplotlib "g"
    blue: tuple[int, int, int] = (0, 0, 255)  # matplotlib "b"
    red: tuple[int, int, int] = (255, 0, 0)  # matplotlib "r"
    cyan: tuple[int, int, int] = (0, 255, 255)  # matplotlib "cyan"
    yellow: tuple[int, int, int] = (191, 191, 0)  # matplotlib "y"
    magenta: tuple[int, int, int] = (191, 0, 191)  # matplotlib "m"
    black: tuple[int, int, int] = (0, 0, 0)  # matplotlib "k"

    # --- Alpha levels (matplotlib alpha -> 0-255, round(alpha * 255)) --
    alpha_80: int = 204  # 0.8
    alpha_20: int = 51  # 0.2
    alpha_25: int = 64  # 0.25
    alpha_50: int = 128  # 0.5

    # --- Pixel-space sizing ----------------------------------------------
    arrow_len_px: float = 14.0
    arrow_head_px: float = 5.0
    shepherd_radius_px: float = 5.0
    star_radius_px: float = 5.0
    target_star_radius_px: float = 6.0
    hud_font_size: int = 22


DEFAULT_CONFIG = RenderConfig()


def _rgba(rgb: tuple[int, int, int], alpha: int) -> tuple[int, int, int, int]:
    """
    Combine an opaque RGB triple from `RenderConfig` with an alpha level.

    Parameters
    ----------
    rgb : tuple of int
        `(r, g, b)`, each 0-255.
    alpha : int
        Alpha, 0-255 (typically one of `RenderConfig`'s `alpha_*` fields,
        or `255` for fully opaque).

    Returns
    -------
    tuple of int
        `(r, g, b, alpha)`, suitable for any `pygame.draw`/`pygame.gfxdraw`
        color argument.
    """
    return (rgb[0], rgb[1], rgb[2], alpha)


# =============================================================================
# Camera: world <-> screen transform
# =============================================================================


@dataclass(frozen=True)
class Camera:
    """
    Immutable world-to-screen affine transform for one rendered frame.

    Maps the fixed world window `[xmin, xmax] x [ymin, ymax]` (reproducing
    draw.py's `ax.set_xlim`/`ax.set_ylim`, see `Camera.fit`) onto a pixel
    rectangle below a HUD band, using a single UNIFORM scale factor shared
    by both axes (see module docstring, "Deliberate visual changes", item
    1) and a y-flip (world y grows up; screen/pygame y grows down).

    Construct via `Camera.fit`; the field-by-field constructor is exposed
    for testing/introspection, but callers should not normally need it.

    Attributes
    ----------
    xmin, xmax, ymin, ymax : float
        World-space window bounds, in world units.
    scale : float
        Pixels per world unit, identical on both axes.
    plot_rect : pygame.Rect
        Pixel-space rectangle the world window is mapped into (excludes
        the HUD band at the top of the window).
    """

    xmin: float
    xmax: float
    ymin: float
    ymax: float
    scale: float
    plot_rect: pygame.Rect

    @staticmethod
    def fit(
        boundary: tuple[int, int],
        window_size: tuple[int, int],
        hud_height: int,
    ) -> Camera:
        """
        Build a `Camera` that fits draw_single's world window into a pixel
        window, below a HUD band.

        Parameters
        ----------
        boundary : tuple of int
            `(boundary_x, boundary_y)`. The world window is
            `[-boundary_x // 4, boundary_x] x [-boundary_y // 4,
            boundary_y]`, reproducing draw.py's `ax.set_xlim(xmin=
            -boundary[0] // 4, xmax=boundary[0])` / `ax.set_ylim(...)`
            (draw.py:283-284) exactly -- including Python's operator
            precedence: `-boundary_x // 4` parses as `(-boundary_x) // 4`
            (a *floor* division of the negated value), NOT
            `-(boundary_x // 4)`. For `boundary_x = 825` this is `-207`,
            not `-206`.
        window_size : tuple of int
            `(width, height)` of the full render target in pixels, HUD
            band included.
        hud_height : int
            Height in pixels of the HUD band reserved at the top of the
            window; the plot area is `window_size[1] - hud_height` tall.

        Returns
        -------
        Camera
            A camera whose `scale` is the largest value that fits the
            whole world window inside the plot rect on both axes
            simultaneously (uniform aspect -- see class docstring).

        Notes
        -----
        Pure function; `boundary`/`window_size` are read, never mutated.
        """
        boundary_x, boundary_y = boundary
        # NOTE the precedence -- see docstring above.
        xmin = float(-boundary_x // 4)
        xmax = float(boundary_x)
        ymin = float(-boundary_y // 4)
        ymax = float(boundary_y)
        world_w = xmax - xmin
        world_h = ymax - ymin

        plot_rect = pygame.Rect(
            0, hud_height, window_size[0], window_size[1] - hud_height
        )

        # Design: UNIFORM scale (min of the two per-axis fits), unlike
        # draw.py's matplotlib Axes, which never set an equal aspect ratio
        # -- see module docstring, "Deliberate visual changes", item 1.
        scale = min(plot_rect.width / world_w, plot_rect.height / world_h)

        return Camera(
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, scale=scale, plot_rect=plot_rect
        )

    def to_screen(self, xy) -> np.ndarray:
        """
        Convert world-space point(s) to float pixel coordinates.

        Parameters
        ----------
        xy : array_like, shape (..., 2)
            World-space point(s); the trailing axis holds (x, y). Accepts
            a single point (shape `(2,)`) or a batch (shape `(N, 2)`, or
            any shape whose last axis has length 2).

        Returns
        -------
        numpy.ndarray, shape (..., 2), dtype float64
            Pixel-space point(s) with the same leading shape as `xy`. Note
            the y-flip: world y grows up, pygame y grows down, so this is
            NOT a pure scale-and-translate on the y axis.

        Notes
        -----
        Pure function; does not mutate `xy`. `xy` is converted via
        `np.asarray(xy, dtype=np.float64)` (a view when `xy` is already a
        float64 ndarray, a copy otherwise); the returned array is always
        freshly allocated.
        """
        points = np.asarray(xy, dtype=np.float64)
        screen_x = self.plot_rect.left + (points[..., 0] - self.xmin) * self.scale
        # y-flip: world y grows up, screen y grows down.
        screen_y = self.plot_rect.bottom - (points[..., 1] - self.ymin) * self.scale
        return np.stack([screen_x, screen_y], axis=-1)

    def px(self, world_len: float) -> float:
        """
        Convert a world-space length (e.g. a radius) to a pixel length.

        Parameters
        ----------
        world_len : float
            A length in world units. This is a length, not a point -- no
            translation or y-flip is applied, only the uniform `scale`
            factor.

        Returns
        -------
        float
            The equivalent length in pixels.
        """
        return world_len * self.scale


# =============================================================================
# Drawing primitives (module-internal; pixel-space inputs only)
# =============================================================================


# pygame.gfxdraw's C API takes signed 16-bit (Sint16) coordinates and raises
# OverflowError beyond them; int(round(...)) of a NaN raises ValueError.
# Primitives whose coordinates fail _coords_drawable are skipped outright
# (matplotlib clipped instead -- see the module docstring's
# deliberate-visual-changes item 5). Kept comfortably under the 32767 hard
# limit so radii/arrowhead offsets added after the check cannot overflow.
_COORD_LIMIT_PX = 30000.0


def _coords_drawable(*values: float) -> bool:
    """
    Whether every value is finite and within `+/-_COORD_LIMIT_PX`.

    Parameters
    ----------
    *values : float
        Pixel-space coordinates (or lengths) about to be handed to a
        drawing primitive.

    Returns
    -------
    bool
        `True` iff all values are safe to int-round and pass to
        `pygame.gfxdraw`/`pygame.draw`.
    """
    return all(math.isfinite(v) and abs(v) <= _COORD_LIMIT_PX for v in values)


def _alpha_circle(
    surface: pygame.Surface,
    center,
    radius: float,
    fill_rgba: tuple[int, int, int, int] | None = None,
    edge_rgba: tuple[int, int, int, int] | None = None,
) -> None:
    """
    Draw a circle with an alpha-blended fill and/or an antialiased 1px
    edge.

    Parameters
    ----------
    surface : pygame.Surface
        Target surface, mutated in place -- the only mutation any function
        in this module performs (never `swarm`/`shepherds`).
    center : array_like, shape (2,)
        Pixel-space center. Rounded to the nearest int pixel.
    radius : float
        Pixel-space radius. Rounded to the nearest int pixel and clamped
        to a minimum of 1, so a world radius mapping to under half a pixel
        (e.g. a very small `camera.px(...)` result at a small window size)
        still renders as a visible dot instead of vanishing.
    fill_rgba : tuple of int, optional
        `(r, g, b, a)`, each 0-255. `None` (default) draws no fill --
        matplotlib's `facecolor="none"` equivalent.
    edge_rgba : tuple of int, optional
        `(r, g, b, a)`, each 0-255. `None` (default) draws no edge.

    Notes
    -----
    Uses `pygame.gfxdraw.filled_circle` for the fill and
    `pygame.gfxdraw.aacircle` for the edge. `gfxdraw` blends RGBA directly
    into the target surface -- no intermediate per-primitive scratch
    surface is needed -- and z-order is exactly call order. (Note that
    matplotlib's was NOT: it sorted artists by `zorder`; see the module
    docstring's deliberate-visual-changes item 4.)

    A center that is non-finite or outside `+/-_COORD_LIMIT_PX` makes this
    a guarded no-op: gfxdraw's C API takes signed 16-bit coordinates and
    raises OverflowError beyond them (and `int(nan)` raises ValueError),
    where matplotlib merely clipped -- see deliberate-visual-changes item 5.
    """
    cx_f, cy_f, r_f = float(center[0]), float(center[1]), float(radius)
    if not _coords_drawable(cx_f, cy_f, r_f):
        return
    cx = int(round(cx_f))
    cy = int(round(cy_f))
    r = max(1, int(round(r_f)))
    if fill_rgba is not None:
        pygame.gfxdraw.filled_circle(surface, cx, cy, r, fill_rgba)
    if edge_rgba is not None:
        pygame.gfxdraw.aacircle(surface, cx, cy, r, edge_rgba)


# Barb half-angle for arrowheads, in radians. A module constant rather than
# an `_arrow` parameter or `RenderConfig` field because the spec fixes this
# value; it is not meant to be tuned per call site.
_ARROW_BARB_SPREAD_RAD = 0.45


def _arrow(
    surface: pygame.Surface,
    start,
    world_angle: float,
    length_px: float,
    color: tuple[int, int, int, int],
    head_px: float,
) -> None:
    """
    Draw a fixed-length heading arrow: an antialiased shaft plus a
    two-barb head.

    Ported from draw.py's per-row `ax.quiver` heading indicator
    (draw.py:112-124 for sheep, draw.py:139-151 for shepherds), but as a
    constant PIXEL length rather than a world-space quiver vector --
    `ax.quiver(..., scale_units="inches", scale=10)` at draw.py's fixed
    `DRAW_DPI` renders as a roughly fixed-size screen glyph, which a fixed
    pixel length approximates directly without needing a DPI concept.

    Parameters
    ----------
    surface : pygame.Surface
        Target surface, mutated in place.
    start : array_like, shape (2,)
        Pixel-space arrow origin (the agent's screen position).
    world_angle : float
        Heading, in radians, WORLD convention (y-up; matches
        `swarm`/`shepherds` column 2). Converted internally to screen
        direction `(cos(theta), -sin(theta))` -- see module docstring's
        y-flip note. Callers must pass the raw world-frame heading; this
        function performs the flip so it is never duplicated ad hoc.
    length_px : float
        Shaft length in pixels.
    color : tuple of int
        `(r, g, b, a)`, each 0-255. Applied to both shaft and barbs.
    head_px : float
        Length in pixels of each of the two head barbs.

    Notes
    -----
    Pure side effect on `surface`; no return value.
    """
    sx, sy = float(start[0]), float(start[1])
    # Guarded no-op for exploded/NaN agents (see _coords_drawable): a NaN
    # heading would also propagate NaN endpoints into pygame.draw.aaline.
    if not _coords_drawable(sx, sy) or not math.isfinite(world_angle):
        return
    shaft_dx = math.cos(world_angle)
    shaft_dy = -math.sin(world_angle)  # y-flip: world angle -> screen dir
    tip_x = sx + length_px * shaft_dx
    tip_y = sy + length_px * shaft_dy
    pygame.draw.aaline(surface, color, (sx, sy), (tip_x, tip_y))

    # Two head barbs, drawn backward from the tip. Each barb's own
    # direction is derived from a world angle offset by
    # +/- _ARROW_BARB_SPREAD_RAD from the shaft's, y-flipped the same way
    # the shaft is, then subtracted from the tip (not added) so the barbs
    # trail back toward the shaft's origin, forming the usual arrowhead V.
    for sign in (-1.0, 1.0):
        barb_angle = world_angle + sign * _ARROW_BARB_SPREAD_RAD
        barb_dx = math.cos(barb_angle)
        barb_dy = -math.sin(barb_angle)  # y-flip applies to the barb too
        barb_x = tip_x - head_px * barb_dx
        barb_y = tip_y - head_px * barb_dy
        pygame.draw.aaline(surface, color, (tip_x, tip_y), (barb_x, barb_y))


# 5-point star geometry: 10 vertices alternating outer/inner radius.
_STAR_POINTS = 5
_STAR_INNER_RATIO = 0.42


def _star(
    surface: pygame.Surface,
    center,
    radius_px: float,
    color: tuple[int, int, int, int],
) -> None:
    """
    Draw a filled, antialiased-edge 5-point star (the pixel-space
    equivalent of matplotlib's `"*"` marker, used throughout draw.py for
    center-of-mass, hull-center, visibility-center, and target markers).

    Parameters
    ----------
    surface : pygame.Surface
        Target surface, mutated in place.
    center : array_like, shape (2,)
        Pixel-space center.
    radius_px : float
        Outer-vertex radius in pixels. Inner vertices sit at
        `_STAR_INNER_RATIO * radius_px`.
    color : tuple of int
        `(r, g, b, a)`, each 0-255. Used for both fill and edge.

    Notes
    -----
    Built as a 10-vertex polygon alternating outer and inner radius,
    starting from the top (screen-up) vertex, so the star always renders
    "point up" regardless of any agent heading -- matplotlib's `"*"`
    marker is likewise heading-independent.
    """
    cx, cy = float(center[0]), float(center[1])
    # Guarded no-op for non-finite/out-of-range centers (gfxdraw polygons
    # share the Sint16 coordinate limit; see _coords_drawable).
    if not _coords_drawable(cx, cy):
        return
    inner_radius_px = _STAR_INNER_RATIO * radius_px

    points = []
    n_vertices = 2 * _STAR_POINTS
    for k in range(n_vertices):
        # theta = pi/2 is screen-up (see the y-flip below); stepping by
        # 1/n_vertices of a full turn per vertex traces 5 outer + 5 inner
        # vertices into a 5-pointed star.
        theta = math.pi / 2 + k * (2 * math.pi / n_vertices)
        r = radius_px if k % 2 == 0 else inner_radius_px
        x = cx + r * math.cos(theta)
        y = cy - r * math.sin(theta)  # y-flip
        # gfxdraw's polygon primitives require integer vertex coordinates.
        points.append((int(round(x)), int(round(y))))

    pygame.gfxdraw.filled_polygon(surface, points, color)
    pygame.gfxdraw.aapolygon(surface, points, color)


def _dashed_line(
    surface: pygame.Surface,
    color: tuple[int, int, int, int],
    p0,
    p1,
    dash_px: float = 6,
    gap_px: float = 4,
    width: int = 2,
) -> None:
    """
    Draw a dashed line segment from `p0` to `p1`.

    Parameters
    ----------
    surface : pygame.Surface
        Target surface, mutated in place.
    color : tuple of int
        `(r, g, b, a)`, expected to be opaque (`a == 255`) -- every caller
        of this primitive in `render_frame` ports a `linestyle=":"` use
        from draw.py, all of which are opaque in the original, so this
        primitive draws with `pygame.draw.line` (no alpha-blend cost)
        rather than `gfxdraw`.
    p0, p1 : array_like, shape (2,)
        Pixel-space endpoints.
    dash_px : float, optional
        Length of each dash, in pixels. Default 6.
    gap_px : float, optional
        Length of each gap between dashes, in pixels. Default 4.
    width : int, optional
        Line width in pixels. Default 2.

    Notes
    -----
    A zero-length segment (`p0 == p1`, within floating-point tolerance) is
    a guarded no-op: its direction cannot be normalized, and there is
    nothing meaningful to dash.
    """
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    # Guarded no-op for non-finite endpoints: an infinite endpoint would
    # make `length` infinite and the dash loop below unbounded.
    if not (
        math.isfinite(x0) and math.isfinite(y0)
        and math.isfinite(x1) and math.isfinite(y1)
    ):
        return
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return
    ux, uy = dx / length, dy / length
    period = dash_px + gap_px

    dash_start = 0.0
    while dash_start < length:
        dash_end = min(dash_start + dash_px, length)
        start_pt = (x0 + ux * dash_start, y0 + uy * dash_start)
        end_pt = (x0 + ux * dash_end, y0 + uy * dash_end)
        pygame.draw.line(surface, color, start_pt, end_pt, width)
        dash_start += period


def _dashed_polygon(
    surface: pygame.Surface,
    color: tuple[int, int, int, int],
    points,
    dash_px: float = 6,
    gap_px: float = 4,
    width: int = 2,
) -> None:
    """
    Draw a closed dashed polygon through `points`, in order.

    Parameters
    ----------
    surface : pygame.Surface
        Target surface, mutated in place.
    color : tuple of int
        Opaque `(r, g, b, a)`, forwarded to `_dashed_line` per edge.
    points : array_like, shape (N, 2)
        Pixel-space vertices, in order. The polygon is closed
        automatically -- an edge is drawn from the last point back to the
        first.
    dash_px, gap_px, width
        Forwarded to `_dashed_line`.

    Notes
    -----
    No-op if `points` is empty. This module's callers only ever invoke
    this with 2 or more points (see the hull-drawing size guards in
    `render_frame`); a single point would trivially no-op (its one "edge"
    is a zero-length point-to-itself segment, itself guarded by
    `_dashed_line`).
    """
    n = len(points)
    if n == 0:
        return
    for i in range(n):
        _dashed_line(
            surface, color, points[i], points[(i + 1) % n], dash_px, gap_px, width
        )


# Number of sampled points used to approximate a circular arc as a
# polyline. See `_arc_lines` docstring for why this is sampled rather than
# drawn with `pygame.draw.arc`.
_ARC_SAMPLES = 64


def _arc_lines(
    surface: pygame.Surface,
    color: tuple[int, int, int, int],
    center,
    radius_px: float,
    theta0: float,
    theta1: float,
    width: int = 2,
) -> None:
    """
    Draw a circular arc from `theta0` to `theta1` as a sampled polyline.

    Parameters
    ----------
    surface : pygame.Surface
        Target surface, mutated in place.
    color : tuple of int
        Opaque `(r, g, b, a)`.
    center : array_like, shape (2,)
        Pixel-space arc center.
    radius_px : float
        Pixel-space arc radius.
    theta0, theta1 : float
        Start/end angle, in radians, WORLD convention (y-up); converted to
        screen coordinates with the same `cos`/`-sin` y-flip used
        elsewhere in this module. May be given in either order.
    width : int, optional
        Line width in pixels. Default 2.

    Notes
    -----
    Design: deliberately NOT `pygame.draw.arc`. That primitive draws
    visible moire "holes" along the arc at `width > 1`, and its angle
    convention does not compose cleanly with this module's y-flip (it is
    easy to get the sweep direction backwards). Sampling `_ARC_SAMPLES`
    points ourselves with `numpy.linspace` and drawing them with
    `pygame.draw.lines` sidesteps both problems, and reuses the same
    angle-sampling approach as `_star`.
    """
    cx, cy = float(center[0]), float(center[1])
    thetas = np.linspace(theta0, theta1, _ARC_SAMPLES)
    xs = cx + radius_px * np.cos(thetas)
    ys = cy - radius_px * np.sin(thetas)  # y-flip
    points = list(zip(xs.tolist(), ys.tolist()))
    pygame.draw.lines(surface, color, False, points, width)


_FONT_CACHE: dict[int, pygame.font.Font] = {}


def _font(size: int) -> pygame.font.Font:
    """
    Return a cached `pygame.font.Font` at `size`, lazily initializing the
    font subsystem on first use.

    Parameters
    ----------
    size : int
        Point size, forwarded to `pygame.font.Font(None, size)`. `None`
        selects pygame's bundled default font, which has no fontconfig or
        system-font dependency and so behaves identically in this
        no-display, potentially font-less container environment.

    Returns
    -------
    pygame.font.Font
        A font object, reused across calls with the same `size`.

    Notes
    -----
    Calls `pygame.font.init()` (never `pygame.init()` -- see module
    docstring) on first use only. The font subsystem does not require a
    display and is safe under this module's display-free contract.
    """
    if size not in _FONT_CACHE:
        if not pygame.font.get_init():
            pygame.font.init()
        _FONT_CACHE[size] = pygame.font.Font(None, size)
    return _FONT_CACHE[size]


# =============================================================================
# Public helpers
# =============================================================================


def flock_center(swarm: np.ndarray) -> tuple[float, float] | None:
    """
    Compute the center of mass of currently-moving sheep.

    Parameters
    ----------
    swarm : numpy.ndarray, shape (N, 25)
        The flock agent array (see module docstring's data contract).
        Column 21 is agent state: `1.0` staying, `0.0` moving. Read-only
        -- never mutated.

    Returns
    -------
    tuple of (float, float), or None
        `(mean_x, mean_y)` over rows with `swarm[:, 21] == 0`, or `None`
        when there are no moving sheep (including when `swarm` has zero
        rows).

    Notes
    -----
    A pure, read-only port of draw.py's numba `calculate_mass_center`
    kernel (draw.py:24-43), with one deliberate behavior change: that
    kernel returned `(0, 0)` for the empty case (its accumulator's initial
    value), which draw.py then plotted as a real star at the world origin --
    misleadingly, since `(0, 0)` was never an actual computed center of
    mass. Returning `None` here lets `render_frame` skip the star outright
    in that case; see module docstring, "Deliberate visual changes", item
    2.

    Examples
    --------
    >>> import numpy as np
    >>> swarm = np.zeros((2, 25))
    >>> swarm[0, :2] = (0.0, 0.0)
    >>> swarm[1, :2] = (10.0, 20.0)
    >>> swarm[:, 21] = 0.0  # both moving
    >>> flock_center(swarm)
    (5.0, 10.0)
    >>> swarm[:, 21] = 1.0  # both staying -> no moving sheep
    >>> flock_center(swarm) is None
    True
    """
    moving = swarm[swarm[:, 21] == 0]
    if moving.shape[0] == 0:
        return None
    return float(np.mean(moving[:, 0])), float(np.mean(moving[:, 1]))


def _draw_visibility_rays(
    surface: pygame.Surface,
    swarm: np.ndarray,
    shepherd_xy: np.ndarray,
    camera: Camera,
    config: RenderConfig,
) -> None:
    """
    Draw each shepherd's magenta visibility rays and the center-of-mass
    star of the sheep it can currently see. Shared by mode 3 and mode 4.

    Parameters
    ----------
    surface : pygame.Surface
        Target surface, mutated in place.
    swarm : numpy.ndarray, shape (N, 25)
        Flock agent array. Read-only; column 23 is read (never written)
        via a `uint64` view to recover the per-sheep visibility bitmask.
    shepherd_xy : numpy.ndarray, shape (M, 2)
        Pre-computed pixel-space shepherd positions (`camera.to_screen`
        already applied), indexed in the same order as `swarm[:, 23]`'s
        bits -- bit `i` corresponds to `shepherd_xy[i]`.
    camera : Camera
        Used to project visible sheep into pixel space.
    config : RenderConfig
        Supplies the magenta ray/star colors and star radius.

    Notes
    -----
    Ported from draw.py's mode-3 branch (draw.py:200-227) and reused
    verbatim for mode 4 (draw.py:245-274) as a single shared helper here.
    draw.py's own comment at draw.py:245-250 documented why mode 4 must
    `enumerate(shepherds)` for its bit index rather than reusing the
    stale loop variable left over from the flock-hull loop that precedes
    it -- both call sites of this helper pass `enumerate`-derived indices
    implicitly via `shepherd_xy`'s row order, so that bug class cannot
    recur here.

    Deliberate fix vs. draw.py: when a shepherd sees zero sheep, draw.py
    always plotted `np.mean(<empty slice>)` -> `nan` (silently dropped by
    matplotlib, per draw.py:218-221's own comment declining to "fix" that
    there). `_star`'s integer rounding of `nan` would raise in Python
    (`int(nan)` raises `ValueError`), so this port instead skips that
    shepherd's contribution outright when it sees no sheep.
    """
    # Computed once per call (not per shepherd): the bitmask does not
    # depend on the loop variable below, only on `swarm`.
    bitmask = swarm[:, 23].view("uint64")
    ray_color = _rgba(config.magenta, config.alpha_25)
    star_color = _rgba(config.magenta, 255)

    for i in range(shepherd_xy.shape[0]):
        visible_mask = (bitmask & (np.uint64(1) << np.uint64(i))) != 0
        visible = swarm[visible_mask]
        if visible.shape[0] == 0:
            # See "Deliberate fix" note above: skip rather than crash.
            continue

        visible_screen = camera.to_screen(visible[:, :2])
        sx_f = float(shepherd_xy[i, 0])
        sy_f = float(shepherd_xy[i, 1])
        # gfxdraw.line shares the Sint16 coordinate limit -- skip the whole
        # shepherd on a bad origin, and individual rays on bad endpoints
        # (see _coords_drawable).
        if not _coords_drawable(sx_f, sy_f):
            continue
        sx, sy = int(round(sx_f)), int(round(sy_f))
        for j in range(visible_screen.shape[0]):
            ex_f = float(visible_screen[j, 0])
            ey_f = float(visible_screen[j, 1])
            if not _coords_drawable(ex_f, ey_f):
                continue
            pygame.gfxdraw.line(surface, sx, sy, int(round(ex_f)), int(round(ey_f)), ray_color)

        center_world = np.array(
            [np.mean(visible[:, 0]), np.mean(visible[:, 1])]
        )
        center_screen = camera.to_screen(center_world)
        _star(surface, center_screen, config.star_radius_px, star_color)


# =============================================================================
# render_frame / save_frame_png
# =============================================================================


def render_frame(
    surface: pygame.Surface,
    swarm: np.ndarray,
    shepherds: np.ndarray,
    boundary: tuple[int, int],
    target: tuple[int, int, int],
    mode: int,
    *,
    hud: str | None = None,
    fence: tuple[float, float] | None = None,
    config: RenderConfig = DEFAULT_CONFIG,
) -> None:
    """
    Render one frame of the simulation into `surface`.

    A pygame port of `draw_single` (draw.py:47-306); see the module
    docstring for the two deliberate visual differences from that
    matplotlib renderer, and for the read-only / RNG-free guarantees this
    function upholds.

    Parameters
    ----------
    surface : pygame.Surface
        Target surface, mutated in place. Its size need not equal
        `config.window_size` (the `Camera` is built purely from
        `config.window_size`, independent of `surface`'s actual
        dimensions), but callers should pass a matching surface or the
        plot will be visibly cropped or offset; `save_frame_png` always
        allocates a `config.window_size` surface for this reason.
    swarm : numpy.ndarray, shape (N, 25)
        Flock agent array (see module docstring's data contract).
        Read-only -- never written to, including via views.
    shepherds : numpy.ndarray, shape (M, 22)
        Shepherd agent array (see module docstring's data contract).
        Read-only.
    boundary : tuple of int
        `(boundary_x, boundary_y)`, forwarded to `Camera.fit`.
    target : tuple of int
        `(target_x, target_y, target_radius)`.
    mode : int
        Simulation mode, 0-4. Deliberately a runtime argument rather than
        being read from `basic.MODE` (which is import-frozen elsewhere),
        so this function stays testable for every mode within a single
        process.
    hud : str, optional
        HUD text to render, centered, in the top `config.hud_height`-px
        band. `None` (default) draws no HUD.
    fence : tuple of (float, float), optional
        `(fence_middle_angle, gate_angular_width)`, in radians. `None`
        (default) draws no fence gate arc.
    config : RenderConfig, optional
        Colors, sizes, and window geometry. Defaults to `DEFAULT_CONFIG`.

    Returns
    -------
    None
        `surface` is mutated in place; this is the function's only
        externally visible effect (besides lazily populating the
        module-level font cache the first time `hud` is not `None`).

    Notes
    -----
    Read-only on `swarm`/`shepherds` and calls neither `random` nor
    `np.random` -- see module docstring. Every boolean-mask, `argsort`, or
    fancy-index selection used below (e.g. `swarm[swarm[:, 21] == 0]`)
    returns a freshly allocated numpy array and never aliases or writes
    back into the inputs.
    """
    surface.fill(config.white)
    camera = Camera.fit(boundary, config.window_size, config.hud_height)

    n_sheep = swarm.shape[0]
    n_shepherds = shepherds.shape[0]

    # Precompute shepherd screen positions once: reused by steps 3-5, 7,
    # and the mode 3/4 visibility overlay below. Safe even when M == 0 --
    # numpy ops on a (0, 2) array just produce another (0, 2) array.
    shepherd_xy = camera.to_screen(shepherds[:, :2])

    # Precompute the fixed RGBA colors used throughout this frame (alpha
    # levels baked in per RenderConfig / the module docstring's palette).
    green_80 = _rgba(config.green, config.alpha_80)
    blue_80 = _rgba(config.blue, config.alpha_80)
    green_opaque = _rgba(config.green, 255)
    red_opaque = _rgba(config.red, 255)
    cyan_opaque = _rgba(config.cyan, 255)
    yellow_opaque = _rgba(config.yellow, 255)
    black_opaque = _rgba(config.black, 255)
    blue_opaque = _rgba(config.blue, 255)
    red_20 = _rgba(config.red, config.alpha_20)
    blue_20 = _rgba(config.blue, config.alpha_20)
    blue_50 = _rgba(config.blue, config.alpha_50)

    # --- 1-2. Sheep bodies + heading arrows --------------------------------
    # Guarded on N > 0: draw.py read `swarm[0, 7]` for the (homogeneous)
    # body radius unconditionally, which crashed on an empty swarm.
    # Skipping the whole block is the deliberate fix (draw.py had no such
    # guard).
    if n_sheep > 0:
        body_radius_px = camera.px(swarm[0, 7])
        sheep_xy = camera.to_screen(swarm[:, :2])

        # draw.py:90-91 -- face is green iff on the convex hull (col 22 !=
        # 0) AND not staying (col 21 != 1); edge is blue if staying, else
        # green. Precomputed as boolean masks so the per-row loop below
        # only does array indexing, not repeated column comparisons.
        in_hull = (swarm[:, 22] != 0) & (swarm[:, 21] != 1)
        staying = swarm[:, 21] == 1

        for i in range(n_sheep):
            fill = green_80 if in_hull[i] else None
            edge = blue_80 if staying[i] else green_80
            _alpha_circle(
                surface, sheep_xy[i], body_radius_px, fill_rgba=fill, edge_rgba=edge
            )

        for i in range(n_sheep):
            _arrow(
                surface,
                sheep_xy[i],
                swarm[i, 2],
                config.arrow_len_px,
                green_opaque,
                config.arrow_head_px,
            )

    # --- 3. Shepherd -> drive/collect point lines --------------------------
    # draw.py:128 plots (col14, col0) x (col15, col1) -- a line from each
    # shepherd's current drive-or-collect point to its own position.
    point_xy = camera.to_screen(shepherds[:, 14:16])
    for i in range(n_shepherds):
        pygame.draw.aaline(surface, cyan_opaque, point_xy[i], shepherd_xy[i])

    # --- 4. Shepherd bodies ---------------------------------------------------
    # Fixed pixel radius (config.shepherd_radius_px) -- see that field's
    # docstring for why this is not a `camera.px(...)` call. An empty
    # `shepherds` naturally no-ops this loop, fixing draw.py:130's
    # `ax.set_prop_cycle(plt.cycler(color=...))` call, which raised on an
    # empty color list when M == 0.
    for i in range(n_shepherds):
        color = red_20 if shepherds[i, 13] == 1 else blue_20
        _alpha_circle(surface, shepherd_xy[i], config.shepherd_radius_px, fill_rgba=color)

    # --- 5. Shepherd heading arrows --------------------------------------------
    for i in range(n_shepherds):
        _arrow(
            surface,
            shepherd_xy[i],
            shepherds[i, 2],
            config.arrow_len_px,
            red_opaque,
            config.arrow_head_px,
        )

    # --- 6. Flock center-of-mass star ------------------------------------------
    # See module docstring, "Deliberate visual changes", item 2: `None`
    # means "skip", never "draw at the origin".
    center = flock_center(swarm)
    if center is not None:
        center_xy = camera.to_screen(np.asarray(center))
        _star(surface, center_xy, config.star_radius_px, red_opaque)

    # --- 7. Collect tethers ------------------------------------------------------
    # draw.py:156-165. Guarded on N > 0: a collecting shepherd's
    # target-sheep index (col 16) is meaningless with no sheep to index.
    if n_sheep > 0:
        for i in range(n_shepherds):
            if shepherds[i, 13] == 0:
                collecting_idx = int(shepherds[i, 16])
                collecting_xy = camera.to_screen(swarm[collecting_idx, :2])
                _dashed_line(surface, yellow_opaque, shepherd_xy[i], collecting_xy, width=2)

    # --- 8. Mode overlays ----------------------------------------------------------
    match mode:
        case 2:
            # draw.py:169-181 -- rows on the convex hull (col 22 != 0),
            # sorted into CCW order by their 1-based hull rank (col 22).
            # Design: checks `np.any(hull_mask)` (a boolean mask) rather
            # than porting draw.py:171's literal `np.any(hull)` (truthiness
            # of the whole selected-row float array); the mask directly
            # and unambiguously answers "are there any hull rows", without
            # relying on selected rows happening to contain a nonzero
            # value somewhere.
            hull_mask = swarm[:, 22] != 0
            if np.any(hull_mask):
                hull_rows = swarm[hull_mask]
                hull_rows = hull_rows[np.argsort(hull_rows[:, 22])]
                center_world = np.array(
                    [np.mean(hull_rows[:, 0]), np.mean(hull_rows[:, 1])]
                )
                _star(surface, camera.to_screen(center_world), config.star_radius_px, black_opaque)
                hull_xy = camera.to_screen(hull_rows[:, :2])
                _dashed_polygon(surface, green_opaque, hull_xy, width=2)

        case 3:
            # draw.py:183-199 -- convex hull of the MOVING sheep only.
            moving = swarm[swarm[:, 21] == 0]
            hull_idx = (
                ConvexHull(moving[:, :2]).vertices
                if moving.shape[0] > 2
                else np.arange(moving.shape[0])
            )
            if hull_idx.shape[0] >= 2:
                hull_xy = camera.to_screen(moving[hull_idx, :2])
                _dashed_polygon(surface, green_opaque, hull_xy, width=2)
            _draw_visibility_rays(surface, swarm, shepherd_xy, camera, config)

        case 4:
            # draw.py:228-243 -- one convex hull per subflock (col 24), 1
            # through max(col 24) inclusive. Guarded on N > 0: `np.max` of
            # an empty column raises `ValueError`.
            if n_sheep > 0:
                max_flock = int(np.max(swarm[:, 24]))
                for flock_id in range(1, max_flock + 1):
                    flock = swarm[swarm[:, 24] == flock_id]
                    hull_idx = (
                        ConvexHull(flock[:, :2]).vertices
                        if flock.shape[0] > 2
                        else np.arange(flock.shape[0])
                    )
                    if hull_idx.shape[0] >= 2:
                        hull_xy = camera.to_screen(flock[hull_idx, :2])
                        _dashed_polygon(surface, green_opaque, hull_xy, width=2)
            # draw.py:245-274 -- same enumerate-based visibility overlay as
            # mode 3, factored into `_draw_visibility_rays` above (see that
            # helper's docstring for draw.py's own note on why this must
            # be `enumerate`-indexed rather than reusing the flock-hull
            # loop's `flock_id`).
            _draw_visibility_rays(surface, swarm, shepherd_xy, camera, config)

        case _:
            # Modes 0/1/anything else: no overlay (draw.py's `match mode:`
            # likewise has no `case` for them).
            pass

    # --- 9. Target -------------------------------------------------------------------
    target_xy = camera.to_screen(np.asarray(target[:2], dtype=np.float64))
    _star(surface, target_xy, config.target_star_radius_px, blue_opaque)
    _alpha_circle(surface, target_xy, camera.px(target[2]), edge_rgba=blue_50)

    # --- 10. Fence gate arc ------------------------------------------------------------
    if fence is not None:
        fence_middle_angle, gate_angular_width = fence
        # draw.py:288-292: both endpoints are rotated by -90 degrees
        # (here, -pi/2 radians) because matplotlib's `patches.Arc` measures
        # theta=0 as pointing down, not right.
        theta0 = fence_middle_angle - gate_angular_width / 2 - math.pi / 2
        theta1 = fence_middle_angle + gate_angular_width / 2 - math.pi / 2
        _arc_lines(surface, red_opaque, target_xy, camera.px(target[2]), theta0, theta1, width=2)

    # --- 11. HUD -----------------------------------------------------------------------
    if hud is not None:
        font = _font(config.hud_font_size)
        text_surface = font.render(hud, True, config.black)
        text_rect = text_surface.get_rect(
            center=(config.window_size[0] // 2, config.hud_height // 2)
        )
        surface.blit(text_surface, text_rect)


def save_frame_png(
    path: str,
    swarm: np.ndarray,
    shepherds: np.ndarray,
    boundary: tuple[int, int],
    target: tuple[int, int, int],
    mode: int,
    *,
    hud: str | None = None,
    fence: tuple[float, float] | None = None,
    config: RenderConfig = DEFAULT_CONFIG,
) -> None:
    """
    Render one frame and save it as a PNG at `path`.

    Allocates a fresh `pygame.Surface(config.window_size)`, calls
    `render_frame` into it, then writes it out via `pygame.image.save`
    (the output format is inferred from `path`'s extension; a `.png`
    extension writes a PNG).

    Parameters
    ----------
    path : str
        Destination file path. Its parent directory is created if missing
        (`os.makedirs(..., exist_ok=True)`); this is skipped entirely
        when `path` has no directory component (a bare filename, written
        to the current working directory).
    swarm, shepherds, boundary, target, mode, hud, fence, config
        Forwarded to `render_frame`; see its docstring.

    Returns
    -------
    None
        The PNG is written to `path` as a side effect.

    Notes
    -----
    Allocates and discards a new `pygame.Surface` per call -- intended for
    one-off snapshots (tests, ad hoc debugging), not a hot render loop; a
    hot loop should call `render_frame` directly into a reused surface.
    """
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    surface = pygame.Surface(config.window_size)
    render_frame(
        surface,
        swarm,
        shepherds,
        boundary,
        target,
        mode,
        hud=hud,
        fence=fence,
        config=config,
    )
    pygame.image.save(surface, path)
