"""
Numba-nopython 2D convex-hull kernels that replace the ``scipy.spatial.ConvexHull``
calls the shepherd drivers previously made from inside ``nb.objmode``.

Motivation
----------
``scipy.spatial.ConvexHull`` cannot be called from numba nopython code, so the
old drivers dropped out of the JIT into an ``nb.objmode`` block for every hull.
The objmode boundary itself is cheap (~0.5 us); the cost is scipy/Qhull spinning
up a general n-dimensional hull engine for a trivial 2-D problem (~30 us/call,
8-22% of herding wall-time). These hand-written 2-D kernels run entirely in
nopython at ~15x the throughput and drop the scipy dependency from the hot path.

Equivalence
-----------
``convex_hull_2d`` returns the same *set* of hull-vertex indices as
``ConvexHull(pts).vertices`` (a different CCW start vertex is allowed).
``visible_chain`` reproduces ``ConvexHull(vstack([p, pts]), qhull_options="QG0")``:
because ``QGn`` leaves the query point out of the hull, the facets "visible from
p" are exactly the near-side arc of ``conv(pts)``, so we build ``conv(pts)`` once
and select the edges whose supporting line has ``p`` on the outer side -- there is
no need to build ``conv(pts u {p})``. Both were fuzz-checked against scipy over
thousands of random and adversarial point sets.
"""

import numba as nb
import numpy as np


@nb.jit(nopython=True)
def _lexsort_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Return the permutation that sorts points lexicographically by (x, then y).

    numba nopython has no ``np.lexsort``; this composes two *stable* argsorts
    (secondary key first). ``kind="mergesort"`` is required -- the default
    quicksort is unstable and would break the tie-ordering the hull relies on.

    @param x: x-coordinates.
    @param y: y-coordinates.
    @return: int64 index array ordering the points by (x, y).
    """
    order_y = np.argsort(y, kind="mergesort")
    return order_y[np.argsort(x[order_y], kind="mergesort")]


@nb.jit(nopython=True)
def convex_hull_2d(pts: np.ndarray) -> np.ndarray:
    """
    Convex hull of 2-D points via Andrew's monotone chain.

    Drop-in for ``scipy.spatial.ConvexHull(pts).vertices``: returns the indices
    (into ``pts``) of the hull vertices in counter-clockwise order. Collinear
    points are dropped (pop on ``cross <= 0``), matching Qhull's convention of
    returning only strictly-extreme vertices. The orientation test is the
    hand-written scalar cross product -- numba's ``cross2d`` heap-allocates per
    call and is 35-105x slower in a hot loop.

    The caller is expected to guard the degenerate case; for ``n <= 2`` this
    returns ``arange(n)`` (every point is a "vertex").

    @param pts: (n, 2) array of point coordinates.
    @return: int64 array of hull-vertex indices into ``pts``, CCW order.
    """
    n = pts.shape[0]
    if n < 3:
        return np.arange(n)

    # Contiguous 1-D copies keep the inner loop off strided column views.
    x = np.ascontiguousarray(pts[:, 0])
    y = np.ascontiguousarray(pts[:, 1])
    order = _lexsort_xy(x, y)

    hull = np.empty(2 * n, dtype=np.int64)
    k = 0

    # Lower hull.
    for ii in range(n):
        i = order[ii]
        while k >= 2:
            ox = x[hull[k - 2]]; oy = y[hull[k - 2]]
            ax = x[hull[k - 1]]; ay = y[hull[k - 1]]
            # cross((o->a), (o->p)); <= 0 means a clockwise/collinear turn -> pop.
            if (ax - ox) * (y[i] - oy) - (ay - oy) * (x[i] - ox) <= 0.0:
                k -= 1
            else:
                break
        hull[k] = i
        k += 1

    # Upper hull (must retain the last point of the lower hull as a base).
    lower_size = k + 1
    for ii in range(n - 2, -1, -1):
        i = order[ii]
        while k >= lower_size:
            ox = x[hull[k - 2]]; oy = y[hull[k - 2]]
            ax = x[hull[k - 1]]; ay = y[hull[k - 1]]
            if (ax - ox) * (y[i] - oy) - (ay - oy) * (x[i] - ox) <= 0.0:
                k -= 1
            else:
                break
        hull[k] = i
        k += 1

    # The last written index duplicates the first vertex; drop it.
    return hull[: k - 1].copy()


@nb.jit(nopython=True)
def visible_chain(pts: np.ndarray, hull_idx: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Indices of the hull vertices bounding the edges visible from external point p.

    Reproduces ``ConvexHull(vstack([p, pts]), qhull_options="QG0")`` reduced to
    the sheep it selects: for a CCW hull, an edge ``a -> b`` is visible from ``p``
    iff ``p`` lies strictly on its outer (right) side; both endpoints of every
    visible edge are returned. The result is **ascending** (via ``np.where`` on a
    membership mask, which also deduplicates), matching the old code's
    ``np.unique`` so that any downstream ``np.mean`` over these indices sums in the
    same order and stays bit-for-bit identical to the scipy path.

    An empty result means ``p`` is inside (or on the boundary of) ``conv(pts)`` --
    no edge is visible -- which the caller handles with its nearest-point fallback.
    Strict ``< 0`` treats an exact on-edge tie as not-visible (conservative).

    @param pts: (n, 2) array of point coordinates.
    @param hull_idx: CCW hull-vertex indices into ``pts`` (from ``convex_hull_2d``).
    @param p: (2,) query point (the shepherd).
    @return: ascending int64 indices into ``pts`` of the visible near-side arc.
    """
    h = hull_idx.shape[0]
    seen = np.zeros(pts.shape[0], dtype=np.bool_)
    px = p[0]
    py = p[1]
    for e in range(h):
        a = hull_idx[e]
        b = hull_idx[(e + 1) % h]
        ax = pts[a, 0]; ay = pts[a, 1]
        bx = pts[b, 0]; by = pts[b, 1]
        # p strictly on the right of the directed CCW edge a->b => edge visible.
        if (bx - ax) * (py - ay) - (by - ay) * (px - ax) < 0.0:
            seen[a] = True
            seen[b] = True
    return np.where(seen)[0]
