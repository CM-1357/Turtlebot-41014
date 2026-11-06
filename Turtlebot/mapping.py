
import heapq, numpy as np
from scipy.ndimage import distance_transform_edt
from config import *

RES = 0.05

class Grid:
    def __init__(self):
        xmin, xmax, ymin, ymax = ROOM; m = 0.3
        self.x0, self.y0 = xmin - m, ymin - m
        self.nx, self.ny = int(round((xmax - xmin + 2 * m) / RES)), int(round((ymax - ymin + 2 * m) / RES))
        xs = self.x0 + (np.arange(self.nx) + .5) * RES; ys = self.y0 + (np.arange(self.ny) + .5) * RES
        self.prior = (xs[:, None] < xmin) | (xs[:, None] > xmax) | (ys[None, :] < ymin) | (ys[None, :] > ymax)
        self.L = np.where(self.prior, 4.0, 0.0)          
        self._ang = np.linspace(0, 2 * np.pi, LIDAR_N, endpoint=False)
        self._s = np.arange(0, LIDAR_RMAX, RES / 2)
        self.dist = None

    def cell(self, x, y): return np.floor((x - self.x0) / RES).astype(int), np.floor((y - self.y0) / RES).astype(int)
    def center(self, i, j): return self.x0 + (i + .5) * RES, self.y0 + (j + .5) * RES
    def ok(self, i, j): return (i >= 0) & (i < self.nx) & (j >= 0) & (j < self.ny)

    def integrate(self, pose, ranges):
        r = np.asarray(ranges, float); a = pose[2] + self._ang
        valid = ~np.isnan(r) & (r >= LIDAR_RMIN); hit = valid & np.isfinite(r) & (r < LIDAR_RMAX - 0.02)
        r_end = np.where(hit, r - RES * 0.7, LIDAR_RMAX); r_end = np.where(valid, r_end, 0)
        m = self._s[None, :] < r_end[:, None]
        px = pose[0] + self._s[None, :] * np.cos(a)[:, None]; py = pose[1] + self._s[None, :] * np.sin(a)[:, None]
        i, j = self.cell(px[m], py[m]); g = self.ok(i, j); free = np.unique(i[g] * self.ny + j[g])
        fi, fj = free // self.ny, free % self.ny; keep = ~self.prior[fi, fj]
        self.L[fi[keep], fj[keep]] = np.maximum(self.L[fi[keep], fj[keep]] - 0.4, -4)
        hx, hy = pose[0] + r[hit] * np.cos(a[hit]), pose[1] + r[hit] * np.sin(a[hit])
        i, j = self.cell(hx, hy); g = self.ok(i, j); hu = np.unique(i[g] * self.ny + j[g])
        hi, hj = hu // self.ny, hu % self.ny; self.L[hi, hj] = np.minimum(self.L[hi, hj] + 0.9, 4)
        self.dist = None

    def clearance(self):
        if self.dist is None: self.dist = distance_transform_edt(self.L < 1.0) * RES
        return self.dist

def astar(grid, start, goal, r_infl):
    dist = grid.clearance(); blocked = dist < r_infl
    si, sj = [int(v) for v in grid.cell(*start)]; gi, gj = [int(v) for v in grid.cell(*goal)]
    for (ci, cj) in ((si, sj), (gi, gj)):                       
        blocked[max(ci - 3, 0):ci + 4, max(cj - 3, 0):cj + 4] &= dist[max(ci - 3, 0):ci + 4, max(cj - 3, 0):cj + 4] < 0.07
    if blocked[gi, gj]: return None
    nb = [(1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1), (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
    g = {(si, sj): 0.0}; par = {}; pq = [(0.0, (si, sj))]; done = set()
    while pq:
        _, u = heapq.heappop(pq)
        if u in done: continue
        done.add(u)
        if u == (gi, gj): break
        for di, dj, c in nb:
            v = (u[0] + di, u[1] + dj)
            if not (0 <= v[0] < grid.nx and 0 <= v[1] < grid.ny) or blocked[v] or v in done: continue
            nc = g[u] + c + 1.5 * max(0.0, 0.40 - dist[v]) / 0.40          
            if nc < g.get(v, 1e18):
                g[v] = nc; par[v] = u; heapq.heappush(pq, (nc + RES * 0 + np.hypot(v[0] - gi, v[1] - gj), v))
    if (gi, gj) not in done: return None
    p = [(gi, gj)]
    while p[-1] != (si, sj): p.append(par[p[-1]])
    return np.array([grid.center(i, j) for i, j in p[::-1]])

def line_free(grid, a, b, r):
    n = max(int(np.hypot(*(b - a)) / (RES / 2)), 2); t = np.linspace(0, 1, n)[:, None]
    q = a + t * (b - a); i, j = grid.cell(q[:, 0], q[:, 1])
    return bool(np.all(grid.ok(i, j)) and np.all(grid.clearance()[i, j] >= r))

def shortcut(grid, path, r):
    out = [path[0]]; k = 0
    while k < len(path) - 1:
        j = len(path) - 1
        while j > k + 1 and not line_free(grid, path[k], path[j], r): j -= 1
        out.append(path[j]); k = j
    return np.array(out)

def chaikin(p, it):
    for _ in range(it):
        q = 0.75 * p[:-1] + 0.25 * p[1:]; r = 0.25 * p[:-1] + 0.75 * p[1:]
        p = np.vstack([p[0], np.stack([q, r], 1).reshape(-1, 2), p[-1]])
    return p

def smooth(grid, path, r_hard):
    sc = shortcut(grid, path, r_hard + 0.02)
    for it in (3, 2, 1):
        sm = chaikin(sc, it)
        i, j = grid.cell(sm[:, 0], sm[:, 1])
        if np.all(grid.clearance()[i, j] >= r_hard): return sm
    return sc

def build_reference(pts, v_nav=0.15, ds=0.02, w_lim=1.2, acc=0.25, v_const=None):
    seg = np.hypot(*np.diff(pts, axis=0).T); s = np.r_[0, np.cumsum(seg)]
    S = np.r_[np.arange(0, s[-1], ds), s[-1]]
    x, y = np.interp(S, s, pts[:, 0]), np.interp(S, s, pts[:, 1])
    th = np.unwrap(np.arctan2(np.gradient(y, S), np.gradient(x, S)))
    kap = np.gradient(th, S); k = 9; kap = np.convolve(np.pad(kap, k // 2, mode='edge'), np.ones(k) / k, 'valid')
    if v_const is None:
        v = np.minimum(v_nav, w_lim / np.maximum(np.abs(kap), 1e-6))
        v = np.minimum(v, 0.05 + np.sqrt(2 * acc * S)); v = np.minimum(v, 0.04 + np.sqrt(2 * acc * (S[-1] - S)))
    else: v = np.full_like(S, v_const)
    return np.stack([x, y, th, v, v * kap], 1), S

class RefTracker:
    def __init__(self, ref, S): self.ref, self.S, self.i = ref, S, 0
    def nearest(self, p):
        w = self.ref[self.i:self.i + 60, :2]
        self.i += int(np.argmin(np.hypot(w[:, 0] - p[0], w[:, 1] - p[1]))); return self.ref[self.i]
    def remaining(self): return self.S[-1] - self.S[self.i]
