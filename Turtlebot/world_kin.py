import numpy as np
from config import *
from markers import corners_world, project, wrap
from navigator import SensorFrame
from models import wheel_speeds

def _ray_boxes(p, d, boxes):
    best = np.full(len(d), np.inf)
    with np.errstate(divide='ignore', invalid='ignore'):
        for cx, cy, hx, hy in boxes:
            t1 = (np.array([cx - hx, cy - hy]) - p) / d; t2 = (np.array([cx + hx, cy + hy]) - p) / d
            tn = np.maximum(np.minimum(t1[:, 0], t2[:, 0]), np.minimum(t1[:, 1], t2[:, 1]))
            tf = np.minimum(np.maximum(t1[:, 0], t2[:, 0]), np.maximum(t1[:, 1], t2[:, 1]))
            hit = (tf >= np.maximum(tn, 0)) & (tn > 0); best = np.where(hit & (tn < best), tn, best)
    return best

class KinWorld:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.R, self.L = R_W * 1.012, L_W * 1.009                      
        self.pose = np.array(START) + np.array([0.03, -0.02, 0.035])     
        self.phi = np.zeros(2); self.w_act = np.zeros(2); self.collisions = 0; self.t = 0.0; self.k = 0
        self.boxes = list(OBSTACLES)

    def step(self, v, w):
        wl, wr = wheel_speeds(v, w); cmd = np.array([wl, wr]); sub = 10; h = DT / sub
        slip = 1 + self.rng.normal(0, 0.02, 2)
        for _ in range(sub):
            self.w_act += (cmd - self.w_act) * (1 - np.exp(-h / 0.04)); self.phi += self.w_act * h
            wa = self.w_act * slip; vt = self.R / 2 * (wa[0] + wa[1]); wt = self.R / self.L * (wa[1] - wa[0])
            th = self.pose[2] + wt * h / 2
            self.pose += np.array([vt * h * np.cos(th), vt * h * np.sin(th), wt * h]); self._collide()
        self.t += DT; self.k += 1

    def _collide(self):
        x, y, _ = self.pose; xmin, xmax, ymin, ymax = ROOM; hit = False
        if x < xmin + ROBOT_R: self.pose[0] = xmin + ROBOT_R; hit = True
        if x > xmax - ROBOT_R: self.pose[0] = xmax - ROBOT_R; hit = True
        if y < ymin + ROBOT_R: self.pose[1] = ymin + ROBOT_R; hit = True
        if y > ymax - ROBOT_R: self.pose[1] = ymax - ROBOT_R; hit = True
        for cx, cy, hx, hy in self.boxes:
            qx, qy = np.clip(self.pose[0], cx - hx, cx + hx), np.clip(self.pose[1], cy - hy, cy + hy)
            d = np.hypot(self.pose[0] - qx, self.pose[1] - qy)
            if d < ROBOT_R and d > 1e-9:
                self.pose[:2] = np.array([qx, qy]) + (self.pose[:2] - [qx, qy]) / d * ROBOT_R; hit = True
        if hit: self.collisions += 1

    def _lidar(self):
        a = self.pose[2] + np.linspace(0, 2 * np.pi, LIDAR_N, endpoint=False); d = np.stack([np.cos(a), np.sin(a)], 1)
        xmin, xmax, ymin, ymax = ROOM; p = self.pose[:2]
        with np.errstate(divide='ignore', invalid='ignore'):
            tx = np.where(d[:, 0] > 0, (xmax - p[0]) / d[:, 0], np.where(d[:, 0] < 0, (xmin - p[0]) / d[:, 0], np.inf))
            ty = np.where(d[:, 1] > 0, (ymax - p[1]) / d[:, 1], np.where(d[:, 1] < 0, (ymin - p[1]) / d[:, 1], np.inf))
        r = np.minimum(np.minimum(tx, ty), _ray_boxes(p, d, self.boxes)) + self.rng.normal(0, LIDAR_SIGMA, LIDAR_N)
        r[r > LIDAR_RMAX] = np.inf; r[r < LIDAR_RMIN] = np.nan; return r

    def _camera(self):
        out = {}; th = self.pose[2]; cam = self.pose[:2] + CAM_OFF * np.array([np.cos(th), np.sin(th)])
        for mid, (mx, my, phi) in MARKERS.items():
            px, zc = project(corners_world(mid), self.pose)
            v = cam - np.array([mx, my]); rng_ = np.hypot(*v)
            if zc.min() < 0.1 or rng_ > 3.0 or np.any(px < 10) or np.any(px[:, 0] > IMG_W - 10) or np.any(px[:, 1] > IMG_H - 10): continue
            if (v @ np.array([np.cos(phi), np.sin(phi)])) / rng_ < np.cos(np.deg2rad(70)): continue
            if np.linalg.norm(px[0] - px[1]) < 20: continue
            d = (np.array([mx, my]) - cam)[None, :] / rng_
            if _ray_boxes(cam, d, self.boxes)[0] < rng_ - 0.02: continue          
            out[mid] = px + self.rng.normal(0, 0.5, px.shape)
        return out or None

    def sense(self):
        q = 2 * np.pi / 4096
        return SensorFrame(self.t, tuple(np.floor(self.phi / q) * q),
                           self._lidar() if self.k % LIDAR_EVERY == 0 else None,
                           self._camera() if self.k % CAM_EVERY == 0 else None)

    def truth(self):                    
        return self.pose.copy()
