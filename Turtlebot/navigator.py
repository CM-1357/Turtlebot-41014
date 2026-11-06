from dataclasses import dataclass
import numpy as np
from config import *
from markers import wrap, pnp_measurement, meas_cov
from ekf import EKF
from lqr import LQRTracker
from mapping import Grid, astar, smooth, build_reference, RefTracker
from models import saturate

@dataclass
class SensorFrame:
    t: float
    enc: tuple                      
    lidar: object = None            
    corners: object = None          

PSI_MAX_RANGE = 1.5                 
R_PLAN, R_HARD = ROBOT_R + 0.05, ROBOT_R + 0.015

class Navigator:
    def __init__(self):
        self.ekf = EKF(START, START_SIGMA); self.grid = Grid(); self.lqr = LQRTracker(DT)
        mx, my, phi = MARKERS[DOCK_ID]; n = np.array([np.cos(phi), np.sin(phi)])
        self.dock_th = wrap(phi + np.pi)
        self.stage = np.array([mx, my]) + STAGE_DIST * n; self.dock_goal = np.array([mx, my]) + DOCK_DIST * n
        self.goals = [np.array(w) for w in WAYPOINTS] + [self.stage]
        self.gi, self.phase, self.trk, self.turn_first = 0, 'NAV', None, False
        self.phi0, self.last_scan, self.t_phase, self.acc_at_align = None, None, 0.0, 0
        self.info = {}

    def step(self, f: SensorFrame):
        e = self.ekf; self.info['e'] = np.full(3, np.nan)
        if self.phi0 is not None: e.predict(f.enc[0] - self.phi0[0], f.enc[1] - self.phi0[1])
        self.phi0 = f.enc
        if f.corners:
            for mid, c in f.corners.items():
                z = pnp_measurement(c)
                if z is None: continue
                use_psi = z[0] < PSI_MAX_RANGE
                e.update_marker(mid, z, meas_cov(z[0]), use_psi)
        if f.lidar is not None:
            self.grid.integrate(e.x, f.lidar); self.last_scan = np.asarray(f.lidar)
        v, w = getattr(self, 'run_' + self.phase.lower())(f)
        self.info.update(phase=self.phase, x=e.x.copy(), P=e.P.copy(), acc=e.n_acc, rej=e.n_rej)
        return saturate(v, w)

    def _blocked_ahead(self):
        r = self.trk.ref[self.trk.i::5, :2]; i, j = self.grid.cell(r[:, 0], r[:, 1])
        return bool(np.any(self.grid.clearance()[i, j] < R_HARD))

    def _front_blocked(self):
        if self.last_scan is None: return False
        a = np.linspace(0, 2 * np.pi, LIDAR_N, endpoint=False); a = wrap(a)
        m = (np.abs(a) < np.deg2rad(30)); r = self.last_scan[m]
        return bool(np.any((r > LIDAR_RMIN) & (r < ROBOT_R + 0.03)))

    def run_nav(self, f):
        x = self.ekf.x; goal = self.goals[self.gi]; last = self.gi == len(self.goals) - 1
        if np.hypot(*(goal - x[:2])) < (0.10 if last else 0.15):
            if last: self.phase, self.t_phase, self.acc_at_align = 'ALIGN', f.t, self.ekf.n_acc; return 0., 0.
            self.gi += 1; self.trk = None; goal = self.goals[self.gi]
        if self.trk is None or self._blocked_ahead():
            path = astar(self.grid, x[:2], goal, R_PLAN)
            if path is None: self.info['plan'] = 'FAILED'; return 0., 0.
            pts = smooth(self.grid, path, R_HARD) if len(path) > 2 else path
            if len(pts) < 2: pts = np.vstack([x[:2], goal])
            ref, S = build_reference(pts); self.trk = RefTracker(ref, S)
            self.turn_first = abs(wrap(ref[0, 2] - x[2])) > np.deg2rad(45); self.info['plan'] = 'ok'
        if self._front_blocked(): self.trk = None; return 0., 0.       
        if self.turn_first:
            eth = wrap(self.trk.ref[0, 2] - x[2])
            if abs(eth) < np.deg2rad(10): self.turn_first = False
            return 0., float(np.clip(2.5 * eth, -1.5, 1.5))
        (v, w), err = self.lqr.control(x, self.trk.nearest(x)); self.info['e'] = err
        return v, w

    
    def run_align(self, f):
        x, P = self.ekf.x, self.ekf.P; eth = wrap(self.dock_th - x[2])
        settled = (self.ekf.n_acc - self.acc_at_align >= 8 and abs(eth) < np.deg2rad(2)
                   and np.sqrt(P[1, 1]) < 0.01 and np.sqrt(P[2, 2]) < np.deg2rad(2))
        if settled or f.t - self.t_phase > 12.0:
            ref, S = build_reference(np.vstack([self.stage, self.dock_goal]), v_const=0.05)   
            ref[:, 2] = self.dock_th; self.trk = RefTracker(ref, S); self.phase = 'DOCK'; return 0., 0.
        return 0., float(np.clip(2.0 * eth, -1.0, 1.0))

    def run_dock(self, f):
        x = self.ekf.x; u = np.array([np.cos(self.dock_th), np.sin(self.dock_th)])
        if (self.dock_goal - x[:2]) @ u <= 0.004: self.phase = 'DONE'; return 0., 0.
        (v, w), err = self.lqr.control(x, self.trk.nearest(x)); self.info['e'] = err
        return v, w

    def run_done(self, f): return 0., 0.
