
import numpy as np
from scipy.linalg import solve_discrete_are
from models import error_model, discretize
from markers import wrap

class LQRTracker:
    def __init__(self, dt, Q=(30., 60., 6.), R=(1.0, 0.4)):
        self.dt, self.Q, self.R = dt, np.diag(Q), np.diag(R); self._cache = {}

    def gain(self, v_r, w_r):
        key = (round(v_r, 3), round(w_r, 2))
        if key not in self._cache:
            Ad, Bd = discretize(*error_model(*key), self.dt)
            P = solve_discrete_are(Ad, Bd, self.Q, self.R)
            self._cache[key] = np.linalg.solve(self.R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
        return self._cache[key]

    def control(self, x_hat, ref):
        xr, yr, thr, vr, wr = ref; c, s = np.cos(x_hat[2]), np.sin(x_hat[2])
        dx, dy = xr - x_hat[0], yr - x_hat[1]
        e = np.array([c * dx + s * dy, -s * dx + c * dy, wrap(thr - x_hat[2])])
        du = -self.gain(max(vr, 0.02), wr) @ e
        return (vr + du[0], wr + du[1]), e
