
import numpy as np
from config import R_W, L_W, CAM_OFF, MARKERS
from markers import wrap

CHI2_3 = 14.16     
K_SLIP = 3e-3     
Q_FLOOR = np.diag([1e-7, 1e-7, 1e-7])

class EKF:
    def __init__(self, x0, sig0):
        self.x = np.array(x0, float); self.P = np.diag(np.square(sig0)); self.n_acc = self.n_rej = 0

    def predict(self, dphi_l, dphi_r):
        ds = R_W / 2 * (dphi_r + dphi_l); dth = R_W / L_W * (dphi_r - dphi_l)
        th = self.x[2]; thm = th + dth / 2; c, s = np.cos(thm), np.sin(thm)
        F = np.array([[1, 0, -ds * s], [0, 1, ds * c], [0, 0, 1.0]])
        a, b = R_W / 2, R_W / L_W                         
        Fu = np.array([[c * a - ds / 2 * s * b, c * a + ds / 2 * s * b],      
                       [s * a + ds / 2 * c * b, s * a - ds / 2 * c * b],
                       [b, -b]])
        Su = np.diag([K_SLIP * abs(dphi_r) + 1e-9, K_SLIP * abs(dphi_l) + 1e-9])
        self.x = self.x + np.array([ds * c, ds * s, dth]); self.x[2] = wrap(self.x[2])
        self.P = F @ self.P @ F.T + Fu @ Su @ Fu.T + Q_FLOOR * (abs(ds) + abs(dth) + 1e-3)

    @staticmethod
    def h(x, mid, use_psi):
        mx, my, phi = MARKERS[mid]; th = x[2]
        d = np.array([mx - x[0] - CAM_OFF * np.cos(th), my - x[1] - CAM_OFF * np.sin(th)])
        r = np.hypot(*d); out = [r, wrap(np.arctan2(d[1], d[0]) - th)]
        if use_psi: out.append(wrap(phi - th))
        return np.array(out), d, r

    def update_marker(self, mid, z, Rm, use_psi=True):
        n = 3 if use_psi else 2
        z = z[:n]; Rm = Rm[:n, :n]
        zh, d, r = self.h(self.x, mid, use_psi); th = self.x[2]
        dd_dth = CAM_OFF * np.array([np.sin(th), -np.cos(th)])
        H = np.zeros((n, 3))
        H[0] = [-d[0] / r, -d[1] / r, d @ dd_dth / r]
        H[1] = [d[1] / r**2, -d[0] / r**2, (-d[1] * dd_dth[0] + d[0] * dd_dth[1]) / r**2 - 1]
        if use_psi: H[2] = [0, 0, -1]
        y = z - zh; y[1] = wrap(y[1]);  y[2:] = wrap(y[2:]) if use_psi else y[2:]
        S = H @ self.P @ H.T + Rm
        if y @ np.linalg.solve(S, y) > (CHI2_3 if n == 3 else 9.21): self.n_rej += 1; return False
        K = self.P @ H.T @ np.linalg.inv(S); I = np.eye(3)
        self.x = self.x + K @ y; self.x[2] = wrap(self.x[2])
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ Rm @ K.T          
        self.P = (self.P + self.P.T) / 2; self.n_acc += 1; return True
