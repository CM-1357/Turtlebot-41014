import numpy as np
from scipy.linalg import expm
from config import R_W, L_W, V_MAX, W_MAX

def f(x, u, dt):
    v, w = u; th = x[2]
    return np.array([x[0] + v * dt * np.cos(th + w * dt / 2), x[1] + v * dt * np.sin(th + w * dt / 2), th + w * dt])

def jac_f(x, u, dt):
    v, w = u; thm = x[2] + w * dt / 2
    A = np.array([[1, 0, -v * dt * np.sin(thm)], [0, 1, v * dt * np.cos(thm)], [0, 0, 1.0]])
    B = np.array([[dt * np.cos(thm), 0], [dt * np.sin(thm), 0], [0, dt]])
    return A, B

def wheel_speeds(v, w):
    return (v - w * L_W / 2) / R_W, (v + w * L_W / 2) / R_W

def saturate(v, w):
    v = np.clip(v, -V_MAX, V_MAX); w = np.clip(w, -W_MAX, W_MAX)
    wl, wr = wheel_speeds(v, w); lim = V_MAX / R_W * 1.0
    m = max(abs(wl), abs(wr))
    if m > lim: v, w = v * lim / m, w * lim / m
    return float(v), float(w)

def error_model(v_r, w_r):
    A = np.array([[0, w_r, 0], [-w_r, 0, v_r], [0, 0, 0.0]])
    B = np.array([[-1, 0], [0, 0], [0, -1.0]])
    return A, B

def discretize(A, B, dt):
    n, m = B.shape; M = np.zeros((n + m, n + m)); M[:n, :n] = A; M[:n, n:] = B
    E = expm(M * dt); return E[:n, :n], E[:n, n:]
