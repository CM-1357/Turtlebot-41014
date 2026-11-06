
import numpy as np, cv2
from config import *

def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi

_H = MARKER_SIDE / 2
OBJ = np.array([[-_H, _H, 0], [_H, _H, 0], [_H, -_H, 0], [-_H, -_H, 0]], np.float64) 

def corners_world(mid):
    x, y, phi = MARKERS[mid]
    c = np.array([x, y, MARKER_Z]); t = np.array([-np.sin(phi), np.cos(phi), 0.]); up = np.array([0, 0, 1.])
    return np.array([c - _H * t + _H * up, c + _H * t + _H * up, c + _H * t - _H * up, c - _H * t - _H * up])

def project(pw, pose):
    x, y, th = pose; c, s = np.cos(th), np.sin(th)
    d = pw[:, :2] - np.array([x, y])
    xr = c * d[:, 0] + s * d[:, 1] - CAM_OFF; yr = -s * d[:, 0] + c * d[:, 1]; zr = pw[:, 2] - CAM_H
    xc, yc, zc = -yr, -zr, xr
    return np.stack([FX * xc / zc + IMG_W / 2, FY * yc / zc + IMG_H / 2], 1), zc

def pnp_measurement(corners):
    ok, rvec, tvec = cv2.solvePnP(OBJ, np.ascontiguousarray(corners, np.float64).reshape(4, 1, 2),
                                  K_CAM, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok: return None
    tx, _, tz = tvec.ravel()
    if tz <= 0: return None
    n = cv2.Rodrigues(rvec)[0][:, 2]                 
    return np.array([np.hypot(tx, tz), np.arctan2(-tx, tz), np.arctan2(-n[0], n[2])])

def meas_cov(r):
    return np.diag([(0.004 + 0.010 * r * r) ** 2, np.deg2rad(0.25 + 0.35 * r) ** 2,
                    np.deg2rad(3.0 + 4.0 * r * r) ** 2])
