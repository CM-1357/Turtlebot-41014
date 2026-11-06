import os, sys, subprocess, numpy as np
from config import *
from markers import wrap, pnp_measurement

def place(w, x, y, th):
    w.d.qpos[0:2] = [x, y]; w.d.qpos[3:7] = [np.cos(th / 2), 0, 0, np.sin(th / 2)]; w.d.qvel[:] = 0
    import mujoco; mujoco.mj_forward(w.m, w.d)

if os.environ.get('_TB3_CHILD') != '1':          
    ok = None
    for flip in range(4):
        env = dict(os.environ, _TB3_CHILD='1', TB3_MARKER_FLIP=str(flip))
        out = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True)
        print(f'--- TB3_MARKER_FLIP={flip}\n{out.stdout}{out.stderr[-400:]}')
        if 'MARKER: PASS' in out.stdout: ok = flip; break
    print(f'\n==> use: export TB3_MARKER_FLIP={ok}' if ok is not None else '\n==> no flip produced a detection: check camera xyaxes / lighting / EGL')
    sys.exit(0)

from world_mj import MujocoWorld
from world_kin import KinWorld
w = MujocoWorld(0); mx, my, phi = MARKERS[0]
pose = (mx - 1.0, my - 0.2, 0.2); place(w, *pose)
c = w._camera(); print('detected ids:', None if c is None else sorted(c))
if c and 0 in c:
    z = pnp_measurement(c[0]); cam = np.array(pose[:2]) + CAM_OFF * np.array([np.cos(pose[2]), np.sin(pose[2])])
    d = np.array([mx, my]) - cam; zt = np.array([np.hypot(*d), wrap(np.arctan2(d[1], d[0]) - pose[2]), wrap(phi - pose[2])])
    print('PnP  r,beta,psi =', z[0].round(3), np.rad2deg(z[1:]).round(1)); print('true r,beta,psi =', zt[0].round(3), np.rad2deg(zt[1:]).round(1))
    good = abs(z[0] - zt[0]) < .03 and abs(wrap(z[1] - zt[1])) < np.deg2rad(1.5) and abs(wrap(z[2] - zt[2])) < np.deg2rad(8)
    print('MARKER:', 'PASS' if good else 'CHECK (detected but pose off: camera axes / FOV / marker size)')
else: print('MARKER: FAIL (not detected)')

place(w, 0.3, -0.2, 0.7); kin = KinWorld(0); kin.pose[:] = [*w.truth()]; kin.rng = np.random.default_rng(1)
a, b = w._lidar(), kin._lidar(); m = np.isfinite(a) & np.isfinite(b)
print(f'LIDAR: median |mj - analytic| = {np.nanmedian(np.abs(a[m]-b[m]))*100:.1f} cm over {m.sum()} rays ->', 'PASS' if np.nanmedian(np.abs(a[m]-b[m])) < .03 else 'CHECK')

place(w, -1.5, -1.5, 0.0); f0 = w.sense(); p0 = w.truth()
for _ in range(int(3 / DT)): w.step(0.1, 0.0)
f1 = w.sense(); ds_odo = R_W / 2 * sum(np.array(f1.enc) - np.array(f0.enc)); ds_true = np.hypot(*(w.truth()[:2] - p0[:2]))
ratio = ds_true / max(ds_odo, 1e-9)
print(f'DRIVE: odometry {ds_odo:.3f} m vs truth {ds_true:.3f} m, truth/odometry = {ratio:.3f} (expect ~1.012: hidden wheel-radius mismatch); collisions={w.collisions}',
      '-> PASS' if 0.97 < ratio < 1.05 and ds_true > 0.25 else '-> CHECK (robot dragging/slipping? casters or floor friction)')
