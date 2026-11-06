import sys, time, argparse, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from config import *
from markers import wrap
from navigator import Navigator

ap = argparse.ArgumentParser(); ap.add_argument('--backend', default='kin', choices=['kin', 'mujoco'])
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--out', default='result.png')
ap.add_argument('--no-vision', action='store_true', help='ablation: drop camera updates (odometry-only EKF)')
ap.add_argument('--tmax', type=float, default=240); a = ap.parse_args()
if a.backend == 'kin':
    from world_kin import KinWorld; world = KinWorld(a.seed)
else:
    from world_mj import MujocoWorld; world = MujocoWorld(a.seed)

nav = Navigator(); log = []; frame = world.sense(); t0 = time.time(); done_t = None
while world.t < a.tmax:
    v, w = nav.step(frame)                                  
    world.step(v, w); frame = world.sense()
    if a.no_vision: frame.corners = None
    xt = world.truth(); xe = nav.info['x']; P = nav.info['P']                
    err = xe - xt; err[2] = wrap(err[2])
    log.append([world.t, *xt, *xe, np.sqrt(P[0, 0]), np.sqrt(P[1, 1]), np.sqrt(P[2, 2]),
                float(err @ np.linalg.solve(P, err)), v, w, *(nav.info.get('e', [np.nan] * 3))])
    if nav.phase == 'DONE':
        done_t = done_t or world.t
        if world.t - done_t > 1.0: break
L = np.array(log); t = L[:, 0]; xt, xe = L[:, 1:4], L[:, 4:7]
mx, my, phi = MARKERS[DOCK_ID]; g = np.array([mx, my]) + DOCK_DIST * np.array([np.cos(phi), np.sin(phi)]); gth = wrap(phi + np.pi)
d = xt[-1, :2] - g; u = np.array([np.cos(gth), np.sin(gth)]); lon, lat = d @ u, d @ np.array([-u[1], u[0]])
pe = np.hypot(*(xe[:, :2] - xt[:, :2]).T)
print(f"backend={a.backend} sim_time={t[-1]:.1f}s wall={time.time()-t0:.1f}s phase={nav.phase} collisions={world.collisions}")
print(f"DOCK: along-track err {lon*100:+.1f} cm, lateral err {lat*100:+.1f} cm, heading err {np.rad2deg(wrap(xt[-1,2]-gth)):+.1f} deg  (truth, evaluation only)")
print(f"EKF: pos RMSE {np.sqrt(np.mean(pe**2))*100:.1f} cm (max {pe.max()*100:.1f}), final {pe[-1]*100:.1f} cm; mean NEES {L[:,10].mean():.2f} (3 dof, ideal ~3); "
      f"marker updates accepted/rejected {nav.info['acc']}/{nav.info['rej']}")
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
A = ax[0]; xmin, xmax, ymin, ymax = ROOM; A.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], 'k-')
for cx, cy, hx, hy in OBSTACLES: A.add_patch(plt.Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy, color='0.6'))
for k, (x, y, p) in MARKERS.items(): A.plot(x, y, 's', color='tab:red' if k == DOCK_ID else 'tab:orange'); A.text(x, y, f' id{k}', fontsize=8)
A.plot(xt[:, 0], xt[:, 1], 'b', label='truth'); A.plot(xe[:, 0], xe[:, 1], 'r--', label='EKF estimate')
A.plot(*zip(*WAYPOINTS), 'g^', label='waypoints'); A.plot(*g, 'k*', ms=12, label='dock goal'); A.set_aspect('equal'); A.legend(fontsize=8); A.set_title('Trajectory')
B = ax[1]; B.plot(t, (xe[:, 0] - xt[:, 0]) * 100, label='x err'); B.plot(t, (xe[:, 1] - xt[:, 1]) * 100, label='y err')
B.fill_between(t, -3 * L[:, 7] * 100, 3 * L[:, 7] * 100, alpha=.15, color='k', label='±3σ (y)'); B.set_ylabel('cm'); B.set_xlabel('t [s]'); B.legend(fontsize=8); B.set_title('Estimation error vs truth')
C = ax[2]; C.plot(t, L[:, 13] * 100, label='e_x (cm)'); C.plot(t, L[:, 14] * 100, label='e_y (cm)'); C.plot(t, np.rad2deg(L[:, 15]), label='e_θ (deg)')
C.set_ylim(-30, 30); C.set_xlabel('t [s]'); C.legend(fontsize=8); C.set_title('LQR tracking error (est. frame)')
plt.tight_layout(); plt.savefig(a.out, dpi=110); print('saved', a.out)
