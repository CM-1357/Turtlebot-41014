import argparse, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.animation as an
from matplotlib.patches import Rectangle, Circle, Polygon, Ellipse
from config import *
from markers import wrap
from navigator import Navigator
from world_kin import KinWorld
from mapping import RES

ap = argparse.ArgumentParser(); ap.add_argument('--out', default='demo.mp4'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--every', type=int, default=2, help='render every N control ticks (20 Hz sim)'); ap.add_argument('--fps', type=int, default=20)
ap.add_argument('--no-vision', action='store_true'); a = ap.parse_args()


world, nav = KinWorld(a.seed), Navigator(); frame = world.sense(); R = []; last_c, c_age = None, 99.0; done_t = None
while world.t < 240:
    v, w = nav.step(frame); world.step(v, w); frame = world.sense()
    if a.no_vision: frame.corners = None
    if frame.corners: last_c, c_age = frame.corners, 0.0
    else: c_age += DT
    if len(R) % 1 == 0 and world.k % a.every == 0:
        R.append(dict(t=world.t, xt=world.truth(), xe=nav.info['x'].copy(), P=nav.info['P'][:2, :2].copy(), ph=nav.phase,
                      scan=None if nav.last_scan is None else nav.last_scan.copy(), L=nav.grid.L.copy(), cor=last_c if c_age < 0.25 else None))
    if nav.phase == 'DONE':
        done_t = done_t or world.t
        if world.t - done_t > 2.0: break
print(f'simulated {world.t:.1f}s, {len(R)} frames')
T = np.array([r['t'] for r in R]); XT = np.array([r['xt'] for r in R]); XE = np.array([r['xe'] for r in R])
ERR = np.hypot(*(XE[:, :2] - XT[:, :2]).T) * 100; SIG3 = np.array([3 * np.sqrt(np.linalg.eigvalsh(r['P']).max()) for r in R]) * 100


fig = plt.figure(figsize=(13, 6.4)); gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1.15, 1])
M = fig.add_subplot(gs[:, 0]); C = fig.add_subplot(gs[0, 1]); E = fig.add_subplot(gs[1, 1])
xmin, xmax, ymin, ymax = ROOM; g0 = Navigator().grid
img = M.imshow(np.zeros((g0.ny, g0.nx)), origin='lower', cmap='gray', vmin=0, vmax=1, zorder=0,
               extent=[g0.x0, g0.x0 + g0.nx * RES, g0.y0, g0.y0 + g0.ny * RES])
M.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], 'k-', lw=2, zorder=2)
for cx, cy, hx, hy in OBSTACLES: M.add_patch(Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy, fill=False, ec='tab:brown', lw=1, ls=':', zorder=2))
for k, (x, y, p) in MARKERS.items():
    M.plot(x, y, 's', ms=7, color='tab:red' if k == DOCK_ID else 'tab:orange', zorder=3); M.text(x - .05, y + .1 * np.sign(-y) if abs(y) > 1.9 else y, f'ID{k}', fontsize=8, ha='right' if x > 0 else 'left')
mx, my, phi = MARKERS[DOCK_ID]; goal = np.array([mx, my]) + DOCK_DIST * np.array([np.cos(phi), np.sin(phi)])
M.plot(*goal, 'k*', ms=13, zorder=3, label='dock goal'); M.plot(*zip(*WAYPOINTS), 'g^', ms=9, zorder=3, label='waypoints')
tr_t, = M.plot([], [], 'b-', lw=1.5, label='truth'); tr_e, = M.plot([], [], 'r--', lw=1.2, label='EKF estimate')
hits = M.scatter([], [], s=4, c='tab:red', zorder=3); body = Circle((0, 0), ROBOT_R, fc='tab:blue', alpha=.35, ec='b', zorder=4); M.add_patch(body)
hd, = M.plot([], [], 'b-', lw=2, zorder=5); fov = Polygon(np.zeros((3, 2)), fc='yellow', alpha=.25, ec='none', zorder=1); M.add_patch(fov)
cov = Ellipse((0, 0), 0, 0, fc='none', ec='r', lw=1.2, zorder=5); M.add_patch(cov)
M.set_xlim(g0.x0, g0.x0 + g0.nx * RES); M.set_ylim(g0.y0, g0.y0 + g0.ny * RES); M.set_aspect('equal'); M.legend(loc='lower left', fontsize=8, ncol=2)
title = M.set_title(''); M.set_xlabel('x [m]'); M.set_ylabel('y [m]')
C.set_xlim(0, IMG_W); C.set_ylim(IMG_H, 0); C.set_facecolor('0.15'); C.set_xticks([]); C.set_yticks([]); C.set_title('Camera view (detected ArUco)')
camtxt = C.text(IMG_W / 2, IMG_H / 2, 'no marker in view', color='w', ha='center', va='center'); cam_artists = []
E.plot(T, ERR, 'r', lw=1, alpha=.25); E.set_xlim(0, T[-1]); E.set_ylim(0, max(10, min(ERR.max() * 1.2, SIG3[10:].max() * 1.1)))
el, = E.plot([], [], 'r', lw=1.5, label='EKF position error'); es, = E.plot([], [], 'k--', lw=1, label='3σ bound'); E.legend(fontsize=8, loc='upper right')
E.set_xlabel('t [s]'); E.set_ylabel('cm'); E.set_title('Estimation error vs truth (truth for display only)')
cur, = E.plot([], [], 'ko', ms=4)

def draw(i):
    r = R[i]; xt, xe = r['xt'], r['xe']
    img.set_data(np.where(r['L'] > 1, 0.05, np.where(r['L'] < -0.3, 1.0, 0.82)).T)
    tr_t.set_data(XT[:i + 1, 0], XT[:i + 1, 1]); tr_e.set_data(XE[:i + 1, 0], XE[:i + 1, 1])
    body.center = xt[:2]; hd.set_data([xt[0], xt[0] + ROBOT_R * np.cos(xt[2])], [xt[1], xt[1] + ROBOT_R * np.sin(xt[2])])
    if r['scan'] is not None:
        s = r['scan']; ok = np.isfinite(s); ang = xe[2] + np.linspace(0, 2 * np.pi, LIDAR_N, endpoint=False)[ok]    
        hits.set_offsets(np.c_[xe[0] + s[ok] * np.cos(ang), xe[1] + s[ok] * np.sin(ang)])
    h = np.deg2rad(31); fov.set_xy([xt[:2], xt[:2] + 1.2 * np.array([np.cos(xt[2] + h), np.sin(xt[2] + h)]), xt[:2] + 1.2 * np.array([np.cos(xt[2] - h), np.sin(xt[2] - h)])])
    w_, v_ = np.linalg.eigh(r['P']); cov.set_center(xe[:2]); cov.set_width(6 * np.sqrt(w_[1])); cov.set_height(6 * np.sqrt(w_[0])); cov.angle = np.rad2deg(np.arctan2(v_[1, 1], v_[0, 1]))
    title.set_text(f"t = {r['t']:5.1f} s    phase: {r['ph']}    (red ellipse = EKF 3σ)")
    for art in cam_artists: art.remove()
    cam_artists.clear()
    if r['cor']:
        camtxt.set_visible(False)
        for mid, c in r['cor'].items():
            cam_artists.append(C.add_patch(Polygon(c, fc='none', ec='lime', lw=2))); cam_artists.append(C.text(c[:, 0].mean(), c[:, 1].mean(), f'ID {mid}', color='lime', ha='center', va='center', fontsize=9))
    else: camtxt.set_visible(True)
    el.set_data(T[:i + 1], ERR[:i + 1]); es.set_data(T[:i + 1], SIG3[:i + 1]); cur.set_data([T[i]], [ERR[i]])
    return []


out = a.out
try:
    import imageio_ffmpeg; matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception: pass
if out.endswith('.mp4') and not an.writers.is_available('ffmpeg'):
    out = out[:-4] + '.gif'; print('ffmpeg not found -> writing GIF instead (pip install imageio-ffmpeg for mp4)')
ani = an.FuncAnimation(fig, draw, frames=len(R), blit=False)
ani.save(out, writer=an.FFMpegWriter(fps=a.fps, bitrate=3000) if out.endswith('.mp4') else an.PillowWriter(fps=a.fps), dpi=90)
print('saved', out)
