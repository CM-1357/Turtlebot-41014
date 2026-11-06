import numpy as np, mujoco
from config import *
from world_mj import MujocoWorld

w = MujocoWorld(0, render=False); m, d = w.m, w.d
gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f'geom{g}'
dof = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ('wheel_l', 'wheel_r')]
d.qpos[0:2] = [-1.5, -1.5]; d.qpos[3:7] = [1, 0, 0, 0]; d.qvel[:] = 0; mujoco.mj_forward(m, d)

def report(tag):
    R = d.xmat[w.base].reshape(3, 3); f = np.zeros(6)
    print(f'[{tag}] t={w.t:4.2f}s z={d.qpos[2]:.4f} pitch={np.rad2deg(-np.arcsin(R[2,0])):+.2f} roll={np.rad2deg(np.arctan2(R[2,1],R[2,2])):+.2f} deg | '
          f'base vx={d.qvel[0]:+.3f} m/s | wheel w=({d.qvel[dof[0]]:+.2f},{d.qvel[dof[1]]:+.2f}) rad/s (cmd {0.1/R_W:.2f}) | x={d.qpos[0]:+.3f}')
    for i in range(d.ncon):
        c = d.contact[i]; mujoco.mj_contactForce(m, d, i, f)
        print(f'      contact {gname(c.geom1):>14s} - {gname(c.geom2):<14s} normal={f[0]:6.2f} N  tangent=({f[1]:+.2f},{f[2]:+.2f})  dist={c.dist:+.4f}')

for _ in range(int(0.5 / DT)): w.step(0.0, 0.0)
report('settled, stationary')
x0 = d.qpos[0]; phi0 = np.array([d.qpos[a] for a in w.qa])
for k in range(int(3 / DT)):
    w.step(0.1, 0.0)
    if k in (1, 5, 20, 59): report(f'driving k={k}')
phi = np.array([d.qpos[a] for a in w.qa]) - phi0
print(f'\nwheel rotation {phi.round(2)} rad  -> rolling distance at true radius {phi.mean()*R_W*1.012:.3f} m;  base moved {d.qpos[0]-x0:.3f} m')
