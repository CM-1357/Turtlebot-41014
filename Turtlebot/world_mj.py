import os
if 'DISPLAY' not in os.environ: os.environ.setdefault('MUJOCO_GL', 'egl')      
import numpy as np, cv2, mujoco
from config import *
from markers import wrap
from navigator import SensorFrame

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
PX, PAD = 200, 25                                       
FLIP = int(os.environ.get('TB3_MARKER_FLIP', '0'))      
HALF = (MARKER_SIDE / 2) * (PX + 2 * PAD) / PX          
R_TRUE, TRACK_TRUE = R_W * 1.012, L_W * 1.009           

def write_marker_pngs():
    os.makedirs(ASSETS, exist_ok=True)
    for mid in MARKERS:
        im = cv2.aruco.generateImageMarker(DICT, mid, PX)
        im = cv2.copyMakeBorder(im, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=255)
        if FLIP & 1: im = np.flipud(im)
        if FLIP & 2: im = np.fliplr(im)
        cv2.imwrite(os.path.join(ASSETS, f'marker{mid}.png'), np.ascontiguousarray(im))

def build_xml(start):
    write_marker_pngs(); xmin, xmax, ymin, ymax = ROOM; wh = 0.15
    tex = ''.join(f'<texture name="t{m}" type="2d" file="{ASSETS}/marker{m}.png"/>' for m in MARKERS)
    mat = ''.join(f'<material name="m{m}" texture="t{m}" texuniform="false" emission="0.8" specular="0"/>' for m in MARKERS)
    walls = (f'<geom type="box" pos="{xmax+.05} 0 {wh}" size=".05 {ymax+.1} {wh}" rgba=".75 .75 .7 1"/>'
             f'<geom type="box" pos="{xmin-.05} 0 {wh}" size=".05 {ymax+.1} {wh}" rgba=".75 .75 .7 1"/>'
             f'<geom type="box" pos="0 {ymax+.05} {wh}" size="{xmax+.1} .05 {wh}" rgba=".75 .75 .7 1"/>'
             f'<geom type="box" pos="0 {ymin-.05} {wh}" size="{xmax+.1} .05 {wh}" rgba=".75 .75 .7 1"/>')
    obst = ''.join(f'<geom type="box" pos="{cx} {cy} {wh}" size="{hx} {hy} {wh}" rgba=".55 .45 .35 1"/>' for cx, cy, hx, hy in OBSTACLES)
    marks = ''
    for m, (x, y, phi) in MARKERS.items():       
        marks += (f'<geom name="marker{m}" type="plane" group="3" contype="0" conaffinity="0" material="m{m}" '
                  f'pos="{x} {y} {MARKER_Z}" xyaxes="{-np.sin(phi)} {np.cos(phi)} 0 0 0 1" size="{HALF} {HALF} 0.01"/>')
    y_w = TRACK_TRUE / 2; sx, sy, sth = start
    return f"""<mujoco model="tb3_dock">
 <compiler autolimits="true"/><option timestep="0.002" gravity="0 0 -9.81"/>
 <visual><global offwidth="{IMG_W}" offheight="{IMG_H}"/></visual>
 <asset>{tex}{mat}<texture name="grid" type="2d" builtin="checker" width="256" height="256" rgb1=".85 .85 .85" rgb2=".78 .78 .78"/>
  <material name="floor" texture="grid" texrepeat="10 8" reflectance="0"/></asset>
 <worldbody>
  <light pos="0 0 3" dir="0 0 -1" diffuse=".7 .7 .7"/><light pos="1 1 3" dir="-.2 -.2 -1" diffuse=".3 .3 .3"/>
  <geom name="floor" type="plane" size="4 4 .1" material="floor"/>
  {walls}{obst}{marks}
  <body name="base" pos="{sx} {sy} {R_TRUE}">
   <freejoint name="root"/>
   <geom name="chassis" type="cylinder" size="{ROBOT_R} .05" pos="0 0 .045" mass="1.0" rgba=".2 .3 .9 1"/>
   <geom name="lidar_cap" type="cylinder" size=".035 .012" pos="-.03 0 .107" mass=".1" contype="0" conaffinity="0" rgba=".1 .1 .1 1"/>
   <site name="lidar" pos="0 0 {0.12 - R_TRUE}"/>
   <camera name="front" pos="{CAM_OFF} 0 {CAM_H - R_TRUE}" xyaxes="0 -1 0 0 0 1" fovy="{np.rad2deg(FOVY)}"/>
   <geom name="caster_f" type="sphere" size=".008" pos=".085 0 {-R_TRUE+.008}" mass=".02" friction=".001 .001 .0001"/>
   <geom name="caster_r" type="sphere" size=".008" pos="-.085 0 {-R_TRUE+.008}" mass=".02" friction=".001 .001 .0001"/>
   <body name="wl" pos="0 {y_w} 0"><joint name="wheel_l" type="hinge" axis="0 1 0" damping="0.001"/>
     <geom type="cylinder" size="{R_TRUE} .009" euler="90 0 0" mass=".05" friction="1.5 .005 .0001" rgba=".1 .1 .1 1"/></body>
   <body name="wr" pos="0 {-y_w} 0"><joint name="wheel_r" type="hinge" axis="0 1 0" damping="0.001"/>
     <geom type="cylinder" size="{R_TRUE} .009" euler="90 0 0" mass=".05" friction="1.5 .005 .0001" rgba=".1 .1 .1 1"/></body>
  </body>
 </worldbody>
 <actuator><velocity name="act_l" joint="wheel_l" kv=".3" forcerange="-1 1" ctrlrange="-8 8"/>
           <velocity name="act_r" joint="wheel_r" kv=".3" forcerange="-1 1" ctrlrange="-8 8"/></actuator>
</mujoco>"""

class MujocoWorld:
    def __init__(self, seed=0, render=True):
        self.rng = np.random.default_rng(seed)
        start = (START[0] + 0.03, START[1] - 0.02, START[2] + 0.035)         
        self.m = mujoco.MjModel.from_xml_string(build_xml(start)); self.d = mujoco.MjData(self.m)
        self.d.qpos[3:7] = [np.cos(start[2] / 2), 0, 0, np.sin(start[2] / 2)]; mujoco.mj_forward(self.m, self.d)
        g = lambda t, n: mujoco.mj_name2id(self.m, t, n)
        self.base = g(mujoco.mjtObj.mjOBJ_BODY, 'base'); self.site = g(mujoco.mjtObj.mjOBJ_SITE, 'lidar')
        self.qa = [self.m.jnt_qposadr[g(mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ('wheel_l', 'wheel_r')]
        self.chassis = g(mujoco.mjtObj.mjOBJ_GEOM, 'chassis'); self.floor = g(mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        self.sub = int(round(DT / self.m.opt.timestep)); self.k = 0; self.t = 0.0; self.collisions = 0
        self.renderer = mujoco.Renderer(self.m, IMG_H, IMG_W) if render else None
        p = cv2.aruco.DetectorParameters(); p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.det = cv2.aruco.ArucoDetector(DICT, p)
        self.ang = np.linspace(0, 2 * np.pi, LIDAR_N, endpoint=False)
        self.gmask = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)            

    def step(self, v, w):
        wl, wr = (v - w * L_W / 2) / R_W, (v + w * L_W / 2) / R_W            
        self.d.ctrl[:] = [wl, wr]
        for _ in range(self.sub): mujoco.mj_step(self.m, self.d)
        for c in self.d.contact[:self.d.ncon]:                                
            if self.chassis in (c.geom1, c.geom2) and self.floor not in (c.geom1, c.geom2): self.collisions += 1; break
        self.t += DT; self.k += 1

    def _yaw(self):
        R = self.d.xmat[self.base].reshape(3, 3); return np.arctan2(R[1, 0], R[0, 0])

    def _lidar(self):
        p = self.d.site_xpos[self.site].copy(); a = self._yaw() + self.ang
        vec = np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], 1)
        dist = np.full(LIDAR_N, -1.0); gid = np.full(LIDAR_N, -1, dtype=np.int32)
        try:
            mujoco.mj_multiRay(self.m, self.d, p, vec.ravel(), self.gmask, 1, self.base, gid, dist, LIDAR_N, LIDAR_RMAX)
        except Exception:                                                     
            for i in range(LIDAR_N):
                g1 = np.zeros(1, dtype=np.int32)
                dist[i] = mujoco.mj_ray(self.m, self.d, p, vec[i], self.gmask, 1, self.base, g1)
        r = np.where((dist >= 0) & (dist <= LIDAR_RMAX), dist + self.rng.normal(0, LIDAR_SIGMA, LIDAR_N), np.inf)
        r[r < LIDAR_RMIN] = np.nan; return r

    def _camera(self):
        self.renderer.update_scene(self.d, camera='front'); img = self.renderer.render()
        c, ids, _ = self.det.detectMarkers(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        if ids is None: return None
        out = {int(i): cc.reshape(4, 2) for cc, i in zip(c, ids.ravel()) if int(i) in MARKERS}
        return out or None

    def sense(self):
        q = 2 * np.pi / 4096; phi = np.array([self.d.qpos[a] for a in self.qa])
        return SensorFrame(self.t, tuple(np.floor(phi / q) * q),
                           self._lidar() if self.k % LIDAR_EVERY == 0 else None,
                           self._camera() if (self.renderer and self.k % CAM_EVERY == 0) else None)

    def truth(self):                       
        return np.array([self.d.qpos[0], self.d.qpos[1], self._yaw()])

if __name__ == '__main__':                  
    open('scene.xml', 'w').write(build_xml(START)); print('wrote scene.xml  ->  python -m mujoco.viewer --mjcf=scene.xml')
