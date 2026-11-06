
import numpy as np


R_W, L_W = 0.033, 0.160          
V_MAX, W_MAX = 0.22, 2.84        
ROBOT_R = 0.105                 


DT = 0.05                        
CAM_EVERY, LIDAR_EVERY = 2, 4    


CAM_OFF, CAM_H = 0.05, 0.10      
IMG_W, IMG_H = 640, 480
FOVY = np.deg2rad(48.8)          
FY = (IMG_H / 2) / np.tan(FOVY / 2); FX = FY
K_CAM = np.array([[FX, 0, IMG_W / 2], [0, FY, IMG_H / 2], [0, 0, 1.0]])
MARKER_SIDE, MARKER_Z = 0.15, 0.10   


LIDAR_N, LIDAR_RMIN, LIDAR_RMAX, LIDAR_SIGMA = 360, 0.12, 3.5, 0.01


ROOM = (-2.5, 2.5, -2.0, 2.0)                       
MARKERS = {0: (2.498, 0.8, np.pi),                  
           1: (-2.498, 1.0, 0.0),
           2: (-0.5, 1.998, -np.pi / 2),
           3: (1.0, -1.998, np.pi / 2)}
DOCK_ID, DOCK_DIST, STAGE_DIST = 0, 0.25, 0.70       
OBSTACLES = [(-0.8, 0.3, 0.25, 0.25), (0.6, -0.8, 0.10, 0.25),
             (1.2, 1.5, 0.15, 0.15), (-1.2, -0.9, 0.20, 0.20)]   
START = (-2.0, -1.4, 0.3)                           
START_SIGMA = (0.05, 0.05, np.deg2rad(5))
WAYPOINTS = [(-1.6, 0.9), (0.0, 0.2)]               
