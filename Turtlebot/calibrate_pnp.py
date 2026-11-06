from markers import *
rng = np.random.default_rng(0)
print(" r    view   sd_r[m]  sd_beta[deg]  sd_psi[deg]  bias_psi[deg]")
for r in [0.3, 0.5, 0.8, 1.2, 1.8, 2.5]:
    for view in [0, 30]:        
        e = []
        for _ in range(300):
            phi = np.pi; a = np.deg2rad(view) + rng.uniform(-.05, .05)
            m = np.array(MARKERS[0][:2]); n = np.array([np.cos(phi), np.sin(phi)])
            p = m + r * np.array([np.cos(phi + a), np.sin(phi + a)]) - CAM_OFF * 0
            th = np.arctan2(*(m - p)[::-1]) + rng.uniform(-.2, .2)
            pose = (p[0] - CAM_OFF * np.cos(th), p[1] - CAM_OFF * np.sin(th), th)
            px, zc = project(corners_world(0), pose)
            if np.any(zc <= 0) or np.any(np.abs(px[:, 0] - 320) > 320) : continue
            z = pnp_measurement(px + rng.normal(0, 0.5, px.shape))
            if z is None: continue
            d = m - (np.array(pose[:2]) + CAM_OFF * np.array([np.cos(th), np.sin(th)]))
            zt = np.array([np.hypot(*d), wrap(np.arctan2(d[1], d[0]) - th), wrap(phi - th)])
            e.append([z[0] - zt[0], wrap(z[1] - zt[1]), wrap(z[2] - zt[2])])
        e = np.array(e)
        print(f"{r:4.1f} {view:4d}   {e[:,0].std():.4f}   {np.rad2deg(e[:,1].std()):.3f}        "
              f"{np.rad2deg(e[:,2].std()):.2f}        {np.rad2deg(e[:,2].mean()):.2f}   (model: {meas_cov(r)[0,0]**.5:.4f}, {np.rad2deg(meas_cov(r)[1,1]**.5):.2f}, {np.rad2deg(meas_cov(r)[2,2]**.5):.1f})")
