"""Detent-latch bottle simulation: two torque-controlled Panda arms, one free bottle."""
import math
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from .scene import scene

# Free cameras (rgb_robot_v3 contract): azimuth/elevation in degrees, distance in metres.
VIEWS = {
    'agentview': dict(azimuth=120, elevation=-24, distance=1.65, lookat=[-.05, 0, 1.]),
    'frontview': dict(azimuth=60, elevation=-8, distance=.30, lookat=[0, -.02, 1.04]),
    'topview': dict(azimuth=110, elevation=-40, distance=.29, lookat=[0, -.02, 1.05]),
}
WRISTS = ('left_wrist', 'right_wrist')
SIDES = ('left', 'right')
WORKSPACE = np.array([[-.30, -.40, .81], [.35, .40, 1.40]])
DETENT = (.018, .01, .018, .002)
HOME = [0, -.45, 0, -2.15, 0, 1.7, .7854]


class DetentBottleSim:
    def __init__(self, *, stop_on_success=True, max_steps=2400, wrist_cameras='hand', bottle_pose=None):
        self.stop_on_success = stop_on_success
        self.max_steps = max_steps
        self.latch = 'detent'
        self.xml = scene(wrist_cameras, bottle_pose)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model); self.ik = mujoco.MjData(self.model)
        m = self.model
        self.ids = [[m.joint(f'{s}_joint{i}').id for i in range(1, 8)] for s in SIDES]
        self.qids = [m.jnt_qposadr[j] for j in self.ids]
        self.dids = [m.jnt_dofadr[j] for j in self.ids]
        self.tips = [m.site(s + '_tip').id for s in SIDES]
        self.wrists = [m.site(s + '_wrist').id for s in SIDES]
        self.finger_qids = [[m.jnt_qposadr[m.joint(f'{s}_finger_{f}').id] for f in ('a', 'b')] for s in SIDES]
        self.finger_dids = [[m.jnt_dofadr[m.joint(f'{s}_finger_{f}').id] for f in ('a', 'b')] for s in SIDES]
        catch = m.joint('catch_hinge').id; self.catch_q, self.catch_d = m.jnt_qposadr[catch], m.jnt_dofadr[catch]
        self.renderers = {}
        self.reset()

    def reset(self):
        mj, m = mujoco, self.model
        mj.mj_resetData(m, self.data); mj.mj_resetData(m, self.ik)
        for ids in self.finger_qids: self.data.qpos[ids] = .045
        self.targets = np.array([[-.12, -.18, 1.12], [.13, .16, 1.12]])
        self.rotations = np.array([np.diag([1., -1., -1.])] * 2)
        self.grip_targets = np.full(2, .045)
        self.qtargets = []
        self.steps = 0; self.success = False; self.reason = None; self.open_time = 0.; self.overload_time = 0.
        for q in self.qids: self.data.qpos[q] = HOME
        mj.mj_forward(m, self.data)
        for a in range(2):
            self.data.qpos[self.qids[a]] = self.solve_ik(a, self.targets[a], self.rotations[a], 300)
            mj.mj_forward(m, self.data)
        self.qtargets = [self.data.qpos[q].copy() for q in self.qids]
        for _ in range(10): self.step(check=False)
        self.baseline = self.wrenches_world()
        self.data.time = 0.; self.steps = 0
        self.start_bottle = self.data.body('bottle').xpos.copy()

    def solve_ik(self, arm, target, rotation, iterations=20, q0=None):
        mj = mujoco
        for qi in self.qids + self.finger_qids: self.ik.qpos[qi] = self.data.qpos[qi]
        if q0 is not None: self.ik.qpos[self.qids[arm]] = q0
        qi, di, site = self.qids[arm], self.dids[arm], self.tips[arm]
        jp = np.zeros((3, self.model.nv)); jr = jp.copy()
        for _ in range(iterations):
            mj.mj_kinematics(self.model, self.ik); mj.mj_comPos(self.model, self.ik)
            rot = self.ik.site_xmat[site].reshape(3, 3)
            err = np.r_[target - self.ik.site_xpos[site], .3 * Rotation.from_matrix(rotation @ rot.T).as_rotvec()]
            if np.linalg.norm(err) < 1e-5: break
            mj.mj_jacSite(self.model, self.ik, jp, jr, site)
            J = np.vstack([jp[:, di], .3 * jr[:, di]])
            dq = J.T @ np.linalg.solve(J @ J.T + np.eye(6) * .0001, err)
            self.ik.qpos[qi] += np.clip(dq, -.08, .08)
            self.ik.qpos[qi] = np.clip(self.ik.qpos[qi], self.model.jnt_range[self.ids[arm], 0] + .01, self.model.jnt_range[self.ids[arm], 1] - .01)
        return self.ik.qpos[qi].copy()

    def preview(self, arm, position, rpy, steps=40):
        """Robot-only kinematic dry run of a target from the current joints: same per-tick
        waypoints and IK as step(), perfect tracking assumed, bottle ignored, physics untouched."""
        mj = mujoco; target = np.asarray(position, float)
        rotation = Rotation.from_euler('xyz', rpy, degrees=True).as_matrix()
        site, ids = self.tips[arm], self.ids[arm]; lo, hi = self.model.jnt_range[ids].T
        q = self.data.qpos[self.qids[arm]].copy(); collisions = set()
        for _ in range(steps):
            self.ik.qpos[self.qids[arm]] = q; mj.mj_kinematics(self.model, self.ik)
            current = self.ik.site_xpos[site].copy(); rot = self.ik.site_xmat[site].reshape(3, 3).copy(); delta = target - current
            waypoint = current + delta * min(1., .008 / max(np.linalg.norm(delta), 1e-9))
            dr = Rotation.from_matrix(rotation @ rot.T).as_rotvec()
            intermediate = Rotation.from_rotvec(dr * min(1., .08 / max(np.linalg.norm(dr), 1e-9))).as_matrix() @ rot
            q = self.solve_ik(arm, waypoint, intermediate, q0=q)
            mj.mj_fwdPosition(self.model, self.ik)
            for c in self.ik.contact[:self.ik.ncon]:
                a, b = (self.model.body(self.model.geom_bodyid[g]).name for g in (c.geom1, c.geom2))
                sides = {n.split('_')[0] for n in (a, b) if n.startswith(('left_', 'right_'))}
                if c.dist < 0 and (len(sides) == 2 or (sides and 'world' in (a, b))):
                    collisions.add('left arm vs right arm' if len(sides) == 2 else f'{sides.pop()} arm vs table')
        self.ik.qpos[self.qids[arm]] = q; mj.mj_kinematics(self.model, self.ik)
        error = Rotation.from_matrix(rotation @ self.ik.site_xmat[site].reshape(3, 3).T).magnitude()
        margin = np.minimum(q - lo, hi - q)
        return dict(position_error_mm=round(1000 * float(np.linalg.norm(target - self.ik.site_xpos[site])), 1),
                    rotation_error_deg=round(float(np.degrees(error)), 1),
                    joints_at_limit=[f'joint{i+1}' for i in np.flatnonzero(margin < .05)],
                    robot_collisions=sorted(collisions),
                    predicted_tool_xyz=self.ik.site_xpos[site].round(4).tolist())

    def reach_options(self, arm, position, direction, steps=40, samples=12):
        """Robot-only: sample tool rotations whose +Z points along `direction`, preview each, best first."""
        d = np.asarray(direction, float); d = d / max(np.linalg.norm(d), 1e-9)
        seed = np.array([1., 0, 0]) if abs(d[0]) < .9 else np.array([0, 1., 0])
        x0 = np.cross(d, seed); x0 /= np.linalg.norm(x0)
        rows = []
        for k in range(samples):
            a = 2 * math.pi * k / samples
            x = math.cos(a) * x0 + math.sin(a) * np.cross(d, x0)
            rpy = Rotation.from_matrix(np.column_stack([x, np.cross(d, x), d])).as_euler('xyz', degrees=True)
            p = self.preview(arm, position, rpy, steps)
            p['rpy_deg'] = [round(float(v), 1) for v in rpy]
            p['feasible'] = bool(p['position_error_mm'] <= 20 and p['rotation_error_deg'] <= 10 and not p['joints_at_limit'] and not p['robot_collisions'])
            rows.append(p)
        rows.sort(key=lambda p: (not p['feasible'], p['position_error_mm'] + p['rotation_error_deg']))
        return rows

    def target(self, arm, position, rpy, jaw):
        position = np.asarray(position, float); rpy = np.asarray(rpy, float)
        if position.shape != (3,) or rpy.shape != (3,) or not np.isfinite(np.r_[position, rpy, jaw]).all():
            raise ValueError('Expected finite XYZ, roll/pitch/yaw and jaw gap')
        if np.any(position < WORKSPACE[0]) or np.any(position > WORKSPACE[1]):
            raise ValueError('Outside worktable tool workspace: X[-.30,.35], Y[-.40,.40], Z[.81,1.40] m')
        if not 0 <= jaw <= .09 or np.any(abs(rpy) > 180): raise ValueError('Jaw gap must be 0-0.09 m; angles within ±180 degrees')
        self.targets[arm] = position
        self.rotations[arm] = Rotation.from_euler('xyz', rpy, degrees=True).as_matrix()
        self.grip_targets[arm] = jaw / 2

    def step(self, check=True):
        if self.reason: return
        mj = mujoco
        # Limit commanded Cartesian/rotational travel per 50 ms control tick; no object pose writes.
        for a in range(2):
            current = self.data.site_xpos[self.tips[a]]; delta = self.targets[a] - current
            waypoint = current + delta * min(1., .008 / max(np.linalg.norm(delta), 1e-9))
            rot = self.data.site_xmat[self.tips[a]].reshape(3, 3)
            dr = Rotation.from_matrix(self.rotations[a] @ rot.T).as_rotvec()
            intermediate = Rotation.from_rotvec(dr * min(1., .08 / max(np.linalg.norm(dr), 1e-9))).as_matrix() @ rot
            self.qtargets[a] = self.solve_ik(a, waypoint, intermediate)
        for i in range(round(.05 / self.model.opt.timestep)):
            for a in range(2):
                qi, di = self.qids[a], self.dids[a]
                tau = 450 * (self.qtargets[a] - self.data.qpos[qi]) - 40 * self.data.qvel[di] + self.data.qfrc_bias[di]
                self.data.ctrl[a*7:(a+1)*7] = np.clip(tau, [-80]*5 + [-12]*2, [80]*5 + [12]*2)
            self.data.ctrl[14:] = np.repeat(self.grip_targets, 2)
            self.data.qfrc_applied[self.catch_d] = self.detent_torque()
            mj.mj_step(self.model, self.data)
        mj.mj_forward(self.model, self.data); mj.mj_rnePostConstraint(self.model, self.data)
        self.steps += 1
        if check:
            state = self.mechanism()
            upright = self.data.body('bottle').xmat.reshape(3, 3)[2, 2] > .90
            self.open_time = self.open_time + .05 if state['lid_deg'] > 60 and upright else 0.
            self.success = self.success or self.open_time >= .4
            force = max(np.linalg.norm(self.data.sensor(s + '_force').data) for s in SIDES)
            self.overload_time = self.overload_time + .05 if force > 100 else 0.
            if any(w.number for w in self.data.warning) or not np.isfinite(self.data.qpos).all(): self.reason = 'numerical_instability'
            elif self.data.body('bottle').xpos[2] < .76: self.reason = 'bottle_fell'
            elif self.overload_time >= .2: self.reason = 'wrist_force_limit'
            elif self.success and self.stop_on_success: self.reason = 'success'
            elif self.steps >= self.max_steps: self.reason = 'time_limit'

    def detent_torque(self):
        # Gravity compensated everywhere. Below 30 deg: drive to 0; above 150: drive to 180; between: damped hold.
        # F0 clears the 0.012 N m lid-lip ramp at ~8 deg and the 0.002 N m hinge frictionloss; C sets ~1 rad/s, overdamped.
        F0, K, C, CH = DETENT
        th, v = self.data.qpos[self.catch_q], self.data.qvel[self.catch_d]; tau = self.data.qfrc_bias[self.catch_d]
        if th < math.radians(30): return tau - F0 - K * th - C * v
        if th > math.radians(150): return tau + F0 + K * (math.pi - th) - C * v
        return tau - CH * v

    def mechanism(self):
        return {name + '_deg': round(math.degrees(float(self.data.joint(joint).qpos[0])), 2)
                for name, joint in [('lid', 'lid_hinge'), ('loop', 'catch_hinge'), ('button', 'button_hinge')]}

    def wrenches_world(self):
        result = []
        for side, site in zip(SIDES, self.wrists):
            R = self.data.site_xmat[site].reshape(3, 3)
            result.append(np.r_[R @ self.data.sensor(side + '_force').data, R @ self.data.sensor(side + '_torque').data])
        return np.array(result)

    def finger_forces(self):
        return np.array([[float(self.data.sensor(f'{s}_finger_{f}_normal').data[0]) for f in ('a', 'b')] for s in SIDES])

    def openings(self):
        return np.array([self.data.qpos[qi].sum() for qi in self.finger_qids])

    def loads(self):
        return (self.wrenches_world() - self.baseline)[:, 2]

    def bottle_qpos(self):
        """Free-joint pose plus lid, button and loop hinge angles (privileged)."""
        m = self.model
        adr = [m.jnt_qposadr[m.joint(j).id] for j in ('bottle_free', 'lid_hinge', 'button_hinge', 'catch_hinge')]
        return np.r_[self.data.qpos[adr[0]:adr[0] + 7], self.data.qpos[adr[1:]]]

    def render(self, camera, size=640):
        """RGB uint8 image from a VIEWS name or a model camera name; robot marker sites hidden."""
        if size not in self.renderers: self.renderers[size] = mujoco.Renderer(self.model, height=size, width=size)
        r = self.renderers[size]
        if camera in VIEWS:
            v = VIEWS[camera]; cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = v['lookat']; cam.distance, cam.azimuth, cam.elevation = v['distance'], v['azimuth'], v['elevation']
        else:
            cam = camera
        option = mujoco.MjvOption(); option.sitegroup[:] = 0
        r.update_scene(self.data, camera=cam, scene_option=option)
        return r.render()

    def close(self):
        for r in self.renderers.values(): r.close()
        self.renderers = {}
