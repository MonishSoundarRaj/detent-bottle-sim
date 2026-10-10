"""Agent-facing adapter for the detent bottle: v3 tool schemas, instructions, force guard, touch and locate."""
import base64
import io
import math
import mujoco
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from detent_bottle.core import DetentBottleSim
from .overlay import draw_tools
from .router import HOLD_N, JUMP_N

WRISTS = ['left_wrist', 'right_wrist']
CONTRACT = 'rgb_robot_dual_v1'
VEC3 = {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}
JAW = {'type': 'number', 'minimum': 0, 'maximum': .09}
STEPS = {'type': 'integer', 'minimum': 1, 'maximum': 40}
FORCE_N = 30.
HELD_TORQUE_NM, HELD_LOAD_N, HELD_EMPTY_N = 8., 30., 5.
TOUCH_HOLD = 3
FORCE = {'type': ['number', 'null'], 'minimum': 5, 'maximum': 60}
WAIT = {'type': 'function', 'name': 'wait', 'strict': True,
        'description': 'Hold both arms at their current goals and jaw targets and advance 1-60 control ticks of 0.05 s. '
                       'Use it to let evaluator hold timers elapse with the hands still; returns both arms\' feedback.',
        'parameters': {'type': 'object', 'properties': {'ticks': {'type': 'integer', 'minimum': 1, 'maximum': 60}},
                       'required': ['ticks'], 'additionalProperties': False}}
TARE = {'type': 'function', 'name': 'tare', 'strict': True,
        'description': 'Re-zero load_n: store the current wrist loads of both arms as the baseline, so load_n reads relative to them. No physics advances.',
        'parameters': {'type': 'object', 'properties': {}, 'required': [], 'additionalProperties': False}}
RES = 640
LOCATE_NOTE = ('\nlocate returns the 3D world point under given pixels of one camera from a depth render of that camera; give the pixels as '
               f'[u, v] in the latest {RES}x{RES} image you received from that camera (u right, v down, origin top-left). It advances no physics.')
FORCE_NOTE = ('\ntouch moves one gripper along a straight line until its wrist force or a fingertip force changes and reports the contact point and force.'
              '\nEvery move_arm, move_both_arms, wait and touch result has force_trace: per commanded arm, numbers on how its wrist force, torque, '
              'load_n and fingertip forces changed during that call.')
ROUTER_NOTE = ('\nA camera router omits the wrist image of a hand that is holding steadily and was not '
               'just commanded; a note names each omitted view and why. Use request_views to get one.')
SCAFFOLD = 'uni-1'
TEMPLATE = """You operate two Panda robot arms on a simulated table.
Your observations contain only RGB camera images and robot proprioception: joint/tool
poses, jaw openings, command targets, wrist force/torque and fingertip normal forces.
Object poses, object dimensions, masses, contact identities, material parameters, object
internal states and evaluator results are not provided; infer them from images and physical
interaction. No demonstration trajectory or procedure is supplied. No object controls exist:
objects move only through contact with the robot.
{pose}
Jaw gap range 0-0.09 m; the jaws are position-controlled with bounded force, so a closed jaw
on an object squeezes it. Robot workspace in world metres: {workspace}; these are interface
limits, not object coordinates. Arm bases: {bases}; table top at Z {table:.2f}.
move_arm sets one arm's goal (the other keeps its goal); move_both_arms sets both goals at
once, which you need when both hands hold one object. {tick} An obstructed arm keeps pressing
toward its target, so watch tracking error and forces. {abort} load_n is the tared world-Z
wrist load (positive = supporting weight); wrench_world is world-frame force/torque at each wrist.
Cameras: {views}; all three are aimed at the workspace centre {centre} and carry overlays drawn
from robot proprioception only: a filled dot marks each measured tool XYZ (L blue, R orange),
a ring its commanded target, and arrows at R show world +X (red), +Y (green), +Z (blue), 30 mm
each. Overlays are not object features. 'left_wrist' and 'right_wrist' are RGB cameras rigidly
mounted on each hand (200 mm outboard: right on tool -Y, left on tool +Y; aimed past the
fingertips toward the hand's centre line, fingers at the image bottom, fovy 75); they move with
the hand. Observe, check_reach and find_reach never advance physics. Do not equate a commanded
motion with a result. Verify contact before loading it; resistance is not proof of contact on
the intended part.
Use check_reach to test a specific pose and find_reach to compare the orientations (and the arm)
that reach a position; both advance no physics. Keep update_notes current.{notes}
{scene}
{success}"""
BOTTLE_POSE = """Tool XYZ is the midpoint of the distal fingertip ends; each fingertip is 8 mm wide with a 24 mm
distal contact length. Orientation rpy_deg is fixed-axis XYZ roll/pitch/yaw in degrees; tool local X
is the jaw opening axis and local +Z points from palm toward fingertips ([180,0,0] points the
fingers straight down with the jaw axis along world X; [90,0,0] points them toward world -Y)."""
BOTTLE_TICK = ('Each 0.05 s control tick moves the commanded target at most 8 mm and 0.08 rad toward its goal, '
               'measured from the measured tool pose.')
BOTTLE_SCENE = 'Scene: a grey bottle with a dark lid stands on the table.'
BOTTLE_SUCCESS = """Success (physics only): the lid is open by more than 60 degrees from its closed position while the
bottle stays upright (tilt below 25 degrees), for 0.4 continuous seconds. The episode ends if the
bottle falls off the table."""


ARM_FIELDS = ('name', 'eef', 'rpy_deg', 'joints', 'target', 'target_rpy_deg', 'jaw_target_m', 'opening_mm',
              'wrist_wrench', 'wrench_world', 'load_n', 'finger_forces_n')


def arm_observation(arm):
    return {key: arm[key] for key in ARM_FIELDS if key in arm}


def fn(name, description, properties):
    return {'type': 'function', 'name': name, 'description': description, 'strict': True,
            'parameters': {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}}


def camera_rule(lo, hi, bases, table):
    # Framed on the workspace centre, never on an object; the side facing away from the arm bases is the front.
    c = np.r_[(lo[:2] + hi[:2]) / 2, table + .15]; b = bases[:, :2]
    s = np.array([b[0, 1] - b[1, 1], b[1, 0] - b[0, 0]]); s /= np.linalg.norm(s)
    if s @ (c[:2] - b.mean(0)) < 0: s = -s
    r, az = np.linalg.norm(hi[:2] - lo[:2]) / 2, math.degrees(math.atan2(-s[1], -s[0]))
    return {'agentview': dict(azimuth=az + 45, elevation=-45, distance=float(2 * r), lookat=c.tolist()),
            'frontview': dict(azimuth=az - 45, elevation=-15, distance=float(1.8 * r), lookat=c.tolist()),
            'topview': dict(azimuth=az, elevation=-89.9, distance=float(2.4 * r), lookat=c.tolist())}, c


class BottleTask:
    contract = CONTRACT
    wrists = WRISTS
    lo, hi = np.array([-.30, -.40, .81]), np.array([.35, .40, 1.40])
    ws_text = 'X[-.30,.35], Y[-.40,.40], Z[.81,1.40] m'
    step_m = .008
    rot_key, rot_text = 'rpy_deg', 'roll/pitch/yaw'
    rot_schema = {'type': 'array', 'items': {'type': 'number', 'minimum': -180, 'maximum': 180}, 'minItems': 3, 'maxItems': 3}
    pose_text, tick_text, abort_n, abort_s = BOTTLE_POSE, BOTTLE_TICK, 100, .2
    reach_text = 'sample 12 rotations about the world direction tool_direction that the tool +Z (palm to fingertips) should point'
    reach_props = {'tool_direction': VEC3}

    def __init__(self, seed=0):
        self.id, self.seed = 'custom_bottle:dual_arm_detent', seed
        self.env = DetentBottleSim(stop_on_success=False, wrist_cameras='boom'); self.ikd = self.env.ik; self.renderers = {}
        self.goal_text = 'Open the bottle using the robot grippers.'
        self.blocks, self.finger_abort, self.extra_tools, self.declarations = (BOTTLE_SCENE, BOTTLE_SUCCESS), False, [TARE], {'tare'}
        self.dry_move = dict(arm='right', position=[.1, .1, 1.1], rpy_deg=[180, 0, 0], jaw_gap_m=.05, steps=5)
        self.dry_reach = dict(tool_direction=[0, 0, -1])
        self.init_guard()

    def init_guard(self):
        self.guard, self.over, self.last_limit, self.trace, self.load_zero = None, [0, 0], [FORCE_N, FORCE_N], None, np.zeros(2)
        m = self.env.model; hand = [{m.body(f'{s}_{b}').id for b in ('finger_a', 'finger_b')} for s in ('left', 'right')]
        self.hand_geoms = [[g for g in range(m.ngeom) if m.geom_bodyid[g] in hand[a] and m.geom_contype[g] and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX] for a in range(2)]

    def setup(self):
        e = self.env; m, d = e.model, e.data
        bases = np.array([d.xpos[m.body(s + '_base').id] for s in ('left', 'right')])
        facing = ['+X' if R[0, 0] > .7 else '-X' if R[0, 0] < -.7 else '+Y' if R[1, 0] > 0 else '-Y' for R in (d.xmat[m.body(s + '_base').id].reshape(3, 3) for s in ('left', 'right'))]
        g = m.geom('table').id; table = float(d.geom_xpos[g, 2] + m.geom_size[g, 2])
        self.views, centre = camera_rule(self.lo, self.hi, bases, table)
        self.cameras = list(self.views) + WRISTS
        where = lambda v: np.round(np.asarray(v['lookat']) - v['distance'] * np.array([math.cos(math.radians(v['elevation'])) * math.cos(math.radians(v['azimuth'])),
                                   math.cos(math.radians(v['elevation'])) * math.sin(math.radians(v['azimuth'])), math.sin(math.radians(v['elevation']))]), 2).tolist()
        views = (f"'agentview' (free camera at {where(self.views['agentview'])}, 45 deg down), 'frontview' (free camera at {where(self.views['frontview'])}, "
                 f"15 deg down, from the other front quadrant) and 'topview' (straight down from {where(self.views['topview'])})")
        abort = (f'A wrist force above {self.abort_n} N for {self.abort_s} s aborts the episode'
                 + f'; so do numerical instability and the {e.max_steps}-tick limit.')
        self.instructions = TEMPLATE.format(pose=self.pose_text, workspace=self.ws_text.removesuffix(' m'), table=table, tick=self.tick_text, abort=abort, views=views,
                                            bases=', '.join(f'{s} at ({b[0]:.2f},{b[1]:.2f}) facing {f}' for s, b, f in zip(('left', 'right'), bases, facing)),
                                            centre=f'({centre[0]:.2f},{centre[1]:.2f},{centre[2]:.2f})', notes=LOCATE_NOTE + FORCE_NOTE + ROUTER_NOTE,
                                            scene=self.blocks[0], success=self.blocks[1])
        rot = {self.rot_key: self.rot_schema}
        pose = {'position': VEC3, **rot, 'jaw_gap_m': JAW, 'max_force_n': FORCE}
        arm = {'type': 'string', 'enum': ['left', 'right']}
        guard = (' Force guard, checked every tick: the move of an arm stops early if its wrist force norm stays above max_force_n '
                 '(null = 30 N; 5-60) for 0.1 s, or, when max_force_n is 30 or less, if a fingertip force appears or rises while that '
                 'jaw is not closing. The stopped arm then holds its measured pose and the result reports stopped_by and guard_events. '
                 f'Raise max_force_n for intended pushes or fingertip contact; the {self.abort_n} N abort still applies. '
                 f'The whole move also stops if the arm not being commanded stays above its last max_force_n for 0.1 s (force_uncommanded), or if a '
                 f'hand whose closed jaw holds something sees its wrist torque change by more than {HELD_TORQUE_NM:.0f} N m or its load_n by more than '
                 f'{HELD_LOAD_N:.0f} N within one tick from a |load_n| of at least {HELD_EMPTY_N:.0f} N (held_object_change).')
        self.tools = [fn('observe', 'Observe the RGB cameras and robot proprioception without advancing physics.', {}),
                      fn('move_arm', f'Set one gripper goal (XYZ, {self.rot_text}, jaw gap) and advance 1-40 control ticks of 0.05 s. '
                         'The other arm keeps its goal. Motion through native motor control and contact only.' + guard,
                         {'arm': arm, **pose, 'steps': STEPS}),
                      fn('move_both_arms', f'Set both gripper goals at once (each XYZ, {self.rot_text}, jaw gap) and advance 1-40 control ticks. '
                         'Use for coordinated two-hand motion. max_force_n is per arm.' + guard,
                         {'left': {'type': 'object', 'properties': pose, 'required': list(pose), 'additionalProperties': False},
                          'right': {'type': 'object', 'properties': pose, 'required': list(pose), 'additionalProperties': False}, 'steps': STEPS}),
                      fn('check_reach', 'Preview a move_arm command without moving: robot-only kinematic dry run from the current joints with the '
                         'same per-tick target limits. Reports predicted position/rotation error, joints at limits and arm-arm or arm-table '
                         'collisions. Ignores every non-robot object except the table, and dynamic lag.',
                         {'arm': arm, 'position': VEC3, **rot, 'steps': STEPS}),
                      fn('find_reach', f'Robot-only search for orientations: given a tool position, {self.reach_text}, and preview each from the current '
                         'joints (same kinematic dry run as check_reach). Returns candidates best-first with the orientation, predicted errors, joint limits '
                         'and collisions; feasible=true means usable as a move_arm orientation. Ask for one arm or both to compare which arm can do a job. '
                         'No physics advances.', {'arm': {'type': 'string', 'enum': ['left', 'right', 'both']}, 'position': VEC3, **self.reach_props, 'steps': STEPS}), WAIT,
                      fn('locate', f'RGB-D back-projection: for 1-8 pixels [u, v] of the latest {RES}x{RES} image of one camera (u right, v down, origin '
                         'top-left), render depth from that camera at the current simulation state and return the world XYZ (m) of the first surface '
                         'along each pixel ray, its depth_m along the camera axis, and valid=false where the ray hits nothing. Also returns the camera '
                         'world position and the control step of that image vs now; if they differ, the scene may have changed since the image. No physics advances.',
                         {'camera': {'type': 'string', 'enum': self.cameras},
                          'points': {'type': 'array', 'minItems': 1, 'maxItems': 8,
                                     'items': {'type': 'array', 'minItems': 2, 'maxItems': 2, 'items': {'type': 'integer', 'minimum': 0, 'maximum': RES - 1}}}}),
                      fn('touch', 'Force probe: move one gripper in a straight line along a world-frame direction (normalised by the tool) at up to 8 mm per tick, '
                         'holding its orientation and jaw gap (or the given ones), and stop at the first tick where its world-frame wrist force differs from its value at '
                         'the start of the call by more than force_n (null = 5 N), or a fingertip force that was at most 0.5 N exceeds 0.5 N, or after max_distance_m. The arm then '
                         'holds its measured pose for 3 more ticks; the other arm keeps its goal. Returns contacted, contact_point (the leading surface of the fingers '
                         'along the direction, on the line through the measured tool XYZ), tool_xyz (measured tool XYZ), travel_m (tool displacement along the '
                         'direction), ticks (moving) and hold_ticks, force_on_hand_n (world-frame force the contact exerts on the hand, i.e. minus the change of the wrench_world force) '
                         'as the median over the hold ticks, with its magnitude, impact_peak_n (force magnitude at the stop tick), surface_normal (unit vector of the '
                         'force on the hand at the stop tick; null if impact_peak_n is under 1 N), fingertip forces, guard_events if any, and force_trace. The force guard runs with '
                         f'max_force_n = min(max(2 x force_n, 30), 60); the {self.abort_n} N abort still applies.',
                         {'arm': arm, 'direction': VEC3, 'max_distance_m': {'type': 'number', 'minimum': .005, 'maximum': .30},
                          'force_n': {'type': ['number', 'null'], 'minimum': 1, 'maximum': 30}, self.rot_key: {**self.rot_schema, 'type': ['array', 'null']},
                          'jaw_gap_m': {**JAW, 'type': ['number', 'null']}})] + self.extra_tools

    def goal_pose(self, i):
        return self.env.targets[i], self.env.rotations[i]

    def arms(self):
        e = self.env; loads = self.loads(); world = e.wrenches_world(); openings = e.openings(); fingers = e.finger_forces()
        out = []
        for i, side in enumerate(('left', 'right')):
            out.append(dict(name=side, eef=e.data.site_xpos[e.tips[i]].tolist(),
                            rpy_deg=Rotation.from_matrix(e.data.site_xmat[e.tips[i]].reshape(3, 3)).as_euler('xyz', degrees=True).tolist(),
                            joints=e.data.qpos[e.qids[i]].tolist(), target=self.goal_pose(i)[0].tolist(),
                            target_rpy_deg=Rotation.from_matrix(self.goal_pose(i)[1]).as_euler('xyz', degrees=True).tolist(),
                            jaw_target_m=float(2 * e.grip_targets[i]), opening_mm=float(openings[i] * 1000),
                            wrist_wrench=np.r_[e.data.sensor(side + '_force').data, e.data.sensor(side + '_torque').data].tolist(),
                            wrench_world=world[i].tolist(), load_n=float(loads[i]), finger_forces_n=fingers[i].tolist()))
        return out

    def render_jpeg(self, view, draw=None):
        e = self.env
        if RES not in self.renderers: self.renderers[RES] = mujoco.Renderer(e.model, height=RES, width=RES)
        r = self.renderers[RES]; option = mujoco.MjvOption(); option.sitegroup[:] = 0
        r.update_scene(e.data, camera=self.camera(view), scene_option=option)
        image = Image.fromarray(r.render())
        if draw: image = draw(image, r.scene, e.model.vis.global_.fovy)
        out = io.BytesIO(); image.save(out, 'JPEG', quality=85)
        return 'data:image/jpeg;base64,' + base64.b64encode(out.getvalue()).decode()

    def snapshot(self, images=True):
        arms = self.arms(); snap = dict(arms=arms, images={})
        if images:
            for name in self.views:
                snap['images'][name] = self.render_jpeg(name, lambda image, scene, fovy: draw_tools(image, scene, fovy, arms))
            for name in WRISTS:
                snap['images'][name] = self.render_jpeg(name)
        return snap

    def model_state(self, snap):
        e = self.env
        return dict(observation_contract=CONTRACT, goal=self.goal_text, control_step=e.steps,
                    simulation_time_s=float(e.data.time), arms=[arm_observation(a) for a in snap['arms']])

    def parse_target(self, cmd):
        if self.env.reason: raise ValueError('Episode ended; reset before moving')
        p, rpy, jaw = np.asarray(cmd['position'], float), np.asarray(cmd['rpy_deg'], float), float(cmd['jaw_gap_m'])
        if p.shape != (3,) or rpy.shape != (3,) or not np.isfinite(np.r_[p, rpy, jaw]).all(): raise ValueError('Expected finite XYZ, roll/pitch/yaw and jaw gap')
        if np.any(p < self.lo) or np.any(p > self.hi): raise ValueError('Outside tool workspace: ' + self.ws_text)
        if not 0 <= jaw <= .09 or np.any(abs(rpy) > 180): raise ValueError('Jaw gap must be 0-0.09 m; angles within ±180 degrees')
        if not 5 <= (cmd.get('max_force_n') or FORCE_N) <= 60: raise ValueError('max_force_n must be 5-60 N or null')
        return p, rpy, jaw

    def command(self, name, args):
        pairs = [(args['arm'], args)] if name == 'move_arm' else [('left', args['left']), ('right', args['right'])]
        parsed = [(int(arm == 'right'), self.parse_target(cmd)) for arm, cmd in pairs]
        for a, (p, rpy, jaw) in parsed: self.env.target(a, p, rpy, jaw)
        limits = {int(arm == 'right'): cmd.get('max_force_n') or FORCE_N for arm, cmd in pairs}
        for a, n in limits.items(): self.last_limit[a] = n
        self.arm_guard(limits, args['steps'], True)
        return [arm for arm, _ in pairs]

    def reach_options(self, a, args):
        return self.env.reach_options(a, args['position'], args['tool_direction'], args['steps'])

    def preview(self, args):
        return self.env.preview(int(args['arm'] == 'right'), args['position'], args['rpy_deg'], args['steps'])

    def held(self, a, f, op):
        return f[a].min() > HOLD_N and 2 * self.env.grip_targets[a] < op[a] - .002

    def arm_guard(self, limits, ticks, hold):
        e = self.env; f, op = e.finger_forces(), e.openings(); w, loads = e.wrenches_world(), self.loads()
        self.guard = dict(hold=hold, ticks=ticks, tick=0, events=[], over=[0, 0], fired=set(), prior_stops=self.stops(),
                          prev=[(w[a, 3:].copy(), float(loads[a])) if self.held(a, f, op) else None for a in range(2)], arms={a: dict(limit=float(n), f0=f[a].copy(), stopped=False,
                          fingers=n <= FORCE_N and 2 * e.grip_targets[a] >= op[a] - .002) for a, n in limits.items()})
        self.trace = dict(arms=sorted(limits), w0=w.copy(), f0=f.copy(), ticks=[])

    def force_trace(self):
        t, out = self.trace, {}
        for a in t['arms'] if t and t['ticks'] else ():
            d = np.array([w[a, :3] - t['w0'][a, :3] for w, _ in t['ticks']]); n = np.linalg.norm(d, axis=1); f = [x[a] for _, x in t['ticks']]; k = int(n.argmax())
            first = next((i + 1 for i in range(len(n)) if n[i] > 3 or ((f[i] > HOLD_N) & (t['f0'][a] <= HOLD_N)).any()), None)
            w1 = t['ticks'][-1][0][a]
            out[('left', 'right')[a]] = dict(ticks=len(n), first_contact_tick=first, peak_force_n=round(float(n[k]), 2), peak_tick=k + 1,
                                             peak_direction=(d[k] / n[k]).round(3).tolist() if n[k] > 1e-6 else None, final_force_n=round(float(n[-1]), 2),
                                             finger_forces_n=[t['f0'][a].round(2).tolist(), f[-1].round(2).tolist()],
                                             torque_delta_nm=round(float(np.linalg.norm(w1[3:] - t['w0'][a, 3:])), 3), load_delta_n=round(float(w1[2] - t['w0'][a, 2]), 2))
        return out

    def touch(self, args, tick=None):
        e = self.env; a = int(args['arm'] == 'right')
        if e.reason: raise ValueError('Episode ended; reset before moving')
        d = np.asarray(args['direction'], float); dist, force = float(args['max_distance_m']), float(args['force_n'] or 5.)
        if d.shape != (3,) or not np.isfinite(d).all() or np.linalg.norm(d) < 1e-6: raise ValueError('Expected a finite non-zero direction')
        if not (.005 <= dist <= .30 and 1 <= force <= 30): raise ValueError('max_distance_m must be 0.005-0.30 and force_n 1-30 N or null')
        d = d / np.linalg.norm(d); goal = e.targets[a] + d * dist
        if np.any(goal < self.lo) or np.any(goal > self.hi): raise ValueError('Touch end point outside tool workspace: ' + self.ws_text)
        if args['jaw_gap_m'] is not None: e.grip_targets[a] = min(max(float(args['jaw_gap_m']), 0.), .09) / 2
        if args['rpy_deg'] is not None: e.rotations[a] = Rotation.from_euler('xyz', args['rpy_deg'], degrees=True).as_matrix()
        e.targets[a] = goal
        self.last_limit[a] = limit = min(max(2 * force, FORCE_N), 60.)
        ticks = math.ceil(dist / self.step_m) + 2
        self.arm_guard({a: limit}, ticks, True)
        tick = tick or (lambda: 0 if e.reason else (self.step(), 1)[1])
        w0, f0, p0 = e.wrenches_world()[a].copy(), e.finger_forces()[a].copy(), e.data.site_xpos[e.tips[a]].copy()
        executed, contacted = 0, False
        for _ in range(ticks):
            if not tick(): break
            executed += 1; w, f = e.wrenches_world()[a], e.finger_forces()[a]
            contacted = bool(np.linalg.norm(w[:3] - w0[:3]) > force or ((f > HOLD_N) & (f0 <= HOLD_N)).any())
            if contacted or self.guard['arms'][a]['stopped'] or e.reason: break
        P = -(e.wrenches_world()[a, :3] - w0[:3]); peak = float(np.linalg.norm(P)); held = []
        if not e.reason: self.hold_here(a)
        for _ in range(TOUCH_HOLD if not e.reason else 0):
            if not tick(): break
            held.append(-(e.wrenches_world()[a, :3] - w0[:3]))
        tip = e.data.site_xpos[e.tips[a]].copy(); F = np.median(held, axis=0) if held else -(e.wrenches_world()[a, :3] - w0[:3]); n = float(np.linalg.norm(F))
        corners = [e.data.geom_xpos[g] + e.data.geom_xmat[g].reshape(3, 3) @ (e.model.geom_size[g] * s) for g in self.hand_geoms[a] for s in np.array(np.meshgrid(*[[-1, 1]] * 3)).reshape(3, -1).T]
        lead = max(float((c - tip) @ d) for c in corners)
        return dict(contacted=contacted, contact_point=(tip + d * lead).round(4).tolist(), tool_xyz=tip.round(4).tolist(), travel_m=round(float((tip - p0) @ d), 4),
                    ticks=executed, hold_ticks=len(held), force_on_hand_n=F.round(2).tolist(), force_n=round(n, 2), impact_peak_n=round(peak, 2), surface_normal=(P / peak).round(3).tolist() if contacted and peak >= 1 else None,
                    finger_forces_n=e.finger_forces()[a].round(2).tolist(), force_trace=self.force_trace())

    def stops(self):
        return {('left', 'right').index(v['arm']) for v in self.guard['events'] if v['kind'] in ('force', 'finger_contact')} if self.guard else set()

    def check_guard(self):
        e, g = self.env, self.guard
        w = e.wrenches_world(); norms = np.linalg.norm(w[:, :3], axis=1); f, op, loads = e.finger_forces(), e.openings(), self.loads()
        limits = g['arms'] if g and g['tick'] < g['ticks'] else {}
        lim = [limits[a]['limit'] if a in limits else self.last_limit[a] for a in range(2)]
        for a in range(2): self.over[a] = self.over[a] + 1 if g and g['hold'] and limits and norms[a] > lim[a] else 0
        if not limits: return False
        over = self.over if g['hold'] else g['over']
        if not g['hold']:
            for a in range(2): over[a] = over[a] + 1 if a in limits and norms[a] > limits[a]['limit'] else 0
        g['tick'] += 1
        for a, s in limits.items():
            if s['stopped']: continue
            kind = ('force' if over[a] >= 2 else 'finger_contact' if s['fingers'] and (
                (f[a].max() > HOLD_N and s['f0'].max() <= HOLD_N) or (f[a] - s['f0']).max() > JUMP_N) else None)
            if not kind: continue
            s['stopped'] = True
            g['events'].append(dict(kind=kind, arm=('left', 'right')[a], tick=g['tick'], wrist_force_n=round(float(norms[a]), 1), limit_n=s['limit'],
                                    finger_forces_n=f[a].round(2).tolist(), load_n=round(float(loads[a]), 1)))
            if g['hold']: self.hold_here(a)
        whole = []
        for a in range(2):
            if g['hold'] and a not in limits and self.over[a] >= 2 and 'force' not in g['fired']:
                g['fired'].add('force'); whole.append(dict(kind='force_uncommanded', arm=('left', 'right')[a], tick=g['tick'], wrist_force_n=round(float(norms[a]), 1),
                                                          limit_n=lim[a], finger_forces_n=f[a].round(2).tolist(), load_n=round(float(loads[a]), 1)))
            now = (w[a, 3:].copy(), float(loads[a])) if self.held(a, f, op) else None
            # Unloading toward zero after a force/finger stop in this or the previous call is the intended release, not a held-object change.
            unloading = now and g['prev'][a] and a in g['prior_stops'] | self.stops() and abs(now[1]) < abs(g['prev'][a][1])
            if now and g['prev'][a] and a not in g['fired'] and not unloading and (np.linalg.norm(now[0] - g['prev'][a][0]) > HELD_TORQUE_NM or abs(now[1] - g['prev'][a][1]) > HELD_LOAD_N and abs(g['prev'][a][1]) >= HELD_EMPTY_N):
                g['fired'].add(a); whole.append(dict(kind='held_object_change', arm=('left', 'right')[a], tick=g['tick'],
                                                     torque_before_nm=g['prev'][a][0].round(2).tolist(), torque_after_nm=now[0].round(2).tolist(),
                                                     load_before_n=round(g['prev'][a][1], 1), load_after_n=round(now[1], 1), finger_forces_n=f[a].round(2).tolist()))
            g['prev'][a] = now
        g['events'] += whole
        if g['hold'] and whole:
            for a, s in limits.items():
                if not s['stopped']: s['stopped'] = True; self.hold_here(a)
        if g['hold'] and all(s['stopped'] for s in limits.values()):
            g['ticks'] = 0; return True
        return False

    def hold_here(self, a):
        e = self.env; e.targets[a] = e.data.site_xpos[e.tips[a]].copy(); e.rotations[a] = e.data.site_xmat[e.tips[a]].reshape(3, 3).copy()

    def step(self):
        self.env.step(); stop = self.check_guard()
        if self.trace: self.trace['ticks'].append((self.env.wrenches_world(), self.env.finger_forces()))
        return stop

    def camera(self, name):
        view = self.views.get(name, dict(camera=name))
        if 'camera' in view: return view['camera']
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.lookat[:] = view['lookat']
        cam.distance, cam.azimuth, cam.elevation = view['distance'], view['azimuth'], view['elevation']
        return cam

    def locate(self, args, image_step):
        e = self.env
        if not hasattr(self, 'depth'):  # Single-sample depth: a multisampled resolve reads an off-centre sample.
            q = e.model.vis.quality; n, q.offsamples = q.offsamples, 0
            self.depth = mujoco.Renderer(e.model, height=RES, width=RES); q.offsamples = n
        option = mujoco.MjvOption(); option.sitegroup[:] = 0
        r = self.depth; r.update_scene(e.data, camera=self.camera(args['camera']), scene_option=option)
        gl = mujoco.mjv_averageCamera(r.scene.camera[0], r.scene.camera[1])
        r.enable_depth_rendering(); depth = r.render().copy(); r.disable_depth_rendering()
        pos, fwd, up = (np.asarray(a, float) for a in (gl.pos, gl.forward, gl.up)); right = np.cross(fwd, up)
        tan_y = (gl.frustum_top - gl.frustum_bottom) / 2 / gl.frustum_near
        far = e.model.vis.map.zfar * e.model.stat.extent
        out = []
        for u, v in args['points']:
            if not (0 <= u < RES and 0 <= v < RES):
                out.append(dict(pixel=[u, v], valid=False, xyz=None, depth_m=None)); continue
            z = float(depth[v, u]); x, y = (2 * (u + .5) / RES - 1) * tan_y, (1 - 2 * (v + .5) / RES) * tan_y
            ok = z < .99 * far
            out.append(dict(pixel=[u, v], valid=bool(ok), xyz=(pos + z * (fwd + x * right + y * up)).round(3).tolist() if ok else None,
                            depth_m=round(z, 3) if ok else None))
        result = dict(camera=args['camera'], image_size=[RES, RES], camera_position=pos.round(3).tolist(), points=out,
                      image_control_step=image_step, control_step=e.steps)
        if image_step != e.steps:
            result['note'] = ('No image of this camera was sent yet' if image_step is None else f'{e.steps - image_step} control ticks elapsed since that image') + \
                             '; depth is from the current state, so moved objects or a moved camera no longer match those pixels'
        return result

    def loads(self):
        return self.env.loads() - self.load_zero

    def declare(self, name, args):
        if name == 'touch': return self.touch(args)
        self.load_zero = self.env.loads().copy()
        return dict(note='Baseline re-tared; load_n now reads relative to the current wrist loads; no physics advanced')

    def metrics(self):
        return self.env.mechanism()

    def privileged_poses(self):
        d = self.env.data; b = self.env.model.body('bottle').id; R = d.xmat[b].reshape(3, 3)
        return dict(bottle=dict(position=d.xpos[b].round(4).tolist(), yaw_deg=round(math.degrees(math.atan2(R[1, 0], R[0, 0])), 2),
                                tilt_deg=round(math.degrees(math.acos(min(1., float(R[2, 2])))), 2)), **self.env.mechanism())

    def close(self):
        if hasattr(self, 'depth'): self.depth.close()
        for r in self.renderers.values(): r.close()
        self.env.close()


def make_task(seed=0, sensing='both'):
    task = BottleTask(seed)
    task.sensing = sensing; task.setup()
    drop = {'none': ('locate', 'touch'), 'locate': ('touch',), 'touch': ('locate',), 'both': ()}[sensing]
    task.tools = [t for t in task.tools if t['name'] not in drop]
    for name, note in (('locate', LOCATE_NOTE), ('touch', FORCE_NOTE)):
        if name in drop: task.instructions = task.instructions.replace(note, '')
    return task
