"""v3 agentic episode on the detent bottle: notes, camera router, strategy reviews and a USD budget around a tool-calling model.
--dry uses a fake model (zero API calls); only --live calls the OpenAI Responses API (OPENAI_API_KEY from the environment).
Run from the repo root: python -m agent.run --dry"""
import argparse
import asyncio
from collections import Counter
import base64
import json
import math
import os
import re
from pathlib import Path
import sys
import time
import numpy as np

from agent.task import FORCE_N, SCAFFOLD, arm_observation as robot_arm_observation, make_task
from agent.router import route
from agent.costs import CostMeter

p = argparse.ArgumentParser()
p.add_argument('--seed', type=int, default=0); p.add_argument('--out'); p.add_argument('--budget', type=float)
p.add_argument('--resume', help='continue a crashed run dir in place (seed/sensing/privileged from its run.json); a mode flag overrides its mode')
p.add_argument('--privileged', action='store_true', help='evaluation-only ablation: send true object poses every turn')
p.add_argument('--sensing', choices=['none', 'locate', 'touch', 'both'], default='both')
mode = p.add_mutually_exclusive_group()
mode.add_argument('--dry', action='store_true'); mode.add_argument('--live', action='store_true')
ARGS = p.parse_args()
HERE = Path(__file__).resolve().parent
if ARGS.resume:
    if ARGS.out: p.error('--resume and --out are mutually exclusive')
    out = Path(ARGS.resume); META = json.loads((out / 'run.json').read_text())
    ARGS.seed, ARGS.privileged, ARGS.sensing = META['seed'], META['privileged'], META.get('sensing', 'both')
    MODE = 'dry' if ARGS.dry else 'live' if ARGS.live else META['mode']
else:
    if not (ARGS.dry or ARGS.live): p.error('one of --dry/--live is required')
    MODE = 'dry' if ARGS.dry else 'live'
    out = Path(ARGS.out) if ARGS.out else Path('runs') / f"detent-bottle-{MODE}-s{ARGS.seed}-{time.strftime('%Y%m%d-%H%M%S')}"
    if ARGS.privileged and not out.name.endswith('-PRIV'): out = out.with_name(out.name + '-PRIV')
    out.mkdir(parents=True, exist_ok=False)
sys.stdout = open(out / 'runner.log', 'a' if ARGS.resume else 'w', buffering=1)
print('SCAFFOLD', SCAFFOLD, 'SENSING', ARGS.sensing, flush=True)
if ARGS.privileged: print('=' * 20, 'PRIVILEGED ABLATION: evaluator object poses are sent to the model every turn; not a normal run', '=' * 20, flush=True)

MAX_TURNS = 250
REVIEW_MISSES, REVIEW_EVERY, SAME_ARM_STREAK, NOTES_MAX = 2, 12, 12, 1200
REVIEW = ('Strategy review (tools are disabled for this reply). Trigger: {}. Success criteria, quoted from your task instructions: "{}"\n'
          'Using the images and robot feedback so far, answer briefly: (1) split the success criteria into their separate conditions; for each, judge it '
          'from the CURRENT images and proprioception in the latest observation and mark it satisfied, not satisfied or cannot tell, naming the image '
          'or reading that shows it; (2) whether the current plan still leads to success from this state or must change, and why; (3) the most likely cause of the recent failures or lack of progress; (4) the current plan\'s structure: how the available resources '
          '(each arm, the approach directions, the object\'s placement) are assigned, which assumptions that rests on, and whether the evidence '
          'so far still supports it or points to a different assignment; (5) alternatives not yet tried and whether each is feasible; '
          '(6) your plan for the next few turns. You may plan to use check_reach to test candidate approaches.')
STOP_REVIEW = ('You indicated you are stopping. Before stopping (tools are disabled for this reply): list the overall strategies you have not '
               'yet tried, including different roles for the two arms and different ways of holding or placing the objects. For each, say '
               'whether it is feasible and why, using the evidence so far. If any untried strategy is feasible, name the one you will try next; '
               'robot control resumes after this reply. If none is feasible, say so, and reply without tool calls next turn to confirm stopping.')
MAX_STOP_REVIEWS = 2
WORST_CALL = {'gpt-6-sol': .35}.get(os.environ.get('OPENAI_MODEL'), .70)
MOVES = ('move_arm', 'move_both_arms')


def fn(name, description, properties):
    return {'type': 'function', 'name': name, 'description': description, 'strict': True,
            'parameters': {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}}


def observation(state, images, cameras, notes):
    content = [{'type': 'input_text', 'text': 'Current observation: ' + json.dumps(state)}]
    content += [{'type': 'input_text', 'text': note} for note in notes]
    for camera in cameras:
        content.extend([{'type': 'input_text', 'text': camera}, {'type': 'input_image', 'image_url': images[camera], 'detail': 'auto'}])
    return {'role': 'user', 'content': content}


def fake_model(task):
    usage = {'input_tokens': 1000, 'output_tokens': 100, 'input_tokens_details': {'cached_tokens': 0}}
    def reply(text, calls=()):
        return {'model': 'gpt-6-astra', 'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': text}]}]
                + [{'type': 'function_call', 'name': n, 'call_id': f'c{i}', 'arguments': json.dumps(a)} for i, (n, a) in enumerate(calls)], 'usage': usage}
    async def respond(inputs, on_delta, **_):
        last = inputs[-1].get('content')
        if isinstance(last, str) and 'tools are disabled' in last:
            return reply(f'{MODE} review')
        respond.n += 1
        images = sum(1 for x in last if x['type'] == 'input_image') if isinstance(last, list) else 0
        move = dict(task.dry_move); check = {k: move[k] for k in move if k != 'jaw_gap_m'}
        if respond.n == 2:
            calls = [('update_notes', dict(notes='DRY NOTE: fake model; no API call.')), ('check_reach', check)]
            calls.append(('find_reach', dict(arm='both', position=move['position'], **task.dry_reach, steps=40)))
            if 'locate' in task.tool_names: calls.append(('locate', dict(camera='topview', points=[[320, 320]])))
            if 'touch' in task.tool_names: calls.append(('touch', dict(arm='left', direction=[0, 0, -1], max_distance_m=.15, force_n=5, jaw_gap_m=None, **{task.rot_key: None})))
        elif respond.n >= 30:
            return reply('dry stop')
        else:
            calls = [('check_reach', check), ('move_arm', move)] + ([('wait', dict(ticks=5))] if respond.n == 3 else [])
        return reply(f'dry {images} images', calls)
    respond.n = 0
    return respond


async def main():
    wall = time.time()
    task = make_task(ARGS.seed, ARGS.sensing)
    task.tool_names = {t['name'] for t in task.tools}
    trace = hasattr(task, 'force_trace') and 'touch' in task.tool_names
    e = task.env
    state_of = lambda snap: dict(task.model_state(snap), privileged_poses=task.privileged_poses()) if ARGS.privileged else task.model_state(snap)
    if MODE == 'live':
        from agent.llm import ResponsesAgent
        agent = ResponsesAgent(); model = agent.model
        respond = lambda inputs, on_delta: agent.respond(inputs, on_delta, instructions=instructions, tools=tools)
    else:
        model, fake = 'gpt-6-astra', fake_model(task)
        respond = lambda inputs, on_delta: fake(inputs, on_delta)
    wrists = task.wrists
    tools = task.tools + [
        fn('request_views', 'Ask for wrist camera images that the camera router omitted. They accompany the next observation. No physics advances.',
           {'cameras': {'type': 'array', 'items': {'type': 'string', 'enum': wrists}, 'minItems': 1, 'maxItems': 2}}),
        fn('update_notes', 'Replace your persistent notes (max 1200 characters). Everything else in your context is compacted every five turns; '
                           'these notes are kept verbatim and shown after every compaction and in every strategy review. Record confirmed facts, '
                           'open hypotheses, approaches tried and their outcomes, and the current plan.', {'notes': {'type': 'string'}})]
    instructions = task.instructions
    found = re.search(r'(?:Success|Evaluation) \([^)]*\):\s*(.*?)(?:\s+[^.]*no\s+physics|\Z)', task.instructions, re.S)
    criteria = ' '.join((found.group(1) if found else task.goal_text).split())
    if not ARGS.resume: (out / 'run.json').write_text(json.dumps(dict(task=task.id, seed=ARGS.seed, scaffold=SCAFFOLD, mode=MODE, privileged=ARGS.privileged, sensing=ARGS.sensing, latch_mode=getattr(e, 'latch', None), model=model, contract=task.contract, max_turns=MAX_TURNS, review_criteria=criteria,
                                                  max_steps=e.max_steps, cameras=task.cameras, instructions=instructions, tools=tools), indent=2))
    meter = CostMeter(model, f'agentic-v3-{MODE}', out / 'billing.jsonl')
    if not ARGS.resume: (out / 'prompt.txt').write_text((HERE / 'prompts' / '_template.txt').read_text().strip().format(
        goal=(HERE / 'prompts' / 'bottle.txt').read_text().strip(), ticks=e.max_steps) + '\n')
    prompt = {'role': 'user', 'content': (out / 'prompt.txt').read_text()}
    snap = task.snapshot()
    history, actions, log, qpos, mech = [prompt], [], [], [e.data.qpos.copy()], [task.metrics()]
    if not ARGS.resume: (out / 'before.json').write_text(json.dumps(task.model_state(snap)))
    references, commanded, requested = {}, set(), set()
    misses, last_review, review_text, reach_log, stop_reviews, notes_text, streak = {'left': 0, 'right': 0}, 0, '', [], 0, '', ['', 0]
    review_calls, image_steps, retries, replaying, start = 0, {}, 0, False, 1

    def facts(since):
        lines = []
        for side in ('left', 'right'):
            moves = [a for a in actions if a['turn'] > since and a['name'] in MOVES and a['arguments'].get('arm', 'both') in (side, 'both')]
            checks = [r for r in reach_log if r['turn'] > since and r['arm'] == side]
            bad = [r for r in checks if r['failed']]
            reasons = Counter(x for r in bad for x in r['result']['joints_at_limit'] + r['result']['robot_collisions'])
            lines.append(f"{side} arm: {len(moves)} moves ({sum(a['result']['position_error_m'] > .05 for a in moves)} missed by >50 mm); "
                         f"check_reach: {len(checks)} checks, {len(bad)} predicted to fail"
                         + (' (' + ', '.join(f'{k} x{v}' for k, v in reasons.most_common()) + ')' if reasons else ''))
        return f'Facts since turn {since}: ' + '; '.join(lines) + '.'

    def unreachable():
        return '\n'.join(f"turn {r['turn']} {r['arm']} arm {r['position']} orientation {r['rpy']}: error {r['result']['position_error_mm']} mm, "
                         f"limits {r['result']['joints_at_limit']}, collisions {r['result']['robot_collisions']}" for r in reach_log if r['failed'])[-3000:]

    async def call_model(context, name):
        nonlocal retries
        for attempt in range(4):
            record = meter.begin(dict(context, attempt=attempt) if attempt else context)
            try:
                response = await respond(history, lambda x: print(x, end='', flush=True))
            except BaseException as exc:
                meter.finish(record, getattr(exc, 'response', None), status='failed')
                if not isinstance(exc, Exception): raise
                error = f'{type(exc).__name__}: {exc}'
                if attempt == 3:
                    print('\nPROVIDER ERROR after 3 retries:', error, flush=True)
                    write_final(reason='provider_error', error=error); sys.exit(1)
                retries += 1
                print(f'\nRETRY {attempt + 1}/3 of {name} in {(5, 20, 60)[attempt]} s after {error}', flush=True)
                await asyncio.sleep((5, 20, 60)[attempt]); continue
            meter.finish(record, response)
            (out / name).write_text(json.dumps(response))
            return response

    async def review(turn, kind, text):
        nonlocal review_text, review_calls
        review_calls += 1
        history.append({'role': 'user', 'content': text})
        (out / f'{kind}-input-{turn:02}.json').write_text(json.dumps(history))
        print('REVIEW', kind, turn, flush=True)
        response = await call_model({'turn': turn, 'review': kind}, f'{kind}-response-{turn:02}.json')
        history.extend(response.get('output', []))
        for call in [x for x in response.get('output', []) if x.get('type') == 'function_call']:
            history.append({'type': 'function_call_output', 'call_id': call['call_id'], 'output': json.dumps({'error': 'Tools are disabled during the strategy review'})})
        review_text = ' '.join(c.get('text', '') for o in response.get('output', []) if o.get('type') == 'message' for c in o['content'])
        with (out / 'reviews.jsonl').open('a') as f:
            f.write(json.dumps(dict(turn=turn, step=e.steps, kind=kind, prompt=text, text=review_text)) + '\n')
        history.append({'role': 'user', 'content': 'Robot control resumes now.'})

    def advance(n):
        executed = 0
        for _ in range(n):
            if e.reason: break
            stop = task.step(); executed += 1
            qpos.append(e.data.qpos.copy()); mech.append(task.metrics())
            if stop: break
        return executed

    def guard_events():
        return getattr(task, 'guard', None) and task.guard['events']

    def execute(turn, call, args):
        nonlocal notes_text, misses, streak
        name = call['name']
        if name == 'wait':
            if hasattr(task, 'arm_guard'): task.arm_guard({0: FORCE_N, 1: FORCE_N}, args['ticks'], False)
            executed = advance(args['ticks'])
            result = dict(executed=executed, actual_arms=[robot_arm_observation(a) for a in task.snapshot(images=False)['arms']])
            if trace: result['force_trace'] = task.force_trace()
            if guard_events(): result['guard_events'] = guard_events()
            if e.reason: result['controller_status'] = 'Motion unavailable; operator reset required'
            actions.append(dict(id=call['call_id'], name=name, arguments=args, status='completed', executed=executed, result=result, turn=turn))
            return result
        if name == 'request_views':
            requested.update(args['cameras'])
            return {'requested': args['cameras'], 'note': 'Images accompany the next observation; no physics advanced'}
        if name == 'find_reach':
            result = {}
            for side in (('left', 'right') if args['arm'] == 'both' else (args['arm'],)):
                rows = task.reach_options(int(side == 'right'), args)
                result[side] = dict(feasible=[r for r in rows if r['feasible']][:4], infeasible_count=sum(not r['feasible'] for r in rows),
                                    infeasible_reasons=dict(Counter(x for r in rows if not r['feasible'] for x in (r['joints_at_limit'] + r['robot_collisions']) or ['tracking error'])),
                                    best_if_none=None if any(r['feasible'] for r in rows) else rows[0])
                for r in rows:
                    reach_log.append(dict(turn=turn, arm=side, position=args['position'], rpy=r.get('rpy_deg', r.get('yaw_deg')), result=r, failed=not r['feasible']))
            return result
        if name == 'update_notes':
            notes_text = args['notes'][:NOTES_MAX]
            if not replaying:
                with (out / 'notes.jsonl').open('a') as f:
                    f.write(json.dumps(dict(turn=turn, step=e.steps, notes=notes_text)) + '\n')
            return {'stored_characters': len(notes_text), 'truncated': len(args['notes']) > NOTES_MAX}
        if name == 'check_reach':
            result = task.preview(args)
            reach_log.append(dict(turn=turn, arm=args['arm'], position=args['position'], rpy=args.get('rpy_deg', args.get('yaw_deg')), result=result,
                                  failed=bool(result['position_error_mm'] > 20 or result['rotation_error_deg'] > 10 or result['joints_at_limit'] or result['robot_collisions'])))
            return result
        if name == 'locate' and 'locate' in task.tool_names:
            result = task.locate(args, image_steps.get(args['camera']))
            actions.append(dict(id=call['call_id'], name=name, arguments=args, status='completed', executed=0, result=result, turn=turn, step=e.steps))
            return result
        if name == 'touch' and 'touch' in task.tool_names:
            try:
                result = task.touch(args, lambda: advance(1))
            except ValueError as exc:
                return {'error': str(exc)}
            commanded.add(args['arm'])
            if guard_events(): result['guard_events'] = guard_events()
            if e.reason: result['controller_status'] = 'Motion unavailable; operator reset required'
            actions.append(dict(id=call['call_id'], name=name, arguments=args, status='completed', executed=result['ticks'] + result['hold_ticks'], result=result, turn=turn))
            return result
        if name == 'observe':
            return state_of(task.snapshot())
        if name in task.declarations:
            try:
                result = task.declare(name, args)
            except (ValueError, RuntimeError) as exc:
                return {'error': str(exc)}
            actions.append(dict(id=call['call_id'], name=name, arguments=args, status='completed', executed=0, result=result, turn=turn, step=e.steps))
            return result
        if name in MOVES:
            try:
                arms = task.command(name, args)
            except ValueError as exc:
                return {'error': str(exc)}
            commanded.update(arms)
            executed = advance(args['steps'])
            state = {a['name']: a for a in task.snapshot(images=False)['arms']}
            errors = {arm: math.dist((args[arm] if name == 'move_both_arms' else args)['position'], state[arm]['eef']) for arm in arms}
            result = dict(executed=executed, **({'actual_arm': robot_arm_observation(state[arms[0]])} if len(arms) == 1 else
                                                {'actual_arms': [robot_arm_observation(state[a]) for a in arms]}), position_error_m=max(errors.values()))
            if trace: result['force_trace'] = task.force_trace()
            if guard_events(): result.update(stopped_by=guard_events()[0]['kind'], guard_events=guard_events())
            if e.reason: result['controller_status'] = 'Motion unavailable; operator reset required'
            for arm in arms: misses[arm] = misses[arm] + 1 if errors[arm] > .05 else 0
            who = arms[0] if len(arms) == 1 else 'both'
            streak = [who, streak[1] + 1] if streak[0] == who else [who, 1]
            actions.append(dict(id=call['call_id'], name=name, arguments=args, status='completed', executed=executed, result=result, turn=turn))
            return result
        return {'error': 'Unknown tool'}

    def write_final(**extra):
        (out / 'all-actions.json').write_text(json.dumps(actions, indent=2))
        (out / 'costs.json').write_text(json.dumps(meter.export(), indent=2))
        np.savez_compressed(out / 'trajectory.npz', qpos=np.array(qpos))
        (out / 'evaluator-metrics.json').write_text(json.dumps(mech))
        snap = task.snapshot()
        for name, url in snap['images'].items():
            (out / f'final-{name}.jpg').write_bytes(base64.b64decode(url.split(',', 1)[1]))
        numeric = [k for k in mech[0] if all(isinstance(m[k], (int, float)) for m in mech)]
        final = dict(task=task.id, seed=ARGS.seed, scaffold=SCAFFOLD, mode=MODE, privileged=ARGS.privileged, sensing=ARGS.sensing, latch_mode=getattr(e, 'latch', None), model_api_calls=len(meter.requests) if MODE == 'live' else 0, fake_calls=0 if MODE == 'live' else len(meter.requests),
                     model_turns=sum(r['status'] == 'completed' for r in meter.requests) - review_calls, review_calls=review_calls, steps=e.steps, success=e.success, reason=e.reason, wall_s=round(time.time() - wall, 1),
                     usd={k: v for k, v in meter.summary().items() if k not in ('requests', 'pricing')},
                     peak={k: [min(m[k] for m in mech), max(m[k] for m in mech)] for k in numeric}, final_metrics=task.metrics(),
                     retries=retries, resumed_from_turn=start - 1 if ARGS.resume else None)
        final.update(extra)
        (out / 'final.json').write_text(json.dumps(final, indent=2))
        print('\nFINISHED', json.dumps(final), flush=True)
        print('FINISHED', out, json.dumps({k: final[k] for k in ('scaffold', 'model_api_calls', 'fake_calls', 'model_turns', 'review_calls', 'steps', 'success', 'reason', 'wall_s')}), file=sys.stderr)
        task.close()

    if ARGS.resume:
        replaying, turns = True, sorted(int(x.stem.split('-')[1]) for x in out.glob('response-*.json'))
        last = turns[-1] if turns else 0
        reviews = [json.loads(x) for x in (out / 'reviews.jsonl').open()] if (out / 'reviews.jsonl').exists() else []
        router = {r['turn']: r for r in map(json.loads, (out / 'router.jsonl').open())}
        for t in range(1, last + 2):
            if any(r['turn'] == t and r['kind'] == 'review' for r in reviews): misses, last_review, streak = {'left': 0, 'right': 0}, t, ['', 0]
            if t > last: break
            image_steps.update({c: router[t]['step'] for c in task.cameras if c not in wrists or c in router[t]['sent']})
            commanded, requested = set(), set()
            for call in [x for x in json.loads((out / f'response-{t:02}.json').read_text()).get('output', []) if x.get('type') == 'function_call']:
                args = json.loads(call['arguments'])
                log.append(dict(turn=t, name=call['name'], arguments=args, result=execute(t, call, args)))
        replaying, start, history = False, last + 1, json.loads((out / 'history.json').read_text())
        review_calls, stop_reviews, review_text = len(reviews), sum(r['kind'] == 'stop-review' for r in reviews), reviews[-1]['text'] if reviews else ''
        meter.requests = list({r['request_id']: r for r in (json.loads(x)['request'] for x in (out / 'billing.jsonl').open())}.values())
        if (out / 'final.json').exists(): retries = json.loads((out / 'final.json').read_text()).get('retries', 0)
        if MODE != 'live': fake.n = last
        saved_q, saved_m = np.load(out / 'trajectory.npz')['qpos'], json.loads((out / 'evaluator-metrics.json').read_text())
        q_last = float(np.max(np.abs(e.data.qpos - saved_q[-1])))
        q_all = float(np.max(np.abs(np.array(qpos) - saved_q))) if len(qpos) == len(saved_q) else f'length {len(qpos)} vs {len(saved_q)}'
        m_last, m_all = json.loads(json.dumps(mech[-1])) == saved_m[-1], json.loads(json.dumps(mech)) == saved_m
        a_same = json.loads(json.dumps(actions)) == json.loads((out / 'all-actions.json').read_text())
        summary = meter.summary()
        banner = (f'RESUMED resumed_from_turn {last} mode {MODE}: replayed {len(actions)} actions / {len(log)} tool calls to step {e.steps}; '
                  f'qpos max error vs trajectory.npz last row {q_last}, all rows {q_all}; evaluator metrics equal last row {m_last}, all rows {m_all}; '
                  f'all-actions equal {a_same}; notes {len(notes_text)} chars; reviews {review_calls}; prior requests {len(meter.requests)} usd_high {summary["usd_high"]:.4f}; prior retries {retries}')
        print('=' * 20, banner, '=' * 20, flush=True); print(banner, file=sys.stderr)
        if q_last > 1e-6 or not m_last: sys.exit('resume replay does not match the saved run')
        actions[:], qpos[:], mech[:] = json.loads((out / 'all-actions.json').read_text()), list(saved_q), saved_m

    if ARGS.budget is not None or not ARGS.resume: (out / 'budget.txt').write_text(str(ARGS.budget or 10.))
    for turn in range(start, MAX_TURNS + 1):
        budget = float((out / 'budget.txt').read_text())
        summary = meter.summary()
        spent = summary['usd_high'] + WORST_CALL * (summary['unknown_requests'] + summary['pending_requests'])
        if spent + WORST_CALL > budget:
            print(f'BUDGET stop before turn {turn}: spent ${spent:.4f}', flush=True); break
        if e.reason or e.success:
            break
        if (turn > 1 and (turn - 1) % 5 == 0) or turn == start > 1:
            notes = [p['text'] for item in history if item.get('type') == 'message' for p in item.get('content', []) if p.get('text')]
            recent = [x for x in log if x['turn'] >= turn - 8]
            history = [prompt, {'role': 'user', 'content': 'Your recent statements (interpretations, not ground truth):\n' + '\n'.join(notes[-5:])
                                + '\nYour tool calls and their actual results from the last 8 turns:\n' + json.dumps(recent)
                                + ('\nYour latest strategy review:\n' + review_text if review_text else '')
                                + ('\nYour notes (kept verbatim):\n' + notes_text if notes_text else '\nYour notes are empty; use update_notes.')
                                + ('\nPoses check_reach predicted to fail so far:\n' + unreachable() if unreachable() else '')}]
            references = {}
        if hasattr(task, 'privileged_poses'):
            with (out / 'truth.jsonl').open('a') as f:
                f.write(json.dumps(dict(turn=turn, control_step=e.steps, poses=task.privileged_poses())) + '\n')
        snap = task.snapshot()
        send, omitted = route(snap['arms'], references, commanded, requested)
        with (out / 'router.jsonl').open('a') as f:
            f.write(json.dumps(dict(turn=turn, step=e.steps, sent=send, omitted=omitted, commanded=sorted(commanded), requested=sorted(requested))) + '\n')
        cameras = [c for c in task.cameras if c not in wrists] + [c for c in task.cameras if c in send]
        image_steps.update({c: e.steps for c in cameras})
        history += [{'role': 'user', 'content': f'Turn {turn}. Control ticks remaining: {e.max_steps - e.steps}. Estimated model budget remaining: ${budget - spent:.2f}.'
                                                + (" Evaluator-measured object poses are provided this episode under 'privileged_poses'; treat them as ground truth." if ARGS.privileged else '')},
                    observation(state_of(snap), snap['images'], cameras, omitted)]
        trigger = (f'the {max(misses, key=misses.get)} arm missed its commanded position by more than 50 mm in {max(misses.values())} consecutive moves'
                   if max(misses.values()) >= REVIEW_MISSES else
                   f'the {streak[0]} arm has been the only arm commanded for {streak[1]} consecutive commands' if streak[1] >= SAME_ARM_STREAK and streak[0] != 'both' else
                   f'{turn - last_review} turns since the last review' if turn - last_review >= REVIEW_EVERY else None)
        if trigger and spent + 2 * WORST_CALL <= budget:
            await review(turn, 'review', REVIEW.format(trigger, criteria) + '\n' + facts(last_review)
                         + ('\nYour notes:\n' + notes_text if notes_text else '')
                         + ('\nPoses check_reach predicted to fail so far:\n' + unreachable() if unreachable() else ''))
            misses, last_review, streak = {'left': 0, 'right': 0}, turn, ['', 0]
        (out / f'input-{turn:02}.json').write_text(json.dumps(history))
        print('TURN', turn, 'step', e.steps, flush=True)
        response = await call_model({'turn': turn, 'images_sent': sum(1 for i in history if isinstance(i.get('content'), list) for p in i['content'] if p.get('type') == 'input_image')},
                                    f'response-{turn:02}.json')
        history.extend(response.get('output', []))
        calls = [x for x in response.get('output', []) if x.get('type') == 'function_call']
        commanded, requested = set(), set()
        for call in calls:
            args = json.loads(call['arguments'])
            result = execute(turn, call, args)
            log.append(dict(turn=turn, name=call['name'], arguments=args, result=result))
            history.append({'type': 'function_call_output', 'call_id': call['call_id'], 'output': json.dumps(result)})
        print('\nSTEP', e.steps, flush=True)
        (out / 'all-actions.json').write_text(json.dumps(actions, indent=2))
        (out / 'history.json').write_text(json.dumps(history))
        (out / 'costs.json').write_text(json.dumps(meter.export(), indent=2))
        np.savez_compressed(out / 'trajectory.npz', qpos=np.array(qpos))
        (out / 'evaluator-metrics.json').write_text(json.dumps(mech))
        if not calls:
            summary = meter.summary()
            if e.success or stop_reviews >= MAX_STOP_REVIEWS or summary['usd_high'] + 2 * WORST_CALL > budget:
                break
            stop_reviews += 1
            await review(turn, 'stop-review', STOP_REVIEW + '\n' + facts(0) + ('\nYour notes:\n' + notes_text if notes_text else '')
                         + ('\nPoses check_reach predicted to fail so far:\n' + unreachable() if unreachable() else ''))
            (out / 'history.json').write_text(json.dumps(history))
            (out / 'costs.json').write_text(json.dumps(meter.export(), indent=2))
    write_final()

asyncio.run(main())
