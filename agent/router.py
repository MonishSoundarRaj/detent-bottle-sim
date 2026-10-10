"""Pick which wrist images to send from robot proprioception already in the observation."""
HOLD_N, JUMP_N, JAW_MM = .5, 1.5, 2.


def route(arms, references, commanded, requested):
    """references: per-arm state when that wrist image was last sent (updated in place)."""
    send, notes = [], []
    for arm in arms:
        name, f = arm['name'], arm['finger_forces_n']
        ref = references.get(name)
        if min(f) <= HOLD_N:
            reason = 'not holding'
        elif name in commanded:
            reason = 'commanded last turn'
        elif name + '_wrist' in requested:
            reason = 'requested'
        elif ref is None:
            reason = 'first view of this hold'
        elif max(abs(a - b) for a, b in zip(f, ref['finger_forces_n'])) > JUMP_N:
            reason = 'hold force changed'
        elif abs(arm['opening_mm'] - ref['opening_mm']) > JAW_MM:
            reason = 'jaw moved'
        else:
            notes.append(f"{name}_wrist omitted: {name} hold steady ({f[0]:.2f}/{f[1]:.2f} N, jaw {arm['opening_mm']:.0f} mm; "
                         f"within {JUMP_N} N and {JAW_MM:.0f} mm of the last {name}_wrist image). Use request_views to get it.")
            continue
        send.append(name + '_wrist')
        references[name] = dict(finger_forces_n=list(f), opening_mm=arm['opening_mm'], reason=reason)
    return send, notes
