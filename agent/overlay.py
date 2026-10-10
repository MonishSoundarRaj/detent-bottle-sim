"""Draw the robot's own tool poses on fixed-camera RGB; uses no object state."""
import numpy as np
from PIL import ImageDraw, ImageFont

COLORS = {'left': (40, 140, 255), 'right': (255, 140, 0)}
AXES = [((1, 0, 0), (255, 60, 60), '+X'), ((0, 1, 0), (40, 200, 60), '+Y'), ((0, 0, 1), (60, 110, 255), '+Z')]


def project(scene, fovy, size, points):
    cam = scene.camera
    pos = (np.array(cam[0].pos) + np.array(cam[1].pos)) / 2
    forward, up = np.array(cam[0].forward), np.array(cam[0].up)
    d = np.asarray(points, float) - pos
    z = d @ forward
    t = np.tan(np.radians(fovy) / 2)
    x, y = d @ np.cross(forward, up) / z / t, d @ up / z / t
    return np.c_[(x + 1) / 2 * size, (1 - y) / 2 * size], z > 0


def draw_tools(image, scene, fovy, arms):
    size = image.size[0]
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=13)
    for arm in arms:
        color = COLORS[arm['name']]
        (tip, target), ok = project(scene, fovy, size, [arm['eef'], arm['target']])
        if not ok.all():
            continue
        draw.line([*tip, *target], fill=color, width=1)
        draw.ellipse([*(target - 6), *(target + 6)], outline=color, width=2)
        draw.ellipse([*(tip - 4), *(tip + 4)], fill=color, outline=(0, 0, 0))
        draw.text(tuple(tip + [6, -16]), arm['name'][0].upper(), fill=color, font=font, stroke_width=2, stroke_fill=(0, 0, 0))
        if arm['name'] == 'right':
            ends, _ = project(scene, fovy, size, [np.add(arm['eef'], np.multiply(axis, .03)) for axis, _, _ in AXES])
            for end, (_, c, label) in zip(ends, AXES):
                draw.line([*tip, *end], fill=c, width=2)
                draw.text(tuple(end + [2, -6]), label, fill=c, font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    return image
