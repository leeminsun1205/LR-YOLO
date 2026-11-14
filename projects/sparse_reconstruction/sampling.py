"""Pure-Python sampling geometry; no global RNG or image/annotation mutation."""
import random
import math


def sample_regions(height, width, boxes, count=2, core=192, halo=32,
                   alignment=16, object_probability=0.5, seed=42):
    outer = core + 2 * halo
    if (count < 1 or core < 16 or halo < 0 or alignment < 2
            or alignment % 2 or core % alignment or halo % alignment
            or height % alignment or width % alignment
            or outer > min(height, width) or not 0 <= object_probability <= 1):
        raise ValueError('Invalid aligned crop budget/shape/probability')
    rng = random.Random(seed)
    valid = []
    for box in boxes:
        if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
            raise ValueError('Boxes must contain four finite coordinates')
        x1,y1,x2,y2 = map(float,box)
        if x2 > x1 and y2 > y1:
            valid.append((x1,y1,x2,y2))
    result = []
    for _ in range(count):
        guided = bool(valid) and rng.random() < object_probability
        if guided:
            x1, y1, x2, y2 = rng.choice(valid)
            # Uniform objects, never proportional to their area; retain jitter.
            cx = (x1 + x2) / 2 + rng.uniform(-core / 4, core / 4)
            cy = (y1 + y2) / 2 + rng.uniform(-core / 4, core / 4)
            left = round((cx - core / 2) / alignment) * alignment
            top = round((cy - core / 2) / alignment) * alignment
            left = max(0, min(left, width - core))
            top = max(0, min(top, height - core))
        else:
            left = rng.randrange((width - core) // alignment + 1) * alignment
            top = rng.randrange((height - core) // alignment + 1) * alignment
        x = max(0, min(left - halo, width - outer))
        y = max(0, min(top - halo, height - outer))
        result.append(dict(x=x, y=y, size=outer, core_x=left-x,
                           core_y=top-y, core=core, guided=guided))
    return result


def decoder_bounds(region, height, width):
    """Two Z cells protect BOTH late 3x3 convolutions and PixelShuffle phase.

    height/width refer to RGB output; Z has half that resolution. At real
    canvas boundaries preserve original zero padding rather than inventing halo.
    """
    x, y, size = region['x'], region['y'], region['size']
    if any(v % 2 for v in (x, y, size, height, width)):
        raise ValueError('PixelShuffle requires even aligned coordinates')
    x0, y0 = max(0, x // 2 - 2), max(0, y // 2 - 2)
    x1 = min(width // 2, (x + size) // 2 + 2)
    y1 = min(height // 2, (y + size) // 2 + 2)
    return x0, y0, x1, y1, x - 2*x0, y - 2*y0
