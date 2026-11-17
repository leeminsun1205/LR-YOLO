"""Pure-Python sampling geometry; no global RNG or image/annotation mutation."""
import random
import math


def sample_regions(height, width, boxes, count=2, core=192, halo=32,
                   alignment=16, object_probability=0.5, seed=42,
                   stratified_pair=False):
    outer = core + 2 * halo
    if (count < 1 or core < 16 or halo < 0 or alignment < 2
            or alignment % 2 or core % alignment or halo % alignment
            or height % alignment or width % alignment
            or outer > min(height, width) or not 0 <= object_probability <= 1):
        raise ValueError('Invalid aligned crop budget/shape/probability')
    if stratified_pair and count != 2:
        raise ValueError('Stratified pair requires exactly two crops')
    rng = random.Random(seed)
    valid = []
    for box in boxes:
        if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
            raise ValueError('Boxes must contain four finite coordinates')
        x1,y1,x2,y2 = map(float,box)
        if x2 > x1 and y2 > y1:
            valid.append((x1,y1,x2,y2))
    result = []
    for index in range(count):
        guided = (bool(valid) and index == 0 if stratified_pair
                  else bool(valid) and rng.random() < object_probability)
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


def sample_coverage_regions(height, width, boxes, count=2, core=192, halo=32,
                            alignment=16, seed=42, candidates=8):
    """S8: SAME S6 anchor/uniform draws, geometric selection before decoding.

    Lexicographic objective: maximize selected visible GT box fraction in L1
    core, then minimize overlap with the unchanged uniform core. The original
    S6 crop is always a candidate, so anchor coverage cannot decrease. This is
    not an unbiased dense-loss estimator or a full-object-containment guarantee.
    """
    if not isinstance(candidates,int) or isinstance(candidates,bool) or candidates<2:
        raise ValueError('S8 needs at least two candidate locations')
    valid=[]
    for box in boxes:
        if len(box)!=4 or not all(math.isfinite(float(v)) for v in box):
            raise ValueError('Boxes must contain four finite coordinates')
        x1,y1,x2,y2=map(float,box)
        # Native augmented boxes are clipped already. Defensively exclude
        # nonvisible objects without changing legacy S0-S6 sampling behavior.
        x1,y1=max(0.,x1),max(0.,y1)
        x2,y2=min(float(width),x2),min(float(height),y2)
        if x2>x1 and y2>y1:valid.append((x1,y1,x2,y2))
    pair=sample_regions(height,width,valid,count=count,core=core,halo=halo,
                        alignment=alignment,seed=seed,stratified_pair=True)
    if not valid:return pair
    # In S6 the first random operation is uniformly selecting an anchor box.
    anchor=random.Random(seed).choice(valid)
    original,uniform=pair
    base=(original['x']+original['core_x'],original['y']+original['core_y'])
    ul,ut=uniform['x']+uniform['core_x'],uniform['y']+uniform['core_y']
    x1,y1,x2,y2=anchor;area=(x2-x1)*(y2-y1)
    rng=random.Random(seed+17000021)  # Does not perturb S6 uniform/degradation RNG.

    def origin(cx,cy):
        return (max(0,min(round((cx-core/2)/alignment)*alignment,width-core)),
                max(0,min(round((cy-core/2)/alignment)*alignment,height-core)))

    locations=[base,origin((x1+x2)/2,(y1+y2)/2)]
    for _ in range(candidates-2):
        locations.append(origin((x1+x2)/2+rng.uniform(-core/4,core/4),
                                (y1+y2)/2+rng.uniform(-core/4,core/4)))
    locations=list(dict.fromkeys(locations))

    def score(location):
        left,top=location
        covered=max(0.,min(left+core,x2)-max(left,x1))*max(0.,min(top+core,y2)-max(top,y1))
        overlap=max(0,min(left+core,ul+core)-max(left,ul))*max(0,min(top+core,ut+core)-max(top,ut))
        return covered,-overlap

    left,top=max(locations,key=score)
    outer=core+2*halo
    x,y=max(0,min(left-halo,width-outer)),max(0,min(top-halo,height-outer))
    selected,baseline=score((left,top)),score(base)
    pair[0]=dict(x=x,y=y,size=outer,core_x=left-x,core_y=top-y,core=core,guided=True,
        anchor_box=list(anchor),anchor_coverage=selected[0]/area,
        baseline_anchor_coverage=baseline[0]/area,core_overlap_fraction=-selected[1]/core**2,
        baseline_core_overlap_fraction=-baseline[1]/core**2,
        candidates_scored=len(locations),candidate_budget=candidates,policy='coverage_then_overlap')
    return pair
