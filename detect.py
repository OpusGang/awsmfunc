import vapoursynth as vs
from vapoursynth import core
import fvsfunc as fvf
from functools import partial
from vsutil import iterate, get_y
from vsutil import plane as fplane

def __vs_out_updated(c, t):
    if c == t:
        print("Frame: {}/{}".format(c, t), end="\n")
    else:
        print("Frame: {}/{}".format(c, t), end="\r")


def bandmask(clip, thr=1000, pix=3, left=1, mid=1, right=1, dec=2, exp=None, plane=0):
    """
    A mask that finds areas lacking in grain by simply binarizing the gradient.
    :param clip: Clip to be processed
    :param thr: Binarize threshold.
    :param pix: Pixels to be shifted by.
    :param left: Left shift.
    :param mid: Middle shift.
    :param right: Right shift.
    :param dec: Amount of minimize calls.
    :param exp: Amount of maximize calls, defaults to dec + pix.
    :param plane: Plane to be processed.
    :return: Grayscale mask of areas with gradient smaller than thr.
    """
    if len([plane]) != 1:
        raise ValueError("bandmask: Only one plane can be processed at once!")
    if clip.format.bits_per_sample < 16:
        clip = fvf.Depth(clip, 16)
    if exp is None:
        exp = dec + pix
    pln = fplane(clip, plane)

    def comp(c, mat):
        orig = c
        for _ in range(1, pix):
            c = c.std.Convolution(mat)
        diff = core.std.Expr([orig, c], "x y - abs").std.Binarize(thr, 65535, 0)
        decreased = iterate(diff, core.std.Minimum, dec)
        return iterate(decreased, core.std.Maximum, exp)

    v1 = comp(pln, 6 * [0] + [left, mid, right])
    v2 = comp(pln, [left, mid, right] + 6 * [0])
    h1 = comp(pln, [left] + 2 * [0] + [mid] + 2 * [0] + [right] + 2 * [0])
    h2 = comp(pln, 2 * [0] + [left] + 2 * [0] + [mid] + 2 * [0] + [right])
    return core.std.Expr([v1, v2, h1, h2], "x y + z + a +")

def merge_detections(input, output, cycle=1, min_zone_len=0, delim=" "):
    import numpy as np

    def consecutive(data, cycle=cycle):
        return np.split(data, np.where(np.diff(data) != cycle)[0]+1)

    with open(input, 'r') as in_f:
        a = np.array(in_f.read().splitlines(), dtype=np.int)
        c = consecutive(a, cycle=cycle)

        zones = []
        actual_cycle = cycle - 1

        if min_zone_len >= cycle:
            min_zone = min_zone_len - 1 if min_zone_len % cycle == 0 else min_zone_len
        else:
            min_zone = min_zone_len - 1 if cycle % min_zone_len == 0 else min_zone_len

        for dtc in c:
            start = dtc[0]
            end = dtc[-1] + actual_cycle

            if end - start >= min_zone:
                zone = "{}{delim}{}\n".format(start, end, delim=delim)
                zones.append(zone)

        if zones:
            with open(output, 'w') as out_f:
                out_f.writelines(zones)
                print("Merged frames into zonefile: {}".format(output))

def banddtct(clip, output="banding-frames.txt", thr=150, hi=0.90, lo=0.10, trim=False, cycle=1, merge=True, min_zone_len=0):
    import os
    import sys
    import time

    def detect(n, f, clip, hi, lo, detections):
        if f.props.PlaneStatsAverage >= lo and f.props.PlaneStatsAverage <= hi:
            detections.append(n*cycle)

        return clip

    clip = bandmask(clip, thr=thr, pix=3, left=1, mid=2, right=1, dec=3, exp=None, plane=0)

    if trim and cycle > 1:
        clip = clip.std.SelectEvery(cycle=cycle, offsets=0)

    clip = clip.std.PlaneStats()
    total_frames = clip.num_frames
    
    detected_frames = []

    with open(os.devnull, 'wb') as f:
        processed = core.std.FrameEval(clip, partial(detect, clip=clip, hi=hi, lo=lo, detections=detected_frames), prop_src=clip)
        start = time.time()
        processed.output(f, progress_update=__vs_out_updated)

    end = time.time()
    print("Elapsed: {:0.2f} seconds ({:0.2f} fps)".format(end - start, total_frames / float(end - start)))

    with open(output, 'w') as out_file:
        for f in detected_frames:
            out_file.write("{}\n".format(f))

    if merge:
        merged_output = "merged-{}".format(output)
        merge_detections(output, merged_output, cycle=cycle, min_zone_len=min_zone_len)

    quit("Finished detecting banding, output file: {}".format(output))

    return None


def detect_dirty_lines(clip, output, num, ori=None, thr=.1, merged_output=None, cycle=1):
    import os
    import sys
    import time

    luma = get_y(clip)

    def get_rows(luma, ori, num):
        if ori == "row":
            clip_a = luma.std.Crop(top=num, bottom=luma.height - num - 1)
            if num + 1 < luma.height:
                clip_b = luma.std.Crop(top=num + 1, bottom=luma.height - num - 2)
            else:
                clip_b = luma.std.Crop(top=num - 1, right=1)
        elif ori == "column" or ori == "col":
            clip_a = luma.std.Crop(left=num, right=luma.width - num - 1)
            if num + 1 < luma.width:
                clip_b = luma.std.Crop(left=num + 1, right=luma.width - num - 2)
            else:
                clip_b = luma.std.Crop(left=num - 1, right=1)
        else:
            raise ValueError("blckdtct: ori must be either row, column, or None for both!")
        return core.std.PlaneStats(clip_a, clip_b)

    def detect(n, f, clip, thr, detections):
        if f.props.PlaneStatsDiff > thr:
            detections.append(n * cycle)
        return clip

    def updated(c, t):
        if c == t:
            print("Frame: {}/{}".format(c, t), end="\n")
        else:
            print("Frame: {}/{}".format(c, t), end="\r")

    total_frames = clip.num_frames

    detected_frames = []

    if isinstance(num, int):
        num = [num]

    with open(os.devnull, 'wb') as f:
        for i in num:
            if ori is None:
                clip_diff = get_rows(luma, "row", i)
                processed = core.std.FrameEval(clip, partial(detect, clip=clip, thr=thr, detections=detected_frames),
                                               prop_src=clip_diff)
                clip_diff = get_rows(luma, "col", i)
                processed = core.std.FrameEval(processed,
                                               partial(detect, clip=processed, thr=thr, detections=detected_frames),
                                               prop_src=clip_diff)
            else:
                clip_diff = get_rows(luma, ori, i)
                processed = core.std.FrameEval(clip, partial(detect, clip=clip, thr=thr, detections=detected_frames),
                                               prop_src=clip_diff)
        start = time.time()
        processed.output(f, progress_update=updated)

    end = time.time()
    print("Elapsed: {:0.2f} seconds ({:0.2f} fps)".format(end - start, total_frames / float(end - start)))

    with open(output, 'w') as out_file:
        for f in detected_frames:
            out_file.write("{}\n".format(f))

    if merged_output:
        merge_detections(output, merged_output, cycle=cycle)

    return None


def dirtdtct(clip, output="dirty-frames.txt", num=[0], ori=None, thr=.1, trim=False, cycle=24, merge=True):
    if trim and cycle > 1:
        clip = mask.std.SelectEvery(cycle=cycle, offsets=0)
    else:
        cycle = 1

    if merge:
        merge = "merged-{}".format(output)

    dtc = detect_dirty_lines(clip, output, num, ori, thr, merge, cycle)
    quit("Finished detecting dirty lines, output file: {}".format(output))

    return dtc
