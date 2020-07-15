import vapoursynth as vs
from vapoursynth import core

from functools import partial
from vsutil import iterate, get_y
from vsutil import plane as fplane
from vsutil import depth as Depth


def __vs_out_updated(c, t):
    if c == t:
        print("Frame: {}/{}".format(c, t), end="\n")
    else:
        print("Frame: {}/{}".format(c, t), end="\r")


def bandmask(clip, thr=1000, pix=3, left=1, mid=1, right=1, dec=2, exp=None, plane=0, darkthr=None, brightthr=None,
             blankthr=None):
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
    :param darkthr: If set, values under darkthr will be ignored and set black.
    :param brightthr: If set, values above brightthr will be ignored and set to black.
                      If either darkthr or brightthr is set and the other isn't, the other will default to TV range.
    :param blankthr: If set, values with less change than this will be ignored and set black.
    :return: Grayscale mask of areas with gradient smaller than thr.
    """
    if len([plane]) != 1:
        raise ValueError("bandmask: Only one plane can be processed at once!")
    depth = clip.format.bits_per_sample
    hi = 65535
    if depth < 16:
        clip = Depth(clip, 16)
    elif depth == 32:
        hi = 1
        if thr >= 1:
            thr = thr / 65535
    if exp is None:
        exp = dec + pix
    pln = fplane(clip, plane)
    if not darkthr and brightthr:
        darkthr = 4096
    if darkthr and not brightthr:
        if plane == 0:
            brightthr = 60160
        else:
            brightthr = 61440

    def comp(c, mat):
        orig = c
        for _ in range(1, pix):
            c = c.std.Convolution(mat)
        if not blankthr:
            diff = core.std.Expr([orig, c], "x y - abs").std.Binarize(thr, hi, 0)
        else:
            diff = core.std.Expr([orig, c], "x y - abs").std.Expr("x {} > x {} < thr {} 0 ?".format(blankthr, thr, hi))
        decreased = iterate(diff, core.std.Minimum, dec)
        return iterate(decreased, core.std.Maximum, exp)

    v1 = comp(pln, 6 * [0] + [left, mid, right])
    v2 = comp(pln, [left, mid, right] + 6 * [0])
    h1 = comp(pln, [left] + 2 * [0] + [mid] + 2 * [0] + [right] + 2 * [0])
    h2 = comp(pln, 2 * [0] + [left] + 2 * [0] + [mid] + 2 * [0] + [right])

    if darkthr:
        return core.std.Expr([v1, v2, h1, h2, pln], "b {} > b {} < and x y + z + a + 0 ?".format(darkthr, brightthr))
    else:
        return core.std.Expr([v1, v2, h1, h2], "x y + z + a +")


def merge_detections(input, output, cycle=1, min_zone_len=1, delim=" ", tolerance=0):
    import numpy as np

    def consecutive(data, cycle=cycle):
        return np.split(data, np.where(np.diff(data) > cycle + tolerance)[0] + 1)

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

            if end - start >= min_zone and start != end:
                zone = "{}{delim}{}\n".format(start, end, delim=delim)
                zones.append(zone)

        if zones:
            with open(output, 'w') as out_f:
                out_f.writelines(zones)
                print("Merged frames into zonefile: {}".format(output))


def banddtct(clip, output="banding-frames.txt", thr=150, hi=0.90, lo=0.10, trim=False, cycle=1, merge=True,
             min_zone_len=1, tolerance=0, check_next=True, diff=0.10, darkthr=5632, brightthr=60160, blankthr=None,
             debug=False):
    """
    :param clip: Input clip cropped to disclude black bars.  Can resize beforehand as an easy speed-up.
    :param output: Output text file.
    :param thr: The maximum variation for something to be picked up as banded; the higher this is, the more will be 
                detected. Decent default values for live action are 100-200, 10-75 for anime.
    :param hi: The maximum threshold, which is used to filter out black frames, credits et cetera; changing this won’t
               usually be necessary. If you find that some skies are being ignored, raise this.
    :param lo: The minimum threshold, which is used to determine when a frame can be considered to include banding.
               If there are small banded areas that are missed by the default, lower this. Raising this is a bit like
               raising thr.
    :param trim: If True and cycle > 1, adds a SelectEvery call.
    :param cycle: Allows setting SelectEvery(cycle=cycle, offsets=0). This can speed things up, but beware that because
                  frames need to be decoded, this isn’t that insanely helpful. If you want to use your own SelectRangeEvery
                  call or whatever, still set cycle, but set trim=False!
    :param merge: Whether to merge the detected frames into zones (start end) in a separate file.
    :param min_zone_len: Minimum number of consecutive frames for a zone.
    :param tolerance: Sets additional tolerance for zone detection; if detected frames are cycle + tolerance apart,
                      they’ll be merged into a zone.
    :param check_next: Whether to check the next frame with more lenient settings to make sure zones are picked up properly.
    :param diff: Difference from previous frame for check_next. This is a budget scene change detection. If the next
                 frame is too different from the previous one, it won’t use the lenient detection settings.
    :param darkthr: Threshold under which pixels will be ignored. If you want use an 8-bit value, shift it to 16-bit via
                    e.g. 16 << 8.
    :param brightthr: Threshold above which pixels will be ignored.
    :param blankthr: Threshold under which changes will be ignored. I haven’t tested it yet, but this with a higher pix
                     could be useful for anime.
    :param debug: Setting this to True will output the currently used bandmask in the input clip’s format with the current
                  frame’s value printed in the top left corner; if it falls between the thresholds, the text will turn from green to red.
    :return: None
    """
    import os
    import sys
    import time

    def detect(n, f, clip, hi, lo, detections, diff, check_next):
        if f[0].props.PlaneStatsAverage >= lo and f[0].props.PlaneStatsAverage <= hi:
            if check_next:
                detections.add(n * cycle)
                if f[1].props.PlaneStatsDiff < diff and f[2].props.PlaneStatsAverage >= lo / 2 and f[
                    2].props.PlaneStatsAverage <= hi:
                    detections.add((n + 1) * cycle)
            else:
                detections.add(n * cycle)

        return clip


    def debug_detect(n, f, clip, hi, lo):
        if f.props.PlaneStatsAverage >= lo and f.props.PlaneStatsAverage <= hi:
            return clip.sub.Subtitle(f"{f.props.PlaneStatsAverage}\nBanding detected!", style="sans-serif,20,&H000000FF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,7,10,10,10,1")
        else:
            return clip.sub.Subtitle(f.props.PlaneStatsAverage, style="sans-serif,20,&H0000FF00,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,7,10,10,10,1")


    original_format = clip.format

    clip = bandmask(clip, thr=thr, pix=3, left=1, mid=2, right=1, dec=3, exp=None, plane=0, darkthr=darkthr,
                    brightthr=brightthr, blankthr=blankthr)

    if debug:
        clip = clip.resize.Point(format=original_format)
        return clip.std.FrameEval(partial(debug_detect, clip=clip, hi=hi, lo=lo), clip.std.PlaneStats())

    if trim and cycle > 1:
        clip = clip.std.SelectEvery(cycle=cycle, offsets=0)

    next_frame = clip[1:]

    clip_diff = core.std.PlaneStats(clip, next_frame)
    clip = clip.std.PlaneStats()
    next_frame = next_frame.std.PlaneStats()
    total_frames = clip.num_frames

    detected_frames = set([])

    with open(os.devnull, 'wb') as f:
        processed = core.std.FrameEval(clip,
                                       partial(detect, clip=clip, hi=hi, lo=lo, detections=detected_frames, diff=diff,
                                               check_next=check_next),
                                       prop_src=[clip, clip_diff, next_frame])
        start = time.time()
        processed.output(f, progress_update=__vs_out_updated)

    # Sort frames because multithreading likely made them weird
    detected_frames = list(detected_frames)
    detected_frames.sort()

    end = time.time()
    print("Elapsed: {:0.2f} seconds ({:0.2f} fps)".format(end - start, total_frames / float(end - start)))
    print("Detected frames: {}".format(len(detected_frames)))

    if detected_frames:
        with open(output, 'w') as out_file:
            for f in detected_frames:
                out_file.write("{}\n".format(f))

        if merge:
            merged_output = "merged-{}".format(output)
            merge_detections(output, merged_output, cycle=cycle, min_zone_len=min_zone_len, tolerance=tolerance)

        quit("Finished detecting banding, output file: {}".format(output))

    return None


def detect_dirty_lines(clip, output, left, top, right, bottom, thr=.1, merged_output=None, cycle=1, tolerance=0):
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
                clip_b = luma.std.Crop(top=num - 1, bottom=1)
        elif ori == "column" or ori == "col":
            clip_a = luma.std.Crop(left=num, right=luma.width - num - 1)
            if num + 1 < luma.width:
                clip_b = luma.std.Crop(left=num + 1, right=luma.width - num - 2)
            else:
                clip_b = luma.std.Crop(left=num - 1, right=1)
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

    column_list = []
    if left:
        column_list.append(left)
    if right:
        column_list.append(right)
    row_list = []
    if top:
        row_list.append(top)
    if bottom:
        row_list.append(bottom)

    with open(os.devnull, 'wb') as f:
        for _ in column_list:
            for i in _:
                clip_diff = get_rows(luma, "col", i)
                processed = core.std.FrameEval(clip, partial(detect, clip=clip, thr=thr, detections=detected_frames),
                                               prop_src=clip_diff)
        for _ in row_list:
            for i in _:
                clip_diff = get_rows(luma, "row", i)
                processed = core.std.FrameEval(clip, partial(detect, clip=clip, thr=thr, detections=detected_frames),
                                               prop_src=clip_diff)
        start = time.time()
        processed.output(f, progress_update=updated)

    end = time.time()
    print("Elapsed: {:0.2f} seconds ({:0.2f} fps)".format(end - start, total_frames / float(end - start)))
    print("Detected frames: {}".format(len(detected_frames)))

    if detected_frames:
        with open(output, 'w') as out_file:
            for f in detected_frames:
                out_file.write("{}\n".format(f))

        if merged_output:
            merge_detections(output, merged_output, cycle=cycle, tolerance=tolerance)

    return None


def dirtdtct(clip, output="dirty-frames.txt", left=None, top=None, right=None, bottom=None, thr=.1, trim=False,
             cycle=1, merge=True, tolerance=0):
    """
    :param clip: Input clip cropped to disclude black bars.
    :param output: Output text file.
    :param left, top, right, bottom: Which rows/columns to look at. For column 1919, set right=0, right=1 for 1918 etc.
    :param thr: Minimum difference for it to be considered a dirty line. Set way lower for live action (something
                like .01 for weak dirty lines, around .05 for stronger ones). Lower means more gets picked up.
    :param trim: If True and cycle > 1, adds a SelectEvery call.
    :param cycle: Allows setting SelectEvery(cycle=cycle, offsets=0). This can speed things up, but beware that because
                  frames need to be decoded, this isn’t that insanely helpful. If you want to use your own SelectRangeEvery
                  call or whatever, still set cycle, but set trim=False!
    :param merge: Whether to merge the detected frames into zones (start end) in a separate file.
    :param tolerance: Sets additional tolerance for zone detection; if detected frames are cycle + tolerance apart,
                      they’ll be merged into a zone.
    :return: None
    """
    if isinstance(left, int):
        left = [left]
    if isinstance(top, int):
        top = [top]
    if isinstance(right, int):
        right = [clip.width - right - 1]
    elif isinstance(right, list):
        for _ in range(len(right)):
            right[_] = clip.width - right[_] - 1
    if isinstance(bottom, int):
        bottom = [clip.height - bottom - 1]
    elif isinstance(bottom, list):
        for _ in range(len(bottom)):
            bottom[_] = clip.height - bottom[_] - 1

    if trim and cycle > 1:
        clip = clip.std.SelectEvery(cycle=cycle, offsets=0)
    else:
        cycle = 1

    if merge:
        merge = "merged-{}".format(output)

    detect_dirty_lines(clip, output, left, top, right, bottom, thr, merge, cycle, tolerance=tolerance)
    quit("Finished detecting dirty lines, output file: {}".format(output))

    return None


def brdrdtct(clip, output="bordered-frames.txt", range=4, left=0, right=0, top=0, bottom=0, color=[0, 123, 123],
             color_second=[21, 133, 133], trim=False, cycle=1, merge=True, min_zone_len=1, tolerance=0):
    """
    :param clip: Input clip cropped to disclude black bars.
    :param output: Output text file.
    :param range, left, right, top, bottom, color, color_second: https://github.com/Irrational-Encoding-Wizardry/vapoursynth-autocrop/wiki/CropValues
    :param trim: If True and cycle > 1, adds a SelectEvery call.
    :param cycle: Allows setting SelectEvery(cycle=cycle, offsets=0). This can speed things up, but beware that because
                  frames need to be decoded, this isn’t that insanely helpful. If you want to use your own SelectRangeEvery
                  call or whatever, still set cycle, but set trim=False!
    :param merge: Whether to merge the detected frames into zones (start end) in a separate file.
    :param min_zone_len: Minimum number of consecutive frames for a zone.
    :param tolerance: Sets additional tolerance for zone detection; if detected frames are cycle + tolerance apart,
                      they’ll be merged into a zone.
    :return: None
    """
    import os
    import sys
    import time
    
    def detect(n, f, clip, detections):
        if (f.props.CropTopValue > 0 and f.props.CropBottomValue > 0) or (f.props.CropLeftValue > 0 and f.props.CropRightValue > 0):
            detections.add(n * cycle)
        return clip
    
    if trim and cycle > 1:
        clip = clip.std.SelectEvery(cycle=cycle, offsets=0)
    
    total_frames = clip.num_frames
    
    detected_frames = set([])
    
    clip = clip.acrop.CropValues(range=range, left=left, right=right, top=top, bottom=bottom, color=color,
                                 color_second=color_second)

    with open(os.devnull, 'wb') as f:
        processed = core.std.FrameEval(clip,
                                       partial(detect, clip=clip, detections=detected_frames),
                                       prop_src=clip)
        start = time.time()
        processed.output(f, progress_update=__vs_out_updated)
    
    # Sort frames because multithreading likely made them weird
    detected_frames = list(detected_frames)
    detected_frames.sort()

    end = time.time()
    print("Elapsed: {:0.2f} seconds ({:0.2f} fps)".format(end - start, total_frames / float(end - start)))
    print("Detected frames: {}".format(len(detected_frames)))

    if detected_frames:
        with open(output, 'w') as out_file:
            for f in detected_frames:
                out_file.write("{}\n".format(f))

        if merge:
            merged_output = "merged-{}".format(output)
            merge_detections(output, merged_output, cycle=cycle, min_zone_len=min_zone_len, tolerance=tolerance)

        quit("Finished detecting bordered frames, output file: {}".format(output))

    return None
