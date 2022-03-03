import vapoursynth as vs
from vapoursynth import core

import os
import time
from os import PathLike
from typing import Callable, Dict, List, Set, Union, Optional, Any, Iterable

from functools import partial
from vsutil import iterate, get_y
from vsutil import plane as fplane
from vsutil import depth as Depth

from .base import SUBTITLE_DEFAULT_STYLE


def __vs_out_updated(current: int, total: int):
    if current == total:
        print("Frame: {}/{}".format(current, total), end="\n")
    else:
        print("Frame: {}/{}".format(current, total), end="\r")


def __detect(clip: vs.VideoNode, func: Callable[[Set[int]], vs.VideoNode], options: Dict):
    total_frames = clip.num_frames
    detections: Set[int] = set([])

    output = options['output']
    merge = options['merge']

    with open(os.devnull, 'wb') as f:
        processed = func(detections)

        start = time.time()
        processed.output(f, progress_update=__vs_out_updated)

    # Sort frames because multithreading likely made them weird
    detections_list = list(detections)
    detections_list.sort()

    end = time.time()
    print("Elapsed: {:0.2f} seconds ({:0.2f} fps)".format(end - start, total_frames / float(end - start)))
    print("Detected frames: {}".format(len(detections_list)))

    if detections_list:
        with open(output, 'w') as out_file:
            for d in detections_list:
                out_file.write(f"{d}\n")

        if merge:
            merged_output = "merged-{}".format(output)
            merge_detections(output,
                             merged_output,
                             cycle=options['cycle'],
                             min_zone_len=options['min_zone_len'],
                             tolerance=options['tolerance'])

        quit("Finished detecting, output file: {}".format(output))


def bandmask(clip: vs.VideoNode,
             thr: int = 1000,
             pix: int = 3,
             left: int = 1,
             mid: int = 1,
             right: int = 1,
             dec: int = 2,
             exp: Optional[int] = None,
             plane: int = 0,
             darkthr: Optional[int] = None,
             brightthr: Optional[int] = None,
             blankthr: Optional[int] = None) -> vs.VideoNode:
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
            thr = int(thr / 65535)

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


def merge_detections(input: Union[str, PathLike],
                     output: Union[str, PathLike],
                     cycle: int = 1,
                     min_zone_len: int = 1,
                     delim: str = ' ',
                     tolerance: int = 0) -> None:
    import numpy as np

    def consecutive(data: Iterable[int], cycle: int = cycle):
        return np.split(data, np.where(np.diff(data) > cycle + tolerance)[0] + 1)

    with open(input, 'r') as in_f:
        a = np.array(in_f.read().splitlines(), dtype=np.uint)
        c = consecutive(a, cycle=cycle)

        zones = []
        actual_cycle = cycle - 1

        if min_zone_len >= cycle:
            min_zone = min_zone_len - 1 if min_zone_len % cycle == 0 else min_zone_len
        else:
            min_zone = min_zone_len - 1 if cycle % min_zone_len == 0 else min_zone_len

        for dtc in c:
            start = int(dtc[0])
            end = int(dtc[-1] + actual_cycle)

            if end - start >= min_zone and start != end:
                zone = "{}{delim}{}\n".format(start, end, delim=delim)
                zones.append(zone)

        if zones:
            with open(output, 'w') as out_f:
                out_f.writelines(zones)
                print("Merged frames into zonefile: {}".format(output))


def banddtct(clip: vs.VideoNode,
             output: Union[str, PathLike] = "banding-frames.txt",
             thr: int = 150,
             hi: float = 0.90,
             lo: float = 0.10,
             trim: bool = False,
             cycle: int = 1,
             merge: bool = True,
             min_zone_len: int = 1,
             tolerance: int = 0,
             check_next: bool = True,
             diff: float = 0.10,
             darkthr: int = 5632,
             brightthr: int = 60160,
             blankthr: Optional[int] = None,
             debug: bool = False) -> None:
    """
    :param clip: Input clip cropped to disclude black bars.  Can resize beforehand as an easy speed-up.
    :param output: Output text file.
    :param thr: The maximum variation for something to be picked up as banded; the higher this is, the more will be
                detected. Decent default values for live action are 100-200, 10-75 for anime.
    :param hi: The maximum threshold, which is used to filter out black frames, credits et cetera; changing this won't
               usually be necessary. If you find that some skies are being ignored, raise this.
    :param lo: The minimum threshold, which is used to determine when a frame can be considered to include banding.
               If there are small banded areas that are missed by the default, lower this. Raising this is a bit like
               raising thr.
    :param trim: If True and cycle > 1, adds a SelectEvery call.
    :param cycle: Allows setting SelectEvery(cycle=cycle, offsets=0). This can speed things up, but beware that because
                  frames need to be decoded, this isn’t that insanely helpful.
                  If you want to use your own SelectRangeEvery call or whatever, still set cycle, but set trim=False!
    :param merge: Whether to merge the detected frames into zones (start end) in a separate file.
    :param min_zone_len: Minimum number of consecutive frames for a zone.
    :param tolerance: Sets additional tolerance for zone detection; if detected frames are cycle + tolerance apart,
                      they’ll be merged into a zone.
    :param check_next: Whether to check the next frame with more lenient settings
                        to make sure zones are picked up properly.
    :param diff: Difference from previous frame for check_next. This is a budget scene change detection. If the next
                 frame is too different from the previous one, it won’t use the lenient detection settings.
    :param darkthr: Threshold under which pixels will be ignored. If you want use an 8-bit value, shift it to 16-bit via
                    e.g. 16 << 8.
    :param brightthr: Threshold above which pixels will be ignored.
    :param blankthr: Threshold under which changes will be ignored. I haven’t tested it yet, but this with a higher pix
                     could be useful for anime.
    :param debug: Setting this to True will output the currently used bandmask in the input clip’s format
                  with the current frame’s value printed in the top left corner.
                  If it falls between the thresholds, the text will turn from green to red.
    :return: None
    """

    options = locals()

    def debug_detect(n, f, clip, hi, lo):
        if f.props.PlaneStatsAverage >= lo and f.props.PlaneStatsAverage <= hi:
            return clip.sub.Subtitle(f"{f.props.PlaneStatsAverage}\nDetected banding!", style=SUBTITLE_DEFAULT_STYLE)
        else:
            return clip.sub.Subtitle(f.props.PlaneStatsAverage, style=SUBTITLE_DEFAULT_STYLE)

    original_format = clip.format
    clip = bandmask(clip,
                    thr=thr,
                    pix=3,
                    left=1,
                    mid=2,
                    right=1,
                    dec=3,
                    exp=None,
                    plane=0,
                    darkthr=darkthr,
                    brightthr=brightthr,
                    blankthr=blankthr)

    if debug:
        clip = clip.resize.Point(format=original_format)
        return clip.std.FrameEval(partial(debug_detect, clip=clip, hi=hi, lo=lo), clip.std.PlaneStats())

    if trim and cycle > 1:
        clip = clip.std.SelectEvery(cycle=cycle, offsets=0)

    next_frame = clip[1:]

    clip_diff = core.std.PlaneStats(clip, next_frame)
    clip = clip.std.PlaneStats()
    next_frame = next_frame.std.PlaneStats()

    prop_src = [clip, clip_diff, next_frame]

    def detect_func(detections):

        def banding_detect(n, f, clip, detections, hi, lo, diff, check_next):
            if f[0].props.PlaneStatsAverage >= lo and f[0].props.PlaneStatsAverage <= hi:
                if check_next:
                    detections.add(n * cycle)
                    if f[1].props.PlaneStatsDiff < diff and f[2].props.PlaneStatsAverage >= lo / 2 and f[
                            2].props.PlaneStatsAverage <= hi:
                        detections.add((n + 1) * cycle)
                else:
                    detections.add(n * cycle)
            return clip

        return core.std.FrameEval(clip,
                                  partial(banding_detect,
                                          clip=clip,
                                          detections=detections,
                                          hi=hi,
                                          lo=lo,
                                          diff=diff,
                                          check_next=check_next),
                                  prop_src=prop_src)

    __detect(clip, detect_func, options)


def cambidtct(clip: vs.VideoNode,
              output: Union[str, PathLike] = "banding-frames.txt",
              thr: float = 5.0,
              thr_next: float = 4.5,
              cambi_args: Union[None, dict] = None,
              trim: bool = False,
              cycle: int = 1,
              merge: bool = True,
              min_zone_len: int = 1,
              tolerance: int = 0,
              check_next: bool = True,
              diff: float = 0.10,
              debug: bool = False) -> None:

    options = locals()

    def debug_detect(n, f, clip):
        if f.props['CAMBI'] >= thr:
            return clip.sub.Subtitle(f"{f.props['CAMBI']}\nDetected banding!", style=SUBTITLE_DEFAULT_STYLE)
        else:
            return clip.sub.Subtitle(f.props['CAMBI'], style=SUBTITLE_DEFAULT_STYLE)

    cambi_dict: Dict[str, Any] = dict(topk=0.1, tvi_threshold=0.012)
    if cambi_args is not None:
        cambi_dict |= cambi_args
    clip = core.akarin.Cambi(clip, **cambi_dict)

    if debug:
        return clip.std.FrameEval(partial(debug_detect, clip=clip), clip.std.PlaneStats())

    if trim and cycle > 1:
        clip = clip.std.SelectEvery(cycle=cycle, offsets=0)

    next_frame = clip[1:]

    clip_diff = core.std.PlaneStats(clip, next_frame)
    clip = clip.std.PlaneStats()
    next_frame = next_frame.std.PlaneStats()

    prop_src = [clip, clip_diff, next_frame]

    def detect_func(detections):

        def banding_detect(n, f, clip, detections, diff, check_next):
            if f[0].props['CAMBI'] >= thr:
                if check_next:
                    detections.add(n * cycle)
                    if f[1].props.PlaneStatsDiff < diff and f[2].props['CAMBI'] >= thr_next:
                        detections.add((n + 1) * cycle)
                else:
                    detections.add(n * cycle)
            return clip

        return core.std.FrameEval(clip,
                                  partial(banding_detect,
                                          clip=clip,
                                          detections=detections,
                                          diff=diff,
                                          check_next=check_next),
                                  prop_src=prop_src)

    __detect(clip, detect_func, options)


def __detect_dirty_lines(clip: vs.VideoNode,
                         output: Union[str, PathLike],
                         left: Optional[Union[int, List[int]]],
                         top: Optional[Union[int, List[int]]],
                         right: Optional[Union[int, List[int]]],
                         bottom: Optional[Union[int, List[int]]],
                         thr: float,
                         cycle: int,
                         merge: bool = True,
                         min_zone_len: int = 1,
                         tolerance: int = 0) -> None:
    options = locals()

    luma = get_y(clip)

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

    def detect_func(detections):

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

        def line_detect(n, f, clip, detections, thr):
            if f.props.PlaneStatsDiff > thr:
                detections.add(n * cycle)
            return clip

        for _ in column_list:
            for i in _:
                clip_diff = get_rows(luma, "col", i)
                processed = core.std.FrameEval(clip,
                                               partial(line_detect, clip=clip, detections=detections, thr=thr),
                                               prop_src=clip_diff)
        for _ in row_list:
            for i in _:
                clip_diff = get_rows(luma, "row", i)
                processed = core.std.FrameEval(clip,
                                               partial(line_detect, clip=clip, detections=detections, thr=thr),
                                               prop_src=clip_diff)
        return processed

    __detect(clip, detect_func, options)


def dirtdtct(clip: vs.VideoNode,
             output: Union[str, PathLike] = "dirty-frames.txt",
             left: Optional[Union[int, List[int]]] = None,
             top: Optional[Union[int, List[int]]] = None,
             right: Optional[Union[int, List[int]]] = None,
             bottom: Optional[Union[int, List[int]]] = None,
             thr: float = 0.1,
             trim: bool = False,
             cycle: int = 1,
             merge: bool = True,
             min_zone_len: int = 1,
             tolerance: int = 0) -> None:
    """
    :param clip: Input clip cropped to disclude black bars.
    :param output: Output text file.
    :param left, top, right, bottom: Which rows/columns to look at. For column 1919, set right=0, right=1 for 1918 etc.
    :param thr: Minimum difference for it to be considered a dirty line. Set way lower for live action (something
                like .01 for weak dirty lines, around .05 for stronger ones). Lower means more gets picked up.
    :param trim: If True and cycle > 1, adds a SelectEvery call.
    :param cycle: Allows setting SelectEvery(cycle=cycle, offsets=0). This can speed things up, but beware that because
                  frames need to be decoded, this isn’t that insanely helpful.
                  If you want to use your own SelectRangeEvery call or whatever, still set cycle, but set trim=False!
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

    __detect_dirty_lines(clip, output, left, top, right, bottom, thr, cycle, merge, min_zone_len, tolerance)


def brdrdtct(clip: vs.VideoNode,
             output: Union[str, PathLike] = "bordered-frames.txt",
             range: int = 4,
             left: int = 0,
             right: int = 0,
             top: int = 0,
             bottom: int = 0,
             color: List[int] = [0, 123, 123],
             color_second: List[int] = [21, 133, 133],
             trim: bool = False,
             cycle: int = 1,
             merge: bool = True,
             min_zone_len: int = 1,
             tolerance: int = 0) -> None:
    """
    :param clip: Input clip cropped to disclude black bars.
    :param output: Output text file.
    :param range, left, right, top, bottom, color, color_second:
        https://github.com/Irrational-Encoding-Wizardry/vapoursynth-autocrop/wiki/CropValues
    :param trim: If True and cycle > 1, adds a SelectEvery call.
    :param cycle: Allows setting SelectEvery(cycle=cycle, offsets=0). This can speed things up, but beware that because
                  frames need to be decoded, this isn’t that insanely helpful.
                  If you want to use your own SelectRangeEvery call or whatever, still set cycle, but set trim=False!
    :param merge: Whether to merge the detected frames into zones (start end) in a separate file.
    :param min_zone_len: Minimum number of consecutive frames for a zone.
    :param tolerance: Sets additional tolerance for zone detection; if detected frames are cycle + tolerance apart,
                      they’ll be merged into a zone.
    :return: None
    """

    options = locals()

    if trim and cycle > 1:
        clip = clip.std.SelectEvery(cycle=cycle, offsets=0)

    clip = clip.acrop.CropValues(range=range,
                                 left=left,
                                 right=right,
                                 top=top,
                                 bottom=bottom,
                                 color=color,
                                 color_second=color_second)

    def detect_func(detections):

        def border_detect(n, f, clip, detections):
            if ((f.props.CropTopValue > 0 and f.props.CropBottomValue > 0)
                    or (f.props.CropLeftValue > 0 and f.props.CropRightValue > 0)):
                detections.add(n * cycle)

            return clip

        processed = core.std.FrameEval(clip, partial(border_detect, clip=clip, detections=detections), prop_src=clip)

        return processed

    __detect(clip, detect_func, options)


#####################
#      Exports      #
#####################

__all__ = [
    "banddtct",
    "bandmask",
    "brdrdtct",
    "cambidtct",
    "dirtdtct",
    "merge_detections",
]
