import os
import time
from enum import Enum
from functools import partial
from os import PathLike
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Union

import vapoursynth as vs
import vsutil
from vapoursynth import core

from .base import SUBTITLE_DEFAULT_STYLE
from .types import placebo
from .types.dovi import HdrMeasurement
from .types.misc import DetectProgressState


def awf_init_progress_state() -> DetectProgressState:
    return {
        "start_time": time.monotonic(),
        "frames_done": 0,
        "fps": 0.0,
        "last_fps_report_time": time.monotonic(),
    }


def awf_vs_out_updated(current: int, total: int, state: Optional[DetectProgressState] = None):
    progress = f"Frame: {current}/{total}"
    ratio = float(current) / float(total)

    if state:
        state["frames_done"] += 1

        current = time.monotonic()
        elapsed = current - state["last_fps_report_time"]
        elapsed_from_start = current - state["start_time"]

        if elapsed > 0.5:
            state["last_fps_report_time"] = current
        if elapsed_from_start > 8:
            state["fps"] = state["frames_done"] / elapsed_from_start

            progress += f' ({state["fps"]:0.2f} fps)'

    progress += f' {ratio:0.2%}'

    if current == total:
        print(progress, end="\n")
    else:
        print(progress, end="\r")


def _detect(clip: vs.VideoNode,
            name: str,
            func: Callable[[Set[int]], vs.VideoNode],
            options: Optional[Dict] = None,
            quit_after: bool = True,
            print_detections: bool = True) -> Optional[List[int]]:
    total_frames = clip.num_frames
    detections: Set[int] = set([])

    if options is None:
        options = {}

    output = options.get('output', None)
    merge = options.get('merge', False)
    filter_consecutives = options.get('filter_consecutives', False)

    print(f"Running {name} detection...")

    with open(os.devnull, 'wb') as f:
        processed = func(detections)

        state = awf_init_progress_state()
        processed.output(f, progress_update=partial(awf_vs_out_updated, state=state))

    # Sort frames because multithreading likely made them weird
    detections_list = list(detections)
    detections_list.sort()

    if filter_consecutives:
        detections_list = merge_detections(detections_list,
                                           cycle=options.get('cycle', 1),
                                           min_zone_len=options.get('min_zone_len', 1),
                                           tolerance=options.get('tolerance', 0),
                                           start_only=True)

    start = state["start_time"]
    end = time.monotonic()
    print(f"\nElapsed: {end - start:0.2f} seconds ({total_frames / float(end - start):0.2f} fps)")

    if print_detections:
        print(f"Detected frames: {detections_list}")

    if detections_list is not None:
        if output:
            with open(output, 'w') as out_file:
                for d in detections_list:
                    out_file.write(f"{d}\n")

            if merge:
                merged_output = f"merged-{output}"
                merge_detections(output,
                                 output=merged_output,
                                 cycle=options.get('cycle', 1),
                                 min_zone_len=options.get('min_zone_len', 1),
                                 tolerance=options.get('tolerance', 0))

        if quit_after:
            quit(f"Finished detecting, output file: {output}")

        return detections_list


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
    :param plane: Plane index to be processed.
    :param darkthr: If set, values under darkthr will be ignored and set black.
    :param brightthr: If set, values above brightthr will be ignored and set to black.
                      If either darkthr or brightthr is set and the other isn't, the other will default to TV range.
    :param blankthr: If set, values with less change than this will be ignored and set black.
    :return: Grayscale mask of areas with gradient smaller than thr.
    """

    depth = clip.format.bits_per_sample
    hi = 65535

    if depth < 16:
        clip = vsutil.depth(clip, 16)
    elif depth == 32:
        hi = 1
        if thr >= 1:
            thr = int(thr / 65535)

    if exp is None:
        exp = dec + pix

    pln = vsutil.plane(clip, plane)

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
            diff = core.std.Expr([orig, c], "x y - abs").std.Expr(f"x {blankthr} > x {thr} < thr {hi} 0 ?")
        decreased = vsutil.iterate(diff, core.std.Minimum, dec)
        return vsutil.iterate(decreased, core.std.Maximum, exp)

    v1 = comp(pln, 6 * [0] + [left, mid, right])
    v2 = comp(pln, [left, mid, right] + 6 * [0])
    h1 = comp(pln, [left] + 2 * [0] + [mid] + 2 * [0] + [right] + 2 * [0])
    h2 = comp(pln, 2 * [0] + [left] + 2 * [0] + [mid] + 2 * [0] + [right])

    if darkthr:
        return core.std.Expr([v1, v2, h1, h2, pln], f"b {darkthr} > b {brightthr} < and x y + z + a + 0 ?")

    return core.std.Expr([v1, v2, h1, h2], "x y + z + a +")


def get_min_diff_consecutives(input_arg: Union[str, PathLike, List[int]],
                              cycle: int = 1,
                              tolerance: int = 0) -> List[int]:
    import numpy as np

    def consecutive(data: Iterable[int], cycle: int, tolerance: int):
        return np.split(data, np.where(np.diff(data) > cycle + tolerance)[0] + 1)

    if isinstance(input_arg, list):
        return consecutive(input_arg, cycle, tolerance)

    with open(input_arg, 'r') as in_f:
        a = np.array(in_f.read().splitlines(), dtype=np.uint)
        return consecutive(a, cycle, tolerance)


def merge_detections(input_arg: Union[str, PathLike, List[int]],
                     output: Optional[Union[str, PathLike]] = None,
                     cycle: int = 1,
                     min_zone_len: int = 1,
                     delim: str = ' ',
                     tolerance: int = 0,
                     start_only: bool = False) -> Optional[List[int]]:
    c = get_min_diff_consecutives(input_arg, cycle=cycle, tolerance=tolerance)

    zones = []
    actual_cycle = cycle - 1

    if min_zone_len >= cycle:
        min_zone = min_zone_len - 1 if min_zone_len % cycle == 0 else min_zone_len
    else:
        min_zone = min_zone_len - 1 if cycle % min_zone_len == 0 else min_zone_len

    for dtc in c:
        if len(dtc) == 0:
            continue

        start = int(dtc[0])
        end = int(dtc[-1] + actual_cycle)

        if end - start >= min_zone and start != end:
            if start_only:
                zones.append(start)
            else:
                zone = f"{start}{delim}{end}\n"
                zones.append(zone)

    if output and zones:
        with open(output, 'w') as out_f:
            out_f.writelines(zones)
            print(f"Merged frames into zonefile: {output}")

    return zones


def banddtct(
        clip: vs.VideoNode,
        output: Union[str, PathLike] = "banding-frames.txt",  # pylint: disable=unused-argument
        thr: int = 150,
        hi: float = 0.90,
        lo: float = 0.10,
        trim: bool = False,
        cycle: int = 1,
        merge: bool = True,  # pylint: disable=unused-argument
        min_zone_len: int = 1,  # pylint: disable=unused-argument
        tolerance: int = 0,  # pylint: disable=unused-argument
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
                  frames need to be decoded, this isn't that insanely helpful.
                  If you want to use your own SelectRangeEvery call or whatever, still set cycle, but set trim=False!
    :param merge: Whether to merge the detected frames into zones (start end) in a separate file.
    :param min_zone_len: Minimum number of consecutive frames for a zone.
    :param tolerance: Sets additional tolerance for zone detection; if detected frames are cycle + tolerance apart,
                      they'll be merged into a zone.
    :param check_next: Whether to check the next frame with more lenient settings
                        to make sure zones are picked up properly.
    :param diff: Difference from previous frame for check_next. This is a budget scene change detection. If the next
                 frame is too different from the previous one, it won't use the lenient detection settings.
    :param darkthr: Threshold under which pixels will be ignored. If you want use an 8-bit value, shift it to 16-bit via
                    e.g. 16 << 8.
    :param brightthr: Threshold above which pixels will be ignored.
    :param blankthr: Threshold under which changes will be ignored. I haven't tested it yet, but this with a higher pix
                     could be useful for anime.
    :param debug: Setting this to True will output the currently used bandmask in the input clip's format
                  with the current frame's value printed in the top left corner.
                  If it falls between the thresholds, the text will turn from green to red.
    :return: None
    """

    options = locals()

    def debug_detect(_n, f, clip, hi, lo):
        if f.props.PlaneStatsAverage >= lo and f.props.PlaneStatsAverage <= hi:
            return clip.sub.Subtitle(f"{f.props.PlaneStatsAverage}\nDetected banding!", style=SUBTITLE_DEFAULT_STYLE)

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
        return core.std.FrameEval(clip, partial(debug_detect, clip=clip, hi=hi, lo=lo), clip.std.PlaneStats())

    if trim and cycle > 1:
        clip = clip.std.SelectEvery(cycle=cycle, offsets=0)

    next_frame = clip[1:]

    clip_diff = core.std.PlaneStats(clip, next_frame)
    clip = clip.std.PlaneStats()
    next_frame = next_frame.std.PlaneStats()

    prop_src = [clip, clip_diff, next_frame]

    def detect_func(detections: Set[int]):

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

    _detect(clip, "Banding", detect_func, options=options)


def cambidtct(
        clip: vs.VideoNode,
        output: Union[str, PathLike] = "banding-frames.txt",  # pylint: disable=unused-argument
        thr: float = 5.0,
        thr_next: float = 4.5,
        cambi_args: Union[None, dict] = None,
        trim: bool = False,
        cycle: int = 1,
        merge: bool = True,  # pylint: disable=unused-argument
        min_zone_len: int = 1,  # pylint: disable=unused-argument
        tolerance: int = 0,  # pylint: disable=unused-argument
        check_next: bool = True,
        diff: float = 0.10,
        debug: bool = False) -> None:

    options = locals()

    def debug_detect(_n, f, clip):
        if f.props['CAMBI'] >= thr:
            return clip.sub.Subtitle(f"{f.props['CAMBI']}\nDetected banding!", style=SUBTITLE_DEFAULT_STYLE)

        return clip.sub.Subtitle(f.props['CAMBI'], style=SUBTITLE_DEFAULT_STYLE)

    cambi_dict: Dict[str, Any] = dict(topk=0.1, tvi_threshold=0.012)
    if cambi_args is not None:
        cambi_dict |= cambi_args
    clip = core.akarin.Cambi(clip, **cambi_dict)

    if debug:
        return core.std.FrameEval(clip, partial(debug_detect, clip=clip), prop_src=clip.std.PlaneStats())

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

    _detect(clip, "CAMBI", detect_func, options=options)


def detect_dirty_lines(
        clip: vs.VideoNode,
        output: Union[str, PathLike],  # pylint: disable=unused-argument
        left: Optional[Union[int, List[int]]],
        top: Optional[Union[int, List[int]]],
        right: Optional[Union[int, List[int]]],
        bottom: Optional[Union[int, List[int]]],
        thr: float,
        cycle: int,
        merge: bool = True,  # pylint: disable=unused-argument
        min_zone_len: int = 1,  # pylint: disable=unused-argument
        tolerance: int = 0) -> None:  # pylint: disable=unused-argument
    options = locals()

    luma = vsutil.get_y(clip)

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
            elif ori in ("column", "col"):
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

    _detect(clip, "Dirty lines", detect_func, options=options)


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
                  frames need to be decoded, this isn't that insanely helpful.
                  If you want to use your own SelectRangeEvery call or whatever, still set cycle, but set trim=False!
    :param merge: Whether to merge the detected frames into zones (start end) in a separate file.
    :param tolerance: Sets additional tolerance for zone detection; if detected frames are cycle + tolerance apart,
                      they'll be merged into a zone.
    :return: None
    """
    if isinstance(left, int):
        left = [left]
    if isinstance(top, int):
        top = [top]
    if isinstance(right, int):
        right = [clip.width - right - 1]
    elif isinstance(right, list):
        for (i, _) in enumerate(right):
            right[i] = clip.width - right[i] - 1
    if isinstance(bottom, int):
        bottom = [clip.height - bottom - 1]
    elif isinstance(bottom, list):
        for (i, _) in enumerate(bottom):
            bottom[i] = clip.height - bottom[i] - 1

    if trim and cycle > 1:
        clip = clip.std.SelectEvery(cycle=cycle, offsets=0)

    detect_dirty_lines(clip, output, left, top, right, bottom, thr, cycle, merge, min_zone_len, tolerance)


def brdrdtct(
        clip: vs.VideoNode,
        output: Union[str, PathLike] = "bordered-frames.txt",  # pylint: disable=unused-argument
        ac_range: int = 4,
        left: int = 0,
        right: int = 0,
        top: int = 0,
        bottom: int = 0,
        color: Optional[List[int]] = None,
        color_second: Optional[List[int]] = None,
        trim: bool = False,
        cycle: int = 1,
        merge: bool = True,  # pylint: disable=unused-argument
        min_zone_len: int = 1,  # pylint: disable=unused-argument
        tolerance: int = 0) -> None:  # pylint: disable=unused-argument
    """
    :param clip: Input clip cropped to disclude black bars.
    :param output: Output text file.
    :param ac_range, left, right, top, bottom, color, color_second:
        https://github.com/Irrational-Encoding-Wizardry/vapoursynth-autocrop/wiki/CropValues
    :param trim: If True and cycle > 1, adds a SelectEvery call.
    :param cycle: Allows setting SelectEvery(cycle=cycle, offsets=0). This can speed things up, but beware that because
                  frames need to be decoded, this isn't that insanely helpful.
                  If you want to use your own SelectRangeEvery call or whatever, still set cycle, but set trim=False!
    :param merge: Whether to merge the detected frames into zones (start end) in a separate file.
    :param min_zone_len: Minimum number of consecutive frames for a zone.
    :param tolerance: Sets additional tolerance for zone detection; if detected frames are cycle + tolerance apart,
                      they'll be merged into a zone.
    :return: None
    """

    if color is None:
        color = [0, 123, 123]
    if color_second is None:
        color_second = [21, 133, 133]

    options = locals()

    if trim and cycle > 1:
        clip = clip.std.SelectEvery(cycle=cycle, offsets=0)

    clip = clip.acrop.CropValues(range=ac_range,
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

    _detect(clip, "Borders", detect_func, options=options)


def _av_scenechange_detect(command: List[str], clip: vs.VideoNode) -> List[int]:
    import json
    from subprocess import PIPE, Popen

    with Popen(command, stdin=PIPE, stdout=PIPE, stderr=PIPE) as proc:
        state = awf_init_progress_state()
        clip.output(proc.stdin, y4m=True, progress_update=partial(awf_vs_out_updated, state=state))

        start = state["start_time"]
        end = time.monotonic()
        print(f"\nElapsed: {end - start:0.2f} seconds ({clip.num_frames / float(end - start):0.2f} fps)")

        stdout, stderr = proc.communicate()
        if stderr:
            print(stderr)

        result_json = json.loads(stdout.decode("utf-8"))
        scene_changes = result_json["scene_changes"]

        print(f"Scene changes detected: {len(scene_changes)}, speed: {result_json['speed']:.2f} fps")

        return scene_changes


class SceneChangeDetector(str, Enum):
    AvScenechange = 'av-scenechange'
    WWXD = 'wwxd'
    SCXVID = 'scxvid'
    MVTools = 'mvtools'

    def run_detection(self,
                      clip: vs.VideoNode,
                      av_sc_cli: Optional[str] = None,
                      output: Optional[Union[str, PathLike]] = None) -> List[int]:
        if self == SceneChangeDetector.AvScenechange:
            scenechange_cli = [av_sc_cli, "-s", "0", "--min-scenecut", "12", "-"]

            scene_changes = _av_scenechange_detect(scenechange_cli, clip)

            if output:
                with open(output, 'w') as out_file:
                    for sc in scene_changes:
                        out_file.write(f"{sc}\n")

            return scene_changes

        prop = "_SceneChangePrev"
        options = dict(output=output)

        if self == SceneChangeDetector.WWXD:
            prop = "Scenechange"
            scd_clip = clip.wwxd.WWXD()
        elif self == SceneChangeDetector.SCXVID:
            clip = core.resize.Spline36(clip, format=vs.YUV420P8, dither_type="error_diffusion")

            scd_clip = clip.scxvid.Scxvid()
        elif self == SceneChangeDetector.MVTools:
            sup = core.mv.Super(clip)
            vectors = core.mv.Analyse(sup)
            scd_clip = core.mv.SCDetection(clip, vectors)

            options = options | dict(filter_consecutives=True, cycle=12, min_zone_len=12)

        def props_scenechange_detect(detections: Set[int]):

            def get_scd_prop(n: int, f: vs.VideoFrame, clip: vs.VideoNode):
                if prop in f.props and f.props[prop] == 1:
                    detections.add(n)

                return clip

            return core.std.FrameEval(clip, partial(get_scd_prop, clip=clip), prop_src=scd_clip)

        return _detect(scd_clip, f"Scene changes {self}", props_scenechange_detect, options=options, quit_after=False)


def run_scenechange_detect(clip: vs.VideoNode,
                           detector: SceneChangeDetector = SceneChangeDetector.AvScenechange,
                           tonemap: bool = True,
                           brighten: bool = True,
                           preview: bool = False,
                           av_sc_cli: str = "av-scenechange",
                           output: Optional[Union[str, PathLike]] = None) -> Union[List[int], vs.VideoNode]:
    """
    Run scene change detection using specified detector.

    Dependencies:
      - vs-placebo: https://github.com/Lypheo/vs-placebo (tonemapping)

      - Detectors (chosen must be available):
            - av-scenechange: https://github.com/rust-av/av-scenechange
            - vapoursynth-wwxd: https://github.com/dubhater/vapoursynth-wwxd
            - vapoursynth-scxvid: https://github.com/dubhater/vapoursynth-scxvid
            - vapoursynth-mvtools: https://github.com/dubhater/vapoursynth-mvtools


    The input clip is expected to be limited range.
    It is scaled down to 270px max width for scene change detection.

    :param tonemap: Tonemap the input clip using libplacebo
        The input clip is expected to be PQ, BT.2020, limited range
    :param brighten: Brighten the clip's gamma
    :param preview: Return the final detection clip
    :param av_sc_cli: Path to `av-scenechange` executable
    :param output: Output file to store the scene change frames

    The output is a list of the scene change frames.
    """
    from .base import DynamicTonemap, zresize

    # Motion vectors work better at higher res
    if detector != SceneChangeDetector.MVTools:
        clip = zresize(clip, preset=1080 / 4, kernel="point")

    if tonemap:
        pl_opts = placebo.PlaceboTonemapOpts(source_colorspace=placebo.PlaceboColorSpace.HDR10,
                                             target_colorspace=placebo.PlaceboColorSpace.SDR,
                                             peak_detect=True,
                                             gamut_mode=placebo.PlaceboGamutMode.Clip,
                                             tone_map_function=placebo.PlaceboTonemapFunction.BT2390,
                                             tone_map_param=2.0,
                                             tone_map_mode=placebo.PlaceboTonemapMode.Hybrid,
                                             use_dovi=False).with_static_peak_detect()
        clip = DynamicTonemap(clip, src_fmt=True, libplacebo=True, target_nits=203, placebo_opts=pl_opts)

    clip = vsutil.depth(clip, 10)

    if brighten:
        clip = clip.std.Levels(gamma=1.50, min_in=64, max_in=940, min_out=64, max_out=940, planes=0)

    if not preview:
        return detector.run_detection(clip, av_sc_cli=av_sc_cli, output=output)

    return clip


def measure_hdr10_content_light_level(clip: vs.VideoNode,
                                      outlier_rejection: bool = True,
                                      downscale: bool = False,
                                      hlg: bool = False,
                                      max_percentile: Optional[float] = None) -> List[HdrMeasurement]:
    """
    Measure the clip to extract the global MaxCLL and MaxFALL brightness values.
    The input clip is expected to be PQ or HLG, BT.2020, limited range

    :param outlier_rejection: Reject outlier pixels by using percentiles
    :param kwargs: Arguments passed to `add_hdr_measurement_props`
    """
    import numpy as np

    from .base import (ST2084_PEAK_LUMINANCE, add_hdr_measurement_props, st2084_eotf)

    measurements: List[HdrMeasurement] = []

    percentile = 100.0
    if outlier_rejection:
        percentile = 99.99

    # Allow overriding frame max percentile
    if max_percentile is not None:
        percentile = max_percentile

    clip = add_hdr_measurement_props(clip,
                                     measurements=measurements,
                                     store_float=True,
                                     as_nits=False,
                                     percentile=percentile,
                                     downscale=downscale,
                                     hlg=hlg,
                                     no_planestats=True)

    def do_it(_detections: Set[int]):
        return clip

    _detect(clip, "HDR10 content light level measurements", do_it, quit_after=False, print_detections=False)

    maxcll_measurement = max(measurements, key=lambda m: m.max)
    maxfall_measurement = max(measurements, key=lambda m: float(m.fall))

    if outlier_rejection:
        maxrgb_values = list(map(lambda m: m.max, measurements))
        fall_values = list(map(lambda m: float(m.fall), measurements))

        maxcll = np.percentile(maxrgb_values, 99.5)
        maxfall = np.percentile(fall_values, 99.75)
    else:
        maxcll = maxcll_measurement.max
        maxfall = maxfall_measurement.fall

    maxcll_nits = st2084_eotf(maxcll) * ST2084_PEAK_LUMINANCE
    maxfall_nits = st2084_eotf(maxfall) * ST2084_PEAK_LUMINANCE

    print(f"\nMaxCLL: {maxcll_nits:0.2f} nits"
          f"\n  Max brightness frame: {maxcll_measurement.frame}, PQ value: {maxcll_measurement.max}")
    print(f"MaxFALL: {maxfall_nits:0.2f} nits"
          f"\n  Frame: {maxfall_measurement.frame}, PQ value: {maxfall_measurement.fall}")
    print(f"Note: Max PQ values according to {percentile}% percentile of the frame's MaxRGB values")

    return measurements


#####################
#      Exports      #
#####################

__all__ = [
    "awf_init_progress_state",
    "awf_vs_out_updated",
    "banddtct",
    "bandmask",
    "brdrdtct",
    "cambidtct",
    "dirtdtct",
    "measure_hdr10_content_light_level",
    "merge_detections",
    "run_scenechange_detect",
    "SceneChangeDetector",
]
