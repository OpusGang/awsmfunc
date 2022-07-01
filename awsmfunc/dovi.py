import vapoursynth as vs

import json
import time
from functools import partial
from typing import List, Union
from subprocess import PIPE, Popen

import vsutil

from .base import zresize, DynamicTonemap, st2084_eotf, ST2084_PEAK_LUMINANCE
from .detect import awf_init_progress_state, awf_vs_out_updated

from .types.dovi import HdrMeasurement
from .types import placebo


def run_scenechange_detect(clip: vs.VideoNode,
                           bin_cli: str = "av-scenechange",
                           preview: bool = False) -> Union[List[int], vs.VideoNode]:
    """
    Run scene change detection using `av-scenechange`.
    Requires:
      - av-scenechange: https://github.com/rust-av/av-scenechange
      - vs-placebo: https://github.com/Lypheo/vs-placebo

    The input clip is expected to be PQ, BT.2020, limited range.
    It is scaled down to 270px width, tone mapped to SDR for detection.

    :param bin_cli: Path to `av-scenechange` binary
    :param preview: Return the tonemapped clip

    The output is a list of the scene change frames.
    """

    scenechange_cli = [bin_cli, "-s", "0", "--min-scenecut", "12", "-"]
    pl_opts = placebo.PlaceboTonemapOpts(source_colorspace=placebo.PlaceboColorSpace.HDR10,
                                         target_colorspace=placebo.PlaceboColorSpace.SDR,
                                         peak_detect=True,
                                         gamut_mode=placebo.PlaceboGamutMode.Clip,
                                         tone_map_function=placebo.PlaceboTonemapFunction.BT2390,
                                         tone_map_param=2.0,
                                         tone_map_mode=placebo.PlaceboTonemapMode.Hybrid,
                                         use_dovi=False).with_static_peak_detect()

    clip = zresize(clip, preset=1080 / 4, kernel="point")
    clip = DynamicTonemap(clip, src_fmt=True, libplacebo=True, target_nits=203, placebo_opts=pl_opts)

    clip = vsutil.depth(clip, 10)
    clip = clip.std.Levels(gamma=1.50, min_in=64, max_in=940, min_out=64, max_out=940, planes=0)

    if not preview:
        with Popen(scenechange_cli, stdin=PIPE, stdout=PIPE, stderr=PIPE) as proc:
            state = awf_init_progress_state()
            clip.output(proc.stdin, y4m=True, progress_update=partial(awf_vs_out_updated, state=state))

            start = state["start_time"]
            end = time.monotonic()
            print("\nElapsed: {:0.2f} seconds ({:0.2f} fps)".format(end - start, clip.num_frames / float(end - start)))

            stdout, stderr = proc.communicate()
            if stderr:
                print(stderr)

            result_json = json.loads(stdout.decode("utf-8"))
            scene_changes = result_json["scene_changes"]

            print(f"Scene changes detected: {len(scene_changes)}, speed: {result_json['speed']:.2f} fps")

            return scene_changes

    return clip


def generate_dovi_config(clip: vs.VideoNode,
                         measurements: List[HdrMeasurement],
                         scene_changes: List[int],
                         hlg: bool = False) -> dict:
    """
    Generates a Dolby Vision metadata generation config for `dovi_tool`.

    :param measurements: List of the measured frame brightness info
    :param scene_changes: List of the scene change frames
    :param hlg: Whether the metadata should be generated for an HLG video

    The output is a regular dictionary, it can be dumped to JSON.

    Example:
    >>> measurements = []
    >>> clip = awf.add_hdr_measurement_props(clip, measurements=measurements)
    >>> scene_changes = awf.run_scenechange_detect(clip)
    >>> dovi_generate_config = awf.generate_dovi_config(clip, measurements, scene_changes)
    >>>
    >>> with open("dovi_generate_config.json", "w") as f:
            json.dump(dovi_generate_config, f, indent=4)
    """

    shots = []
    for i, scene_change in enumerate(scene_changes):
        shot = {
            "start": scene_change,
        }

        if i == len(scene_changes) - 1:
            end = clip.num_frames - 1
        else:
            end = scene_changes[i + 1] - 1

        shot["duration"] = end - shot["start"] + 1

        measurements_for_scene = [m for m in measurements if m.frame >= shot["start"] and m.frame <= end]
        max_measurement = max(measurements_for_scene, key=lambda m: m.max)

        min_pq = int(round((max_measurement.min / 65535) * 4095.0))
        max_pq = int(round((max_measurement.max / 65535) * 4095.0))
        avg_pq = int(round(max_measurement.avg * 4095.0))

        shot["metadata_blocks"] = [{
            "Level1": {
                "min_pq": min_pq,
                "max_pq": max_pq,
                "avg_pq": avg_pq,
            }
        }]
        shots.append(shot)

    maxcll = st2084_eotf(max(measurements, key=lambda m: m.max).max / 65535.0) * ST2084_PEAK_LUMINANCE
    maxfall = st2084_eotf(max(measurements, key=lambda m: m.avg).avg) * ST2084_PEAK_LUMINANCE

    shots = sorted(shots, key=lambda s: s["start"])

    profile = "8.1"

    if hlg:
        profile = "8.4"

    return {
        "cm_version": "V40",
        "profile": profile,
        "length": clip.num_frames,
        "level6": {
            "max_display_mastering_luminance": 1000,
            "min_display_mastering_luminance": 1,
            "max_content_light_level": round(maxcll),
            "max_frame_average_light_level": round(maxfall)
        },
        "shots": shots,
    }


#####################
#      Exports      #
#####################

__all__ = [
    "generate_dovi_config",
    "run_scenechange_detect",
]
