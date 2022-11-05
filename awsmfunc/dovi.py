from typing import Dict, List
import numpy as np

import vapoursynth as vs

from .base import ST2084_PEAK_LUMINANCE, st2084_eotf
from .types.dovi import HdrMeasurement


def generate_dovi_config(clip: vs.VideoNode,
                         measurements: List[HdrMeasurement],
                         scene_changes: List[int],
                         hlg: bool = False,
                         normalized: bool = False,
                         with_l4=False) -> dict:
    """
    Generates a Dolby Vision metadata generation config for `dovi_tool`.

    :param measurements: List of the measured frame brightness info
    :param scene_changes: List of the scene change frames
    :param hlg: Whether the metadata should be generated for an HLG video
    :param normalized: Whether the measurements are already normalized to [0-1]
         - The average is always assumed to be normalized.
    :param with_l4: Compute L4 metadata for every frame.
         - Requires that the measurements have MaxFALL and MaxFALL stdev values.

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

    l4_dict = None
    if with_l4:
        l4_dict = __calc_dovi_l4(clip, measurements, scene_changes, normalized=normalized)

    if scene_changes and scene_changes[0] != 0:
        scene_changes.insert(0, 0)

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

        if not measurements_for_scene:
            continue

        max_measurement = max(measurements_for_scene, key=lambda m: m.max)

        min_pq = max_measurement.min
        max_pq = max_measurement.max
        avg_pq = max_measurement.avg

        if not normalized:
            min_pq /= 65535.0
            max_pq /= 65535.0

        min_pq = int(np.clip(round(min_pq * 4095.0), 0.0, 4095.0))
        max_pq = int(np.clip(round(max_pq * 4095.0), 0.0, 4095.0))
        avg_pq = int(np.clip(round(avg_pq * 4095.0), 0.0, 4095.0))

        shot["metadata_blocks"] = [{
            "Level1": {
                "min_pq": min_pq,
                "max_pq": max_pq,
                "avg_pq": avg_pq,
            }
        }]

        if l4_dict:
            l4_for_shot = {i: l4_dict.get(i) for i in range(shot["start"], end + 1)}

            shot["frame_edits"] = [{
                "edit_offset":
                offset,
                "metadata_blocks": [{
                    "Level4": {
                        "anchor_pq": l4["tf_pq_mean"],
                        "anchor_power": l4["tf_pq_stdev"]
                    }
                }]
            } for offset, (_, l4) in enumerate(l4_for_shot.items())]

        shots.append(shot)

    maxcll = max(measurements, key=lambda m: m.max).max
    if not normalized:
        maxcll /= 65535.0

    maxcll = st2084_eotf(maxcll) * ST2084_PEAK_LUMINANCE
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


def __calc_dovi_l4(clip: vs.VideoNode,
                   measurements: List[HdrMeasurement],
                   scene_changes: List[int],
                   normalized: bool = False) -> Dict:
    framerate = clip.fps

    l4_norm_dict = {}

    measurements = sorted(measurements, key=lambda k: k.frame)
    measurements_dict = dict(enumerate(measurements))

    for i, m in enumerate(measurements):
        scene_boundary = i in scene_changes
        prev_measurement = measurements_dict.get(i - 1)

        if prev_measurement:
            prev_l4 = l4_norm_dict.get(i - 1)
            prev_tf_pq_norm_mean = prev_l4["tf_pq_mean"]
            prev_tf_pq_norm_stdev = prev_l4["tf_pq_stdev"]

            sc = int(scene_boundary)
            prev_maxfall = prev_measurement.fall
            cur_maxfall = m.fall
            cur_max_stdev = m.max_stdev

            if not normalized:
                prev_maxfall /= 65535.0
                cur_maxfall /= 65535.0
                cur_max_stdev /= 65535.0

            mean_diff = sc * abs(cur_maxfall - prev_maxfall) * 8.0 + 0.1
            alpha = min(1, min(1, mean_diff * 24.0 / framerate))

            tf_pq_norm_mean = prev_tf_pq_norm_mean * (1.0 - alpha) + cur_maxfall * alpha
            tf_pq_norm_stdev = prev_tf_pq_norm_stdev * (1.0 - alpha) + cur_max_stdev * alpha
        else:
            tf_pq_norm_mean = 0.0
            tf_pq_norm_stdev = 0.0

        l4_norm_dict[i] = {'tf_pq_mean': tf_pq_norm_mean, 'tf_pq_stdev': tf_pq_norm_stdev}

    return {
        i: {
            "tf_pq_mean": int(np.clip(round(l4_norm["tf_pq_mean"] * 4095.0), 0.0, 4095.0)),
            "tf_pq_stdev": int(np.clip(round(l4_norm["tf_pq_stdev"] * 4095.0), 0.0, 4095.0))
        }
        for i, l4_norm in l4_norm_dict.items()
    }


#####################
#      Exports      #
#####################

__all__ = [
    "generate_dovi_config",
]
