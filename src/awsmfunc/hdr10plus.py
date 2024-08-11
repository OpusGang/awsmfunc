from importlib.metadata import version
from typing import Any, List

import vapoursynth as vs

from .base import ST2084_PEAK_LUMINANCE, st2084_eotf
from .types.dovi import HdrMeasurement

HDR10PLUS_DISTRIBUTION_INDEX_LIST = [1, 5, 10, 25, 50, 75, 90, 95, 99]


def generate_hdr10plus_json(
    clip: vs.VideoNode,
    measurements: List[HdrMeasurement],
    scene_changes: List[int],
) -> dict:
    """
    Generates a HDR10+ metadata JSON file.
    For the average brightness, FALL is prioritized over regular Average

    :param measurements: List of the measured frame brightness info
    :param scene_changes: List of the scene change frames

    Example:
    >>> measurements = []
    >>> clip = awf.add_hdr_measurement_props(
    >>>     clip,
    >>>     measurements=measurements,
    >>>     no_planestats=True,
    >>>     as_nits=False,
    >>>     compute_hdr10plus=True
    >>> )
    >>> scene_changes = awf.run_scenechange_detect(clip)
    >>> hdr10plus_json = awf.generate_hdr10plus_json(clip, measurements, scene_changes)
    >>>
    >>> with open("hdr10plus_metadata.json", "w") as f:
            json.dump(hdr10plus_json, f, indent=4)
    """

    shots = []

    if scene_changes and scene_changes[0] != 0:
        scene_changes.insert(0, 0)

    for i, scene_change in enumerate(scene_changes):
        shot: Any = {
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

        if any(m.hdr10plus_maxscl is None or m.hdr10plus_histogram is None for m in measurements_for_scene):
            raise Exception

        max_measurement = max(measurements_for_scene, key=lambda m: m.hdr10plus_histogram.distribution_y_99)
        if max_measurement.hdr10plus_maxscl is None or max_measurement.hdr10plus_histogram is None:
            raise Exception

        measurements_len = float(len(measurements_for_scene))

        average_maxrgb_pq = (
            sum(m.fall if m.fall is not None else 0.0 for m in measurements_for_scene) / measurements_len
        )
        shot["average_maxrgb"] = int(round(st2084_eotf(average_maxrgb_pq) * ST2084_PEAK_LUMINANCE) * 10)

        # Histogram for the brightest frame in shot
        distribution_values = max_measurement.hdr10plus_histogram.to_hdr10plus_distribution()

        # DistributionY100nit value is an average value of DistributionY100nit values within a scene
        distribution_y_100nit_sum = sum(
            m.hdr10plus_histogram.distribution_y_100nit * 100.0 for m in measurements_for_scene
        )
        distribution_values[2] = int(round(distribution_y_100nit_sum / measurements_len))

        shot["distribution_values"] = distribution_values
        shot["maxscl"] = [
            int(round(st2084_eotf(c) * ST2084_PEAK_LUMINANCE) * 10) for c in max_measurement.hdr10plus_maxscl
        ]

        shots.append(shot)

    shots = sorted(shots, key=lambda s: s["start"])
    scene_first_frame_indices = [shot["start"] for shot in shots]
    scene_frame_numbers = [shot["duration"] for shot in shots]

    scene_info = []

    sequence_frame_index = 0
    for shot_index, shot in enumerate(shots):
        for shot_frame_index in range(0, shot["duration"]):
            frame_info = {
                "LuminanceParameters": {
                    "AverageRGB": shot["average_maxrgb"],
                    "LuminanceDistributions": {
                        "DistributionIndex": HDR10PLUS_DISTRIBUTION_INDEX_LIST,
                        "DistributionValues": shot["distribution_values"],
                    },
                    "MaxScl": shot["maxscl"],
                },
                "NumberOfWindows": 1,
                "TargetedSystemDisplayMaximumLuminance": 0,
                "SceneFrameIndex": shot_frame_index,
                "SceneId": shot_index,
                "SequenceFrameIndex": sequence_frame_index,
            }
            sequence_frame_index += 1

            scene_info.append(frame_info)

    return {
        "JSONInfo": {"HDR10plusProfile": "A", "Version": "1.0"},
        "SceneInfo": scene_info,
        "SceneInfoSummary": {
            "SceneFirstFrameIndex": scene_first_frame_indices,
            "SceneFrameNumbers": scene_frame_numbers,
        },
        "ToolInfo": {"Tool": "awsmfunc", "Version": version("awsmfunc")},
    }


#####################
#      Exports      #
#####################

__all__ = [
    "generate_hdr10plus_json",
]
