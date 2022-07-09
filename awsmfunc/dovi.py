import vapoursynth as vs

from typing import List

from .base import st2084_eotf, ST2084_PEAK_LUMINANCE
from .types.dovi import HdrMeasurement


def generate_dovi_config(clip: vs.VideoNode,
                         measurements: List[HdrMeasurement],
                         scene_changes: List[int],
                         hlg: bool = False,
                         normalized: bool = False) -> dict:
    """
    Generates a Dolby Vision metadata generation config for `dovi_tool`.

    :param measurements: List of the measured frame brightness info
    :param scene_changes: List of the scene change frames
    :param hlg: Whether the metadata should be generated for an HLG video
    :param normalized: Whether the measurements are already normalized to [0-1]
         - The average is always assumed to be normalized.

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
        max_measurement = max(measurements_for_scene, key=lambda m: m.max)

        if not normalized:
            min_pq = int(round((max_measurement.min / 65535) * 4095.0))
            max_pq = int(round((max_measurement.max / 65535) * 4095.0))
            avg_pq = int(round(max_measurement.avg * 4095.0))
        else:
            min_pq = int(round(max_measurement.min * 4095.0))
            max_pq = int(round(max_measurement.max * 4095.0))
            avg_pq = int(round(max_measurement.avg * 4095.0))

        shot["metadata_blocks"] = [{
            "Level1": {
                "min_pq": min_pq,
                "max_pq": max_pq,
                "avg_pq": avg_pq,
            }
        }]
        shots.append(shot)

    if not normalized:
        maxcll = st2084_eotf(max(measurements, key=lambda m: m.max).max / 65535.0) * ST2084_PEAK_LUMINANCE
    else:
        maxcll = st2084_eotf(max(measurements, key=lambda m: m.max).max) * ST2084_PEAK_LUMINANCE

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
]
