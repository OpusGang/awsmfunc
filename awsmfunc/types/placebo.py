from enum import IntEnum
from typing import Dict, NamedTuple, Optional


class PlaceboColorSpace(IntEnum):
    SDR = 0
    HDR10 = 1
    HLG = 2
    DOVI = 3
    """Process profile 5/7/8 Dolby Vision"""


class PlaceboTonemapFunction(IntEnum):
    Auto = 0
    Clip = 1
    BT2390 = 2
    BT2446a = 3
    Spline = 4
    Reinhard = 5
    Mobius = 6
    Hable = 7
    Gamma = 8
    Linear = 9


class PlaceboGamutMode(IntEnum):
    Clip = 0
    Warn = 1
    Darken = 2
    Desaturate = 3


class PlaceboTonemapMode(IntEnum):
    Auto = 0
    RGB = 1
    Max = 2
    Hybrid = 3
    Luma = 4


class PlaceboTonemapOpts(NamedTuple):
    """Options for vs-placebo Tonemap. For use with awsmfunc.DynamicTonemap.

    Attributes:
        `source_colorspace`: Input clip colorpsace. Defaults to HDR10 (PQ + BT.2020)
        `target_colorspace`: Output clip colorpsace. Defaults to SDR (BT.1886 + BT.709)
        `peak_detect`: Use libplacebo's dynamic peak detection instead of FrameEval
        `gamut_mode`: How to handle out-of-gamut colors when changing the content primaries
        `tone_map_function`: Tone map function to use for luma
        `tone_map_param`: Parameter for the tone map function
        `tone_map_mode`: Tone map mode to map colours/chroma
        `tone_map_crosstalk`: Extra crosstalk factor to apply before tone mapping
        `use_dovi`: Whether to use the Dolby Vision RPU for ST2086 metadata
            This does not do extra processing, only uses the RPU for extra metadata
        `smoothing_period`, `scene_threshold_low`, `scene_threshold_high`: Peak detection parameters
        `use_planestats`: When peak detection is disabled, whether to use the frame max RGB for tone mapping
    """

    source_colorspace: PlaceboColorSpace = PlaceboColorSpace.HDR10
    """Input clip colorpsace. Defaults to HDR10 (PQ + BT.2020)"""
    target_colorspace: PlaceboColorSpace = PlaceboColorSpace.SDR
    """Output clip colorpsace. Defaults to SDR (BT.1886 + BT.709)"""

    peak_detect: bool = True
    """Use libplacebo's dynamic peak detection instead of FrameEval"""
    gamut_mode: Optional[PlaceboGamutMode] = None
    """How to handle out-of-gamut colors when changing the content primaries"""
    tone_map_function: Optional[PlaceboTonemapFunction] = None
    """Tone map function to use for luma"""
    tone_map_param: Optional[float] = None
    """Parameter for the tone map function"""
    tone_map_mode: Optional[PlaceboTonemapMode] = None
    """Tone map mode to map colours/chroma"""
    tone_map_crosstalk: Optional[float] = None
    """Extra crosstalk factor to apply before tone mapping"""
    use_dovi: Optional[bool] = None
    """Whether to use the Dolby Vision RPU for ST2086 metadata

        This does not do extra processing, only uses the RPU for extra metadata
    """

    smoothing_period: Optional[float] = None
    scene_threshold_low: Optional[float] = None
    scene_threshold_high: Optional[float] = None

    use_planestats: bool = True
    """When peak detection is disabled, whether to use the frame max RGB for tone mapping"""

    def with_static_peak_detect(self):
        """Ignore peak detect smoothing and scene detection"""
        # pylint: disable-next=no-member
        return self._replace(smoothing_period=-1, scene_threshold_low=-1, scene_threshold_high=-1)

    def vsplacebo_dict(self) -> Dict:
        return {
            'src_csp': self.source_colorspace,
            'dst_csp': self.target_colorspace,
            'dynamic_peak_detection': self.peak_detect,
            'gamut_mode': self.gamut_mode,
            'tone_mapping_function': self.tone_map_function,
            'tone_mapping_param': self.tone_map_param,
            'tone_mapping_mode': self.tone_map_mode,
            'tone_mapping_crosstalk': self.tone_map_crosstalk,
            'use_dovi': self.use_dovi,
            'smoothing_period': self.smoothing_period,
            'scene_threshold_low': self.scene_threshold_low,
            'scene_threshold_high': self.scene_threshold_high,
        }

    def is_dovi_src(self) -> bool:
        """Whether the options process the clip as Dolby Vision"""
        return self.source_colorspace == PlaceboColorSpace.DOVI

    def is_hdr10_src(self) -> bool:
        return self.source_colorspace == PlaceboColorSpace.HDR10

    def is_sdr_target(self) -> bool:
        return self.target_colorspace == PlaceboColorSpace.SDR
