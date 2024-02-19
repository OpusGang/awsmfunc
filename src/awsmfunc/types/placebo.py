from enum import Enum, IntEnum
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
    ST2094_40 = 2
    ST2094_10 = 3
    BT2390 = 4
    BT2446a = 5
    Spline = 6
    Reinhard = 7
    Mobius = 8
    Hable = 9
    Gamma = 10
    Linear = 11


class PlaceboTonemapFunctionName(str, Enum):
    Auto = "auto"
    Clip = "clip"
    ST2094_40 = "st2094-40"
    ST2094_10 = "st2094-10"
    BT2390 = "bt2390"
    BT2446a = "bt2446a"
    Spline = "spline"
    Reinhard = "reinhard"
    Mobius = "mobius"
    Hable = "hable"
    Gamma = "gamma"
    Linear = "linear"


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


class PlaceboHdrMetadataType(IntEnum):
    Any = 0
    """Use any available metadata"""
    PlNone = 1
    """PL_HDR_METADATA_NONE"""
    HDR10 = 2
    """Static HDR10 metadata only"""
    HDR10Plus = 3
    """Dynamic HDR10+ metadata, scene_max and scene_avg"""
    CIE_Y = 4
    """Dynamic PQ luminance metadata, max_pq_y and avg_pq_y"""


class PlaceboTonemapOpts(NamedTuple):
    """Options for vs-placebo Tonemap. For use with awsmfunc.DynamicTonemap.

    Attributes:
        `source_colorspace`: Input clip colorpsace. Defaults to HDR10 (PQ + BT.2020)
        `target_colorspace`: Output clip colorpsace. Defaults to SDR (BT.1886 + BT.709)
        `dst_max`: Output maximum brightness. Defaults to 203 nits.
        `dst_min`: Output minimum brightness. Defaults to 1000:1 contrast.
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

    target_primaries: Optional[int] = None
    """Target color primaries"""

    dst_max: float = 203.0
    """Target peak brightness, in nits"""
    dst_min: float = dst_max / 1000
    """Target black point, in nits"""

    peak_detect: bool = True
    """Use libplacebo's dynamic peak detection instead of FrameEval"""
    gamut_mode: Optional[PlaceboGamutMode] = None
    """How to handle out-of-gamut colors when changing the content primaries"""
    tone_map_function: Optional[PlaceboTonemapFunction] = None
    """Tone map function to use for luma"""
    tone_map_function_s: Optional[PlaceboTonemapFunctionName] = None
    """Tone map function to use for luma, string name version"""
    tone_map_param: Optional[float] = None
    """Parameter for the tone map function"""
    tone_map_mode: Optional[PlaceboTonemapMode] = None
    """Tone map mode to map colours/chroma"""
    tone_map_crosstalk: Optional[float] = None
    """Extra crosstalk factor to apply before tone mapping"""
    hdr_metadata_type: Optional[PlaceboHdrMetadataType] = None
    """HDR metadata type to use"""

    use_dovi: Optional[bool] = None
    """Whether to use the Dolby Vision RPU for ST2086 metadata

        This does not do extra processing, only uses the RPU for extra metadata
    """

    smoothing_period: Optional[float] = None
    """Scene change smoothing period for peak detection"""
    scene_threshold_low: Optional[float] = None
    scene_threshold_high: Optional[float] = None

    percentile: Optional[float] = None
    """Percentile to use for detected peak"""
    show_clipping: Optional[bool] = None
    """Highlight hard-clipped pixels from tone-mapping"""
    visualize_lut: Optional[bool] = None
    """Display a (PQ-PQ) graph of the active tone-mapping LUT"""

    use_planestats: bool = True
    """When peak detection is disabled, whether to use the frame max RGB for tone mapping"""

    def with_static_peak_detect(self):
        """Ignore peak detect smoothing and scene detection"""
        return self._replace(smoothing_period=0, scene_threshold_low=0, scene_threshold_high=0)

    def vsplacebo_dict(self) -> Dict:
        all_args = {
            "src_csp": self.source_colorspace,
            "dst_csp": self.target_colorspace,
            "dst_prim": self.target_primaries,
            "dst_max": self.dst_max,
            "dst_min": self.dst_min,
            "dynamic_peak_detection": self.peak_detect,
            "gamut_mode": self.gamut_mode,
            "tone_mapping_function": self.tone_map_function,
            "tone_mapping_function_s": self.tone_map_function_s,
            "tone_mapping_param": self.tone_map_param,
            "tone_mapping_mode": self.tone_map_mode,
            "tone_mapping_crosstalk": self.tone_map_crosstalk,
            "use_dovi": self.use_dovi,
            "smoothing_period": self.smoothing_period,
            "scene_threshold_low": self.scene_threshold_low,
            "scene_threshold_high": self.scene_threshold_high,
            "show_clipping": self.show_clipping,
            "percentile": self.percentile,
            "metadata": self.hdr_metadata_type,
            "visualize_lut": self.visualize_lut,
        }

        return {k: v for k, v in all_args.items() if v is not None}

    def is_dovi_src(self) -> bool:
        """Whether the options process the clip as Dolby Vision"""
        return self.source_colorspace == PlaceboColorSpace.DOVI

    def is_hdr10_src(self) -> bool:
        return self.source_colorspace == PlaceboColorSpace.HDR10

    def is_sdr_target(self) -> bool:
        return self.target_colorspace == PlaceboColorSpace.SDR
