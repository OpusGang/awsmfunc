from typing import NamedTuple, Optional

from .misc import ST2084_PEAK_LUMINANCE, Hdr10PlusHistogram, st2084_eotf, st2084_inverse_eotf


class HdrMeasurement(NamedTuple):
    """
    Measurement taken by `awsmfunc.add_hdr_measurement_props`
    """

    frame: int
    """Frame number of this measurement"""
    min: float
    """Min PQ value of the frame"""
    max: float
    """Max PQ value of the frame"""
    avg: float
    """Average PQ value of the frame"""
    fall: Optional[float]
    """Frame average light level, mean of the pixels' MaxRGB"""
    max_stdev: Optional[float]
    """Standard deviation of the frame's MaxRGB"""

    hdr10plus_maxscl: Optional[list[float]]
    hdr10plus_histogram: Optional[Hdr10PlusHistogram]
    """HDR10+ histogram metadata for the frame"""

    def human_readable_str(self, precision=6):
        formatted_str = (
            f"{self.frame}, min: {self.min:0.{precision}f}, max: {self.max:0.{precision}f}, "
            f"avg: {self.avg:0.{precision}f}"
        )

        if self.fall is not None:
            formatted_str += f", MaxFALL: {self.fall:0.{precision}f}"
        if self.max_stdev is not None:
            formatted_str += f", MaxStd: {self.max_stdev:0.{precision}f}"
        if self.hdr10plus_maxscl is not None and self.hdr10plus_histogram is not None:
            maxscl = [f"{x:0.{precision}f}" for x in self.hdr10plus_maxscl]
            histogram_list = [f"{x:0.{precision}f}" for x in self.hdr10plus_histogram.to_list()]
            formatted_str += f", HDR10+ MaxSCL: [{', '.join(maxscl)}], histogram: [{', '.join(histogram_list)}]"

        return formatted_str

    def __str__(self):
        return self.human_readable_str()

    def to_nits(self, normalized: bool = False, inverse: bool = False):
        """
        Returns the measurements converted to nits (cd/m^2)

        :param normalized: Whether the measurements are already normalized to [0-1]
          - The average is always assumed to be normalized.
        :param inverse: Convert from nits to PQ code values instead.
        """

        def _convert_fn(value: float) -> float:
            if inverse:
                return st2084_inverse_eotf(value)

            return st2084_eotf(value) * ST2084_PEAK_LUMINANCE

        min_v = self.min
        max_v = self.max
        avg = self.avg
        fall = self.fall
        max_stdev = self.max_stdev

        hdr10plus_maxscl = self.hdr10plus_maxscl
        hdr10plus_histogram = self.hdr10plus_histogram

        if not normalized and not inverse:
            min_v /= 65535.0
            max_v /= 65535.0

            if fall is not None:
                fall /= 65535.0
            if max_stdev is not None:
                max_stdev /= 65535.0

        min_v = _convert_fn(min_v)
        max_v = _convert_fn(max_v)
        avg = _convert_fn(avg)

        if fall is not None:
            fall = _convert_fn(fall)
        if max_stdev is not None:
            max_stdev = _convert_fn(max_stdev)
        if hdr10plus_maxscl is not None:
            hdr10plus_maxscl = [_convert_fn(x) for x in hdr10plus_maxscl]
        if hdr10plus_histogram is not None:
            hdr10plus_histogram = hdr10plus_histogram.to_nits(inverse=inverse)

        return HdrMeasurement(
            frame=self.frame,
            min=min_v,
            max=max_v,
            avg=avg,
            fall=fall,
            max_stdev=max_stdev,
            hdr10plus_maxscl=hdr10plus_maxscl,
            hdr10plus_histogram=hdr10plus_histogram,
        )
