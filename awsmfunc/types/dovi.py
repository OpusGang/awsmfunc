from typing import NamedTuple, Optional

from .misc import st2084_eotf, ST2084_PEAK_LUMINANCE


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

    def __str__(self):
        formatted_str = f'{self.frame}, min: {self.min:0.6f}, max: {self.max:0.6f}, avg: {self.avg:0.6f}'
        if self.fall:
            formatted_str += f', MaxFALL: {self.fall:0.6f}'

        return formatted_str

    def to_nits(self, normalized: bool = False):
        """
        Returns the measurements converted to nits (cd/m^2)

        :param normalized: Whether the measurements are already normalized to [0-1]
          - The average is always assumed to be normalized.
        """

        min_v = self.min
        max_v = self.max
        avg = self.avg
        fall = self.fall

        if not normalized:
            min_v /= 65535.0
            max_v /= 65535.0

            if fall:
                fall /= 65535.0

        min_v = st2084_eotf(min_v) * ST2084_PEAK_LUMINANCE
        max_v = st2084_eotf(max_v) * ST2084_PEAK_LUMINANCE
        avg = st2084_eotf(avg) * ST2084_PEAK_LUMINANCE

        if fall:
            fall = st2084_eotf(fall) * ST2084_PEAK_LUMINANCE

        # pylint: disable-next=no-member
        return self._replace(min=min_v, max=max_v, avg=avg, fall=fall)
