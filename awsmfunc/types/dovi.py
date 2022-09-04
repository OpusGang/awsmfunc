from typing import NamedTuple, Optional

from .misc import st2084_eotf, ST2084_PEAK_LUMINANCE


class HdrMeasurement(NamedTuple):
    """Measurement taken by `awsmfunc.add_hdr_measurement_props`"""
    """Frame number of this measurement"""
    frame: int
    """Min PQ value of the frame"""
    min: float
    """Max PQ value of the frame"""
    max: float
    """Average PQ value of the frame"""
    avg: float
    """Frame average light level, mean of the pixels' MaxRGB"""
    fall: Optional[float]

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

        min = self.min
        max = self.max
        avg = self.avg
        fall = self.fall

        if not normalized:
            min /= 65535.0
            max /= 65535.0

            if fall:
                fall /= 65535.0

        min = st2084_eotf(min) * ST2084_PEAK_LUMINANCE
        max = st2084_eotf(max) * ST2084_PEAK_LUMINANCE
        avg = st2084_eotf(avg) * ST2084_PEAK_LUMINANCE

        if fall:
            fall = st2084_eotf(fall) * ST2084_PEAK_LUMINANCE

        return self._replace(min=min, max=max, avg=avg, fall=fall)
