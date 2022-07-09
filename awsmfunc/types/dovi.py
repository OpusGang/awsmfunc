from typing import NamedTuple, Optional


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
