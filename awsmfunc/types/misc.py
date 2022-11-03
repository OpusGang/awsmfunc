import math
from typing import TypedDict

ST2084_PEAK_LUMINANCE = 10000
ST2084_M1 = 2610.0 / 16384.0
ST2084_M2 = (2523.0 / 4096.0) * 128.0
ST2084_C1 = 3424.0 / 4096.0
ST2084_C2 = (2413.0 / 4096.0) * 32.0
ST2084_C3 = (2392.0 / 4096.0) * 32.0


class DetectProgressState(TypedDict):
    start_time: float
    frames_done: int
    fps: float
    last_fps_report_time: float


def st2084_eotf(x: float) -> float:
    y = float(0.0)
    if x > 0.0:
        xpow = math.pow(x, float(1.0) / ST2084_M2)
        num = max(xpow - ST2084_C1, float(0.0))
        den = max(ST2084_C2 - ST2084_C3 * xpow, float('-inf'))
        y = float(math.pow(num / den, float(1.0) / ST2084_M1))

    return y


def st2084_inverse_eotf(x: float) -> float:
    y = x / ST2084_PEAK_LUMINANCE

    return math.pow((ST2084_C1 + (ST2084_C2 * math.pow(y, ST2084_M1))) / (1 + (ST2084_C3 * math.pow(y, ST2084_M1))),
                    ST2084_M2)
