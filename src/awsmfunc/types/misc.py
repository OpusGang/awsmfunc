import math
from typing import NamedTuple, TypedDict

ST2084_PEAK_LUMINANCE = 10000
ST2084_M1 = 2610.0 / 16384.0
ST2084_M2 = (2523.0 / 4096.0) * 128.0
ST2084_C1 = 3424.0 / 4096.0
ST2084_C2 = (2413.0 / 4096.0) * 32.0
ST2084_C3 = (2392.0 / 4096.0) * 32.0


def st2084_eotf(x: float) -> float:
    y = float(0.0)
    if x > 0.0:
        xpow = math.pow(x, float(1.0) / ST2084_M2)
        num = max(xpow - ST2084_C1, float(0.0))
        den = max(ST2084_C2 - ST2084_C3 * xpow, float("-inf"))
        y = float(math.pow(num / den, float(1.0) / ST2084_M1))

    return y


def st2084_inverse_eotf(x: float) -> float:
    y = x / ST2084_PEAK_LUMINANCE

    return math.pow(
        (ST2084_C1 + (ST2084_C2 * math.pow(y, ST2084_M1))) / (1 + (ST2084_C3 * math.pow(y, ST2084_M1))), ST2084_M2
    )


class DetectProgressState(TypedDict):
    start_time: float
    frames_done: int
    fps: float
    last_fps_report_time: float


class Hdr10PlusHistogram(NamedTuple):
    percentile_1: float
    """1% percentile maxRGB value"""
    distribution_y_99: float
    """99.99% percentile maxRGB value"""
    distribution_y_100nit: float
    """% of pixels below or equal to 100 nits"""
    percentile_25: float
    """25% percentile maxRGB value"""
    percentile_50: float
    """50% percentile maxRGB value"""
    percentile_75: float
    """75% percentile maxRGB value"""
    percentile_90: float
    """90% percentile maxRGB value"""
    percentile_95: float
    """95% percentile maxRGB value"""
    percentile_99_98: float
    """99.98% percentile maxRGB value"""

    def to_nits(self, inverse: bool = False):
        def _convert_fn(value: float) -> float:
            if inverse:
                return st2084_inverse_eotf(value)

            return st2084_eotf(value) * ST2084_PEAK_LUMINANCE

        percentiles = self.to_list()
        distribution_y_100nit = percentiles.pop(2)

        percentiles = [_convert_fn(x) for x in percentiles]

        return Hdr10PlusHistogram(
            percentile_1=percentiles[0],
            distribution_y_99=percentiles[1],
            distribution_y_100nit=distribution_y_100nit,
            percentile_25=percentiles[2],
            percentile_50=percentiles[3],
            percentile_75=percentiles[4],
            percentile_90=percentiles[5],
            percentile_95=percentiles[6],
            percentile_99_98=percentiles[7],
        )

    def to_list(self) -> list[float]:
        return [
            self.percentile_1,
            self.distribution_y_99,
            self.distribution_y_100nit,
            self.percentile_25,
            self.percentile_50,
            self.percentile_75,
            self.percentile_90,
            self.percentile_95,
            self.percentile_99_98,
        ]

    @classmethod
    def from_list(cls, percentiles: list[float]):
        return cls(
            percentile_1=percentiles[0],
            distribution_y_99=percentiles[1],
            distribution_y_100nit=percentiles[2],
            percentile_25=percentiles[3],
            percentile_50=percentiles[4],
            percentile_75=percentiles[5],
            percentile_90=percentiles[6],
            percentile_95=percentiles[7],
            percentile_99_98=percentiles[8],
        )

    def to_hdr10plus_distribution(self, convert_nits=True):
        distribution = self.to_nits().to_list() if convert_nits else self.to_list()

        # DistributionY100nit as percentage, rounded
        distribution_y_100nit = int(round(distribution.pop(2) * 100.0))

        # Scale to 100_000 max, re-add DistributionY100nit
        distribution = [int(round(x * 10.0)) for x in distribution]
        distribution.insert(2, distribution_y_100nit)

        return distribution
