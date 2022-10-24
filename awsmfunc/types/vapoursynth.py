from enum import IntEnum


class Matrix(IntEnum):
    RGB = 0
    GBR = RGB
    BT709 = 1
    UNKNOWN = 2
    FCC = 4
    BT470BG = 5
    BT601 = BT470BG
    SMPTE170M = 6
    SMPTE240M = 7
    BT2020NC = 9
    BT2020C = 10
    SMPTE2085 = 11
    CHROMA_DERIVED_NC = 12
    CHROMA_DERIVED_C = 13
    ICTCP = 14


class Transfer(IntEnum):
    BT709 = 1
    UNKNOWN = 2
    BT470M = 4
    BT470BG = 5
    BT601 = 6
    ST240M = 7
    LINEAR = 8
    LOG_100 = 9
    LOG_316 = 10
    XVYCC = 11
    SRGB = 13
    BT2020_10bits = 14
    BT2020_12bits = 15
    ST2084 = 16
    ARIB_B67 = 18


class Primaries(IntEnum):
    BT709 = 1
    UNKNOWN = 2
    BT470M = 4
    BT470BG = 5
    ST170M = 6
    ST240M = 7
    FILM = 8
    BT2020 = 9
    ST428 = 10
    ST431_2 = 11
    ST432_1 = 12
    EBU3213E = 22
